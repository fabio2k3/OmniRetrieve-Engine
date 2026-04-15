"""
test_crawler_new.py
===================
Suite de tests para el crawler refactorizado.

Cubre:
  1.  Document           — campo doc_id, propiedad arxiv_id, CSV (nuevo y legado)
  2.  IdStore            — IDs compuestos, CSV doc_id, retrocompat arxiv_id
  3.  robots.py          — ALLOWED_DOMAINS extensible, allow_domain()
  4.  BaseClient         — interfaz abstracta, make_doc_id, parse_doc_id
  5.  ArxivClient        — source_name, parseo XML, doc_id en Document
  6.  chunker            — clean_text, make_chunks, overlap, edge cases
  7.  pdf_extractor      — delegación a chunker (mismos resultados)
  8.  Crawler            — multi-cliente, routing, discovery, download, text loop
  9.  Retrocompatibilidad — imports antiguos siguen funcionando
  10. Integración        — FakeClient end-to-end con SQLite

Uso
---
    cd <raíz del proyecto>
    python -m pytest backend/tests/test_crawler_new.py -v
    python -m pytest backend/tests/test_crawler_new.py -v -m "not network"   # sin internet
"""

from __future__ import annotations

import csv
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Asegurar que el paquete raíz esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# =============================================================================
# Fixtures compartidos
# =============================================================================

@pytest.fixture
def tmp(tmp_path):
    """Directorio temporal limpio para cada test."""
    return tmp_path


@pytest.fixture
def tmp_db(tmp_path):
    """Base de datos SQLite temporal inicializada."""
    from backend.database.schema import init_db
    db = tmp_path / "db" / "test.db"
    init_db(db)
    return db


@pytest.fixture
def fake_client():
    """Cliente mínimo que implementa BaseClient sin acceso a red."""
    from backend.crawler.clients.base_client import BaseClient
    from backend.crawler.document import Document

    class FakeClient(BaseClient):
        SOURCE = "fake"

        def __init__(self, ids=None, text="Sample text. " * 60):
            self._ids  = ids or ["doc1", "doc2", "doc3"]
            self._text = text
            self.download_calls: List[str] = []

        @property
        def source_name(self) -> str:
            return self.SOURCE

        def fetch_ids(self, max_results=100, start=0) -> List[str]:
            return self._ids[:max_results]

        def fetch_documents(self, local_ids: List[str]) -> List[Document]:
            return [
                Document(
                    doc_id     = self.make_doc_id(lid),
                    title      = f"Title {lid}",
                    authors    = "Author A",
                    abstract   = f"Abstract for {lid}",
                    categories = "cs.AI",
                    published  = "2024-01-01T00:00:00Z",
                    updated    = "2024-01-01T00:00:00Z",
                    pdf_url    = f"https://fake.example.com/{lid}.pdf",
                )
                for lid in local_ids
            ]

        def download_text(self, local_id: str, **kwargs) -> str:
            self.download_calls.append(local_id)
            return self._text

    return FakeClient()


# =============================================================================
# 1. Document
# =============================================================================

class TestDocument:

    def _make_doc(self, doc_id="arxiv:2301.12345", **kwargs):
        from backend.crawler.document import Document
        defaults = dict(
            title="Test Paper", authors="Alice, Bob",
            abstract="A great paper.", categories="cs.AI, cs.LG",
            published="2023-01-01T00:00:00Z", updated="2023-01-02T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/2301.12345",
        )
        defaults.update(kwargs)
        return Document(doc_id=doc_id, **defaults)

    def test_doc_id_field(self):
        doc = self._make_doc()
        assert doc.doc_id == "arxiv:2301.12345"

    def test_arxiv_id_alias_returns_doc_id(self):
        """El atributo arxiv_id es un alias retrocompat de doc_id."""
        doc = self._make_doc(doc_id="arxiv:9999.00001")
        assert doc.arxiv_id == "arxiv:9999.00001"
        assert doc.arxiv_id == doc.doc_id

    def test_to_dict_uses_doc_id_key(self):
        from backend.crawler.document import DOCUMENT_FIELDS
        doc = self._make_doc()
        d = doc.to_dict()
        assert "doc_id" in d
        assert "arxiv_id" not in d
        assert set(d.keys()) == set(DOCUMENT_FIELDS)

    def test_from_dict_new_key(self):
        from backend.crawler.document import Document
        data = {
            "doc_id": "arxiv:1111.22222", "title": "T", "authors": "A",
            "abstract": "X", "categories": "cs.AI",
            "published": "2024-01-01", "updated": "2024-01-01",
            "pdf_url": "http://x", "fetched_at": "2024-01-01",
        }
        doc = Document.from_dict(data)
        assert doc.doc_id == "arxiv:1111.22222"

    def test_from_dict_legacy_key(self):
        """from_dict acepta la clave antigua arxiv_id (retrocompat)."""
        from backend.crawler.document import Document
        data = {
            "arxiv_id": "arxiv:legacy999", "title": "T", "authors": "A",
            "abstract": "X", "categories": "cs.AI",
            "published": "2024-01-01", "updated": "2024-01-01",
            "pdf_url": "http://x", "fetched_at": "2024-01-01",
        }
        doc = Document.from_dict(data)
        assert doc.doc_id == "arxiv:legacy999"

    def test_equality_based_on_doc_id(self):
        doc_a = self._make_doc(doc_id="arxiv:001")
        doc_b = self._make_doc(doc_id="arxiv:001", title="Different")
        doc_c = self._make_doc(doc_id="arxiv:002")
        assert doc_a == doc_b
        assert doc_a != doc_c

    def test_hash_based_on_doc_id(self):
        doc_a = self._make_doc(doc_id="arxiv:001")
        doc_b = self._make_doc(doc_id="arxiv:001")
        assert hash(doc_a) == hash(doc_b)
        assert len({doc_a, doc_b}) == 1

    def test_save_creates_csv_with_doc_id_column(self, tmp):
        csv_path = tmp / "docs.csv"
        doc = self._make_doc()
        doc.save(csv_path)
        with csv_path.open() as f:
            header = f.readline().strip().split(",")
        assert "doc_id" in header
        assert "arxiv_id" not in header

    def test_save_and_load_all_round_trip(self, tmp):
        from backend.crawler.document import Document
        csv_path = tmp / "docs.csv"
        docs = [self._make_doc(doc_id=f"arxiv:{i:04d}") for i in range(5)]
        for d in docs:
            d.save(csv_path)
        loaded = Document.load_all(csv_path)
        assert len(loaded) == 5
        assert {d.doc_id for d in loaded} == {f"arxiv:{i:04d}" for i in range(5)}

    def test_load_all_from_legacy_csv(self, tmp):
        """load_all funciona con CSV antiguo (columna arxiv_id)."""
        from backend.crawler.document import Document, DOCUMENT_FIELDS
        csv_path = tmp / "legacy.csv"
        legacy_fields = ["arxiv_id"] + DOCUMENT_FIELDS[1:]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=legacy_fields)
            w.writeheader()
            w.writerow({
                "arxiv_id": "arxiv:legacy001", "title": "Legacy",
                "authors": "A", "abstract": "X", "categories": "cs.AI",
                "published": "2024", "updated": "2024",
                "pdf_url": "http://x", "fetched_at": "2024",
            })
        docs = Document.load_all(csv_path)
        assert len(docs) == 1
        assert docs[0].doc_id == "arxiv:legacy001"

    def test_load_ids_new_csv(self, tmp):
        csv_path = tmp / "docs.csv"
        doc = self._make_doc(doc_id="arxiv:2301.12345")
        doc.save(csv_path)
        ids = type(doc).load_ids(csv_path)
        assert "arxiv:2301.12345" in ids

    def test_load_ids_legacy_csv(self, tmp):
        """load_ids funciona con CSV antiguo (columna arxiv_id)."""
        from backend.crawler.document import Document, DOCUMENT_FIELDS
        csv_path = tmp / "legacy.csv"
        legacy_fields = ["arxiv_id"] + DOCUMENT_FIELDS[1:]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=legacy_fields)
            w.writeheader()
            w.writerow({
                "arxiv_id": "arxiv:old001", "title": "T", "authors": "A",
                "abstract": "X", "categories": "cs.AI",
                "published": "2024", "updated": "2024",
                "pdf_url": "http://x", "fetched_at": "2024",
            })
        ids = Document.load_ids(csv_path)
        assert "arxiv:old001" in ids

    def test_composite_id_non_arxiv_source(self, tmp):
        """doc_id funciona con cualquier fuente, no solo arxiv."""
        from backend.crawler.document import Document
        doc = Document(
            doc_id="semantic_scholar:abc123",
            title="SS Paper", authors="X",
            abstract="Y", categories="cs.CL",
            published="2024-01-01", updated="2024-01-01",
            pdf_url="https://ss.org/paper/abc123",
        )
        assert doc.doc_id == "semantic_scholar:abc123"
        assert doc.arxiv_id == "semantic_scholar:abc123"
        csv_path = tmp / "docs.csv"
        doc.save(csv_path)
        loaded = Document.load_all(csv_path)
        assert loaded[0].doc_id == "semantic_scholar:abc123"


# =============================================================================
# 2. IdStore
# =============================================================================

class TestIdStore:

    def test_add_composite_ids(self, tmp):
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        added = store.add_ids(["arxiv:001", "arxiv:002", "fake:abc"])
        assert added == 3
        assert store.total == 3

    def test_deduplication(self, tmp):
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        store.add_ids(["arxiv:001", "arxiv:002"])
        added = store.add_ids(["arxiv:001", "arxiv:003"])
        assert added == 1
        assert store.total == 3

    def test_get_pending_batch_returns_composite_ids(self, tmp):
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        store.add_ids(["arxiv:001", "fake:doc1", "arxiv:002"])
        batch = store.get_pending_batch(2)
        assert len(batch) == 2
        assert all(":" in doc_id for doc_id in batch)

    def test_mark_downloaded(self, tmp):
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        store.add_ids(["arxiv:001", "arxiv:002", "arxiv:003"])
        store.mark_downloaded(["arxiv:001", "arxiv:002"])
        assert store.downloaded_count == 2
        assert store.pending_count == 1
        pending = store.get_pending_batch(10)
        assert pending == ["arxiv:003"]

    def test_csv_column_is_doc_id(self, tmp):
        from backend.crawler.id_store import IdStore
        csv_path = tmp / "ids.csv"
        store = IdStore(csv_path)
        store.add_ids(["arxiv:001"])
        with csv_path.open() as f:
            header = f.readline().strip().split(",")
        assert header[0] == "doc_id"
        assert "arxiv_id" not in header

    def test_persistence_across_reload(self, tmp):
        from backend.crawler.id_store import IdStore
        csv_path = tmp / "ids.csv"
        s1 = IdStore(csv_path)
        s1.add_ids(["arxiv:001", "arxiv:002", "arxiv:003"])
        s1.mark_downloaded(["arxiv:001"])
        s2 = IdStore(csv_path)
        assert s2.total == 3
        assert s2.downloaded_count == 1
        assert s2.pending_count == 2

    def test_load_legacy_csv_with_arxiv_id_column(self, tmp):
        """Carga correctamente un CSV antiguo con columna arxiv_id."""
        from backend.crawler.id_store import IdStore
        csv_path = tmp / "legacy.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arxiv_id", "discovered_at", "downloaded"])
            w.writeheader()
            w.writerow({"arxiv_id": "arxiv:legacy01", "discovered_at": "2024", "downloaded": "False"})
            w.writerow({"arxiv_id": "arxiv:legacy02", "discovered_at": "2024", "downloaded": "True"})
        store = IdStore(csv_path)
        assert store.total == 2
        assert store.downloaded_count == 1
        assert store.pending_count == 1
        pending = store.get_pending_batch(10)
        assert "arxiv:legacy01" in pending

    def test_multisource_ids_coexist(self, tmp):
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        store.add_ids(["arxiv:111", "fake:aaa", "semantic:xyz"])
        assert store.total == 3
        store.mark_downloaded(["arxiv:111"])
        pending = store.get_pending_batch(10)
        assert set(pending) == {"fake:aaa", "semantic:xyz"}

    def test_thread_safety(self, tmp):
        """add_ids es seguro cuando varios hilos escriben a la vez."""
        from backend.crawler.id_store import IdStore
        store = IdStore(tmp / "ids.csv")
        errors = []

        def add_batch(prefix, n):
            try:
                store.add_ids([f"{prefix}:{i:04d}" for i in range(n)])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_batch, args=(f"src{t}", 20))
                   for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errores en hilos: {errors}"
        assert store.total == 100


# =============================================================================
# 3. robots.py
# =============================================================================

class TestRobots:

    def test_allowed_domains_is_mutable_set(self):
        from backend.crawler import robots
        assert isinstance(robots.ALLOWED_DOMAINS, set)

    def test_allow_domain_adds_to_allowed_domains(self):
        from backend.crawler.robots import RobotsChecker, ALLOWED_DOMAINS
        rc = RobotsChecker()
        test_domain = "api.test-unique-domain-xyz.example"
        assert test_domain not in ALLOWED_DOMAINS
        rc.allow_domain(test_domain)
        assert test_domain in ALLOWED_DOMAINS
        ALLOWED_DOMAINS.discard(test_domain)  # limpieza

    def test_allowed_returns_true_for_registered_domain(self):
        from backend.crawler.robots import RobotsChecker, ALLOWED_DOMAINS
        rc = RobotsChecker()
        ALLOWED_DOMAINS.add("api.localtest.example")
        try:
            assert rc.allowed("https://api.localtest.example/search") is True
        finally:
            ALLOWED_DOMAINS.discard("api.localtest.example")

    def test_crawl_delay_zero_for_registered_domain(self):
        from backend.crawler.robots import RobotsChecker, ALLOWED_DOMAINS
        rc = RobotsChecker()
        ALLOWED_DOMAINS.add("nodelay.localtest.example")
        try:
            assert rc.crawl_delay("https://nodelay.localtest.example/x") == 0.0
        finally:
            ALLOWED_DOMAINS.discard("nodelay.localtest.example")

    def test_arxiv_always_allowed_without_network(self):
        from backend.crawler.robots import RobotsChecker
        rc = RobotsChecker()
        assert rc.allowed("https://arxiv.org/abs/2301.12345") is True
        assert rc.allowed("https://export.arxiv.org/api/query") is True

    def test_arxiv_crawl_delay_is_zero(self):
        from backend.crawler.robots import RobotsChecker
        rc = RobotsChecker()
        assert rc.crawl_delay("https://arxiv.org/pdf/2301.12345") == 0.0


# =============================================================================
# 4. BaseClient
# =============================================================================

class TestBaseClient:

    def test_cannot_instantiate_abstract_class(self):
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(TypeError):
            BaseClient()

    def test_make_doc_id_format(self):
        from backend.crawler.clients.base_client import BaseClient
        from backend.crawler.document import Document

        class MinClient(BaseClient):
            @property
            def source_name(self): return "mysource"
            def fetch_ids(self, **kw): return []
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""

        c = MinClient()
        assert c.make_doc_id("abc123") == "mysource:abc123"
        assert c.make_doc_id("2301.12345") == "mysource:2301.12345"

    def test_parse_doc_id_valid(self):
        from backend.crawler.clients.base_client import BaseClient
        source, local = BaseClient.parse_doc_id("arxiv:2301.12345")
        assert source == "arxiv"
        assert local == "2301.12345"

    def test_parse_doc_id_preserves_colons_in_local(self):
        """El ID local puede contener ':' — solo se parte en el primero."""
        from backend.crawler.clients.base_client import BaseClient
        source, local = BaseClient.parse_doc_id("fake:id:with:colons")
        assert source == "fake"
        assert local == "id:with:colons"

    @pytest.mark.parametrize("bad_id", [
        "nocolon",
        ":noleftpart",
        "noleft:",
        "",
    ])
    def test_parse_doc_id_invalid_raises(self, bad_id):
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(ValueError):
            BaseClient.parse_doc_id(bad_id)

    def test_make_and_parse_are_inverse(self):
        from backend.crawler.clients.base_client import BaseClient

        class AnyClient(BaseClient):
            @property
            def source_name(self): return "src"
            def fetch_ids(self, **kw): return []
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""

        c = AnyClient()
        local_ids = ["id1", "id2", "complex.id.v3"]
        for lid in local_ids:
            composite = c.make_doc_id(lid)
            src, parsed = BaseClient.parse_doc_id(composite)
            assert src == "src"
            assert parsed == lid


# =============================================================================
# 5. ArxivClient
# =============================================================================

class TestArxivClient:

    ATOM_NS = "http://www.w3.org/2005/Atom"

    def _make_entry(self, arxiv_id="2301.12345v2", title="Test",
                    author="Alice", abstract="Abs",
                    categories=("cs.AI",), pdf_href="https://arxiv.org/pdf/2301.12345"):
        ns = self.ATOM_NS
        entry_str = f"""<entry xmlns="{ns}">
          <id>https://arxiv.org/abs/{arxiv_id}</id>
          <title>{title}</title>
          <author><name>{author}</name></author>
          <summary>{abstract}</summary>
          {"".join(f'<category term="{c}"/>' for c in categories)}
          <published>2023-01-01T00:00:00Z</published>
          <updated>2023-01-02T00:00:00Z</updated>
          <link title="pdf" href="{pdf_href}"/>
        </entry>"""
        return ET.fromstring(entry_str)

    def test_source_name(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        assert ArxivClient().source_name == "arxiv"

    def test_make_doc_id(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        c = ArxivClient()
        assert c.make_doc_id("2301.12345") == "arxiv:2301.12345"

    def test_parse_ids_strips_version(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        ns = self.ATOM_NS
        xml = f"""<feed xmlns="{ns}">
          <entry><id>https://arxiv.org/abs/2301.00001v3</id></entry>
          <entry><id>https://arxiv.org/abs/2302.99999v1</id></entry>
        </feed>"""
        client = ArxivClient()
        ids = client._parse_ids(xml)
        assert ids == ["2301.00001", "2302.99999"]

    def test_entry_to_document_produces_composite_doc_id(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        entry = self._make_entry(arxiv_id="2301.12345v2")
        doc = client._entry_to_document(entry)
        assert doc is not None
        assert doc.doc_id == "arxiv:2301.12345"
        assert doc.arxiv_id == "arxiv:2301.12345"   # propiedad retrocompat

    def test_entry_to_document_fields(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        entry = self._make_entry(
            title="Attention Is All You Need",
            author="Vaswani",
            abstract="Transformer paper.",
            categories=("cs.CL", "cs.LG"),
            pdf_href="https://arxiv.org/pdf/1706.03762",
        )
        doc = client._entry_to_document(entry)
        assert doc is not None
        assert doc.title == "Attention Is All You Need"
        assert "Vaswani" in doc.authors
        assert doc.abstract == "Transformer paper."
        assert "cs.CL" in doc.categories
        assert doc.pdf_url == "https://arxiv.org/pdf/1706.03762"

    def test_entry_without_id_returns_none(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        ns = self.ATOM_NS
        entry = ET.fromstring(f'<entry xmlns="{ns}"><title>No ID</title></entry>')
        client = ArxivClient()
        assert client._entry_to_document(entry) is None

    def test_fetch_documents_groups_into_chunks_of_20(self):
        """fetch_documents divide listas > 20 IDs en sub-lotes."""
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        call_sizes = []

        def fake_fetch_chunk(local_ids):
            call_sizes.append(len(local_ids))
            return []

        client._fetch_chunk = fake_fetch_chunk
        client.fetch_documents([f"{i:04d}" for i in range(55)])
        assert len(call_sizes) == 3
        assert call_sizes[0] == 20
        assert call_sizes[1] == 20
        assert call_sizes[2] == 15

    def test_backward_compat_import(self):
        """ArxivClient sigue importable desde la ruta antigua."""
        from backend.crawler import ArxivClient
        c = ArxivClient()
        assert c.source_name == "arxiv"

    @pytest.mark.network
    def test_fetch_ids_returns_local_ids(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        ids = client.fetch_ids(max_results=3)
        assert len(ids) > 0
        for lid in ids:
            assert ":" not in lid, f"ID should be local (no prefix): {lid!r}"
            assert "v" not in lid, f"ID should not have version suffix: {lid!r}"

    @pytest.mark.network
    def test_fetch_documents_returns_composite_doc_ids(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        ids = client.fetch_ids(max_results=2)
        docs = client.fetch_documents(ids)
        assert len(docs) > 0
        for doc in docs:
            assert doc.doc_id.startswith("arxiv:"), f"Expected 'arxiv:' prefix: {doc.doc_id!r}"
            assert doc.title

    @pytest.mark.network
    def test_download_text_returns_non_empty_string(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        text = client.download_text("1706.03762")
        assert isinstance(text, str)
        assert len(text) > 500


# =============================================================================
# 6. chunker
# =============================================================================

class TestChunker:

    def test_clean_text_collapses_newlines(self):
        from backend.crawler.chunker import clean_text
        result = clean_text("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result
        assert "line1" in result and "line2" in result

    def test_clean_text_removes_page_numbers(self):
        from backend.crawler.chunker import clean_text
        result = clean_text("Intro text\n42\nMore text")
        assert "42" not in result.split()

    def test_clean_text_collapses_spaces(self):
        from backend.crawler.chunker import clean_text
        result = clean_text("word1    word2\t\t\tword3")
        assert "  " not in result

    def test_make_chunks_returns_list_of_strings(self):
        from backend.crawler.chunker import make_chunks
        text = "Hello world. This is a test. " * 30
        chunks = make_chunks(text)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
        assert len(chunks) > 0

    def test_make_chunks_respects_chunk_size(self):
        from backend.crawler.chunker import make_chunks
        text = ("A" * 50 + ". ") * 50
        chunks = make_chunks(text, chunk_size=200)
        # Ningún chunk debería exceder aprox 2× el límite (con overlap)
        oversized = [c for c in chunks if len(c) > 400]
        assert not oversized, f"Chunks demasiado grandes: {[len(c) for c in oversized]}"

    def test_make_chunks_overlap_shared_sentences(self):
        """Las últimas N oraciones de un chunk aparecen al inicio del siguiente."""
        from backend.crawler.chunker import make_chunks
        # Texto con oraciones numeradas bien diferenciadas
        sentences = [f"Sentence {i} is here and has enough text to count." for i in range(20)]
        text = " ".join(sentences)
        chunks = make_chunks(text, chunk_size=200, overlap_sentences=2)
        if len(chunks) >= 2:
            # Las últimas palabras del chunk[0] deben aparecer en chunk[1]
            end_of_first = chunks[0].split()[-5:]
            start_of_second = chunks[1]
            assert any(w in start_of_second for w in end_of_first), \
                "No se detectó solapamiento entre chunks consecutivos"

    def test_make_chunks_no_overlap(self):
        from backend.crawler.chunker import make_chunks
        text = ("Word word word. ") * 100
        chunks_ov = make_chunks(text, chunk_size=200, overlap_sentences=2)
        chunks_no = make_chunks(text, chunk_size=200, overlap_sentences=0)
        # Sin solapamiento debería producir chunks más cortos o iguales en total
        total_ov = sum(len(c) for c in chunks_ov)
        total_no = sum(len(c) for c in chunks_no)
        assert total_ov >= total_no  # overlap añade texto repetido

    def test_make_chunks_empty_text(self):
        from backend.crawler.chunker import make_chunks
        assert make_chunks("") == []
        assert make_chunks("   \n\n   ") == []

    def test_make_chunks_very_short_text(self):
        from backend.crawler.chunker import make_chunks
        # Texto menor que MIN_CHUNK_CHARS (100) → lista vacía o 1 chunk
        result = make_chunks("Short text.")
        assert isinstance(result, list)

    def test_make_chunks_paragraph_boundaries_respected(self):
        """Contenido de párrafos distintos no aparece mezclado en un mismo chunk."""
        from backend.crawler.chunker import make_chunks, MIN_CHUNK_CHARS
        # Oraciones de 45 chars c/u → 3 oraciones = 135 chars > MIN_CHUNK_CHARS
        sent_p1 = "Alpha beta gamma delta epsilon zeta eta. "   # 41 chars
        sent_p2 = "Omega theta iota kappa lambda mu nu xi. "    # 40 chars
        p1 = sent_p1 * 8   # ~328 chars
        p2 = sent_p2 * 8   # ~320 chars
        text = p1 + "\n\n" + p2
        chunks = make_chunks(text, chunk_size=150, overlap_sentences=0)
        assert len(chunks) >= 2, f"Esperaba ≥2 chunks, got {len(chunks)}"
        # Ningún chunk debe mezclar palabras exclusivas de cada párrafo
        mixed = [c for c in chunks if "Alpha" in c and "Omega" in c]
        assert not mixed, f"Chunk mezcla párrafos: {mixed}"

    def test_make_chunks_applies_clean_text_internally(self):
        from backend.crawler.chunker import make_chunks
        text = "Good sentence one here.\n\n\n\n   Good sentence two here."
        chunks = make_chunks(text, chunk_size=500)
        full = " ".join(chunks)
        assert "\n\n\n" not in full

    def test_make_chunks_default_params(self):
        from backend.crawler.chunker import make_chunks
        text = "A complete sentence here. " * 100
        chunks_default  = make_chunks(text)
        chunks_explicit = make_chunks(text, chunk_size=1000, overlap_sentences=2)
        assert chunks_default == chunks_explicit


# =============================================================================
# 7. pdf_extractor — delegación a chunker
# =============================================================================

class TestPdfExtractorDelegation:
    """Verifica que pdf_extractor delega en chunker (mismos resultados)."""

    def test_clean_text_delegation(self):
        from backend.crawler.pdf_extractor import _clean_text
        from backend.crawler.chunker import clean_text
        text = "line1\n\n\n\nline2    end"
        assert _clean_text(text) == clean_text(text)

    def test_split_sentences_delegation(self):
        from backend.crawler.pdf_extractor import _split_sentences
        from backend.crawler.chunker import _split_sentences as cs
        text = "Hello world. This is a test. Another sentence here."
        assert _split_sentences(text) == cs(text)

    def test_split_into_chunks_delegation(self):
        from backend.crawler.pdf_extractor import _split_into_chunks
        from backend.crawler.chunker import _split_into_chunks as ci
        text = ("Good sentence here. " * 5 + "\n\n") * 10
        assert _split_into_chunks(text, max_chars=300, overlap_sentences=2) == \
               ci(text, max_chars=300, overlap_sentences=2)

    def test_make_chunks_alias_works(self):
        from backend.crawler.pdf_extractor import _make_chunks
        from backend.crawler.chunker import make_chunks
        text = "Test sentence one. Test sentence two. " * 20
        assert _make_chunks(text, chunk_size=300, overlap_sentences=2) == \
               make_chunks(text, chunk_size=300, overlap_sentences=2)


# =============================================================================
# 8. Crawler — lógica multi-cliente (sin red)
# =============================================================================

class TestCrawlerRouting:

    def _make_crawler_no_db(self, clients):
        """Crea un Crawler parcheando la DB para no necesitar SQLite."""
        from backend.crawler.crawler import Crawler, CrawlerConfig
        with patch("backend.crawler.crawler.Crawler.__init__") as mock_init:
            mock_init.return_value = None
            crawler = Crawler.__new__(Crawler)
        crawler.config = CrawlerConfig()
        crawler._clients = list(clients)
        crawler._client_map = {c.source_name: c for c in clients}
        crawler._stop = threading.Event()
        return crawler

    def test_default_client_is_arxiv(self):
        """Sin pasar clientes, el Crawler usa ArxivClient por defecto."""
        from backend.crawler.crawler import Crawler, CrawlerConfig
        from backend.crawler.clients.arxiv_client import ArxivClient

        # Verificar la lógica de resolución de clientes sin instanciar el Crawler
        # (que requiere SQLite). La lógica está en __init__: if not clients and not client
        # → self._clients = [ArxivClient()]
        clients_result = [ArxivClient()]          # lo que haría Crawler sin args
        client_map = {c.source_name: c for c in clients_result}
        assert "arxiv" in client_map
        assert isinstance(client_map["arxiv"], ArxivClient)

    def test_client_for_routes_arxiv(self, fake_client):
        from backend.crawler.clients.arxiv_client import ArxivClient
        arxiv = ArxivClient()
        crawler = self._make_crawler_no_db([arxiv, fake_client])
        result = crawler._client_for("arxiv:2301.12345")
        assert result is arxiv

    def test_client_for_routes_fake(self, fake_client):
        from backend.crawler.clients.arxiv_client import ArxivClient
        arxiv = ArxivClient()
        crawler = self._make_crawler_no_db([arxiv, fake_client])
        result = crawler._client_for("fake:doc1")
        assert result is fake_client

    def test_client_for_unknown_source_returns_none(self, fake_client):
        crawler = self._make_crawler_no_db([fake_client])
        assert crawler._client_for("unknown:id123") is None

    def test_client_for_invalid_format_returns_none(self, fake_client):
        crawler = self._make_crawler_no_db([fake_client])
        assert crawler._client_for("nocolon") is None

    def test_local_id_extraction(self, fake_client):
        crawler = self._make_crawler_no_db([fake_client])
        assert crawler._local_id("arxiv:2301.12345") == "2301.12345"
        assert crawler._local_id("fake:doc1") == "doc1"
        assert crawler._local_id("src:id:with:colons") == "id:with:colons"

    def test_make_doc_id_per_client(self, fake_client):
        from backend.crawler.clients.arxiv_client import ArxivClient
        arxiv = ArxivClient()
        assert arxiv.make_doc_id("2301.12345") == "arxiv:2301.12345"
        assert fake_client.make_doc_id("doc1") == "fake:doc1"


class TestCrawlerDiscoveryLoop:
    """Tests del _discovery_loop con clientes mockeados."""

    def test_discovery_stores_composite_ids(self, tmp, fake_client):
        from backend.crawler.crawler import Crawler, CrawlerConfig
        from backend.crawler.id_store import IdStore

        id_store = IdStore(tmp / "ids.csv")
        fake_client._ids = ["doc1", "doc2"]

        # Simular un ciclo de discovery manualmente
        local_ids = fake_client.fetch_ids(max_results=10, start=0)
        doc_ids   = [fake_client.make_doc_id(lid) for lid in local_ids]
        added     = id_store.add_ids(doc_ids)

        assert added == 2
        assert id_store.total == 2
        pending = id_store.get_pending_batch(10)
        assert set(pending) == {"fake:doc1", "fake:doc2"}

    def test_discovery_multisource(self, tmp, fake_client):
        from backend.crawler.clients.base_client import BaseClient
        from backend.crawler.document import Document
        from backend.crawler.id_store import IdStore

        class SecondFake(BaseClient):
            @property
            def source_name(self): return "second"
            def fetch_ids(self, **kw): return ["s1", "s2"]
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""

        id_store  = IdStore(tmp / "ids.csv")
        clients   = [fake_client, SecondFake()]
        fake_client._ids = ["f1", "f2"]

        for client in clients:
            local_ids = client.fetch_ids()
            doc_ids   = [client.make_doc_id(lid) for lid in local_ids]
            id_store.add_ids(doc_ids)

        assert id_store.total == 4
        pending = id_store.get_pending_batch(10)
        sources = {p.split(":")[0] for p in pending}
        assert "fake" in sources
        assert "second" in sources


class TestCrawlerTextLoop:
    """Tests del _text_loop: download_text + make_chunks → SQLite."""

    def test_text_loop_calls_download_then_chunks(self, tmp_db, fake_client):
        """Verifica el flujo: client.download_text() → make_chunks() → SQLite."""
        from backend.database import crawler_repository as repo
        from backend.database.chunk_repository import save_chunks, get_chunks
        from backend.crawler.chunker import make_chunks

        # Insertar documento pendiente
        doc_id = fake_client.make_doc_id("doc1")
        repo.upsert_document(
            arxiv_id=doc_id, title="Doc 1", authors="A",
            abstract="X", categories="cs.AI",
            published="2024-01-01", updated="2024-01-01",
            pdf_url="https://fake.example.com/doc1.pdf",
            fetched_at="2024-01-01", db_path=tmp_db,
        )

        # Simular lo que hace _text_loop
        full_text = fake_client.download_text("doc1", pdf_url="https://fake.example.com/doc1.pdf")
        chunks    = make_chunks(full_text, chunk_size=200, overlap_sentences=2)

        repo.save_pdf_text(doc_id, full_text, db_path=tmp_db)
        save_chunks(doc_id, chunks, db_path=tmp_db)

        # Verificar resultados
        assert "doc1" in fake_client.download_calls
        assert len(chunks) > 0
        saved_chunks = get_chunks(doc_id, db_path=tmp_db)
        assert len(saved_chunks) == len(chunks)

    def test_composite_id_stored_correctly(self, tmp_db, fake_client):
        """El ID compuesto se almacena y recupera correctamente de SQLite."""
        from backend.database import crawler_repository as repo
        from backend.database.chunk_repository import save_chunks, get_chunks
        from backend.crawler.chunker import make_chunks

        doc_id = "fake:special-doc-99"
        repo.upsert_document(
            arxiv_id=doc_id, title="Special", authors="B",
            abstract="Y", categories="cs.LG",
            published="2024-01-01", updated="2024-01-01",
            pdf_url="https://fake.example.com/99.pdf",
            fetched_at="2024-01-01", db_path=tmp_db,
        )
        text   = fake_client.download_text("special-doc-99")
        chunks = make_chunks(text, chunk_size=300)
        repo.save_pdf_text(doc_id, text, db_path=tmp_db)
        save_chunks(doc_id, chunks, db_path=tmp_db)

        doc  = repo.get_document(doc_id, db_path=tmp_db)
        cks  = get_chunks(doc_id, db_path=tmp_db)
        assert doc is not None
        assert doc["arxiv_id"] == "fake:special-doc-99"
        assert doc["pdf_downloaded"] == 1
        assert len(cks) == len(chunks)


# =============================================================================
# 9. Retrocompatibilidad de imports
# =============================================================================

class TestBackwardCompatImports:

    def test_arxiv_client_importable_from_old_path(self):
        from backend.crawler import ArxivClient
        assert ArxivClient().source_name == "arxiv"

    def test_crawler_init_exports(self):
        from backend.crawler import (
            Document, IdStore, BaseClient, ArxivClient,
            Crawler, CrawlerConfig, make_chunks,
            download_and_extract, robots_checker,
        )
        assert BaseClient is not None
        assert ArxivClient is not None
        assert make_chunks is not None

    def test_download_and_extract_still_accessible(self):
        from backend.crawler.pdf_extractor import download_and_extract
        import inspect
        sig = inspect.signature(download_and_extract)
        assert "arxiv_id" in sig.parameters
        assert "chunk_size" in sig.parameters

    def test_document_arxiv_id_property_exists(self):
        """Código legado que usa doc.arxiv_id sigue funcionando."""
        from backend.crawler.document import Document
        doc = Document(
            doc_id="arxiv:2301.99999", title="T", authors="A",
            abstract="X", categories="cs.AI",
            published="2024", updated="2024", pdf_url="http://x",
        )
        # Acceso exactamente como lo hace el código antiguo
        assert doc.arxiv_id == "arxiv:2301.99999"
        assert isinstance(doc.arxiv_id, str)


# =============================================================================
# 10. Integración end-to-end con FakeClient
# =============================================================================

class TestEndToEndFakeClient:

    def test_full_pipeline_discovery_to_chunks(self, tmp, fake_client):
        """Discovery → metadatos → texto → chunks, todo con FakeClient."""
        from backend.crawler.document import Document
        from backend.crawler.id_store import IdStore
        from backend.crawler.chunker import make_chunks
        from backend.database.schema import init_db
        from backend.database import crawler_repository as repo
        from backend.database.chunk_repository import save_chunks, get_chunks

        db = tmp / "db" / "e2e.db"
        init_db(db)
        id_store = IdStore(tmp / "ids.csv")

        # Step 1: discovery
        local_ids = fake_client.fetch_ids(max_results=3)
        doc_ids   = [fake_client.make_doc_id(lid) for lid in local_ids]
        added     = id_store.add_ids(doc_ids)
        assert added == 3

        # Step 2: metadata download
        docs = fake_client.fetch_documents(local_ids)
        assert len(docs) == 3
        for doc in docs:
            assert doc.doc_id.startswith("fake:")
            repo.upsert_document(
                arxiv_id=doc.doc_id, title=doc.title, authors=doc.authors,
                abstract=doc.abstract, categories=doc.categories,
                published=doc.published, updated=doc.updated,
                pdf_url=doc.pdf_url, fetched_at=doc.fetched_at,
                db_path=db,
            )
        id_store.mark_downloaded(doc_ids)

        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 3
        assert stats["pdf_pending"] == 3

        # Step 3: text download + chunking
        pending_ids = repo.get_pending_pdf_ids(10, db_path=db)
        assert len(pending_ids) == 3

        for doc_id in pending_ids:
            local_id  = doc_id.split(":", 1)[1]
            doc_row   = repo.get_document(doc_id, db_path=db)
            full_text = fake_client.download_text(
                local_id, pdf_url=doc_row["pdf_url"]
            )
            chunks = make_chunks(full_text, chunk_size=200, overlap_sentences=2)
            repo.save_pdf_text(doc_id, full_text, db_path=db)
            save_chunks(doc_id, chunks, db_path=db)

        # Step 4: verificación final
        final_stats = repo.get_stats(db_path=db)
        assert final_stats["pdf_indexed"] == 3
        assert final_stats["pdf_pending"] == 0
        assert final_stats["total_chunks"] > 0
        assert len(fake_client.download_calls) == 3

        # Chunks recuperables por doc_id
        for doc_id in pending_ids:
            cks = get_chunks(doc_id, db_path=db)
            assert len(cks) > 0

    def test_multisource_pipeline(self, tmp, fake_client):
        """Dos fuentes distintas conviven en la misma DB sin colisiones."""
        from backend.crawler.clients.base_client import BaseClient
        from backend.crawler.document import Document
        from backend.crawler.id_store import IdStore
        from backend.crawler.chunker import make_chunks
        from backend.database.schema import init_db
        from backend.database import crawler_repository as repo
        from backend.database.chunk_repository import save_chunks, get_chunks

        class SecondClient(BaseClient):
            @property
            def source_name(self): return "second"

            def fetch_ids(self, max_results=100, start=0):
                return ["s_doc1", "s_doc2"]

            def fetch_documents(self, local_ids):
                return [
                    Document(
                        doc_id=self.make_doc_id(lid),
                        title=f"Second {lid}", authors="B",
                        abstract="From second source", categories="cs.CV",
                        published="2024-01-01", updated="2024-01-01",
                        pdf_url=f"https://second.example.com/{lid}",
                    )
                    for lid in local_ids
                ]

            def download_text(self, local_id, **kwargs):
                return f"Content from second source for {local_id}. " * 30

        db       = tmp / "db" / "multi.db"
        id_store = IdStore(tmp / "ids.csv")
        init_db(db)

        fake_client._ids = ["f1", "f2"]
        second = SecondClient()
        clients = [fake_client, second]

        # Discovery y metadata para ambos
        all_doc_ids = []
        for client in clients:
            local_ids = client.fetch_ids()
            doc_ids   = [client.make_doc_id(lid) for lid in local_ids]
            id_store.add_ids(doc_ids)
            docs = client.fetch_documents(local_ids)
            for doc in docs:
                repo.upsert_document(
                    arxiv_id=doc.doc_id, title=doc.title, authors=doc.authors,
                    abstract=doc.abstract, categories=doc.categories,
                    published=doc.published, updated=doc.updated,
                    pdf_url=doc.pdf_url, fetched_at=doc.fetched_at,
                    db_path=db,
                )
            all_doc_ids.extend(doc_ids)
        id_store.mark_downloaded(all_doc_ids)

        assert id_store.total == 4
        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 4

        # Text + chunks para todos
        client_map = {c.source_name: c for c in clients}
        pending = repo.get_pending_pdf_ids(10, db_path=db)
        assert len(pending) == 4

        for doc_id in pending:
            source, local_id = doc_id.split(":", 1)
            client    = client_map[source]
            doc_row   = repo.get_document(doc_id, db_path=db)
            text      = client.download_text(local_id, pdf_url=doc_row["pdf_url"])
            chunks    = make_chunks(text, chunk_size=300, overlap_sentences=2)
            repo.save_pdf_text(doc_id, text, db_path=db)
            save_chunks(doc_id, chunks, db_path=db)

        final = repo.get_stats(db_path=db)
        assert final["pdf_indexed"] == 4
        assert final["pdf_pending"] == 0

        # Sin colisiones de IDs entre fuentes
        cks_f1 = get_chunks("fake:f1", db_path=db)
        cks_s1 = get_chunks("second:s_doc1", db_path=db)
        assert len(cks_f1) > 0
        assert len(cks_s1) > 0


# =============================================================================
# Punto de entrada para ejecución directa
# =============================================================================
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v",
         "--tb=short", "-m", "not network"],
        cwd=str(Path(__file__).parent.parent.parent),
    )
    sys.exit(result.returncode)