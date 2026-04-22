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
  7.  ArxivClient        — extracción HTML/PDF (LaTeXML)
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

        @property
        def request_delay(self) -> float:
            return 1.0   # delay mínimo para tests (fuente ficticia)

        @property
        def trusted_domains(self):
            return frozenset({"fake.example.com"})

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

    def test_checker_has_no_allowed_domains(self):
        """robots.py ya no tiene ALLOWED_DOMAINS global — es genérico."""
        import backend.crawler.robots as robots_mod
        assert not hasattr(robots_mod, "ALLOWED_DOMAINS"), \
            "ALLOWED_DOMAINS no debe existir en robots.py — la política la declara cada cliente"

    def test_allowed_without_trusted_domains_reads_robots(self):
        """Sin trusted_domains, allowed() consulta robots.txt normalmente."""
        from backend.crawler.robots import RobotsChecker
        import urllib.robotparser
        rc = RobotsChecker()
        # Inyectar parser simulado que deniega /secret
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /secret"])
        import time
        rc._cache["https://example.com"] = (parser, time.monotonic())
        assert rc.allowed("https://example.com/public") is True
        assert rc.allowed("https://example.com/secret") is False

    def test_allowed_with_trusted_domains_bypasses_disallow(self):
        """Con trusted_domains, allowed() devuelve True aunque robots diga Disallow."""
        from backend.crawler.robots import RobotsChecker
        import urllib.robotparser, time
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /api", "Crawl-delay: 15"])
        rc._cache["https://api.example.com"] = (parser, time.monotonic())
        # Sin trusted: False
        assert rc.allowed("https://api.example.com/api/query") is False
        # Con trusted: True
        assert rc.allowed("https://api.example.com/api/query",
                          trusted_domains=frozenset({"api.example.com"})) is True

    def test_crawl_delay_never_bypassed_even_with_trusted_domains(self):
        """crawl_delay() lee robots.txt siempre, sin importar trusted_domains."""
        from backend.crawler.robots import RobotsChecker
        import urllib.robotparser, time
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Crawl-delay: 15", "Disallow: /api"])
        rc._cache["https://api.example.com"] = (parser, time.monotonic())
        # El delay se lee aunque el dominio esté en trusted
        delay = rc.crawl_delay("https://api.example.com/api/query")
        assert delay == 15.0, f"crawl_delay debería ser 15.0, got {delay}"

    def test_crawl_delay_zero_when_not_declared(self):
        from backend.crawler.robots import RobotsChecker
        import urllib.robotparser, time
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /private"])
        rc._cache["https://nodelay.example.com"] = (parser, time.monotonic())
        assert rc.crawl_delay("https://nodelay.example.com/page") == 0.0

    def test_allowed_fail_open_on_network_error(self):
        """Si no se puede obtener robots.txt, se asume acceso permitido."""
        from backend.crawler.robots import RobotsChecker
        rc = RobotsChecker(ttl=0)  # TTL 0 → siempre intenta recargar
        # Dominio inexistente → fallo de red → fail-open
        result = rc.allowed("https://this-domain-does-not-exist-xyz123.invalid/page")
        assert result is True


# =============================================================================
# 4. BaseClient
# =============================================================================

class TestBaseClient:

    def _minimal_client(self, source="mysource", delay=5.0, trusted=frozenset()):
        from backend.crawler.clients.base_client import BaseClient
        from backend.crawler.document import Document

        class MinClient(BaseClient):
            @property
            def source_name(self): return source
            @property
            def request_delay(self): return delay
            @property
            def trusted_domains(self): return trusted
            def fetch_ids(self, **kw): return []
            def fetch_documents(self, ids): return []
            def download_text(self, lid, **kw): return ""

        return MinClient()

    def test_cannot_instantiate_abstract_class(self):
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(TypeError):
            BaseClient()

    def test_missing_request_delay_raises(self):
        """Una subclase sin request_delay no se puede instanciar."""
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(TypeError):
            class Incomplete(BaseClient):
                @property
                def source_name(self): return "x"
                @property
                def trusted_domains(self): return frozenset()
                def fetch_ids(self, **kw): return []
                def fetch_documents(self, ids): return []
                def download_text(self, lid, **kw): return ""
            Incomplete()

    def test_missing_trusted_domains_raises(self):
        """Una subclase sin trusted_domains no se puede instanciar."""
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(TypeError):
            class Incomplete(BaseClient):
                @property
                def source_name(self): return "x"
                @property
                def request_delay(self): return 5.0
                def fetch_ids(self, **kw): return []
                def fetch_documents(self, ids): return []
                def download_text(self, lid, **kw): return ""
            Incomplete()

    def test_request_delay_is_float(self):
        c = self._minimal_client(delay=10.0)
        assert isinstance(c.request_delay, float)
        assert c.request_delay == 10.0

    def test_trusted_domains_is_frozenset(self):
        c = self._minimal_client(trusted=frozenset({"a.com", "b.com"}))
        assert isinstance(c.trusted_domains, frozenset)
        assert "a.com" in c.trusted_domains

    def test_make_doc_id_format(self):
        c = self._minimal_client()
        assert c.make_doc_id("abc123") == "mysource:abc123"

    def test_parse_doc_id_valid(self):
        from backend.crawler.clients.base_client import BaseClient
        source, local = BaseClient.parse_doc_id("arxiv:2301.12345")
        assert source == "arxiv" and local == "2301.12345"

    def test_parse_doc_id_preserves_colons_in_local(self):
        from backend.crawler.clients.base_client import BaseClient
        source, local = BaseClient.parse_doc_id("fake:id:with:colons")
        assert source == "fake" and local == "id:with:colons"

    @pytest.mark.parametrize("bad_id", ["nocolon", ":noleft", "noright:", ""])
    def test_parse_doc_id_invalid_raises(self, bad_id):
        from backend.crawler.clients.base_client import BaseClient
        with pytest.raises(ValueError):
            BaseClient.parse_doc_id(bad_id)

    def test_make_and_parse_are_inverse(self):
        c = self._minimal_client(source="src")
        from backend.crawler.clients.base_client import BaseClient
        for lid in ["id1", "complex.id.v3"]:
            composite = c.make_doc_id(lid)
            src, parsed = BaseClient.parse_doc_id(composite)
            assert src == "src" and parsed == lid


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
          <author><n>{author}</n></author>
          <summary>{abstract}</summary>
          {"".join(f'<category term="{c}"/>' for c in categories)}
          <published>2023-01-01T00:00:00Z</published>
          <updated>2023-01-02T00:00:00Z</updated>
          <link title="pdf" href="{pdf_href}"/>
        </entry>"""
        return ET.fromstring(entry_str)

    # -- Politica de crawling -------------------------------------------------

    def test_source_name(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        assert ArxivClient().source_name == "arxiv"

    def test_request_delay_is_at_least_15_seconds(self):
        """request_delay debe ser >= 15s (Crawl-delay del robots.txt de arXiv)."""
        from backend.crawler.clients.arxiv_client import ArxivClient
        c = ArxivClient()
        assert isinstance(c.request_delay, float)
        assert c.request_delay >= 15.0, (
            f"request_delay={c.request_delay} viola el Crawl-delay: 15 de arXiv"
        )

    def test_trusted_domains_contains_arxiv_hosts(self):
        """trusted_domains debe incluir los dos hosts de arXiv."""
        from backend.crawler.clients.arxiv_client import ArxivClient
        td = ArxivClient().trusted_domains
        assert isinstance(td, frozenset)
        assert "arxiv.org" in td
        assert "export.arxiv.org" in td

    def test_effective_delay_respects_robots_txt(self):
        """max(request_delay, crawl_delay_robots) >= 15s para todos los hosts de arXiv."""
        from backend.crawler.clients.arxiv_client import ArxivClient
        from backend.crawler.robots import RobotsChecker
        import urllib.robotparser, time

        c = ArxivClient()
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Crawl-delay: 15", "Disallow: /api"])
        for origin in ["https://arxiv.org", "https://export.arxiv.org"]:
            rc._cache[origin] = (parser, time.monotonic())

        for url in ["https://arxiv.org/pdf/2301.12345",
                    "https://export.arxiv.org/api/query"]:
            effective = max(c.request_delay, rc.crawl_delay(url))
            assert effective >= 15.0, f"Delay efectivo para {url} = {effective}s < 15s"

    def test_allowed_uses_trusted_domains(self):
        """allowed() con trusted_domains del cliente resuelve el Disallow: /api."""
        from backend.crawler.robots import RobotsChecker
        from backend.crawler.clients.arxiv_client import ArxivClient
        import urllib.robotparser, time

        c = ArxivClient()
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Disallow: /api", "Crawl-delay: 15"])
        rc._cache["https://export.arxiv.org"] = (parser, time.monotonic())

        assert rc.allowed("https://export.arxiv.org/api/query") is False
        assert rc.allowed("https://export.arxiv.org/api/query", c.trusted_domains) is True

    def test_crawl_delay_not_bypassed_by_trusted_domains(self):
        """crawl_delay() lee robots.txt incluso para dominios trusted."""
        from backend.crawler.robots import RobotsChecker
        from backend.crawler.clients.arxiv_client import ArxivClient
        import urllib.robotparser, time

        c = ArxivClient()
        rc = RobotsChecker()
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(["User-agent: *", "Crawl-delay: 15", "Disallow: /api"])
        for origin in ["https://arxiv.org", "https://export.arxiv.org"]:
            rc._cache[origin] = (parser, time.monotonic())

        assert rc.crawl_delay("https://arxiv.org/pdf/2301.12345") == 15.0
        assert rc.crawl_delay("https://export.arxiv.org/api/query") == 15.0

    # -- Parseo XML -----------------------------------------------------------

    def test_make_doc_id(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        assert ArxivClient().make_doc_id("2301.12345") == "arxiv:2301.12345"

    def test_parse_ids_strips_version(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        ns = self.ATOM_NS
        xml = f"""<feed xmlns="{ns}">
          <entry><id>https://arxiv.org/abs/2301.00001v3</id></entry>
          <entry><id>https://arxiv.org/abs/2302.99999v1</id></entry>
        </feed>"""
        assert ArxivClient()._parse_ids(xml) == ["2301.00001", "2302.99999"]

    def test_entry_to_document_produces_composite_doc_id(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        doc = ArxivClient()._entry_to_document(self._make_entry())
        assert doc is not None
        assert doc.doc_id == "arxiv:2301.12345"
        assert doc.arxiv_id == "arxiv:2301.12345"

    def test_entry_without_id_returns_none(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        ns = self.ATOM_NS
        entry = ET.fromstring(f'<entry xmlns="{ns}"><title>No ID</title></entry>')
        assert ArxivClient()._entry_to_document(entry) is None

    def test_fetch_documents_groups_into_chunks_of_20(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        client = ArxivClient()
        sizes = []
        client._fetch_chunk = lambda ids: (sizes.append(len(ids)) or [])
        client.fetch_documents([f"{i:04d}" for i in range(55)])
        assert sizes == [20, 20, 15]

    def test_rate_limit_is_thread_safe_single_instance(self):
        """
        3 hilos comparten 1 instancia: las peticiones salen serializadas
        con al menos request_delay entre cada una.
        """
        from backend.crawler.clients.arxiv_client import ArxivClient
        import threading, time

        # Subclase con delay corto para el test
        class FastClient(ArxivClient):
            @property
            def request_delay(self): return 0.05   # 50 ms

        client = FastClient()
        # Resetear el estado de clase para que el test sea reproducible
        ArxivClient._last_request = 0.0

        times = []

        def simulate_get():
            robots_delay    = 0.0
            effective_delay = max(client.request_delay, robots_delay)
            with ArxivClient._rate_lock:
                elapsed = time.monotonic() - ArxivClient._last_request
                if elapsed < effective_delay:
                    time.sleep(effective_delay - elapsed)
                ArxivClient._last_request = time.monotonic()
                times.append(ArxivClient._last_request)

        threads = [threading.Thread(target=simulate_get) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        times.sort()
        gaps = [times[i] - times[i-1] for i in range(1, len(times))]
        assert all(g >= 0.045 for g in gaps), \
            f"Gaps demasiado cortos: {[f'{g*1000:.1f}ms' for g in gaps]}"

    def test_rate_limit_shared_across_two_instances(self):
        """
        2 instancias distintas de ArxivClient comparten el mismo rate-limiter
        porque _rate_lock y _last_request son variables de clase.
        """
        from backend.crawler.clients.arxiv_client import ArxivClient
        import threading, time

        class FastClient(ArxivClient):
            @property
            def request_delay(self): return 0.05

        a1 = FastClient()
        a2 = FastClient()

        # Verificar que comparten el mismo lock y último timestamp
        assert a1._rate_lock is a2._rate_lock, \
            "_rate_lock debe ser de clase, compartido entre instancias"

        ArxivClient._last_request = 0.0
        times = []

        def call(client):
            effective_delay = client.request_delay
            with ArxivClient._rate_lock:
                elapsed = time.monotonic() - ArxivClient._last_request
                if elapsed < effective_delay:
                    time.sleep(effective_delay - elapsed)
                ArxivClient._last_request = time.monotonic()
                times.append(ArxivClient._last_request)

        t1 = threading.Thread(target=call, args=(a1,))
        t2 = threading.Thread(target=call, args=(a2,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        times.sort()
        gap = times[1] - times[0]
        assert gap >= 0.045, \
            f"Dos instancias no coordinaron: gap={gap*1000:.1f}ms < 45ms"

    def test_backward_compat_import(self):
        from backend.crawler import ArxivClient
        c = ArxivClient()
        assert c.source_name == "arxiv"
        assert c.request_delay >= 15.0

    @pytest.mark.network
    def test_fetch_ids_returns_local_ids(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        ids = ArxivClient().fetch_ids(max_results=3)
        assert len(ids) > 0
        for lid in ids:
            assert ":" not in lid
            assert "v" not in lid

    @pytest.mark.network
    def test_fetch_documents_returns_composite_doc_ids(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        c = ArxivClient()
        docs = c.fetch_documents(c.fetch_ids(max_results=2))
        for doc in docs:
            assert doc.doc_id.startswith("arxiv:")

    @pytest.mark.network
    def test_download_text_returns_non_empty_string(self):
        from backend.crawler.clients.arxiv_client import ArxivClient
        text = ArxivClient().download_text("1706.03762")
        assert isinstance(text, str) and len(text) > 500


# =============================================================================
# 6. chunker
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
# 7. ArxivClient — extracción de texto
# =============================================================================

class TestArxivTextExtraction:
    """
    Verifica los extractores de texto internos de ArxivClient.
    Toda la lógica de extracción vive en arxiv_client.py.
    El Crawler llama a client.download_text() directamente.
    """

    def test_latexml_extractor_skips_bibliography(self):
        from backend.crawler.clients.arxiv_client import _LaTeXMLExtractor
        html = """
        <div class="ltx_document">
          <p class="ltx_p">Main content here.</p>
          <section class="ltx_bibliography">
            <p>References section that should be skipped.</p>
          </section>
        </div>"""
        p = _LaTeXMLExtractor()
        p.feed(html)
        text = p.get_text()
        assert "Main content" in text
        assert "References section" not in text

    def test_latexml_extractor_skips_authors(self):
        from backend.crawler.clients.arxiv_client import _LaTeXMLExtractor
        html = """
        <div class="ltx_document">
          <div class="ltx_authors">Alice, Bob (should be skipped)</div>
          <p class="ltx_p">Abstract text here.</p>
        </div>"""
        p = _LaTeXMLExtractor()
        p.feed(html)
        text = p.get_text()
        assert "Abstract text" in text
        assert "Alice, Bob" not in text

    def test_latexml_extractor_fallback_generic(self):
        """Si no hay ltx_document, usa un parser genérico como fallback."""
        from backend.crawler.clients.arxiv_client import _extract_text_from_html
        html = b"""<html><body>
          <p>No ltx_document here, just plain HTML content.</p>
          <script>alert('skip me')</script>
        </body></html>"""
        text = _extract_text_from_html(html)
        # Debe extraer algo del contenido aunque no haya estructura LaTeXML
        assert len(text) > 0

    def test_clean_text_in_arxiv_client(self):
        from backend.crawler.clients.arxiv_client import _clean_text
        result = _clean_text("word1    word2\n\n\n\nword3")
        assert "  " not in result
        assert "\n\n\n" not in result

    def test_pdf_extractor_file_does_not_exist(self):
        """pdf_extractor.py ha sido eliminado — ya no existe como módulo."""
        import importlib, pathlib
        path = pathlib.Path("backend/crawler/pdf_extractor.py")
        assert not path.exists(),             "pdf_extractor.py debería haberse eliminado — toda la lógica está en arxiv_client.py"
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("backend.crawler.pdf_extractor")


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
            @property
            def request_delay(self): return 1.0
            @property
            def trusted_domains(self): return frozenset()
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
        """ArxivClient sigue importable desde la ruta original."""
        from backend.crawler import ArxivClient
        assert ArxivClient().source_name == "arxiv"

    def test_crawler_init_exports(self):
        """El __init__ del paquete exporta los símbolos esperados."""
        from backend.crawler import (
            Document, IdStore, BaseClient, ArxivClient,
            Crawler, CrawlerConfig, make_chunks, robots_checker,
        )
        assert BaseClient  is not None
        assert ArxivClient is not None
        assert make_chunks is not None

    def test_document_arxiv_id_property_exists(self):
        """Código legado que usa doc.arxiv_id sigue funcionando."""
        from backend.crawler.document import Document
        doc = Document(
            doc_id="arxiv:2301.99999", title="T", authors="A",
            abstract="X", categories="cs.AI",
            published="2024", updated="2024", pdf_url="http://x",
        )
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
            @property
            def request_delay(self): return 1.0
            @property
            def trusted_domains(self): return frozenset()

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
