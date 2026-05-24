"""Tests de retrocompatibilidad e integración end-to-end con FakeClient."""
from __future__ import annotations
import pytest


# ── Retrocompat ──────────────────────────────────────────────────────────────

def test_arxiv_client_importable_from_old_path():
    from backend.crawler import ArxivClient
    assert ArxivClient().source_name == "arxiv"


def test_crawler_init_exports():
    from backend.crawler import (
        Document, IdStore, BaseClient, ArxivClient,
        Crawler, CrawlerConfig, make_chunks, robots_checker,
    )
    assert BaseClient is not None
    assert ArxivClient is not None
    assert make_chunks is not None


def test_document_arxiv_id_property_exists():
    from backend.crawler.document import Document
    doc = Document(
        doc_id="arxiv:2301.99999", title="T", authors="A",
        abstract="X", categories="cs.AI",
        published="2024", updated="2024", pdf_url="http://x",
    )
    assert doc.arxiv_id == "arxiv:2301.99999"


# ── Integración end-to-end ───────────────────────────────────────────────────

def test_full_pipeline_discovery_to_chunks(tmp_path, fake_client):
    from backend.crawler.id_store import IdStore
    from backend.crawler.chunker import make_chunks
    from backend.database.schema import init_db
    from backend.database import crawler_repository as repo
    from backend.database.chunk_repository import save_chunks, get_chunks

    db = tmp_path / "db" / "e2e.db"
    init_db(db)
    id_store = IdStore(tmp_path / "ids.csv")

    # Discovery
    local_ids = fake_client.fetch_ids(max_results=3)
    doc_ids = [fake_client.make_doc_id(lid) for lid in local_ids]
    assert id_store.add_ids(doc_ids) == 3

    # Metadata
    docs = fake_client.fetch_documents(local_ids)
    for doc in docs:
        assert doc.doc_id.startswith("fake:")
        repo.upsert_document(
            arxiv_id=doc.doc_id, title=doc.title, authors=doc.authors,
            abstract=doc.abstract, categories=doc.categories,
            published=doc.published, updated=doc.updated,
            pdf_url=doc.pdf_url, fetched_at=doc.fetched_at, db_path=db,
        )
    id_store.mark_downloaded(doc_ids)

    stats = repo.get_stats(db_path=db)
    assert stats["total_documents"] == 3
    assert stats["pdf_pending"] == 3

    # Text + chunks
    pending_ids = repo.get_pending_pdf_ids(10, db_path=db)
    for doc_id in pending_ids:
        local_id = doc_id.split(":", 1)[1]
        doc_row = repo.get_document(doc_id, db_path=db)
        full_text = fake_client.download_text(local_id, pdf_url=doc_row["pdf_url"])
        chunks = make_chunks(full_text, chunk_size=200, overlap_sentences=2)
        repo.save_pdf_text(doc_id, full_text, db_path=db)
        save_chunks(doc_id, chunks, db_path=db)

    final = repo.get_stats(db_path=db)
    assert final["pdf_indexed"] == 3
    assert final["pdf_pending"] == 0
    assert final["total_chunks"] > 0
    assert len(fake_client.download_calls) == 3

    for doc_id in pending_ids:
        assert len(get_chunks(doc_id, db_path=db)) > 0


def test_multisource_pipeline(tmp_path, fake_client):
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
        def fetch_ids(self, max_results=100, start=0): return ["s_doc1", "s_doc2"]
        def fetch_documents(self, local_ids):
            return [
                Document(
                    doc_id=self.make_doc_id(lid), title=f"Second {lid}", authors="B",
                    abstract="From second source", categories="cs.CV",
                    published="2024-01-01", updated="2024-01-01",
                    pdf_url=f"https://second.example.com/{lid}",
                )
                for lid in local_ids
            ]
        def download_text(self, local_id, **kwargs):
            return f"Content from second source for {local_id}. " * 30

    db = tmp_path / "db" / "multi.db"
    init_db(db)
    id_store = IdStore(tmp_path / "ids.csv")
    fake_client._ids = ["f1", "f2"]
    clients = [fake_client, SecondClient()]

    all_doc_ids = []
    for client in clients:
        local_ids = client.fetch_ids()
        doc_ids = [client.make_doc_id(lid) for lid in local_ids]
        id_store.add_ids(doc_ids)
        for doc in client.fetch_documents(local_ids):
            repo.upsert_document(
                arxiv_id=doc.doc_id, title=doc.title, authors=doc.authors,
                abstract=doc.abstract, categories=doc.categories,
                published=doc.published, updated=doc.updated,
                pdf_url=doc.pdf_url, fetched_at=doc.fetched_at, db_path=db,
            )
        all_doc_ids.extend(doc_ids)
    id_store.mark_downloaded(all_doc_ids)

    assert repo.get_stats(db_path=db)["total_documents"] == 4

    client_map = {c.source_name: c for c in clients}
    for doc_id in repo.get_pending_pdf_ids(10, db_path=db):
        source, local_id = doc_id.split(":", 1)
        client = client_map[source]
        doc_row = repo.get_document(doc_id, db_path=db)
        text = client.download_text(local_id, pdf_url=doc_row["pdf_url"])
        chunks = make_chunks(text, chunk_size=300, overlap_sentences=2)
        repo.save_pdf_text(doc_id, text, db_path=db)
        save_chunks(doc_id, chunks, db_path=db)

    final = repo.get_stats(db_path=db)
    assert final["pdf_indexed"] == 4
    assert final["pdf_pending"] == 0
    assert len(get_chunks("fake:f1", db_path=db)) > 0
    assert len(get_chunks("second:s_doc1", db_path=db)) > 0
