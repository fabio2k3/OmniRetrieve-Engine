"""Tests del Crawler — routing multi-cliente, discovery loop, text loop."""
from __future__ import annotations
import threading
from unittest.mock import patch
import pytest

from backend.crawler._routing import client_for, local_id


def _make_crawler_no_db(clients):
    from backend.crawler.crawler import Crawler, CrawlerConfig
    with patch("backend.crawler.crawler.Crawler.__init__") as mock_init:
        mock_init.return_value = None
        crawler = Crawler.__new__(Crawler)
    crawler.config = CrawlerConfig()
    crawler._clients = list(clients)
    crawler._client_map = {c.source_name: c for c in clients}
    crawler._stop = threading.Event()
    return crawler


# ── Routing via _routing module ──────────────────────────────────────────────

def test_client_for_routes_arxiv(fake_client):
    from backend.crawler.clients import ArxivClient
    arxiv = ArxivClient()
    crawler = _make_crawler_no_db([arxiv, fake_client])
    assert client_for("arxiv:2301.12345", crawler._client_map) is arxiv


def test_client_for_routes_fake(fake_client):
    from backend.crawler.clients import ArxivClient
    crawler = _make_crawler_no_db([ArxivClient(), fake_client])
    assert client_for("fake:doc1", crawler._client_map) is fake_client


def test_client_for_unknown_source_returns_none(fake_client):
    crawler = _make_crawler_no_db([fake_client])
    assert client_for("unknown:id123", crawler._client_map) is None


def test_client_for_invalid_format_returns_none(fake_client):
    crawler = _make_crawler_no_db([fake_client])
    assert client_for("nocolon", crawler._client_map) is None


def test_local_id_extraction():
    assert local_id("arxiv:2301.12345") == "2301.12345"
    assert local_id("fake:doc1") == "doc1"
    assert local_id("src:id:with:colons") == "id:with:colons"


def test_local_id_invalid_raises():
    with pytest.raises(ValueError):
        local_id("nocolon")


# ── Discovery loop ───────────────────────────────────────────────────────────

def test_discovery_stores_composite_ids(tmp_path, fake_client):
    from backend.crawler.id_store import IdStore
    id_store = IdStore(tmp_path / "ids.csv")
    fake_client._ids = ["doc1", "doc2"]

    local_ids = fake_client.fetch_ids(max_results=10, start=0)
    doc_ids = [fake_client.make_doc_id(lid) for lid in local_ids]
    assert id_store.add_ids(doc_ids) == 2
    assert set(id_store.get_pending_batch(10)) == {"fake:doc1", "fake:doc2"}


def test_discovery_multisource(tmp_path, fake_client):
    from backend.crawler.clients.base_client import BaseClient
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

    id_store = IdStore(tmp_path / "ids.csv")
    fake_client._ids = ["f1", "f2"]
    for c in [fake_client, SecondFake()]:
        id_store.add_ids([c.make_doc_id(lid) for lid in c.fetch_ids()])

    assert id_store.total == 4
    sources = {p.split(":")[0] for p in id_store.get_pending_batch(10)}
    assert {"fake", "second"} == sources


# ── Text loop ────────────────────────────────────────────────────────────────

def test_text_loop_calls_download_then_chunks(tmp_path, fake_client):
    from backend.database import crawler_repository as repo
    from backend.database.chunk_repository import save_chunks, get_chunks
    from backend.crawler.chunker import make_chunks
    from backend.database.schema import init_db

    db = tmp_path / "db" / "test.db"
    init_db(db)
    doc_id = fake_client.make_doc_id("doc1")
    repo.upsert_document(
        arxiv_id=doc_id, title="Doc 1", authors="A",
        abstract="X", categories="cs.AI",
        published="2024-01-01", updated="2024-01-01",
        pdf_url="https://fake.example.com/doc1.pdf",
        fetched_at="2024-01-01", db_path=db,
    )

    full_text = fake_client.download_text("doc1", pdf_url="https://fake.example.com/doc1.pdf")
    chunks = make_chunks(full_text, chunk_size=200, overlap_sentences=2)
    repo.save_pdf_text(doc_id, full_text, db_path=db)
    save_chunks(doc_id, chunks, db_path=db)

    assert "doc1" in fake_client.download_calls
    assert len(chunks) > 0
    assert len(get_chunks(doc_id, db_path=db)) == len(chunks)