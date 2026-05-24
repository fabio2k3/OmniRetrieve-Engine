"""Fixtures compartidos para los tests del módulo crawler."""
from __future__ import annotations
from typing import List
import pytest


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def tmp_db(tmp_path):
    from backend.database.schema import init_db
    db = tmp_path / "db" / "test.db"
    init_db(db)
    return db


@pytest.fixture
def fake_client():
    from backend.crawler.clients.base_client import BaseClient
    from backend.crawler.document import Document

    class FakeClient(BaseClient):
        SOURCE = "fake"

        def __init__(self, ids=None, text="Sample text. " * 60):
            self._ids = ids or ["doc1", "doc2", "doc3"]
            self._text = text
            self.download_calls: List[str] = []

        @property
        def source_name(self) -> str:
            return self.SOURCE

        @property
        def request_delay(self) -> float:
            return 1.0

        @property
        def trusted_domains(self):
            return frozenset({"fake.example.com"})

        def fetch_ids(self, max_results=100, start=0) -> List[str]:
            return self._ids[:max_results]

        def fetch_documents(self, local_ids: List[str]) -> List[Document]:
            return [
                Document(
                    doc_id=self.make_doc_id(lid),
                    title=f"Title {lid}", authors="Author A",
                    abstract=f"Abstract for {lid}", categories="cs.AI",
                    published="2024-01-01T00:00:00Z", updated="2024-01-01T00:00:00Z",
                    pdf_url=f"https://fake.example.com/{lid}.pdf",
                )
                for lid in local_ids
            ]

        def download_text(self, local_id: str, **kwargs) -> str:
            self.download_calls.append(local_id)
            return self._text

    return FakeClient()
