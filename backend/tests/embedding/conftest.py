"""Fixtures y mocks compartidos para los tests de embedding."""
from __future__ import annotations
from pathlib import Path
import sqlite3
import numpy as np
import pytest

DIM = 64

SAMPLE_DOCS = [
    {"arxiv_id": "2401.00001", "title": "Attention Is All You Need",
     "full_text": "The transformer architecture relies entirely on attention mechanisms.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2401.00002", "title": "BERT: Bidirectional Transformers",
     "full_text": "BERT applies bidirectional training of transformers for language understanding.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2401.00003", "title": "GPT-3: Language Models are Few-Shot Learners",
     "full_text": "Large language models can perform tasks with few examples.",
     "pdf_downloaded": 1},
]

SAMPLE_CHUNKS = {
    "2401.00001": [
        "The transformer model uses self-attention to compute representations.",
        "Multi-head attention allows the model to attend to different positions.",
        "Positional encoding is added to give the model information about sequence order.",
    ],
    "2401.00002": [
        "BERT is trained on masked language modeling and next sentence prediction.",
        "The bidirectional context helps BERT understand word meaning in context.",
    ],
    "2401.00003": [
        "GPT-3 has 175 billion parameters and is trained on internet text.",
        "Few-shot learning allows the model to generalize from minimal examples.",
        "Prompt engineering is key to eliciting good responses from GPT-3.",
        "In-context learning happens at inference time without gradient updates.",
    ],
}


class MockEmbedder:
    dim = DIM
    model_name = "mock-model-v0"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.random.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def create_test_db(path: Path) -> None:
    from backend.database.schema import init_db
    from backend.database.embedding_repository import init_embedding_schema
    from backend.database.chunk_repository import save_chunks

    init_db(path)
    init_embedding_schema(path)
    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT OR IGNORE INTO documents "
        "(arxiv_id, title, full_text, pdf_downloaded, abstract, categories, published, updated, pdf_url, fetched_at) "
        "VALUES (:arxiv_id, :title, :full_text, :pdf_downloaded, '', '', '', '', '', '')",
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()
    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        save_chunks(arxiv_id, texts, db_path=path)


def total_chunks() -> int:
    return sum(len(v) for v in SAMPLE_CHUNKS.values())


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test_embedding.db"
    create_test_db(p)
    return p


@pytest.fixture
def faiss_dir(tmp_path):
    d = tmp_path / "faiss"
    d.mkdir()
    return d


@pytest.fixture
def mock_embedder():
    return MockEmbedder()
