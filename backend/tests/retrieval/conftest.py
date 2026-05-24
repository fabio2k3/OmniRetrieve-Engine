"""Fixtures para los tests del módulo retrieval."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest

SAMPLE_DOCS = [
    ("2301.001", "Attention Is All You Need",
     "We propose transformer architecture with self-attention mechanisms "
     "for sequence to sequence tasks in natural language processing"),
    ("2301.002", "BERT: Bidirectional Transformers",
     "Pre-training deep bidirectional transformers for language understanding "
     "using masked language model and next sentence prediction objectives"),
    ("2301.003", "Gradient Descent Optimization",
     "Stochastic gradient descent convergence in deep neural networks "
     "with adaptive learning rate methods including Adam and RMSprop"),
    ("2301.004", "Reinforcement Learning Policy Gradient",
     "Policy gradient methods for deep reinforcement learning agents "
     "using actor-critic architectures and proximal policy optimization"),
    ("2301.005", "Convolutional Neural Networks for Vision",
     "Deep convolutional networks for image classification recognition "
     "using residual connections and batch normalization layers"),
]


def create_retrieval_db(path: Path) -> None:
    from backend.database.schema import init_db
    init_db(path)
    conn = sqlite3.connect(str(path))
    for arxiv_id, title, full_text in SAMPLE_DOCS:
        conn.execute(
            "INSERT INTO documents "
            "(arxiv_id, title, abstract, authors, full_text, text_length, "
            "pdf_downloaded, published, pdf_url) "
            "VALUES (?, ?, '', '', ?, ?, 1, '2023-01-01', '')",
            (arxiv_id, title, full_text, len(full_text)),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test_retrieval.db"
    create_retrieval_db(p)
    return p


@pytest.fixture
def indexed_db(db_path):
    from backend.indexing.pipeline import IndexingPipeline
    from backend.crawler.chunker import make_chunks
    from backend.database.chunk_repository import save_chunks

    IndexingPipeline(db_path=db_path, field="full_text", batch_size=50).run(reindex=True)

    # Create chunks so LSIRetriever._expand_to_chunks() returns results
    conn = sqlite3.connect(str(db_path))
    docs = conn.execute(
        "SELECT arxiv_id, full_text FROM documents WHERE pdf_downloaded=1"
    ).fetchall()
    conn.close()
    for arxiv_id, full_text in docs:
        if full_text:
            chunks = make_chunks(full_text, chunk_size=300, overlap_sentences=1)
            save_chunks(arxiv_id, chunks or [full_text], db_path=db_path)

    return db_path


@pytest.fixture
def lsi_model(indexed_db, tmp_path):
    from backend.retrieval.lsi_model import LSIModel
    model = LSIModel(k=3)
    # min_df=1: the test corpus is too small for the default min_df=20
    model.build(db_path=indexed_db, min_df=1)
    model_path = tmp_path / "model.pkl"
    model.save(path=model_path)
    return model, model_path, indexed_db


@pytest.fixture
def db_with_terms(tmp_path):
    """DB real con terms tabla para test_lsi_query (evita patching frágil)."""
    from backend.database.schema import init_db
    p = tmp_path / "terms.db"
    init_db(p)
    conn = sqlite3.connect(str(p))
    for i in range(1, 6):
        conn.execute(
            "INSERT OR IGNORE INTO terms (term_id, word, df) VALUES (?, ?, ?)",
            (i, f"word_{i}", 2),
        )
    conn.commit()
    conn.close()
    return p