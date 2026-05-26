"""Test de flujo completo (end-to-end): crawler → indexing → retrieval."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest

CORPUS = [
    {"arxiv_id": "2301.001", "title": "Attention Is All You Need",
     "full_text": "We propose a new simple network architecture, the Transformer, "
                  "based solely on attention mechanisms. Multi-head self-attention "
                  "allows the model to jointly attend to information from different "
                  "representation subspaces at different positions.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2301.002", "title": "BERT: Bidirectional Transformers for Language Understanding",
     "full_text": "We introduce BERT, a method of pre-training language representations "
                  "using bidirectional transformers. Masked language model enables deep "
                  "bidirectional pre-training. Fine-tuning yields state-of-the-art results "
                  "on natural language processing tasks.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2301.003", "title": "Adam: A Method for Stochastic Optimization",
     "full_text": "We introduce Adam, an algorithm for first-order gradient-based "
                  "optimization of stochastic objective functions. The method computes "
                  "adaptive learning rates for different parameters from estimates of "
                  "first and second moments of the gradients.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2301.004", "title": "Deep Residual Learning for Image Recognition",
     "full_text": "We present a residual learning framework to ease the training of "
                  "deep neural networks. Shortcut connections perform identity mapping "
                  "added to the outputs of stacked layers.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2301.005", "title": "Generative Adversarial Networks",
     "full_text": "We propose a generative adversarial network framework. A generative "
                  "model captures the data distribution while a discriminative model "
                  "estimates the probability that a sample came from the training data.",
     "pdf_downloaded": 1},
    {"arxiv_id": "2301.006", "title": "Pending Download Article",
     "full_text": None, "pdf_downloaded": 0},
]


def _create_db(path: Path) -> None:
    from backend.database.schema import init_db
    init_db(path)
    conn = sqlite3.connect(str(path))
    for doc in CORPUS:
        conn.execute(
            "INSERT OR IGNORE INTO documents "
            "(arxiv_id, title, abstract, authors, full_text, pdf_downloaded, published, pdf_url) "
            "VALUES (?, ?, '', '', ?, ?, '2023-01-01', '')",
            (doc["arxiv_id"], doc["title"], doc["full_text"], doc["pdf_downloaded"]),
        )
    conn.commit()
    conn.close()


def _add_chunks(db_path: Path) -> None:
    """Crea chunks para cada documento con PDF descargado."""
    from backend.crawler.chunker import make_chunks
    from backend.database.chunk_repository import save_chunks
    conn = sqlite3.connect(str(db_path))
    docs = conn.execute(
        "SELECT arxiv_id, full_text FROM documents WHERE pdf_downloaded=1 AND full_text IS NOT NULL"
    ).fetchall()
    conn.close()
    for arxiv_id, full_text in docs:
        chunks = make_chunks(full_text, chunk_size=300, overlap_sentences=1)
        save_chunks(arxiv_id, chunks or [full_text], db_path=db_path)


@pytest.fixture
def full_db(tmp_path):
    p = tmp_path / "db" / "test.db"
    p.parent.mkdir()
    _create_db(p)
    return p


@pytest.fixture
def model_path(tmp_path):
    p = tmp_path / "models" / "lsi_model.pkl"
    p.parent.mkdir()
    return p


# ── Schema ───────────────────────────────────────────────────────────────────

def test_schema_documents_inserted(full_db):
    conn = sqlite3.connect(str(full_db))
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert count == len(CORPUS)


def test_schema_only_pdf_docs_ready(full_db):
    conn = sqlite3.connect(str(full_db))
    count = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1 AND full_text IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert count == 5


# ── Indexing ─────────────────────────────────────────────────────────────────

@pytest.fixture
def indexed_db(full_db):
    from backend.indexing.pipeline import IndexingPipeline
    stats = IndexingPipeline(db_path=full_db, field="full_text").run(reindex=False)
    assert stats["docs_processed"] == 5
    _add_chunks(full_db)
    return full_db


def test_indexing_processes_only_pdf_docs(full_db):
    from backend.indexing.pipeline import IndexingPipeline
    stats = IndexingPipeline(db_path=full_db, field="full_text").run(reindex=False)
    assert stats["docs_processed"] == 5


def test_indexing_builds_vocabulary(indexed_db):
    from backend.database.index_repository import get_index_stats
    idx = get_index_stats(db_path=indexed_db)
    assert idx["vocab_size"] > 0
    assert idx["total_docs"] == 5


def test_indexing_incremental_skips_processed(indexed_db):
    from backend.indexing.pipeline import IndexingPipeline
    stats2 = IndexingPipeline(db_path=indexed_db, field="full_text").run(reindex=False)
    assert stats2["docs_processed"] == 0


# ── Retrieval ─────────────────────────────────────────────────────────────────

@pytest.fixture
def model_and_retriever(indexed_db, model_path):
    from backend.retrieval.lsi_model import LSIModel
    from backend.retrieval.lsi_retriever import LSIRetriever
    model = LSIModel(k=3)
    # min_df=1: test corpus is too small for the default min_df=20
    model.build(db_path=indexed_db, min_df=1)
    model.save(path=model_path)
    retriever = LSIRetriever()
    retriever.load(model_path=model_path, db_path=indexed_db)
    return model, retriever, indexed_db


def test_lsi_model_builds_correctly(model_and_retriever):
    model, _, _ = model_and_retriever
    assert len(model.doc_ids) == 5
    assert model.docs_latent.shape == (5, 3)


def test_retriever_returns_sorted_results(model_and_retriever):
    _, retriever, _ = model_and_retriever
    results = retriever.retrieve("attention transformer mechanism", top_n=3)
    assert len(results) > 0
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_results_have_required_fields(model_and_retriever):
    _, retriever, _ = model_and_retriever
    results = retriever.retrieve("attention transformer", top_n=3)
    for r in results:
        assert hasattr(r, "score")
        assert hasattr(r, "arxiv_id")
        assert hasattr(r, "chunk_id")


# ── Incremental ───────────────────────────────────────────────────────────────

def test_incremental_new_document(indexed_db, model_path):
    from backend.database.crawler_repository import upsert_document, save_pdf_text
    from backend.indexing.pipeline import IndexingPipeline
    from backend.retrieval.lsi_model import LSIModel
    from backend.retrieval.lsi_retriever import LSIRetriever
    from backend.crawler.chunker import make_chunks
    from backend.database.chunk_repository import save_chunks

    new_text = (
        "Denoising diffusion probabilistic models achieve high quality image synthesis. "
        "The forward process adds Gaussian noise incrementally. Results surpass GANs."
    )
    upsert_document(
        arxiv_id="2301.007", title="Diffusion Models for Image Synthesis",
        authors="Ho et al.", abstract="Denoising diffusion probabilistic models.",
        categories="cs.CV", published="2023-01-07", updated="2023-01-07",
        pdf_url="https://arxiv.org/pdf/2301.007", fetched_at="2023-01-07",
        db_path=indexed_db,
    )
    save_pdf_text("2301.007", new_text, db_path=indexed_db)
    chunks = make_chunks(new_text, chunk_size=300, overlap_sentences=1)
    save_chunks("2301.007", chunks or [new_text], db_path=indexed_db)

    stats = IndexingPipeline(db_path=indexed_db, field="full_text").run(reindex=False)
    assert stats["docs_processed"] == 1

    model = LSIModel(k=3)
    model.build(db_path=indexed_db, min_df=1)
    assert len(model.doc_ids) == 6
    model.save(path=model_path)

    retriever = LSIRetriever()
    retriever.load(model_path=model_path, db_path=indexed_db)
    results = retriever.retrieve("diffusion noise image generation", top_n=6)
    ids = [r.arxiv_id for r in results]
    assert "2301.007" in ids