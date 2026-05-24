"""Tests de LSIRetriever — carga, retrieve, relevancia semántica, lsi_log."""
from __future__ import annotations
import sqlite3
import pytest


@pytest.fixture
def retriever(lsi_model):
    from backend.retrieval.lsi_retriever import LSIRetriever
    model, model_path, db_path = lsi_model
    r = LSIRetriever()
    r.load(model_path=model_path, db_path=db_path)
    return r, db_path


def test_retriever_has_vectorizer(retriever):
    r, _ = retriever
    assert r._vectorizer is not None


def test_retriever_vectorizer_has_word_index(retriever):
    r, _ = retriever
    assert len(r._vectorizer._word_index) > 0


def test_retrieve_returns_sorted_results(retriever):
    r, _ = retriever
    results = r.retrieve("attention transformer mechanism", top_n=3)
    assert len(results) > 0
    scores = [x.score for x in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_result_has_required_fields(retriever):
    r, _ = retriever
    results = r.retrieve("attention transformer", top_n=3)
    for result in results:
        assert hasattr(result, "score")
        assert hasattr(result, "arxiv_id")
        assert hasattr(result, "chunk_id")


def test_retrieve_semantic_attention(retriever):
    r, _ = retriever
    results = r.retrieve("attention transformer mechanism", top_n=5)
    assert len(results) > 0
    scores = [x.score for x in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_semantic_gradient(retriever):
    r, _ = retriever
    results = r.retrieve("gradient optimization learning rate", top_n=5)
    assert len(results) > 0
    scores = [x.score for x in results]
    assert scores == sorted(scores, reverse=True)


def test_lsi_log_registered(retriever):
    _, db_path = retriever
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT * FROM lsi_log ORDER BY id DESC LIMIT 1").fetchall()
    conn.close()
    assert len(rows) > 0