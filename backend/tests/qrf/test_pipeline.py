"""Tests del QueryPipeline — search, session, refine, clear_session."""
from __future__ import annotations
import numpy as np
import pytest

from .conftest import DIM, MockLSIModel, MockEmbedder, MockFaissIndex, build_mock_word_index


def _build_pipeline(db_path, chunk_ids):
    from backend.qrf.pipeline import QueryPipeline
    from backend.qrf.query_expander import QueryExpander

    pipeline = QueryPipeline(db_path=db_path, expand=True, top_k_initial=len(chunk_ids))
    pipeline._embedder = MockEmbedder()

    faiss_mock = MockFaissIndex(dim=DIM)
    rng = np.random.RandomState(0)
    for cid in chunk_ids:
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        faiss_mock.add(v.reshape(1, -1), [cid])
    pipeline._faiss_mgr = faiss_mock

    mock_model = MockLSIModel()
    expander = QueryExpander(
        lsi_model=mock_model, top_dims=2, top_terms_per_dim=5,
        min_correlation=0.1, max_expansion=4,
    )
    expander._model = mock_model
    expander._word_index, expander._idx_to_word = build_mock_word_index(mock_model)
    pipeline._expander = expander
    pipeline._expand_enabled = True
    return pipeline


def test_search_returns_results(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    results = pipeline.search("attention transformer mechanism", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_search_result_fields(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    results = pipeline.search("attention transformer mechanism", top_k=3)
    for r in results:
        assert "chunk_id" in r
        assert "score" in r
        assert "arxiv_id" in r
        assert "text" in r
        assert "title" in r
        assert "expanded_terms" in r


def test_search_with_session_returns_session_id(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    results, sid = pipeline.search_with_session("graph neural networks", top_k=4)
    assert isinstance(sid, str) and len(sid) > 0
    assert sid in pipeline._session_vectors


def test_refine_with_relevant_id(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    results1, sid = pipeline.search_with_session("diffusion models", top_k=4)
    if results1:
        results2 = pipeline.refine(sid, [results1[0]["chunk_id"]], [], top_k=4)
        assert isinstance(results2, list)
        assert len(results2) <= 4


def test_refine_invalid_session_raises(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    with pytest.raises(KeyError):
        pipeline.refine("nonexistent_session_id", [chunk_ids[0]], [], top_k=5)


def test_clear_session_individual(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    _, sid_a = pipeline.search_with_session("diffusion models", top_k=3)
    _, sid_b = pipeline.search_with_session("attention heads", top_k=3)

    pipeline.clear_session(sid_a)
    assert sid_a not in pipeline._session_vectors
    assert sid_b in pipeline._session_vectors


def test_clear_session_all(db_with_chunks):
    db_path, chunk_ids = db_with_chunks
    pipeline = _build_pipeline(db_path, chunk_ids)
    for q in ["query one", "query two", "query three"]:
        pipeline.search_with_session(q, top_k=3)
    pipeline.clear_session()
    assert len(pipeline._session_vectors) == 0
