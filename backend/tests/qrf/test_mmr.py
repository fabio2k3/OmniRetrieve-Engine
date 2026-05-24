"""Tests del MMRReranker — diversidad, fallback, parámetros inválidos."""
from __future__ import annotations
import numpy as np
import pytest

from .conftest import DIM


@pytest.fixture
def rng():
    return np.random.RandomState(42)


def _norm_vec(rng, dim=DIM):
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_mmr_basic_rerank(db_with_chunks, rng):
    from backend.qrf.mmr import MMRReranker
    db_path, chunk_ids = db_with_chunks
    mmr = MMRReranker(lambda_=0.6)
    qvec = _norm_vec(rng)
    candidates = [{"chunk_id": cid, "score": float(i) * 0.1} for i, cid in enumerate(chunk_ids)]
    reranked = mmr.rerank(candidates, qvec, top_n=4, db_path=db_path)
    assert len(reranked) <= 4
    for r in reranked:
        assert "mmr_score" in r
        assert isinstance(r["mmr_score"], float)


def test_mmr_no_duplicate_ids(db_with_chunks, rng):
    from backend.qrf.mmr import MMRReranker
    db_path, chunk_ids = db_with_chunks
    mmr = MMRReranker(lambda_=0.6)
    qvec = _norm_vec(rng)
    candidates = [{"chunk_id": cid, "score": float(i) * 0.1} for i, cid in enumerate(chunk_ids)]
    reranked = mmr.rerank(candidates, qvec, top_n=4, db_path=db_path)
    ids = [r["chunk_id"] for r in reranked]
    assert len(ids) == len(set(ids))


def test_mmr_diversity_vs_relevance(db_with_chunks, rng):
    from backend.qrf.mmr import MMRReranker
    db_path, chunk_ids = db_with_chunks
    qvec = _norm_vec(rng)
    candidates = [{"chunk_id": cid, "score": float(i) * 0.1} for i, cid in enumerate(chunk_ids)]
    diverse = MMRReranker(lambda_=0.1).rerank(candidates, qvec, top_n=3, db_path=db_path)
    relevant = MMRReranker(lambda_=0.9).rerank(candidates, qvec, top_n=3, db_path=db_path)
    for r in diverse + relevant:
        assert "chunk_id" in r
    # Both must have unique IDs
    assert len({r["chunk_id"] for r in diverse}) == len(diverse)
    assert len({r["chunk_id"] for r in relevant}) == len(relevant)


def test_mmr_fallback_without_embeddings(db_with_chunks, rng):
    from backend.qrf.mmr import MMRReranker
    db_path, _ = db_with_chunks
    mmr = MMRReranker(lambda_=0.6)
    qvec = _norm_vec(rng)
    fake_candidates = [{"chunk_id": 999990 + i, "score": float(i)} for i in range(5)]
    result = mmr.rerank(fake_candidates, qvec, top_n=3, db_path=db_path)
    assert len(result) == 3
    assert result[0]["chunk_id"] == 999990


def test_mmr_invalid_lambda_raises():
    from backend.qrf.mmr import MMRReranker
    for lam in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            MMRReranker(lambda_=lam)
