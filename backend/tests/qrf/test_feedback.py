"""Tests de BlindRelevanceFeedback y RocchioFeedback."""
from __future__ import annotations
import numpy as np
import pytest

from .conftest import DIM


@pytest.fixture
def rng():
    return np.random.RandomState(7)


def _norm_vec(rng, dim=DIM):
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── BlindRelevanceFeedback ───────────────────────────────────────────────────

def test_brf_adjust_normalizes_vector(db_with_chunks, rng):
    from backend.qrf.brf import BlindRelevanceFeedback
    db_path, chunk_ids = db_with_chunks
    brf = BlindRelevanceFeedback(alpha=0.75, top_k_rf=3)
    qvec = _norm_vec(rng)
    adjusted = brf.adjust(qvec, [{"chunk_id": cid} for cid in chunk_ids[:5]], db_path)
    assert adjusted.shape == (DIM,)
    assert abs(np.linalg.norm(adjusted) - 1.0) < 1e-5


def test_brf_adjust_positive_correlation(db_with_chunks, rng):
    from backend.qrf.brf import BlindRelevanceFeedback
    from backend.qrf._feedback_utils import cosine_similarity
    db_path, chunk_ids = db_with_chunks
    brf = BlindRelevanceFeedback(alpha=0.75, top_k_rf=3)
    qvec = _norm_vec(rng)
    adjusted = brf.adjust(qvec, [{"chunk_id": cid} for cid in chunk_ids[:5]], db_path)
    assert cosine_similarity(qvec, adjusted) > 0


def test_brf_fallback_without_embeddings(db_with_chunks, rng):
    from backend.qrf.brf import BlindRelevanceFeedback
    db_path, _ = db_with_chunks
    brf = BlindRelevanceFeedback(alpha=0.8)
    qvec = _norm_vec(rng)
    result = brf.adjust(qvec, [{"chunk_id": 999999}, {"chunk_id": 999998}], db_path)
    assert np.allclose(result, qvec)


def test_brf_invalid_alpha_raises():
    from backend.qrf.brf import BlindRelevanceFeedback
    for alpha in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            BlindRelevanceFeedback(alpha=alpha)


# ── RocchioFeedback ──────────────────────────────────────────────────────────

def test_rocchio_adjust_normalizes(db_with_chunks, rng):
    from backend.qrf.rocchio import RocchioFeedback
    db_path, chunk_ids = db_with_chunks
    rocchio = RocchioFeedback(alpha=0.6, beta=0.4, gamma=0.1)
    qvec = _norm_vec(rng)
    adjusted = rocchio.adjust("s1", qvec, chunk_ids[:2], chunk_ids[2:3], db_path)
    assert abs(np.linalg.norm(adjusted) - 1.0) < 1e-5


def test_rocchio_no_feedback_returns_original(db_with_chunks, rng):
    from backend.qrf.rocchio import RocchioFeedback
    db_path, _ = db_with_chunks
    rocchio = RocchioFeedback()
    qvec = _norm_vec(rng)
    result = rocchio.adjust("s_empty", qvec, [], [], db_path)
    assert np.allclose(result, qvec)


def test_rocchio_session_cache(db_with_chunks, rng):
    from backend.qrf.rocchio import RocchioFeedback
    db_path, chunk_ids = db_with_chunks
    rocchio = RocchioFeedback(alpha=0.6, beta=0.4, gamma=0.1)
    qvec = _norm_vec(rng)
    adjusted = rocchio.adjust("session_1", qvec, chunk_ids[:2], [], db_path)

    cached = rocchio.get_cached("session_1")
    assert cached is not None
    assert np.allclose(cached, adjusted)
    assert rocchio.get_cached("nonexistent") is None


def test_rocchio_clear_cache(db_with_chunks, rng):
    from backend.qrf.rocchio import RocchioFeedback
    db_path, chunk_ids = db_with_chunks
    rocchio = RocchioFeedback()
    for i in range(3):
        qvec = _norm_vec(rng)
        rocchio.adjust(f"s{i}", qvec, chunk_ids[:1], [], db_path)

    assert len(rocchio.cached_queries) == 3
    rocchio.clear_cache("s0")
    assert rocchio.get_cached("s0") is None
    assert len(rocchio.cached_queries) == 2
    rocchio.clear_cache()
    assert len(rocchio.cached_queries) == 0
