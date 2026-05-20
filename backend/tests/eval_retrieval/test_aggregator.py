"""
test_aggregator.py
==================
Tests de aggregator.aggregate().

Verifica el desglose por tipo de caso y los valores numéricos agregados.
"""

import math
import pytest
from backend.eval.retrieval._types import RawHit
from backend.eval.retrieval.aggregator import aggregate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(case_type: str, rank: int | None, top_k: int = 10) -> RawHit:
    return RawHit(
        case_id=f"{case_type}_000",
        case_type=case_type,
        expected_chunk_id=1,
        found=rank is not None,
        rank=rank,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_overall_all_found(self):
        hits = [_hit("exact", 1), _hit("exact", 1)]
        m = aggregate(hits, top_k=10)
        assert m.overall.hit_at_k == pytest.approx(1.0)
        assert m.overall.mrr      == pytest.approx(1.0)

    def test_overall_none_found(self):
        hits = [_hit("exact", None), _hit("semantic", None)]
        m = aggregate(hits, top_k=10)
        assert m.overall.hit_at_k == pytest.approx(0.0)
        assert m.overall.mrr      == pytest.approx(0.0)

    def test_split_by_type(self):
        hits = [
            _hit("exact",    rank=1),
            _hit("exact",    rank=None),
            _hit("semantic", rank=2),
        ]
        m = aggregate(hits, top_k=10)
        assert m.exact    is not None
        assert m.semantic is not None
        assert m.exact.n_cases    == 2
        assert m.semantic.n_cases == 1

    def test_no_semantic_cases_yields_none(self):
        hits = [_hit("exact", 1), _hit("exact", 2)]
        m = aggregate(hits, top_k=10)
        assert m.semantic is None

    def test_no_exact_cases_yields_none(self):
        hits = [_hit("semantic", 1)]
        m = aggregate(hits, top_k=10)
        assert m.exact is None

    def test_exact_hit_rate(self):
        hits = [_hit("exact", 1), _hit("exact", None)]  # 1/2 = 0.5
        m = aggregate(hits, top_k=10)
        assert m.exact.hit_at_k == pytest.approx(0.5)

    def test_top_k_propagated(self):
        hits = [_hit("exact", 1)]
        m = aggregate(hits, top_k=5)
        assert m.top_k == 5

    def test_ndcg_rank_1(self):
        hits = [_hit("exact", 1)]
        m = aggregate(hits, top_k=10)
        expected = 1.0 / math.log2(2)
        assert m.overall.ndcg_at_k == pytest.approx(expected)

    def test_empty_hits(self):
        m = aggregate([], top_k=10)
        assert m.overall.n_cases == 0
        assert m.overall.hit_at_k == pytest.approx(0.0)
        assert m.exact    is None
        assert m.semantic is None
