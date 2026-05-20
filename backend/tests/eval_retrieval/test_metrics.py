"""
test_metrics.py
===============
Tests de las funciones matemáticas puras de metrics.py.

Sin mocks ni fixtures externas — solo valores primitivos.
"""

import math
import pytest
from backend.eval.retrieval.metrics import hit_at_k, mrr, ndcg_at_k


class TestHitAtK:
    def test_all_found_at_rank_1(self):
        assert hit_at_k([1, 1, 1], k=10) == 1.0

    def test_none_found(self):
        assert hit_at_k([None, None], k=10) == 0.0

    def test_partial(self):
        assert hit_at_k([1, None, 3], k=10) == pytest.approx(2 / 3)

    def test_rank_exactly_at_k(self):
        assert hit_at_k([10], k=10) == 1.0

    def test_rank_beyond_k(self):
        assert hit_at_k([11], k=10) == 0.0

    def test_empty_list(self):
        assert hit_at_k([], k=10) == 0.0

    def test_k_zero(self):
        assert hit_at_k([1, 2], k=0) == 0.0


class TestMRR:
    def test_rank_1(self):
        assert mrr([1]) == pytest.approx(1.0)

    def test_rank_2(self):
        assert mrr([2]) == pytest.approx(0.5)

    def test_rank_5(self):
        assert mrr([5]) == pytest.approx(0.2)

    def test_not_found(self):
        assert mrr([None]) == pytest.approx(0.0)

    def test_mixed(self):
        # (1.0 + 0.5 + 0.0) / 3
        assert mrr([1, 2, None]) == pytest.approx(0.5)

    def test_empty(self):
        assert mrr([]) == pytest.approx(0.0)

    def test_all_not_found(self):
        assert mrr([None, None, None]) == pytest.approx(0.0)


class TestNDCGAtK:
    def test_rank_1(self):
        # 1 / log2(2) = 1.0
        expected = 1.0 / math.log2(2)
        assert ndcg_at_k([1], k=10) == pytest.approx(expected)

    def test_rank_2(self):
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k([2], k=10) == pytest.approx(expected)

    def test_not_found(self):
        assert ndcg_at_k([None], k=10) == pytest.approx(0.0)

    def test_rank_beyond_k(self):
        assert ndcg_at_k([11], k=10) == pytest.approx(0.0)

    def test_rank_exactly_at_k(self):
        expected = 1.0 / math.log2(11)  # rank=10
        assert ndcg_at_k([10], k=10) == pytest.approx(expected)

    def test_averaged_over_multiple_cases(self):
        # rank 1 → 1/log2(2); None → 0  →  mean = 0.5/log2(2)
        r1 = 1.0 / math.log2(2)
        expected = (r1 + 0.0) / 2
        assert ndcg_at_k([1, None], k=10) == pytest.approx(expected)

    def test_empty(self):
        assert ndcg_at_k([], k=10) == pytest.approx(0.0)

    def test_k_zero(self):
        assert ndcg_at_k([1], k=0) == pytest.approx(0.0)
