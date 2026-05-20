"""
test_scorer.py
==============
Tests de scorer.score_case().

Verifica la lógica de detección de chunk_id en la lista de resultados.
"""

import pytest
from backend.eval.schema import EvalCase
from backend.eval.retrieval.scorer import score_case
from backend.retrieval.protocols import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(expected_chunk_id: int = 42) -> EvalCase:
    return EvalCase(
        case_id="exact_0001",
        case_type="exact",
        query="attention mechanism",
        expected_chunk_id=expected_chunk_id,
        expected_arxiv_id="2401.00001",
        expected_chunk_index=0,
        source_text="Some text.",
        fragment_used="attention mechanism",
    )


def _make_result(chunk_id: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        arxiv_id="2401.00001",
        chunk_index=0,
        text="Result text.",
        score=score,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreCase:
    def test_found_at_rank_1(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(42), _make_result(99)]
        hit = score_case(case, results, top_k=10)
        assert hit.found is True
        assert hit.rank == 1

    def test_found_at_rank_3(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(1), _make_result(2), _make_result(42)]
        hit = score_case(case, results, top_k=10)
        assert hit.found is True
        assert hit.rank == 3

    def test_not_found(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(1), _make_result(2)]
        hit = score_case(case, results, top_k=10)
        assert hit.found is False
        assert hit.rank is None

    def test_found_beyond_top_k_not_counted(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(1), _make_result(2), _make_result(42)]
        hit = score_case(case, results, top_k=2)   # top_k=2 → rank 3 fuera de ventana
        assert hit.found is False
        assert hit.rank is None

    def test_found_exactly_at_top_k(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(1), _make_result(42)]
        hit = score_case(case, results, top_k=2)   # rank 2 == top_k=2 → dentro
        assert hit.found is True
        assert hit.rank == 2

    def test_empty_results(self):
        case = _make_case(expected_chunk_id=42)
        hit  = score_case(case, [], top_k=10)
        assert hit.found is False

    def test_preserves_case_metadata(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(42)]
        hit = score_case(case, results, top_k=10)
        assert hit.case_id   == case.case_id
        assert hit.case_type == case.case_type

    def test_reciprocal_rank_when_found(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(99), _make_result(42)]
        hit = score_case(case, results, top_k=10)
        assert hit.reciprocal_rank == pytest.approx(0.5)

    def test_reciprocal_rank_when_not_found(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(99)]
        hit = score_case(case, results, top_k=10)
        assert hit.reciprocal_rank == pytest.approx(0.0)

    def test_n_results_returned_reflects_full_list(self):
        case    = _make_case(expected_chunk_id=42)
        results = [_make_result(i) for i in range(15)]
        hit = score_case(case, results, top_k=10)
        assert hit.n_results_returned == 15
