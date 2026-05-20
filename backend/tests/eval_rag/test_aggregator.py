"""
test_aggregator.py
==================
Tests de aggregator.aggregate() para el módulo RAG.
"""

import pytest
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.aggregator import aggregate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dim(score: float, dimension="faithfulness") -> DimensionScore:
    raw = round(score * 4) + 1  # invertir normalización aproximada
    return DimensionScore.from_raw(raw=raw, reason="ok", dimension=dimension)


def _judgement(
    case_id:   str,
    case_type: str,
    f_score:   float | None = 0.75,
    ar_score:  float | None = 0.75,
    cr_score:  float | None = 0.75,
    error:     str | None   = None,
) -> RAGJudgement:
    return RAGJudgement(
        case_id=case_id,
        case_type=case_type,
        query="q",
        answer="a",
        faithfulness=_dim(f_score,  "faithfulness")      if f_score  is not None else None,
        answer_relevance=_dim(ar_score, "answer_relevance") if ar_score is not None else None,
        context_relevance=_dim(cr_score,"context_relevance")if cr_score is not None else None,
        judge_error=error,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_returns_rag_aggregated_metrics(self):
        from backend.eval.rag._types import RAGAggregatedMetrics
        m = aggregate([_judgement("e1", "exact")])
        assert isinstance(m, RAGAggregatedMetrics)

    def test_n_total_counts_all_cases(self):
        js = [_judgement(f"e{i}", "exact") for i in range(5)]
        m  = aggregate(js)
        assert m.n_total == 5

    def test_n_errors_counts_cases_with_judge_error(self):
        js = [
            _judgement("e1", "exact"),
            _judgement("e2", "exact", error="faithfulness"),
        ]
        m = aggregate(js)
        assert m.n_errors == 1

    def test_faithfulness_mean(self):
        js = [
            _judgement("e1", "exact", f_score=1.0),
            _judgement("e2", "exact", f_score=0.0),
        ]
        m = aggregate(js)
        assert m.faithfulness is not None
        assert m.faithfulness.mean == pytest.approx(0.5, abs=0.1)

    def test_by_type_split(self):
        js = [
            _judgement("e1", "exact"),
            _judgement("s1", "semantic"),
        ]
        m = aggregate(js)
        assert "exact"    in m.by_type
        assert "semantic" in m.by_type
        assert m.by_type["exact"].n_total    == 1
        assert m.by_type["semantic"].n_total == 1

    def test_only_exact_no_semantic_key(self):
        js = [_judgement("e1", "exact"), _judgement("e2", "exact")]
        m  = aggregate(js)
        assert "semantic" not in m.by_type

    def test_none_score_excluded_from_stats(self):
        """Un caso sin puntuación de faithfulness no debe contar en n_cases de esa dim."""
        js = [
            _judgement("e1", "exact", f_score=None),
            _judgement("e2", "exact", f_score=1.0),
        ]
        m = aggregate(js)
        assert m.faithfulness is not None
        assert m.faithfulness.n_cases == 1

    def test_all_none_scores_yields_none_stats(self):
        js = [_judgement("e1", "exact", f_score=None, ar_score=None, cr_score=None)]
        m  = aggregate(js)
        assert m.faithfulness      is None
        assert m.answer_relevance  is None
        assert m.context_relevance is None

    def test_min_max(self):
        js = [
            _judgement("e1", "exact", f_score=0.0),
            _judgement("e2", "exact", f_score=1.0),
        ]
        m = aggregate(js)
        assert m.faithfulness.minimum == pytest.approx(0.0, abs=0.01)
        assert m.faithfulness.maximum == pytest.approx(1.0, abs=0.01)

    def test_empty_list(self):
        m = aggregate([])
        assert m.n_total          == 0
        assert m.faithfulness     is None
        assert m.by_type          == {}
