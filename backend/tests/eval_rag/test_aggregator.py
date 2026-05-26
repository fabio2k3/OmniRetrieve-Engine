"""
test_aggregator.py
==================
Tests de aggregator.aggregate() para el módulo RAG.
Adaptado: RAGJudgement usa query_id (no case_id), sin context_relevance ni by_type.
"""

import pytest
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.aggregator import aggregate


def _dim(score: float, dimension="faithfulness") -> DimensionScore:
    raw = round(score * 4) + 1
    return DimensionScore.from_raw(raw=raw, reason="ok", dimension=dimension)


def _judgement(
    query_id:  str,
    f_score:   float | None = 0.75,
    ar_score:  float | None = 0.75,
    error:     str | None   = None,
) -> RAGJudgement:
    return RAGJudgement(
        query_id=query_id,
        query="q",
        answer="a",
        faithfulness=_dim(f_score, "faithfulness") if f_score is not None else None,
        answer_relevance=_dim(ar_score, "answer_relevance") if ar_score is not None else None,
        judge_error=error,
    )


class TestAggregate:
    def test_returns_rag_aggregated_metrics(self):
        from backend.eval.rag._types import RAGAggregatedMetrics
        m = aggregate([_judgement("q1")])
        assert isinstance(m, RAGAggregatedMetrics)

    def test_n_total_counts_all_cases(self):
        js = [_judgement(f"q{i}") for i in range(5)]
        assert aggregate(js).n_total == 5

    def test_n_errors_counts_cases_with_judge_error(self):
        js = [_judgement("q1"), _judgement("q2", error="faithfulness")]
        assert aggregate(js).n_errors == 1

    def test_faithfulness_mean(self):
        js = [_judgement("q1", f_score=1.0), _judgement("q2", f_score=0.0)]
        m = aggregate(js)
        assert m.faithfulness is not None
        assert m.faithfulness.mean == pytest.approx(0.5, abs=0.1)

    def test_none_score_excluded_from_stats(self):
        js = [_judgement("q1", f_score=None), _judgement("q2", f_score=1.0)]
        m = aggregate(js)
        assert m.faithfulness is not None
        assert m.faithfulness.n_cases == 1

    def test_all_none_scores_yields_none_stats(self):
        js = [_judgement("q1", f_score=None, ar_score=None)]
        m = aggregate(js)
        assert m.faithfulness is None
        assert m.answer_relevance is None

    def test_min_max(self):
        js = [_judgement("q1", f_score=0.0), _judgement("q2", f_score=1.0)]
        m = aggregate(js)
        assert m.faithfulness.minimum == pytest.approx(0.0, abs=0.01)
        assert m.faithfulness.maximum == pytest.approx(1.0, abs=0.01)

    def test_empty_list(self):
        m = aggregate([])
        assert m.n_total == 0
        assert m.faithfulness is None
        assert m.answer_relevance is None