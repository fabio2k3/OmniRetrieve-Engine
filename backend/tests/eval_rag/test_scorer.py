"""
test_scorer.py
==============
Tests de score_rag_query().

El juez OllamaJudge se mockea mediante subclase para que el test
no dependa de Ollama ni del LLM.

Adaptado a la firma real:
    score_rag_query(query_id, query, pipeline_output, judge) -> RAGJudgement
Dimensiones evaluadas: faithfulness, answer_relevance (no context_relevance).
"""

import pytest
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.scorer import score_rag_query


# ---------------------------------------------------------------------------
# Mock judge
# ---------------------------------------------------------------------------

class _FixedJudge:
    """Juez que devuelve siempre una puntuación fija o None."""

    def __init__(self, score: int | None = 4):
        self._score = score

    def evaluate(self, prompt: str, dimension):
        if self._score is None:
            return None
        return DimensionScore.from_raw(self._score, reason="ok", dimension=dimension)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline_output(answer: str = "Attention lets models focus.") -> dict:
    return {
        "query":   "What is attention?",
        "answer":  answer,
        "context": "[1] Attention Is All You Need\nSome context here.",
    }


def _score(score=4):
    return score_rag_query(
        query_id="query_0001",
        query="What is attention?",
        pipeline_output=_pipeline_output(),
        judge=_FixedJudge(score),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreRagQuery:
    def test_returns_rag_judgement(self):
        assert isinstance(_score(), RAGJudgement)

    def test_both_dimensions_scored(self):
        j = _score(3)
        assert j.faithfulness is not None
        assert j.answer_relevance is not None

    def test_no_error_when_all_succeed(self):
        assert _score(4).judge_error is None

    def test_error_recorded_when_judge_returns_none(self):
        j = _score(None)
        assert j.judge_error is not None
        assert "faithfulness" in j.judge_error

    def test_preserves_query_id(self):
        j = score_rag_query("my_query_99", "q", _pipeline_output(), _FixedJudge(4))
        assert j.query_id == "my_query_99"

    def test_scores_dict_has_expected_keys(self):
        s = _score(5).scores()
        assert set(s.keys()) == {"faithfulness", "answer_relevance"}

    def test_scores_dict_all_none_when_judge_fails(self):
        assert all(v is None for v in _score(None).scores().values())

    def test_uses_context_from_pipeline_output(self):
        output = _pipeline_output()
        output["context"] = "Special context text."
        j = score_rag_query("q1", "What is attention?", output, _FixedJudge(3))
        assert isinstance(j, RAGJudgement)

    def test_missing_answer_in_output(self):
        """Si 'answer' no está en pipeline_output, no debe lanzar excepción."""
        j = score_rag_query("q1", "q", {"context": "ctx"}, _FixedJudge(3))
        assert isinstance(j, RAGJudgement)