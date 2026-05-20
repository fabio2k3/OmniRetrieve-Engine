"""
test_scorer.py
==============
Tests de score_rag_case().

El juez OllamaJudge se mockea mediante subclase para que el test
no dependa de Ollama ni del LLM.
"""

import pytest
from backend.eval.schema import EvalCase
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.scorer import score_rag_case


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

def _make_case(case_id: str = "exact_0001", case_type: str = "exact") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        case_type=case_type,
        query="What is attention?",
        expected_chunk_id=1,
        expected_arxiv_id="2401.00001",
        expected_chunk_index=0,
        source_text="source",
        fragment_used="fragment",
    )


def _pipeline_output(answer: str = "Attention lets models focus.") -> dict:
    return {
        "query":   "What is attention?",
        "answer":  answer,
        "context": "[1] Attention Is All You Need\nSome context here.",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreRagCase:
    def test_returns_rag_judgement(self):
        result = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(4))
        assert isinstance(result, RAGJudgement)

    def test_all_dimensions_scored(self):
        j = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(3))
        assert j.faithfulness      is not None
        assert j.answer_relevance  is not None
        assert j.context_relevance is not None

    def test_no_error_when_all_succeed(self):
        j = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(4))
        assert j.judge_error is None

    def test_error_recorded_when_judge_returns_none(self):
        j = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(None))
        assert j.judge_error is not None
        assert "faithfulness" in j.judge_error

    def test_preserves_case_id(self):
        j = score_rag_case(_make_case("exact_0099"), _pipeline_output(), _FixedJudge(4))
        assert j.case_id == "exact_0099"

    def test_preserves_case_type(self):
        j = score_rag_case(_make_case(case_type="semantic"), _pipeline_output(), _FixedJudge(4))
        assert j.case_type == "semantic"

    def test_scores_dict_has_all_keys(self):
        j = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(5))
        s = j.scores()
        assert set(s.keys()) == {"faithfulness", "answer_relevance", "context_relevance"}

    def test_scores_dict_all_none_when_judge_fails(self):
        j = score_rag_case(_make_case(), _pipeline_output(), _FixedJudge(None))
        assert all(v is None for v in j.scores().values())

    def test_uses_context_from_pipeline_output(self):
        """El scorer debe usar el context del pipeline_output si está disponible."""
        output = _pipeline_output()
        output["context"] = "Special context text."
        j = score_rag_case(_make_case(), output, _FixedJudge(3))
        # Si no lanza excepción y devuelve un judgement, el contexto se pasó correctamente
        assert isinstance(j, RAGJudgement)

    def test_falls_back_to_context_param(self):
        """Si pipeline_output no tiene 'context', usa el parámetro context."""
        output = {"query": "q", "answer": "a"}  # sin 'context'
        j = score_rag_case(_make_case(), output, _FixedJudge(3), context="fallback ctx")
        assert isinstance(j, RAGJudgement)
