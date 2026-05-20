"""
test_runner.py
==============
Tests de RAGEvalRunner con pipeline RAG y juez completamente mockeados.
"""

import pytest
from unittest.mock import MagicMock

from backend.eval.schema import EvalCase, EvalDataset
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.runner import RAGEvalRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(case_id: str, case_type: str = "exact") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        case_type=case_type,
        query=f"query_{case_id}",
        expected_chunk_id=1,
        expected_arxiv_id="2401.00001",
        expected_chunk_index=0,
        source_text="src",
        fragment_used="frag",
    )


def _make_dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(cases=list(cases), db_path="/fake")


def _mock_pipeline(answer: str = "Generated answer.") -> MagicMock:
    mock = MagicMock()
    mock.ask.return_value = {
        "query":   "query",
        "answer":  answer,
        "context": "Some context.",
    }
    return mock


def _mock_judge(score: int = 4) -> MagicMock:
    mock = MagicMock()
    mock.evaluate.return_value = DimensionScore.from_raw(
        raw=score, reason="ok", dimension="faithfulness"
    )
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGEvalRunner:
    def test_returns_one_judgement_per_case(self):
        ds       = _make_dataset(_make_case("e1"), _make_case("e2"))
        runner   = RAGEvalRunner(pipeline=_mock_pipeline(), judge=_mock_judge())
        results  = runner.run(ds)
        assert len(results) == 2

    def test_pipeline_called_once_per_case(self):
        pipeline = _mock_pipeline()
        ds       = _make_dataset(_make_case("e1"), _make_case("e2"), _make_case("e3"))
        RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(ds)
        assert pipeline.ask.call_count == 3

    def test_pipeline_called_with_include_debug(self):
        pipeline = _mock_pipeline()
        ds       = _make_dataset(_make_case("e1"))
        RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(ds)
        _, kwargs = pipeline.ask.call_args
        assert kwargs.get("include_debug") is True

    def test_pipeline_kwargs_forwarded(self):
        pipeline = _mock_pipeline()
        ds       = _make_dataset(_make_case("e1"))
        RAGEvalRunner(
            pipeline=pipeline,
            judge=_mock_judge(),
            pipeline_kwargs={"top_k": 5, "max_chunks": 3},
        ).run(ds)
        _, kwargs = pipeline.ask.call_args
        assert kwargs.get("top_k") == 5
        assert kwargs.get("max_chunks") == 3

    def test_pipeline_error_yields_error_judgement(self):
        pipeline = MagicMock()
        pipeline.ask.side_effect = RuntimeError("Pipeline exploded")
        ds = _make_dataset(_make_case("e1"))

        results = RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(ds)
        assert len(results) == 1
        assert results[0].judge_error is not None
        assert "pipeline_error" in results[0].judge_error

    def test_on_progress_callback_fired(self):
        ds    = _make_dataset(*[_make_case(f"e{i}") for i in range(4)])
        calls = []
        def cb(i, total, j): calls.append((i, total))

        RAGEvalRunner(
            pipeline=_mock_pipeline(),
            judge=_mock_judge(),
            on_progress=cb,
        ).run(ds)
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_empty_dataset(self):
        ds  = _make_dataset()
        out = RAGEvalRunner(pipeline=_mock_pipeline(), judge=_mock_judge()).run(ds)
        assert out == []

    def test_judge_error_does_not_abort_run(self):
        """Si el juez falla en un caso, el runner continúa con el siguiente."""
        pipeline = _mock_pipeline()
        judge    = MagicMock()
        judge.evaluate.return_value = None  # juez falla siempre

        ds      = _make_dataset(_make_case("e1"), _make_case("e2"))
        results = RAGEvalRunner(pipeline=pipeline, judge=judge).run(ds)
        assert len(results) == 2
        for r in results:
            assert r.judge_error is not None
