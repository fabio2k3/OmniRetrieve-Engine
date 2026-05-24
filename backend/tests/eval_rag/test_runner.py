"""
test_runner.py
==============
Tests de RAGEvalRunner con RAGQuerySet y juez completamente mockeados.
"""

import pytest
from unittest.mock import MagicMock

from backend.eval.rag.schema import RAGQuery, RAGQuerySet
from backend.eval.rag._types import DimensionScore, RAGJudgement
from backend.eval.rag.runner import RAGEvalRunner


def _make_query(query_id: str) -> RAGQuery:
    return RAGQuery(query_id=query_id, query=f"query_{query_id}")


def _make_queryset(*ids: str) -> RAGQuerySet:
    return RAGQuerySet(queries=[_make_query(qid) for qid in ids])


def _mock_pipeline(answer: str = "Generated answer.") -> MagicMock:
    mock = MagicMock()
    mock.ask.return_value = {"query": "q", "answer": answer, "context": "Some context."}
    return mock


def _mock_judge(score: int = 4) -> MagicMock:
    mock = MagicMock()
    mock.evaluate.return_value = DimensionScore.from_raw(
        raw=score, reason="ok", dimension="faithfulness"
    )
    return mock


class TestRAGEvalRunner:
    def test_returns_one_judgement_per_query(self):
        qs = _make_queryset("q1", "q2")
        results = RAGEvalRunner(pipeline=_mock_pipeline(), judge=_mock_judge()).run(qs)
        assert len(results) == 2

    def test_pipeline_called_once_per_query(self):
        pipeline = _mock_pipeline()
        qs = _make_queryset("q1", "q2", "q3")
        RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(qs)
        assert pipeline.ask.call_count == 3

    def test_pipeline_called_with_include_debug(self):
        pipeline = _mock_pipeline()
        qs = _make_queryset("q1")
        RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(qs)
        _, kwargs = pipeline.ask.call_args
        assert kwargs.get("include_debug") is True

    def test_pipeline_kwargs_forwarded(self):
        pipeline = _mock_pipeline()
        qs = _make_queryset("q1")
        RAGEvalRunner(
            pipeline=pipeline,
            judge=_mock_judge(),
            pipeline_kwargs={"top_k": 5, "max_chunks": 3},
        ).run(qs)
        _, kwargs = pipeline.ask.call_args
        assert kwargs.get("top_k") == 5
        assert kwargs.get("max_chunks") == 3

    def test_pipeline_error_yields_error_judgement(self):
        pipeline = MagicMock()
        pipeline.ask.side_effect = RuntimeError("Pipeline exploded")
        qs = _make_queryset("q1")
        results = RAGEvalRunner(pipeline=pipeline, judge=_mock_judge()).run(qs)
        assert len(results) == 1
        assert results[0].judge_error is not None
        assert "pipeline_error" in results[0].judge_error

    def test_on_progress_callback_fired(self):
        qs = _make_queryset(*[f"q{i}" for i in range(4)])
        calls = []
        def cb(i, total, j): calls.append((i, total))
        RAGEvalRunner(
            pipeline=_mock_pipeline(), judge=_mock_judge(), on_progress=cb
        ).run(qs)
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_empty_queryset(self):
        qs = _make_queryset()
        out = RAGEvalRunner(pipeline=_mock_pipeline(), judge=_mock_judge()).run(qs)
        assert out == []

    def test_judge_error_does_not_abort_run(self):
        pipeline = _mock_pipeline()
        judge = MagicMock()
        judge.evaluate.return_value = None
        qs = _make_queryset("q1", "q2")
        results = RAGEvalRunner(pipeline=pipeline, judge=judge).run(qs)
        assert len(results) == 2
        for r in results:
            assert r.judge_error is not None