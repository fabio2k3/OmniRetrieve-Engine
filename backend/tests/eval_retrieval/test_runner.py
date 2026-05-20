"""
test_runner.py
==============
Tests de EvalRunner con un retriever completamente mockeado.

No toca la BD ni Ollama — verifica solo la lógica de orquestación.
"""

import pytest
from unittest.mock import MagicMock

from backend.eval.schema import EvalCase, EvalDataset
from backend.eval.retrieval.runner import EvalRunner
from backend.retrieval.protocols import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(case_id: str, chunk_id: int, case_type: str = "exact") -> EvalCase:
    return EvalCase(
        case_id=case_id,
        case_type=case_type,
        query=f"query for {case_id}",
        expected_chunk_id=chunk_id,
        expected_arxiv_id="2401.00001",
        expected_chunk_index=0,
        source_text="source text",
        fragment_used="fragment",
    )


def _make_result(chunk_id: int) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        arxiv_id="2401.00001",
        chunk_index=0,
        text="text",
        score=1.0,
    )


def _make_dataset(*cases: EvalCase) -> EvalDataset:
    return EvalDataset(cases=list(cases), db_path="/fake/db")


def _mock_retriever(results_by_query: dict[str, list[RetrievalResult]]):
    """Retriever falso que devuelve resultados predefinidos por query."""
    mock = MagicMock()
    mock.retrieve.side_effect = lambda query, top_n: results_by_query.get(query, [])
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvalRunner:
    def test_returns_one_hit_per_case(self):
        case = _make_case("exact_0001", chunk_id=10)
        ds   = _make_dataset(case)
        retriever = _mock_retriever({"query for exact_0001": [_make_result(10)]})

        runner = EvalRunner(retriever=retriever, top_k=10)
        hits = runner.run(ds)

        assert len(hits) == 1

    def test_found_when_chunk_in_results(self):
        case = _make_case("exact_0001", chunk_id=10)
        ds   = _make_dataset(case)
        retriever = _mock_retriever({"query for exact_0001": [_make_result(10)]})

        hits = EvalRunner(retriever=retriever, top_k=10).run(ds)
        assert hits[0].found is True
        assert hits[0].rank  == 1

    def test_not_found_when_chunk_absent(self):
        case = _make_case("exact_0001", chunk_id=99)
        ds   = _make_dataset(case)
        retriever = _mock_retriever({"query for exact_0001": [_make_result(10)]})

        hits = EvalRunner(retriever=retriever, top_k=10).run(ds)
        assert hits[0].found is False

    def test_retriever_called_once_per_case(self):
        cases = [_make_case(f"exact_{i:04d}", chunk_id=i) for i in range(5)]
        ds    = _make_dataset(*cases)
        retriever = _mock_retriever({})

        EvalRunner(retriever=retriever, top_k=10).run(ds)
        assert retriever.retrieve.call_count == 5

    def test_retriever_error_yields_not_found(self):
        case = _make_case("exact_0001", chunk_id=42)
        ds   = _make_dataset(case)

        mock = MagicMock()
        mock.retrieve.side_effect = RuntimeError("Retriever crashed")

        hits = EvalRunner(retriever=mock, top_k=10).run(ds)
        assert hits[0].found is False
        assert hits[0].rank  is None

    def test_on_progress_callback_called(self):
        cases = [_make_case(f"exact_{i:04d}", chunk_id=i) for i in range(3)]
        ds    = _make_dataset(*cases)
        retriever = _mock_retriever({})

        calls = []
        def cb(i, total, hit): calls.append((i, total))

        EvalRunner(retriever=retriever, top_k=10, on_progress=cb).run(ds)
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_empty_dataset(self):
        ds    = _make_dataset()
        retriever = _mock_retriever({})

        hits = EvalRunner(retriever=retriever, top_k=10).run(ds)
        assert hits == []

    def test_top_k_respected_in_scorer(self):
        """Chunk en posición 3 no debe encontrarse con top_k=2."""
        case = _make_case("exact_0001", chunk_id=42)
        ds   = _make_dataset(case)
        results = [_make_result(1), _make_result(2), _make_result(42)]
        retriever = _mock_retriever({"query for exact_0001": results})

        hits = EvalRunner(retriever=retriever, top_k=2).run(ds)
        assert hits[0].found is False
