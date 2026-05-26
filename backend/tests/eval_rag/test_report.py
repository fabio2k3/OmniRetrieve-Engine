"""
test_report.py
==============
Tests de report.py para el módulo RAG.

Verifica formato de texto, serialización JSON y guardado de veredictos.
Adaptado a la versión simplificada: solo faithfulness + answer_relevance.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.eval.rag._types import (
    DimensionScore, DimensionStats, RAGAggregatedMetrics, RAGJudgement
)
from backend.eval.rag.report import (
    format_summary, save_json, save_judgements
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(mean=0.75) -> DimensionStats:
    return DimensionStats(n_cases=10, mean=mean, minimum=0.25, maximum=1.0)


def _metrics() -> RAGAggregatedMetrics:
    return RAGAggregatedMetrics(
        n_total=20,
        n_errors=2,
        faithfulness=_stats(0.8),
        answer_relevance=_stats(0.7),
    )


def _judgement(query_id: str = "query_0001") -> RAGJudgement:
    dim = lambda d: DimensionScore.from_raw(4, "ok", d)
    return RAGJudgement(
        query_id=query_id,
        query="What is attention?",
        answer="Attention allows focus.",
        faithfulness=dim("faithfulness"),
        answer_relevance=dim("answer_relevance"),
    )


# ---------------------------------------------------------------------------
# Tests — format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_returns_string(self):
        assert isinstance(format_summary(_metrics()), str)

    def test_contains_pipeline_name(self):
        text = format_summary(_metrics(), pipeline_name="TestPipeline")
        assert "TestPipeline" in text

    def test_contains_faithfulness_label(self):
        text = format_summary(_metrics())
        assert "Faithfulness" in text

    def test_contains_answer_relevance_label(self):
        text = format_summary(_metrics())
        assert "Relevance" in text

    def test_contains_percentage(self):
        assert "%" in format_summary(_metrics())

    def test_error_count_shown(self):
        assert "2" in format_summary(_metrics())   # n_errors = 2


# ---------------------------------------------------------------------------
# Tests — save_json
# ---------------------------------------------------------------------------

class TestSaveJson:
    def test_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sub" / "rag_report.json"
            save_json(_metrics(), path=p)
            assert p.exists()

    def test_json_has_core_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            save_json(_metrics(), path=p)
            payload = json.loads(p.read_text())
            for key in ("generated_at", "n_total", "n_errors",
                        "faithfulness", "answer_relevance"):
                assert key in payload

    def test_pipeline_name_stored(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            save_json(_metrics(), path=p, pipeline_name="TestRAG")
            assert json.loads(p.read_text())["pipeline"] == "TestRAG"

    def test_extra_fields_included(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            save_json(_metrics(), path=p, extra={"dataset_path": "/data/ds.json"})
            assert json.loads(p.read_text())["dataset_path"] == "/data/ds.json"


# ---------------------------------------------------------------------------
# Tests — save_judgements
# ---------------------------------------------------------------------------

class TestSaveJudgements:
    def test_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement()], path=p)
            assert p.exists()

    def test_json_structure(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement("q1"), _judgement("q2")], path=p)
            payload = json.loads(p.read_text())
            assert payload["n_judgements"] == 2
            assert len(payload["judgements"]) == 2

    def test_judgement_fields_serialized(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement("query_0001")], path=p)
            j = json.loads(p.read_text())["judgements"][0]
            assert j["query_id"] == "query_0001"
            assert "faithfulness" in j
            assert "answer_relevance" in j

    def test_extra_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement()], path=p, extra={"run_id": "run_001"})
            assert json.loads(p.read_text())["run_id"] == "run_001"