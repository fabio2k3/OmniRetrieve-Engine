"""
test_report.py
==============
Tests de report.py para el módulo RAG.

Verifica formato de texto, serialización JSON y guardado de veredictos.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.eval.rag._types import (
    DimensionScore, DimensionStats, RAGAggregatedMetrics, RAGJudgement
)
from backend.eval.rag.report import (
    format_summary, save_json, load_json, save_judgements
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(mean=0.75) -> DimensionStats:
    return DimensionStats(n_cases=10, mean=mean, minimum=0.25, maximum=1.0)


def _metrics(with_by_type: bool = True) -> RAGAggregatedMetrics:
    m = RAGAggregatedMetrics(
        n_total=20,
        n_errors=2,
        faithfulness=_stats(0.8),
        answer_relevance=_stats(0.7),
        context_relevance=_stats(0.6),
    )
    if with_by_type:
        m.by_type["exact"]    = RAGAggregatedMetrics(10, 1, _stats(0.85), _stats(0.75), _stats(0.65))
        m.by_type["semantic"] = RAGAggregatedMetrics(10, 1, _stats(0.75), _stats(0.65), _stats(0.55))
    return m


def _judgement(case_id: str = "exact_0001") -> RAGJudgement:
    dim = lambda d: DimensionScore.from_raw(4, "ok", d)
    return RAGJudgement(
        case_id=case_id,
        case_type="exact",
        query="What is attention?",
        answer="Attention allows focus.",
        faithfulness=dim("faithfulness"),
        answer_relevance=dim("answer_relevance"),
        context_relevance=dim("context_relevance"),
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

    def test_contains_dimension_labels(self):
        text = format_summary(_metrics())
        assert "Faithfulness" in text
        assert "Relevance" in text

    def test_contains_percentage(self):
        text = format_summary(_metrics())
        assert "%" in text

    def test_contains_exact_section(self):
        text = format_summary(_metrics(with_by_type=True))
        assert "Exact" in text

    def test_contains_semantic_section(self):
        text = format_summary(_metrics(with_by_type=True))
        assert "Semantic" in text

    def test_error_count_shown(self):
        text = format_summary(_metrics())
        assert "2" in text   # n_errors = 2


# ---------------------------------------------------------------------------
# Tests — save_json / load_json
# ---------------------------------------------------------------------------

class TestSaveLoadJson:
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
                        "faithfulness", "answer_relevance", "context_relevance"):
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

    def test_load_json_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            save_json(_metrics(), path=p)
            assert isinstance(load_json(p), dict)


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
            save_judgements([_judgement("e1"), _judgement("e2")], path=p)
            payload = json.loads(p.read_text())
            assert payload["n_judgements"] == 2
            assert len(payload["judgements"]) == 2

    def test_judgement_fields_serialized(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement("exact_0001")], path=p)
            j = json.loads(p.read_text())["judgements"][0]
            assert j["case_id"] == "exact_0001"
            assert "faithfulness" in j
            assert "answer_relevance" in j

    def test_extra_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "j.json"
            save_judgements([_judgement()], path=p, extra={"run_id": "run_001"})
            assert json.loads(p.read_text())["run_id"] == "run_001"
