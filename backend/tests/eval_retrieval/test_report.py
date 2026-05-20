"""
test_report.py
==============
Tests de report.format_summary() y report.save_json() / load_json().

Solo verifica que la salida contiene los campos esperados — no valores exactos.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.eval.retrieval._types import AggregatedMetrics, MetricSet
from backend.eval.retrieval.report import format_summary, save_json, load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metric_set(n=10, hit=0.8, mrr=0.6, ndcg=0.7) -> MetricSet:
    return MetricSet(n_cases=n, hit_at_k=hit, mrr=mrr, ndcg_at_k=ndcg)


def _metrics(with_semantic: bool = True) -> AggregatedMetrics:
    return AggregatedMetrics(
        top_k=10,
        overall=_metric_set(),
        exact=_metric_set(n=6),
        semantic=_metric_set(n=4) if with_semantic else None,
    )


# ---------------------------------------------------------------------------
# Tests — format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_contains_top_k(self):
        text = format_summary(_metrics(), retriever_name="hybrid")
        assert "Top-K = 10" in text

    def test_contains_retriever_name(self):
        text = format_summary(_metrics(), retriever_name="hybrid")
        assert "hybrid" in text

    def test_contains_global_metrics(self):
        text = format_summary(_metrics())
        assert "Hit@K" in text
        assert "MRR" in text
        assert "NDCG@K" in text

    def test_contains_exact_section(self):
        text = format_summary(_metrics())
        assert "Exact" in text or "exact" in text.lower()

    def test_contains_semantic_section(self):
        text = format_summary(_metrics(with_semantic=True))
        assert "Semantic" in text or "semantic" in text.lower()

    def test_no_semantic_section_when_absent(self):
        text = format_summary(_metrics(with_semantic=False))
        # No debe aparecer el bloque semantic
        assert "Semantic" not in text

    def test_delta_shown_when_both_types_present(self):
        text = format_summary(_metrics(with_semantic=True))
        assert "Δ" in text or "delta" in text.lower() or "semantic" in text.lower()

    def test_returns_string(self):
        assert isinstance(format_summary(_metrics()), str)


# ---------------------------------------------------------------------------
# Tests — save_json / load_json
# ---------------------------------------------------------------------------

class TestSaveLoadJson:
    def test_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "subdir" / "report.json"
            save_json(_metrics(), path=p, retriever_name="lsi")
            assert p.exists()

    def test_json_has_expected_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            save_json(_metrics(), path=p)
            payload = json.loads(p.read_text())
            for key in ("generated_at", "top_k", "overall", "exact", "semantic"):
                assert key in payload

    def test_retriever_name_stored(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            save_json(_metrics(), path=p, retriever_name="embedding")
            payload = json.loads(p.read_text())
            assert payload["retriever"] == "embedding"

    def test_extra_fields_included(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            save_json(_metrics(), path=p, extra={"dataset_path": "/data/ds.json"})
            payload = json.loads(p.read_text())
            assert payload["dataset_path"] == "/data/ds.json"

    def test_load_json_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            save_json(_metrics(), path=p)
            result = load_json(p)
            assert isinstance(result, dict)

    def test_semantic_none_when_no_semantic_cases(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            save_json(_metrics(with_semantic=False), path=p)
            payload = json.loads(p.read_text())
            assert payload["semantic"] is None
