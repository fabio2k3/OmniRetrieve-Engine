"""
test_report.py
==============
Tests de report.py del comparador.
"""

import json
import tempfile
from pathlib import Path

import pytest

from backend.eval.compare._types import ComparisonResult, MetricDelta
from backend.eval.compare.report import format_summary, save_json, load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delta(name: str, group: str, base: float, cand: float, status: str) -> MetricDelta:
    delta = cand - base
    pct   = (delta / base * 100) if base != 0 else None
    return MetricDelta(
        name=name, group=group,
        baseline=base, candidate=cand,
        delta=delta, delta_pct=pct,
        status=status,
    )


def _result(n_improved=2, n_degraded=1, n_neutral=3) -> ComparisonResult:
    deltas = (
        [_delta(f"overall.metric_{i}", "overall", 0.70, 0.80, "improved") for i in range(n_improved)]
        + [_delta(f"exact.metric_{i}",   "exact",   0.80, 0.70, "degraded") for i in range(n_degraded)]
        + [_delta(f"semantic.metric_{i}","semantic", 0.65, 0.65, "neutral")  for i in range(n_neutral)]
    )
    return ComparisonResult(
        baseline_label="report_v1.json",
        candidate_label="report_v2.json",
        report_type="retrieval",
        threshold=0.005,
        deltas=deltas,
        generated_at="2026-05-13T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Tests — format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_returns_string(self):
        assert isinstance(format_summary(_result()), str)

    def test_contains_labels(self):
        text = format_summary(_result())
        assert "report_v1.json" in text
        assert "report_v2.json" in text

    def test_contains_status_icons(self):
        text = format_summary(_result())
        assert "✓" in text
        assert "✗" in text
        assert "~" in text

    def test_contains_counts(self):
        text = format_summary(_result(n_improved=2, n_degraded=1, n_neutral=3))
        assert "Improved=2" in text
        assert "Degraded=1" in text
        assert "Neutral=3"  in text

    def test_contains_threshold(self):
        text = format_summary(_result())
        assert "0.005" in text

    def test_contains_report_type(self):
        text = format_summary(_result())
        assert "retrieval" in text

    def test_no_degraded_section_when_empty(self):
        r    = _result(n_degraded=0)
        text = format_summary(r)
        # No debe aparecer la sección de degraded si no hay
        assert "Degraded=0" in text


# ---------------------------------------------------------------------------
# Tests — save_json / load_json
# ---------------------------------------------------------------------------

class TestSaveLoadJson:
    def test_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sub" / "comparison.json"
            save_json(_result(), path=p)
            assert p.exists()

    def test_json_has_core_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            save_json(_result(), path=p)
            payload = json.loads(p.read_text())
            for key in ("generated_at", "baseline_label", "candidate_label",
                        "report_type", "threshold", "summary", "deltas"):
                assert key in payload

    def test_summary_counts_correct(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            save_json(_result(n_improved=2, n_degraded=1, n_neutral=3), path=p)
            s = json.loads(p.read_text())["summary"]
            assert s["n_improved"] == 2
            assert s["n_degraded"] == 1
            assert s["n_neutral"]  == 3

    def test_deltas_serialized(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            save_json(_result(), path=p)
            deltas = json.loads(p.read_text())["deltas"]
            assert len(deltas) == 6   # 2+1+3
            assert "name"   in deltas[0]
            assert "delta"  in deltas[0]
            assert "status" in deltas[0]

    def test_load_json_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            save_json(_result(), path=p)
            assert isinstance(load_json(p), dict)
