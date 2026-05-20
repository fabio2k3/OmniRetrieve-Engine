"""
test_differ.py
==============
Tests de las funciones puras de differ.py.
Sin I/O — todo son dicts en memoria.
"""

import pytest
from backend.eval.compare.differ import (
    detect_type,
    extract_metrics,
    compute_deltas,
    compare_reports,
    DEFAULT_THRESHOLD,
)
from backend.eval.compare._types import ComparisonResult


# ---------------------------------------------------------------------------
# Fixtures de reportes sintéticos
# ---------------------------------------------------------------------------

def _retrieval_report(
    hit=0.74, mrr=0.61, ndcg=0.68,
    exact_hit=0.80, semantic_hit=0.68,
) -> dict:
    return {
        "retriever": "hybrid",
        "top_k": 10,
        "overall":  {"n_cases": 50, "hit_at_k": hit,       "mrr": mrr,  "ndcg_at_k": ndcg},
        "exact":    {"n_cases": 25, "hit_at_k": exact_hit,  "mrr": 0.65, "ndcg_at_k": 0.72},
        "semantic": {"n_cases": 25, "hit_at_k": semantic_hit,"mrr": 0.57, "ndcg_at_k": 0.64},
    }


def _rag_report(f=0.72, ar=0.68, cr=0.70) -> dict:
    return {
        "pipeline": "RAGPipeline",
        "n_total": 50, "n_errors": 2,
        "faithfulness":      {"n_cases": 48, "mean": f,  "minimum": 0.25, "maximum": 1.0},
        "answer_relevance":  {"n_cases": 48, "mean": ar, "minimum": 0.25, "maximum": 1.0},
        "context_relevance": {"n_cases": 48, "mean": cr, "minimum": 0.25, "maximum": 1.0},
        "by_type": {
            "exact":    {"faithfulness": {"mean": 0.75}, "answer_relevance": {"mean": 0.70},
                         "context_relevance": {"mean": 0.73}},
            "semantic": {"faithfulness": {"mean": 0.69}, "answer_relevance": {"mean": 0.66},
                         "context_relevance": {"mean": 0.67}},
        },
    }


# ---------------------------------------------------------------------------
# detect_type
# ---------------------------------------------------------------------------

class TestDetectType:
    def test_identifies_retrieval(self):
        assert detect_type(_retrieval_report()) == "retrieval"

    def test_identifies_rag(self):
        assert detect_type(_rag_report()) == "rag"

    def test_unknown_returns_unknown(self):
        assert detect_type({"foo": "bar"}) == "unknown"

    def test_empty_dict_is_unknown(self):
        assert detect_type({}) == "unknown"


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_retrieval_extracts_overall(self):
        m = extract_metrics(_retrieval_report())
        assert "overall.hit_at_k" in m
        assert "overall.mrr"      in m
        assert "overall.ndcg_at_k" in m

    def test_retrieval_extracts_exact_and_semantic(self):
        m = extract_metrics(_retrieval_report())
        assert "exact.hit_at_k"    in m
        assert "semantic.hit_at_k" in m

    def test_retrieval_values_correct(self):
        m = extract_metrics(_retrieval_report(hit=0.74))
        _, val = m["overall.hit_at_k"]
        assert val == pytest.approx(0.74)

    def test_rag_extracts_all_dimensions(self):
        m = extract_metrics(_rag_report())
        assert "overall.faithfulness"      in m
        assert "overall.answer_relevance"  in m
        assert "overall.context_relevance" in m

    def test_rag_extracts_by_type(self):
        m = extract_metrics(_rag_report())
        assert "exact.faithfulness"    in m
        assert "semantic.faithfulness" in m

    def test_rag_values_correct(self):
        m = extract_metrics(_rag_report(f=0.72))
        _, val = m["overall.faithfulness"]
        assert val == pytest.approx(0.72)

    def test_missing_group_skipped(self):
        report = _retrieval_report()
        del report["semantic"]
        m = extract_metrics(report)
        assert not any(k.startswith("semantic.") for k in m)


# ---------------------------------------------------------------------------
# compute_deltas
# ---------------------------------------------------------------------------

class TestComputeDeltas:
    def test_positive_delta_is_improved(self):
        base = _retrieval_report(hit=0.70)
        cand = _retrieval_report(hit=0.80)
        deltas = compute_deltas(base, cand, threshold=0.005)
        d = next(x for x in deltas if x.name == "overall.hit_at_k")
        assert d.status == "improved"
        assert d.delta  == pytest.approx(0.10)

    def test_negative_delta_is_degraded(self):
        base = _retrieval_report(hit=0.80)
        cand = _retrieval_report(hit=0.70)
        deltas = compute_deltas(base, cand, threshold=0.005)
        d = next(x for x in deltas if x.name == "overall.hit_at_k")
        assert d.status == "degraded"

    def test_tiny_delta_is_neutral(self):
        base = _retrieval_report(hit=0.700)
        cand = _retrieval_report(hit=0.702)
        deltas = compute_deltas(base, cand, threshold=0.005)
        d = next(x for x in deltas if x.name == "overall.hit_at_k")
        assert d.status == "neutral"

    def test_delta_pct_computed(self):
        base = _retrieval_report(hit=0.50)
        cand = _retrieval_report(hit=0.60)
        deltas = compute_deltas(base, cand, threshold=0.005)
        d = next(x for x in deltas if x.name == "overall.hit_at_k")
        assert d.delta_pct == pytest.approx(20.0)

    def test_only_common_metrics_compared(self):
        base = _retrieval_report()
        del base["semantic"]
        cand = _retrieval_report()
        deltas = compute_deltas(base, cand)
        names = {d.name for d in deltas}
        assert not any(n.startswith("semantic.") for n in names)

    def test_identical_reports_all_neutral(self):
        report = _retrieval_report()
        deltas = compute_deltas(report, report, threshold=0.005)
        assert all(d.status == "neutral" for d in deltas)

    def test_custom_threshold(self):
        base = _retrieval_report(hit=0.70)
        cand = _retrieval_report(hit=0.71)
        # Con threshold=0.005 debería ser improved; con threshold=0.02 neutral
        d_strict = compute_deltas(base, cand, threshold=0.005)
        d_loose  = compute_deltas(base, cand, threshold=0.02)
        hit_strict = next(x for x in d_strict if x.name == "overall.hit_at_k")
        hit_loose  = next(x for x in d_loose  if x.name == "overall.hit_at_k")
        assert hit_strict.status == "improved"
        assert hit_loose.status  == "neutral"


# ---------------------------------------------------------------------------
# compare_reports
# ---------------------------------------------------------------------------

class TestCompareReports:
    def test_returns_comparison_result(self):
        r = compare_reports(_retrieval_report(), _retrieval_report())
        assert isinstance(r, ComparisonResult)

    def test_labels_stored(self):
        r = compare_reports(
            _retrieval_report(), _retrieval_report(),
            baseline_label="v1", candidate_label="v2",
        )
        assert r.baseline_label  == "v1"
        assert r.candidate_label == "v2"

    def test_retrieval_type_detected(self):
        r = compare_reports(_retrieval_report(), _retrieval_report())
        assert r.report_type == "retrieval"

    def test_rag_type_detected(self):
        r = compare_reports(_rag_report(), _rag_report())
        assert r.report_type == "rag"

    def test_mixed_type_when_different_reports(self):
        r = compare_reports(_retrieval_report(), _rag_report())
        assert r.report_type == "mixed"

    def test_helper_methods(self):
        base = _retrieval_report(hit=0.60, mrr=0.80)
        cand = _retrieval_report(hit=0.80, mrr=0.60)
        r = compare_reports(base, cand, threshold=0.005)
        assert len(r.improved()) > 0
        assert len(r.degraded()) > 0

    def test_by_group_filters(self):
        r = compare_reports(_retrieval_report(), _retrieval_report())
        overall_deltas = r.by_group("overall")
        assert all(d.group == "overall" for d in overall_deltas)

    def test_generated_at_set(self):
        r = compare_reports(_retrieval_report(), _retrieval_report())
        assert r.generated_at != ""
