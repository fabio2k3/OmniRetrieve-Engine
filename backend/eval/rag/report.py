"""
rag/report.py
=============
Presentación y serialización de resultados de evaluación RAG.

Salidas soportadas
------------------
format_summary()  — texto legible para consola.
save_json()       — JSON estructurado para análisis posterior.
save_judgements() — lista completa de RAGJudgement (trazabilidad).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ._types import DimensionStats, RAGAggregatedMetrics, RAGJudgement


# ---------------------------------------------------------------------------
# Salida texto
# ---------------------------------------------------------------------------

def _fmt_dim(label: str, stats: DimensionStats | None) -> str:
    if stats is None:
        return f"  {label:<22}: N/A\n"
    pct = stats.mean * 100
    return (
        f"  {label:<22}: {stats.mean:.4f}  ({pct:.1f}%)  "
        f"[min={stats.minimum:.2f}  max={stats.maximum:.2f}  n={stats.n_cases}]\n"
    )


def format_summary(metrics: RAGAggregatedMetrics, pipeline_name: str = "") -> str:
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"Pipeline: {pipeline_name}" if pipeline_name else "RAG Evaluation"
    error_note = (
        f"  ⚠  Consultas con errores de juez: {metrics.n_errors}\n"
        if metrics.n_errors else ""
    )
    return (
        "=" * 60 + "\n"
        f"  {header}\n"
        f"  {now}\n"
        "=" * 60 + "\n"
        f"  Total consultas evaluadas: {metrics.n_total}\n"
        + _fmt_dim("Faithfulness      ", metrics.faithfulness)
        + _fmt_dim("Answer Relevance  ", metrics.answer_relevance)
        + error_note
        + "=" * 60
    )


# ---------------------------------------------------------------------------
# Salida JSON — métricas agregadas
# ---------------------------------------------------------------------------

def _stats_to_dict(s: DimensionStats | None) -> dict | None:
    return asdict(s) if s is not None else None


def save_json(
    metrics:       RAGAggregatedMetrics,
    path:          Path,
    pipeline_name: str = "",
    extra:         dict | None = None,
) -> None:
    payload: dict = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "pipeline":        pipeline_name,
        "n_total":         metrics.n_total,
        "n_errors":        metrics.n_errors,
        "faithfulness":    _stats_to_dict(metrics.faithfulness),
        "answer_relevance":_stats_to_dict(metrics.answer_relevance),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Salida JSON — veredictos individuales
# ---------------------------------------------------------------------------

def _judgement_to_dict(j: RAGJudgement) -> dict:
    def _dim(d):
        return asdict(d) if d is not None else None
    return {
        "query_id":        j.query_id,
        "query":           j.query,
        "answer":          j.answer,
        "faithfulness":    _dim(j.faithfulness),
        "answer_relevance":_dim(j.answer_relevance),
        "judge_error":     j.judge_error,
    }


def save_judgements(
    judgements: list[RAGJudgement],
    path:       Path,
    extra:      dict | None = None,
) -> None:
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_judgements": len(judgements),
        "judgements":   [_judgement_to_dict(j) for j in judgements],
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")