"""
report.py
=========
Presentación y serialización de resultados de evaluación RAG.

Responsabilidad única
---------------------
Formatear RAGAggregatedMetrics para consola y guardar/cargar JSON.
Sin lógica de negocio ni llamadas al LLM.

Salidas soportadas
------------------
· format_summary()  — texto legible para consola.
· save_json()       — JSON estructurado para análisis posterior.
· load_json()       — carga un reporte guardado.
· save_judgements() — guarda la lista completa de RAGJudgement (trazabilidad).
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


def _fmt_group(title: str, m: RAGAggregatedMetrics) -> str:
    error_note = f"  ⚠  Casos con errores de juez: {m.n_errors}\n" if m.n_errors else ""
    return (
        f"  {title} ({m.n_total} casos)\n"
        + _fmt_dim("Faithfulness      ", m.faithfulness)
        + _fmt_dim("Answer Relevance  ", m.answer_relevance)
        + _fmt_dim("Context Relevance ", m.context_relevance)
        + error_note
    )


def format_summary(metrics: RAGAggregatedMetrics, pipeline_name: str = "") -> str:
    """
    Devuelve un resumen legible de las métricas RAG.

    Parámetros
    ----------
    metrics       : resultado de aggregator.aggregate().
    pipeline_name : nombre del pipeline evaluado (para la cabecera).
    """
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"Pipeline: {pipeline_name}" if pipeline_name else "RAG Evaluation"

    lines = [
        "=" * 60,
        f"  {header}",
        f"  {now}",
        "=" * 60,
        _fmt_group("Global  ", metrics),
    ]

    for case_type, sub in metrics.by_type.items():
        lines.append(_fmt_group(case_type.capitalize(), sub))

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Salida JSON — métricas agregadas
# ---------------------------------------------------------------------------

def _stats_to_dict(s: DimensionStats | None) -> dict | None:
    return asdict(s) if s is not None else None


def _metrics_to_dict(m: RAGAggregatedMetrics) -> dict:
    return {
        "n_total":           m.n_total,
        "n_errors":          m.n_errors,
        "faithfulness":      _stats_to_dict(m.faithfulness),
        "answer_relevance":  _stats_to_dict(m.answer_relevance),
        "context_relevance": _stats_to_dict(m.context_relevance),
        "by_type": {k: _metrics_to_dict(v) for k, v in m.by_type.items()},
    }


def save_json(
    metrics:       RAGAggregatedMetrics,
    path:          Path,
    pipeline_name: str  = "",
    extra:         dict | None = None,
) -> None:
    """
    Guarda las métricas RAG agregadas en JSON.

    Parámetros
    ----------
    metrics       : resultado de aggregator.aggregate().
    path          : ruta de destino (se crean directorios intermedios).
    pipeline_name : nombre del pipeline evaluado.
    extra         : campos adicionales (ej. dataset_path).
    """
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline":     pipeline_name,
        **_metrics_to_dict(metrics),
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Carga un reporte JSON guardado previamente."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Salida JSON — veredictos individuales (trazabilidad)
# ---------------------------------------------------------------------------

def _judgement_to_dict(j: RAGJudgement) -> dict:
    def _dim(d):
        return asdict(d) if d is not None else None

    return {
        "case_id":           j.case_id,
        "case_type":         j.case_type,
        "query":             j.query,
        "answer":            j.answer,
        "faithfulness":      _dim(j.faithfulness),
        "answer_relevance":  _dim(j.answer_relevance),
        "context_relevance": _dim(j.context_relevance),
        "judge_error":       j.judge_error,
    }


def save_judgements(
    judgements: list[RAGJudgement],
    path:       Path,
    extra:      dict | None = None,
) -> None:
    """
    Guarda la lista completa de veredictos individuales en JSON.

    Útil para auditar qué dijo el juez caso por caso, detectar patrones
    de error y depurar el pipeline.
    """
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_judgements": len(judgements),
        "judgements":   [_judgement_to_dict(j) for j in judgements],
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
