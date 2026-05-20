"""
report.py
=========
Presentación y serialización de AggregatedMetrics.

Responsabilidad única
---------------------
Formatear y guardar resultados — ninguna lógica de cálculo ni retrieval.
Dos salidas soportadas:
  · Texto para consola  (format_summary)
  · JSON estructurado   (save_json / load_json)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ._types import AggregatedMetrics, MetricSet


# ---------------------------------------------------------------------------
# Salida texto
# ---------------------------------------------------------------------------

def _fmt_metric_set(label: str, ms: MetricSet) -> str:
    """Formatea un MetricSet como bloque de texto."""
    return (
        f"  {label} ({ms.n_cases} casos)\n"
        f"    Hit@K    : {ms.hit_at_k:.4f}  ({ms.hit_at_k * 100:.1f}%)\n"
        f"    MRR      : {ms.mrr:.4f}\n"
        f"    NDCG@K   : {ms.ndcg_at_k:.4f}\n"
    )


def format_summary(metrics: AggregatedMetrics, retriever_name: str = "") -> str:
    """
    Devuelve un resumen legible de las métricas de evaluación.

    Parámetros
    ----------
    metrics        : resultado de aggregator.aggregate().
    retriever_name : nombre del retriever evaluado (para la cabecera).
    """
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"Retriever: {retriever_name}" if retriever_name else "Retrieval Evaluation"

    lines = [
        "=" * 52,
        f"  {header}",
        f"  Top-K = {metrics.top_k}   |   {now}",
        "=" * 52,
        _fmt_metric_set("Global  ", metrics.overall),
    ]

    if metrics.exact is not None:
        lines.append(_fmt_metric_set("Exact   ", metrics.exact))

    if metrics.semantic is not None:
        lines.append(_fmt_metric_set("Semantic ", metrics.semantic))

    if metrics.generated is not None:
        lines.append(_fmt_metric_set("Generated", metrics.generated))

    if metrics.exact and metrics.semantic:
        delta = metrics.semantic.hit_at_k - metrics.exact.hit_at_k
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  Δ Hit@K (semantic − exact):   {sign}{delta:.4f}  ({sign}{delta * 100:.1f}%)\n"
        )

    if metrics.exact and metrics.generated:
        delta = metrics.generated.hit_at_k - metrics.exact.hit_at_k
        sign  = "+" if delta >= 0 else ""
        lines.append(
            f"  Δ Hit@K (generated − exact):  {sign}{delta:.4f}  ({sign}{delta * 100:.1f}%)\n"
        )

    lines.append("=" * 52)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Salida JSON
# ---------------------------------------------------------------------------

def _metric_set_to_dict(ms: MetricSet | None) -> dict | None:
    return asdict(ms) if ms is not None else None


def save_json(
    metrics:        AggregatedMetrics,
    path:           Path,
    retriever_name: str = "",
    extra:          dict | None = None,
) -> None:
    """
    Guarda las métricas en un fichero JSON.

    Parámetros
    ----------
    metrics        : resultado de aggregator.aggregate().
    path           : ruta de destino (se crean directorios intermedios).
    retriever_name : nombre del retriever evaluado.
    extra          : campos adicionales a incluir en el JSON (ej. dataset_path).
    """
    payload: dict = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "retriever":      retriever_name,
        "top_k":          metrics.top_k,
        "overall":        _metric_set_to_dict(metrics.overall),
        "exact":          _metric_set_to_dict(metrics.exact),
        "semantic":       _metric_set_to_dict(metrics.semantic),
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Carga un reporte JSON guardado previamente."""
    return json.loads(path.read_text(encoding="utf-8"))
