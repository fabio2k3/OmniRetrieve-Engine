"""
differ.py
=========
Funciones puras para comparar dos reportes de evaluación.

Responsabilidad única
---------------------
Detectar el tipo de reporte, extraer métricas comparables como dict plano
y calcular los deltas.  Sin I/O, sin efectos secundarios.

Tipos de reporte soportados
---------------------------
retrieval : generado por eval.retrieval.report.save_json().
            Claves clave: "overall" → {"hit_at_k", "mrr", "ndcg_at_k"}.

rag       : generado por eval.rag.report.save_json().
            Claves clave: "faithfulness", "answer_relevance",
                          "context_relevance" → {"mean": ...}.

mixed     : uno de cada tipo (se extraen las métricas que existan en ambos).

API pública
-----------
detect_type(report)               → "retrieval" | "rag" | "unknown"
extract_metrics(report)           → dict[metric_name, float]
compute_deltas(baseline, candidate, threshold) → list[MetricDelta]
"""

from __future__ import annotations

from datetime import datetime, timezone

from ._types import ComparisonResult, MetricDelta, Status

# Umbral predeterminado: delta < -0.005 → degraded, delta > 0.005 → improved
DEFAULT_THRESHOLD = 0.005

# ---------------------------------------------------------------------------
# Detección de tipo
# ---------------------------------------------------------------------------

def detect_type(report: dict) -> str:
    """
    Infiere el tipo de reporte a partir de sus claves.

    Devuelve
    --------
    "retrieval" si contiene métricas Hit@K/MRR/NDCG.
    "rag"       si contiene dimensiones faithfulness/answer_relevance.
    "unknown"   si no se puede determinar.
    """
    if isinstance(report.get("overall"), dict) and "hit_at_k" in (report.get("overall") or {}):
        return "retrieval"
    if "faithfulness" in report and isinstance(report.get("faithfulness"), dict):
        return "rag"
    return "unknown"


# ---------------------------------------------------------------------------
# Extracción de métricas
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """Convierte a float de forma segura; None si no es numérico."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_retrieval(report: dict) -> dict[str, tuple[str, float]]:
    """
    Extrae métricas de un reporte de retrieval.

    Devuelve dict  metric_name → (group, value).
    """
    metrics: dict[str, tuple[str, float]] = {}
    dims = ["hit_at_k", "mrr", "ndcg_at_k"]

    # Grupos: overall, exact, semantic
    for group in ("overall", "exact", "semantic"):
        block = report.get(group)
        if not isinstance(block, dict):
            continue
        for dim in dims:
            val = _safe_float(block.get(dim))
            if val is not None:
                metrics[f"{group}.{dim}"] = (group, val)

    return metrics


def _extract_rag(report: dict) -> dict[str, tuple[str, float]]:
    """
    Extrae métricas de un reporte RAG.

    Devuelve dict  metric_name → (group, value).
    """
    metrics: dict[str, tuple[str, float]] = {}
    dims = ["faithfulness", "answer_relevance", "context_relevance"]

    # Nivel global
    for dim in dims:
        block = report.get(dim)
        if isinstance(block, dict):
            val = _safe_float(block.get("mean"))
            if val is not None:
                metrics[f"overall.{dim}"] = ("overall", val)

    # Desglose by_type (exact / semantic)
    by_type = report.get("by_type") or {}
    for case_type, type_block in by_type.items():
        if not isinstance(type_block, dict):
            continue
        for dim in dims:
            dim_block = type_block.get(dim)
            if isinstance(dim_block, dict):
                val = _safe_float(dim_block.get("mean"))
                if val is not None:
                    metrics[f"{case_type}.{dim}"] = (case_type, val)

    return metrics


def extract_metrics(report: dict) -> dict[str, tuple[str, float]]:
    """
    Extrae métricas comparables de un reporte como dict plano.

    Devuelve
    --------
    dict de  metric_name → (group, value)
    donde metric_name es ej. "overall.hit_at_k" o "exact.faithfulness".
    """
    rtype = detect_type(report)
    if rtype == "retrieval":
        return _extract_retrieval(report)
    if rtype == "rag":
        return _extract_rag(report)
    # Intentar ambos y combinar
    merged = {}
    merged.update(_extract_retrieval(report))
    merged.update(_extract_rag(report))
    return merged


# ---------------------------------------------------------------------------
# Cálculo de deltas
# ---------------------------------------------------------------------------

def _status(delta: float, threshold: float) -> Status:
    if delta > threshold:
        return "improved"
    if delta < -threshold:
        return "degraded"
    return "neutral"


def compute_deltas(
    baseline:  dict,
    candidate: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[MetricDelta]:
    """
    Calcula los deltas entre dos reportes.

    Solo se comparan métricas presentes en ambos reportes.  Las métricas
    exclusivas de uno de los dos se ignoran silenciosamente.

    Parámetros
    ----------
    baseline  : reporte de referencia (JSON cargado como dict).
    candidate : reporte candidato.
    threshold : umbral mínimo de |delta| para salir de "neutral".

    Devuelve
    --------
    Lista de MetricDelta ordenada por grupo y nombre de métrica.
    """
    base_metrics = extract_metrics(baseline)
    cand_metrics = extract_metrics(candidate)
    common_names = sorted(set(base_metrics) & set(cand_metrics))

    deltas: list[MetricDelta] = []
    for name in common_names:
        group, base_val = base_metrics[name]
        _,     cand_val = cand_metrics[name]
        delta = cand_val - base_val
        delta_pct = (delta / base_val * 100) if base_val != 0 else None

        deltas.append(MetricDelta(
            name=name,
            group=group,
            baseline=base_val,
            candidate=cand_val,
            delta=delta,
            delta_pct=delta_pct,
            status=_status(delta, threshold),
        ))

    return deltas


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def compare_reports(
    baseline:         dict,
    candidate:        dict,
    baseline_label:   str   = "baseline",
    candidate_label:  str   = "candidate",
    threshold:        float = DEFAULT_THRESHOLD,
) -> ComparisonResult:
    """
    Compara dos reportes y devuelve un ComparisonResult completo.

    Parámetros
    ----------
    baseline / candidate      : reportes JSON cargados como dict.
    baseline_label / candidate_label : etiquetas para el resumen.
    threshold                 : umbral de delta para "improved"/"degraded".
    """
    b_type = detect_type(baseline)
    c_type = detect_type(candidate)
    report_type = b_type if b_type == c_type else "mixed"

    deltas = compute_deltas(baseline, candidate, threshold=threshold)

    return ComparisonResult(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        report_type=report_type,
        threshold=threshold,
        deltas=deltas,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
