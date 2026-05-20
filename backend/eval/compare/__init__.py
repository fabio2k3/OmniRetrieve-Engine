"""
eval.compare
============
Subpaquete para comparar reportes de evaluación entre corridas.

Útil para detectar regresiones al cambiar parámetros del sistema
(rrf_k, modelo de embedding, tamaño de chunk, etc.).

API pública
-----------
compare_reports   — compara dos reportes JSON y devuelve ComparisonResult.
format_summary    — resumen legible en consola.
save_json         — persiste la comparación en JSON.
load_json         — carga una comparación guardada.
MetricDelta       — delta de una métrica individual.
ComparisonResult  — resultado completo de una comparación.
"""

from ._types import MetricDelta, ComparisonResult
from .differ import compare_reports, detect_type, extract_metrics
from .report import format_summary, save_json, load_json

__all__ = [
    "MetricDelta",
    "ComparisonResult",
    "compare_reports",
    "detect_type",
    "extract_metrics",
    "format_summary",
    "save_json",
    "load_json",
]
