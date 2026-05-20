"""
eval.retrieval
==============
Subpaquete de evaluación de retrieval.

API pública
-----------
EvalRunner         — orquesta la evaluación (runner.py).
aggregate          — convierte RawHit[] → AggregatedMetrics (aggregator.py).
format_summary     — resumen legible en texto (report.py).
save_json          — persiste el reporte en JSON (report.py).
RawHit             — resultado individual de un caso (_types.py).
AggregatedMetrics  — métricas agregadas (_types.py).
MetricSet          — bloque de métricas por tipo (_types.py).
"""

from ._types import RawHit, MetricSet, AggregatedMetrics
from .runner import EvalRunner
from .aggregator import aggregate
from .report import format_summary, save_json, load_json

__all__ = [
    "RawHit",
    "MetricSet",
    "AggregatedMetrics",
    "EvalRunner",
    "aggregate",
    "format_summary",
    "save_json",
    "load_json",
]
