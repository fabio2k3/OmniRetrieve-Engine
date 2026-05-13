"""
eval.rag
========
Subpaquete de evaluación RAG end-to-end con LLM-as-judge.

API pública
-----------
OllamaJudge          — juez LLM (judge.py).
RAGEvalRunner        — orquestador (runner.py).
aggregate            — lista de RAGJudgement → RAGAggregatedMetrics (aggregator.py).
format_summary       — resumen legible en texto (report.py).
save_json            — persiste métricas en JSON (report.py).
save_judgements      — persiste veredictos individuales en JSON (report.py).
RAGJudgement         — veredicto de un caso (_types.py).
DimensionScore       — puntuación de una dimensión (_types.py).
RAGAggregatedMetrics — métricas globales (_types.py).
"""

from ._types import DimensionScore, RAGJudgement, DimensionStats, RAGAggregatedMetrics
from .judge import OllamaJudge
from .scorer import score_rag_case
from .aggregator import aggregate
from .report import format_summary, save_json, load_json, save_judgements
from .runner import RAGEvalRunner

__all__ = [
    "DimensionScore",
    "DimensionStats",
    "RAGJudgement",
    "RAGAggregatedMetrics",
    "OllamaJudge",
    "score_rag_case",
    "aggregate",
    "format_summary",
    "save_json",
    "load_json",
    "save_judgements",
    "RAGEvalRunner",
]
