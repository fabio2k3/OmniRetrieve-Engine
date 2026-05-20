"""
aggregator.py
=============
Convierte una lista de RAGJudgement en RAGAggregatedMetrics.

Responsabilidad única
---------------------
Estadística pura: extraer las puntuaciones de cada dimensión, calcular
media/min/max y desglozar por tipo de caso.  Sin I/O, sin LLM.
"""

from __future__ import annotations

from ._types import (
    Dimension,
    DimensionStats,
    RAGAggregatedMetrics,
    RAGJudgement,
)

_DIMENSIONS: list[Dimension] = ["faithfulness", "answer_relevance", "context_relevance"]


def _dimension_stats(
    judgements: list[RAGJudgement],
    dimension:  Dimension,
) -> DimensionStats | None:
    """
    Calcula estadísticas de una dimensión sobre una lista de RAGJudgement.

    Devuelve None si no hay casos con puntuación válida para esa dimensión.
    """
    scores = [
        getattr(j, dimension).score
        for j in judgements
        if getattr(j, dimension) is not None
    ]
    if not scores:
        return None

    return DimensionStats(
        n_cases=len(scores),
        mean=sum(scores) / len(scores),
        minimum=min(scores),
        maximum=max(scores),
    )


def _aggregate_group(
    judgements: list[RAGJudgement],
) -> RAGAggregatedMetrics:
    """Agrega un grupo de RAGJudgement (sin desgloses por tipo)."""
    n_errors = sum(1 for j in judgements if j.judge_error is not None)
    return RAGAggregatedMetrics(
        n_total=len(judgements),
        n_errors=n_errors,
        faithfulness=_dimension_stats(judgements, "faithfulness"),
        answer_relevance=_dimension_stats(judgements, "answer_relevance"),
        context_relevance=_dimension_stats(judgements, "context_relevance"),
    )


def aggregate(judgements: list[RAGJudgement]) -> RAGAggregatedMetrics:
    """
    Agrega una lista de RAGJudgement en métricas globales y por tipo de caso.

    Parámetros
    ----------
    judgements : resultados individuales de score_rag_case().

    Devuelve
    --------
    RAGAggregatedMetrics con desglose overall y por tipo ("exact" / "semantic").
    Los desgloses de tipos inexistentes no se incluyen en by_type.
    """
    result = _aggregate_group(judgements)

    for case_type in ("exact", "semantic"):
        group = [j for j in judgements if j.case_type == case_type]
        if group:
            result.by_type[case_type] = _aggregate_group(group)

    return result
