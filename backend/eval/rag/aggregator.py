"""
rag/aggregator.py
=================
Convierte una lista de RAGJudgement en RAGAggregatedMetrics.

Estadística pura: sin I/O, sin LLM.
"""

from __future__ import annotations

from ._types import Dimension, DimensionStats, RAGAggregatedMetrics, RAGJudgement

_DIMENSIONS: list[Dimension] = ["faithfulness", "answer_relevance"]


def _dimension_stats(
    judgements: list[RAGJudgement],
    dimension:  Dimension,
) -> DimensionStats | None:
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


def aggregate(judgements: list[RAGJudgement]) -> RAGAggregatedMetrics:
    """
    Agrega una lista de RAGJudgement en métricas globales.

    Parámetros
    ----------
    judgements : resultados individuales de score_rag_query().

    Devuelve
    --------
    RAGAggregatedMetrics con faithfulness y answer_relevance agregados.
    """
    n_errors = sum(1 for j in judgements if j.judge_error is not None)
    return RAGAggregatedMetrics(
        n_total=len(judgements),
        n_errors=n_errors,
        faithfulness=_dimension_stats(judgements, "faithfulness"),
        answer_relevance=_dimension_stats(judgements, "answer_relevance"),
    )