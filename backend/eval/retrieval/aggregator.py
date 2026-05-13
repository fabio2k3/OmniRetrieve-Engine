"""
aggregator.py
=============
Convierte una lista de RawHit en AggregatedMetrics.

Responsabilidad única
---------------------
Estadística pura: agrupa hits por tipo de caso, extrae los ranks y
delega el cálculo numérico a metrics.py.  No sabe nada de retrievers,
datasets ni I/O.
"""

from __future__ import annotations

from . import metrics as m
from ._types import AggregatedMetrics, MetricSet, RawHit


def _metric_set(hits: list[RawHit], k: int) -> MetricSet:
    """Construye un MetricSet a partir de un subconjunto de RawHit."""
    ranks = [h.rank for h in hits]
    return MetricSet(
        n_cases=len(hits),
        hit_at_k=m.hit_at_k(ranks, k),
        mrr=m.mrr(ranks),
        ndcg_at_k=m.ndcg_at_k(ranks, k),
    )


def aggregate(hits: list[RawHit], top_k: int) -> AggregatedMetrics:
    """
    Agrega una lista de RawHit en métricas globales y por tipo de caso.

    Parámetros
    ----------
    hits  : resultados individuales de score_case().
    top_k : ventana de evaluación (debe coincidir con la usada en runner).

    Devuelve
    --------
    AggregatedMetrics con desgloses overall / exact / semantic.
    El desglose de un tipo es None si no hay casos de ese tipo.
    """
    exact_hits    = [h for h in hits if h.case_type == "exact"]
    semantic_hits = [h for h in hits if h.case_type == "semantic"]

    return AggregatedMetrics(
        top_k=top_k,
        overall=_metric_set(hits, top_k),
        exact=_metric_set(exact_hits, top_k) if exact_hits else None,
        semantic=_metric_set(semantic_hits, top_k) if semantic_hits else None,
    )
