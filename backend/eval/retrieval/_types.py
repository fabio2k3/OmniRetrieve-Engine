"""
_types.py
=========
Tipos de datos del submódulo de evaluación de retrieval.

Contiene exclusivamente dataclasses — ninguna lógica de negocio.

Clases
------
RawHit           — resultado crudo de evaluar un único EvalCase.
AggregatedMetrics— métricas agregadas sobre un conjunto de RawHit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawHit:
    """
    Resultado de evaluar un único EvalCase contra un retriever.

    Campos
    ------
    case_id            : identificador del EvalCase evaluado.
    case_type          : "exact" | "semantic".
    expected_chunk_id  : chunk que debería aparecer en los resultados.
    found              : True si el chunk apareció dentro del top_k.
    rank               : posición 1-based donde se encontró (None si no encontrado).
    top_k              : ventana de evaluación usada.
    n_results_returned : cuántos resultados devolvió el retriever (puede ser < top_k).
    """

    case_id:             str
    case_type:           str       # "exact" | "semantic"
    expected_chunk_id:   int
    found:               bool
    rank:                int | None   # 1-based; None si not found
    top_k:               int
    n_results_returned:  int = 0

    @property
    def reciprocal_rank(self) -> float:
        """1/rank si encontrado, 0.0 en caso contrario."""
        return 1.0 / self.rank if self.found and self.rank else 0.0


@dataclass
class MetricSet:
    """
    Métricas de retrieval sobre un subconjunto de RawHit.

    Campos
    ------
    n_cases   : número de casos evaluados.
    hit_at_k  : fracción de casos donde el chunk correcto apareció en top-K.
    mrr       : Mean Reciprocal Rank.
    ndcg_at_k : Normalized Discounted Cumulative Gain a K.
    """

    n_cases:   int
    hit_at_k:  float   # Hit@K = Recall@K = Precision@K (ground truth size = 1)
    mrr:       float   # Mean Reciprocal Rank
    ndcg_at_k: float   # NDCG@K


@dataclass
class AggregatedMetrics:
    """
    Métricas agregadas de una corrida de evaluación completa.

    Desglosadas por tipo de caso.

    Campos
    ------
    top_k     : ventana de evaluación.
    overall   : métricas sobre todos los casos.
    exact     : métricas solo sobre casos "exact"     (None si no hay).
    semantic  : métricas solo sobre casos "semantic"  (None si no hay).
    generated : métricas solo sobre casos "generated" (None si no hay).
                Este es el tipo más representativo del uso real del sistema.
    """

    top_k:     int
    overall:   MetricSet
    exact:     MetricSet | None = None
    semantic:  MetricSet | None = None
    generated: MetricSet | None = None
