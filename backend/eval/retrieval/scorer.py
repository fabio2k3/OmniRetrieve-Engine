"""
scorer.py
=========
Puente entre el dominio de evaluación (EvalCase) y el de retrieval
(RetrievalResult).  Produce un RawHit a partir de un caso y sus resultados.

Responsabilidad única
---------------------
Dada una query evaluada, determinar si el chunk esperado aparece en la
lista de resultados y en qué posición — nada más.

El módulo no conoce cómo se generó el EvalCase ni cómo funciona el retriever.
"""

from __future__ import annotations

from backend.eval.schema import EvalCase
from backend.retrieval.protocols import RetrievalResult
from ._types import RawHit


def score_case(
    case:    EvalCase,
    results: list[RetrievalResult],
    top_k:   int,
) -> RawHit:
    """
    Evalúa si el chunk esperado aparece en los resultados del retriever.

    Parámetros
    ----------
    case    : caso de evaluación con el chunk_id que se espera recuperar.
    results : lista ordenada de RetrievalResult (mejor primero).
    top_k   : ventana de evaluación; solo se consideran los primeros top_k.

    Devuelve
    --------
    RawHit con found=True y el rank 1-based si se encontró dentro de top_k,
    o found=False y rank=None en caso contrario.
    """
    window = results[:top_k]

    for rank, result in enumerate(window, start=1):
        if result.chunk_id == case.expected_chunk_id:
            return RawHit(
                case_id=case.case_id,
                case_type=case.case_type,
                expected_chunk_id=case.expected_chunk_id,
                found=True,
                rank=rank,
                top_k=top_k,
                n_results_returned=len(results),
            )

    return RawHit(
        case_id=case.case_id,
        case_type=case.case_type,
        expected_chunk_id=case.expected_chunk_id,
        found=False,
        rank=None,
        top_k=top_k,
        n_results_returned=len(results),
    )
