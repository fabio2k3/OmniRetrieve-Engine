"""
runner.py
=========
Orquestador de la evaluación de retrieval.

Responsabilidad única
---------------------
Iterar sobre los casos de un EvalDataset, llamar al retriever para cada
query y delegar la puntuación a scorer.py.  No contiene métricas ni I/O.

Uso
---
>>> from backend.eval.retrieval.runner import EvalRunner
>>> runner = EvalRunner(retriever=my_retriever, top_k=10)
>>> hits = runner.run(dataset)
"""

from __future__ import annotations

import logging
from typing import Callable

from backend.eval.schema import EvalDataset, EvalCase
from backend.retrieval.protocols import RetrieverProtocol
from .scorer import score_case
from ._types import RawHit

log = logging.getLogger(__name__)


class EvalRunner:
    """
    Ejecuta la evaluación de retrieval sobre un EvalDataset.

    Parámetros
    ----------
    retriever      : cualquier objeto que implemente RetrieverProtocol.
    top_k          : ventana de evaluación (cuántos resultados pedir y juzgar).
    on_progress    : callback opcional llamado después de cada caso con
                     (case_index: int, total: int, hit: RawHit).
                     Útil para mostrar barras de progreso en la CLI.
    """

    def __init__(
        self,
        retriever:   RetrieverProtocol,
        top_k:       int = 10,
        on_progress: Callable[[int, int, RawHit], None] | None = None,
    ) -> None:
        self.retriever   = retriever
        self.top_k       = top_k
        self.on_progress = on_progress

    def run(self, dataset: EvalDataset) -> list[RawHit]:
        """
        Evalúa todos los casos del dataset y devuelve la lista de RawHit.

        El orden de salida coincide con el orden de dataset.cases.
        Los errores de retrieval de un caso individual son capturados y
        registrados, produciendo un RawHit con found=False para ese caso.
        """
        total = len(dataset.cases)
        hits: list[RawHit] = []

        log.info("[runner] Iniciando evaluación: %d casos, top_k=%d", total, self.top_k)

        for i, case in enumerate(dataset.cases):
            hit = self._evaluate_case(case)
            hits.append(hit)

            if self.on_progress:
                self.on_progress(i + 1, total, hit)

            log.debug(
                "[runner] [%d/%d] %s found=%s rank=%s",
                i + 1, total, case.case_id, hit.found, hit.rank,
            )

        found_count = sum(1 for h in hits if h.found)
        log.info(
            "[runner] Evaluación completada: %d/%d encontrados en top-%d",
            found_count, total, self.top_k,
        )
        return hits

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _evaluate_case(self, case: EvalCase) -> RawHit:
        """Llama al retriever y delega la puntuación a scorer."""
        try:
            results = self.retriever.retrieve(case.query, top_n=self.top_k)
        except Exception as exc:
            log.error("[runner] Error en retrieval para case_id=%s: %s", case.case_id, exc)
            return RawHit(
                case_id=case.case_id,
                case_type=case.case_type,
                expected_chunk_id=case.expected_chunk_id,
                found=False,
                rank=None,
                top_k=self.top_k,
                n_results_returned=0,
            )

        return score_case(case=case, results=results, top_k=self.top_k)
