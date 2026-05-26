"""
rag/runner.py
=============
Orquestador de la evaluación RAG end-to-end.

Responsabilidad única
---------------------
Iterar sobre las consultas de un RAGQuerySet, llamar al pipeline RAG
para obtener la respuesta, y delegar la puntuación a scorer.py.
Sin métricas, prompts ni I/O.

Uso
---
>>> from backend.eval.rag.runner import RAGEvalRunner
>>> runner = RAGEvalRunner(pipeline=my_pipeline, judge=my_judge)
>>> judgements = runner.run(query_set)
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

from .schema import RAGQuery, RAGQuerySet
from .scorer import score_rag_query
from .judge import OllamaJudge
from ._types import RAGJudgement

log = logging.getLogger(__name__)


class RAGPipelineProtocol(Protocol):
    """Contrato mínimo que debe cumplir el pipeline RAG evaluado."""
    def ask(self, query: str, **kwargs) -> dict:
        """
        Devuelve al menos: {"query": str, "answer": str}.
        Con include_debug=True añade también "context": str.
        """
        ...


class RAGEvalRunner:
    """
    Ejecuta la evaluación RAG end-to-end sobre un RAGQuerySet.

    Parámetros
    ----------
    pipeline         : pipeline RAG a evaluar.
    judge            : instancia de OllamaJudge configurada.
    on_progress      : callback opcional (index, total, judgement).
    pipeline_kwargs  : argumentos extra para pipeline.ask().
    """

    def __init__(
        self,
        pipeline:        RAGPipelineProtocol,
        judge:           OllamaJudge,
        on_progress:     Callable[[int, int, RAGJudgement], None] | None = None,
        pipeline_kwargs: dict | None = None,
    ) -> None:
        self.pipeline        = pipeline
        self.judge           = judge
        self.on_progress     = on_progress
        self.pipeline_kwargs = pipeline_kwargs or {}

    def run(self, query_set: RAGQuerySet) -> list[RAGJudgement]:
        """
        Evalúa todas las consultas del RAGQuerySet.

        Para cada consulta:
        1. Llama al pipeline RAG con include_debug=True para obtener el
           contexto real usado por el LLM (necesario para faithfulness).
        2. Pasa la salida al scorer que llama al juez LLM.
        3. Registra el RAGJudgement resultante.

        Los errores individuales son capturados sin abortar la corrida.
        """
        total = len(query_set)
        judgements: list[RAGJudgement] = []

        log.info("[rag_runner] Iniciando evaluación RAG: %d consultas", total)

        for i, rq in enumerate(query_set.queries):
            judgement = self._evaluate_query(rq)
            judgements.append(judgement)

            if self.on_progress:
                self.on_progress(i + 1, total, judgement)

            log.debug(
                "[rag_runner] [%d/%d] %s error=%s",
                i + 1, total, rq.query_id, judgement.judge_error,
            )

        n_errors = sum(1 for j in judgements if j.judge_error)
        log.info(
            "[rag_runner] Completado: %d consultas, %d con errores de juez",
            total, n_errors,
        )
        return judgements

    def _evaluate_query(self, rq: RAGQuery) -> RAGJudgement:
        """Llama al pipeline y delega la puntuación al scorer."""
        try:
            output = self.pipeline.ask(
                rq.query,
                include_debug=True,
                **self.pipeline_kwargs,
            )
        except Exception as exc:
            log.error(
                "[rag_runner] Error en pipeline para query_id=%s: %s",
                rq.query_id, exc,
            )
            return RAGJudgement(
                query_id=rq.query_id,
                query=rq.query,
                answer="",
                judge_error=f"pipeline_error: {exc}",
            )

        return score_rag_query(
            query_id=rq.query_id,
            query=rq.query,
            pipeline_output=output,
            judge=self.judge,
        )