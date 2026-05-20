"""
runner.py
=========
Orquestador de la evaluación RAG end-to-end.

Responsabilidad única
---------------------
Iterar sobre los casos de un EvalDataset, llamar al pipeline RAG para
obtener la respuesta, y delegar la puntuación a scorer.py.
No contiene métricas, prompts ni I/O.

El pipeline debe exponer el método ask() compatible con RAGPipeline:
    pipeline.ask(query, include_debug=True) → dict
        {"query": str, "answer": str, "sources": list, "context": str}

El campo "context" solo está disponible cuando include_debug=True, que es
lo que el runner activa para que el juez pueda evaluar context_relevance
y faithfulness con el contexto real usado por el LLM.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

from backend.eval.schema import EvalCase, EvalDataset
from .judge import OllamaJudge
from .scorer import score_rag_case
from ._types import RAGJudgement

log = logging.getLogger(__name__)


class RAGPipelineProtocol(Protocol):
    """Contrato mínimo que debe cumplir el pipeline RAG evaluado."""

    def ask(self, query: str, **kwargs) -> dict:
        """
        Devuelve al menos: {"query": str, "answer": str, "sources": list}.
        Con include_debug=True añade también "context": str.
        """
        ...


class RAGEvalRunner:
    """
    Ejecuta la evaluación RAG end-to-end sobre un EvalDataset.

    Parámetros
    ----------
    pipeline      : pipeline RAG a evaluar (debe cumplir RAGPipelineProtocol).
    judge         : instancia de OllamaJudge configurada.
    on_progress   : callback opcional con firma (index, total, judgement).
    pipeline_kwargs: argumentos extra para pipeline.ask() (ej. top_k, max_chunks).
    """

    def __init__(
        self,
        pipeline:         RAGPipelineProtocol,
        judge:            OllamaJudge,
        on_progress:      Callable[[int, int, RAGJudgement], None] | None = None,
        pipeline_kwargs:  dict | None = None,
    ) -> None:
        self.pipeline         = pipeline
        self.judge            = judge
        self.on_progress      = on_progress
        self.pipeline_kwargs  = pipeline_kwargs or {}

    def run(self, dataset: EvalDataset) -> list[RAGJudgement]:
        """
        Evalúa todos los casos del dataset y devuelve la lista de RAGJudgement.

        Para cada caso:
        1. Llama al pipeline RAG con la query (include_debug=True para obtener
           el contexto real y poder evaluar faithfulness y context_relevance).
        2. Pasa la salida al scorer que llama al juez LLM en tres dimensiones.
        3. Registra el RAGJudgement resultante.

        Los errores individuales (pipeline o juez) son capturados y registrados
        sin abortar la corrida completa.
        """
        total = len(dataset.cases)
        judgements: list[RAGJudgement] = []

        log.info("[rag_runner] Iniciando evaluación RAG: %d casos", total)

        for i, case in enumerate(dataset.cases):
            judgement = self._evaluate_case(case)
            judgements.append(judgement)

            if self.on_progress:
                self.on_progress(i + 1, total, judgement)

            log.debug(
                "[rag_runner] [%d/%d] %s error=%s",
                i + 1, total, case.case_id, judgement.judge_error,
            )

        error_count = sum(1 for j in judgements if j.judge_error)
        log.info(
            "[rag_runner] Completado: %d casos, %d con errores de juez",
            total, error_count,
        )
        return judgements

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _evaluate_case(self, case: EvalCase) -> RAGJudgement:
        """Llama al pipeline RAG y delega la puntuación al scorer."""
        try:
            output = self.pipeline.ask(
                case.query,
                include_debug=True,
                **self.pipeline_kwargs,
            )
        except Exception as exc:
            log.error(
                "[rag_runner] Error en pipeline para case_id=%s: %s",
                case.case_id, exc,
            )
            return RAGJudgement(
                case_id=case.case_id,
                case_type=case.case_type,
                query=case.query,
                answer="",
                judge_error=f"pipeline_error: {exc}",
            )

        return score_rag_case(case=case, pipeline_output=output, judge=self.judge)
