"""
rag/scorer.py
=============
Orquesta las dimensiones del juez para una única consulta RAG.

Responsabilidad única
---------------------
Dado una query y la salida del pipeline, construir los prompts,
llamar al juez y empaquetar el resultado en un RAGJudgement.
Sin dependencia de EvalCase ni ground truth de chunks.
"""

from __future__ import annotations

import logging

from . import prompts
from ._types import RAGJudgement
from .judge import OllamaJudge

log = logging.getLogger(__name__)


def score_rag_query(
    query_id:        str,
    query:           str,
    pipeline_output: dict,
    judge:           OllamaJudge,
) -> RAGJudgement:
    """
    Evalúa la salida RAG de una consulta con el juez LLM.

    Parámetros
    ----------
    query_id        : identificador de la RAGQuery.
    query           : texto de la consulta original.
    pipeline_output : dict devuelto por RAGPipeline.ask(include_debug=True).
                      Debe tener al menos las claves 'answer' y 'context'.
    judge           : instancia de OllamaJudge configurada.

    Devuelve
    --------
    RAGJudgement con las dimensiones puntuadas (o None si el juez falló).
    """
    answer = pipeline_output.get("answer", "")
    ctx    = pipeline_output.get("context", "")
    errors: list[str] = []

    # ── Faithfulness ─────────────────────────────────────────────────────────
    faithfulness = judge.evaluate(
        prompt=prompts.faithfulness_prompt(query=query, answer=answer, context=ctx),
        dimension="faithfulness",
    )
    if faithfulness is None:
        errors.append("faithfulness")
        log.warning("[scorer] query_id=%s → faithfulness sin puntuación", query_id)

    # ── Answer Relevance ──────────────────────────────────────────────────────
    answer_relevance = judge.evaluate(
        prompt=prompts.answer_relevance_prompt(query=query, answer=answer),
        dimension="answer_relevance",
    )
    if answer_relevance is None:
        errors.append("answer_relevance")
        log.warning("[scorer] query_id=%s → answer_relevance sin puntuación", query_id)

    return RAGJudgement(
        query_id=query_id,
        query=query,
        answer=answer,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        judge_error=", ".join(errors) if errors else None,
    )