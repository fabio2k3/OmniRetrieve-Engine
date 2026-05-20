"""
scorer.py
=========
Orquesta las tres dimensiones del juez para un único caso RAG.

Responsabilidad única
---------------------
Dado un EvalCase y la salida del pipeline RAG, construir los tres prompts,
llamar al juez para cada dimensión y empaquetar el resultado en un
RAGJudgement.  No contiene métricas ni I/O.

Flujo
-----
EvalCase + pipeline_output  →  score_rag_case()  →  RAGJudgement
                                    │
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
               faithfulness  answer_rel.  context_rel.
                (prompts.py)  (prompts.py) (prompts.py)
                     │              │              │
                     └──────────────┼──────────────┘
                                    ▼
                              OllamaJudge.evaluate()
"""

from __future__ import annotations

import logging

from backend.eval.schema import EvalCase
from . import prompts
from ._types import RAGJudgement
from .judge import OllamaJudge

log = logging.getLogger(__name__)


def score_rag_case(
    case:            EvalCase,
    pipeline_output: dict,
    judge:           OllamaJudge,
    context:         str = "",
) -> RAGJudgement:
    """
    Evalúa la salida RAG de un único caso con el juez LLM.

    Parámetros
    ----------
    case            : caso de evaluación (contiene la query original).
    pipeline_output : dict con al menos las claves 'answer' y 'query',
                      tal como devuelve RAGPipeline.ask().
    judge           : instancia de OllamaJudge configurada.
    context         : texto del contexto usado por el pipeline RAG
                      (si no está disponible en pipeline_output).

    Devuelve
    --------
    RAGJudgement con las tres dimensiones puntuadas (o None si el juez falló).
    """
    query  = pipeline_output.get("query", case.query)
    answer = pipeline_output.get("answer", "")

    # El contexto puede venir del pipeline_output (modo debug) o como parámetro
    ctx = pipeline_output.get("context") or context

    errors: list[str] = []

    # ── Faithfulness ────────────────────────────────────────────────────
    faithfulness = judge.evaluate(
        prompt=prompts.faithfulness_prompt(query=query, answer=answer, context=ctx),
        dimension="faithfulness",
    )
    if faithfulness is None:
        errors.append("faithfulness")
        log.warning("[scorer] case_id=%s → faithfulness sin puntuación", case.case_id)

    # ── Answer Relevance ────────────────────────────────────────────────
    answer_relevance = judge.evaluate(
        prompt=prompts.answer_relevance_prompt(query=query, answer=answer),
        dimension="answer_relevance",
    )
    if answer_relevance is None:
        errors.append("answer_relevance")
        log.warning("[scorer] case_id=%s → answer_relevance sin puntuación", case.case_id)

    # ── Context Relevance ───────────────────────────────────────────────
    context_relevance = judge.evaluate(
        prompt=prompts.context_relevance_prompt(query=query, context=ctx),
        dimension="context_relevance",
    )
    if context_relevance is None:
        errors.append("context_relevance")
        log.warning("[scorer] case_id=%s → context_relevance sin puntuación", case.case_id)

    return RAGJudgement(
        case_id=case.case_id,
        case_type=case.case_type,
        query=query,
        answer=answer,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_relevance=context_relevance,
        judge_error=", ".join(errors) if errors else None,
    )
