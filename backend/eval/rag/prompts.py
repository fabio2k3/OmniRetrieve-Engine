"""
prompts.py
==========
Plantillas de prompt para el juez LLM.

Reglas de diseño
----------------
· Solo constantes y una función de formateo trivial por dimensión.
· Sin imports del proyecto — cero dependencias.
· Cada plantilla instruye al LLM a responder ÚNICAMENTE con JSON válido:
  {"score": <1-5>, "reason": "<texto breve>"}

  Esto hace el parseo determinista y evita texto adicional que rompa el JSON.

Escala de puntuación (común a todas las dimensiones)
-----------------------------------------------------
  1 — Muy deficiente
  2 — Deficiente
  3 — Aceptable
  4 — Bueno
  5 — Excelente

Dimensiones
-----------
faithfulness       — ¿La respuesta se basa solo en el contexto dado?
answer_relevance   — ¿La respuesta contesta la pregunta?
context_relevance  — ¿El contexto recuperado es pertinente para la pregunta?
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cabecera común
# ---------------------------------------------------------------------------

_JSON_INSTRUCTION = (
    'Respond ONLY with a valid JSON object — no preamble, no markdown, no extra text.\n'
    'Format: {"score": <integer 1-5>, "reason": "<one sentence in the same language as the input>"}'
)

_SCALE = (
    "Score scale:\n"
    "  1 = Very poor\n"
    "  2 = Poor\n"
    "  3 = Acceptable\n"
    "  4 = Good\n"
    "  5 = Excellent\n"
)

# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_TEMPLATE = """\
You are an impartial judge evaluating whether an AI-generated answer is grounded \
in the provided context documents.

Task: Rate the FAITHFULNESS of the answer on a 1-5 scale.
Definition: A faithful answer contains only claims that can be directly supported \
by the context. It does not introduce external knowledge, fabricate facts, or \
contradict the sources.

{scale}
{json_instruction}

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---
{answer}

JSON output:"""


def faithfulness_prompt(query: str, answer: str, context: str) -> str:
    """Devuelve el prompt para evaluar faithfulness."""
    return _FAITHFULNESS_TEMPLATE.format(
        scale=_SCALE,
        json_instruction=_JSON_INSTRUCTION,
        context=context or "[No context provided]",
        query=query,
        answer=answer or "[No answer generated]",
    )


# ---------------------------------------------------------------------------
# Answer Relevance
# ---------------------------------------------------------------------------

_ANSWER_RELEVANCE_TEMPLATE = """\
You are an impartial judge evaluating whether an AI-generated answer addresses \
the user's question.

Task: Rate the ANSWER RELEVANCE on a 1-5 scale.
Definition: A relevant answer directly responds to what was asked. \
It does not go off-topic, answer a different question, or provide only \
tangential information.

{scale}
{json_instruction}

--- QUESTION ---
{query}

--- ANSWER ---
{answer}

JSON output:"""


def answer_relevance_prompt(query: str, answer: str) -> str:
    """Devuelve el prompt para evaluar answer relevance."""
    return _ANSWER_RELEVANCE_TEMPLATE.format(
        scale=_SCALE,
        json_instruction=_JSON_INSTRUCTION,
        query=query,
        answer=answer or "[No answer generated]",
    )


# ---------------------------------------------------------------------------
# Context Relevance
# ---------------------------------------------------------------------------

_CONTEXT_RELEVANCE_TEMPLATE = """\
You are an impartial judge evaluating whether the retrieved context documents \
are useful for answering the user's question.

Task: Rate the CONTEXT RELEVANCE on a 1-5 scale.
Definition: Relevant context contains information that directly helps answer \
the question. Penalize context that is off-topic or contains no useful \
evidence for the question.

{scale}
{json_instruction}

--- QUESTION ---
{query}

--- CONTEXT ---
{context}

JSON output:"""


def context_relevance_prompt(query: str, context: str) -> str:
    """Devuelve el prompt para evaluar context relevance."""
    return _CONTEXT_RELEVANCE_TEMPLATE.format(
        scale=_SCALE,
        json_instruction=_JSON_INSTRUCTION,
        query=query,
        context=context or "[No context provided]",
    )
