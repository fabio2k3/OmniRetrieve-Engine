"""
rag/prompts.py
==============
Plantillas de prompt para el juez LLM.

Reglas de diseño
----------------
· Sin imports del proyecto — cero dependencias.
· Cada plantilla instruye al LLM a responder ÚNICAMENTE con JSON válido.
· Se incluye un ejemplo concreto para que modelos pequeños (3B-7B) no
  olviden poner comillas en el valor de "reason".
· format="json" en el judge es la primera línea de defensa; estos prompts
  son la segunda.

Escala de puntuación (común a todas las dimensiones)
-----------------------------------------------------
  1 — Muy deficiente
  2 — Deficiente
  3 — Aceptable
  4 — Bueno
  5 — Excelente
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Instrucción JSON común
# ---------------------------------------------------------------------------

_JSON_INSTRUCTION = """\
Respond ONLY with a valid JSON object. No preamble, no markdown, no extra text.
Required format (copy exactly, replacing values):
{"score": 3, "reason": "Your one-sentence justification here."}

Rules:
- "score" must be an integer between 1 and 5.
- "reason" must be a quoted string (double quotes).
- Do not add any text before or after the JSON object.\
"""

_SCALE = """\
Score scale:
  1 = Very poor
  2 = Poor
  3 = Acceptable
  4 = Good
  5 = Excellent\
"""

# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_TEMPLATE = """\
You are an impartial judge evaluating whether an AI answer is grounded in the provided context.

Task: Rate the FAITHFULNESS of the answer (1-5).
Definition: A faithful answer only makes claims supported by the context.
It does not introduce external knowledge, fabricate facts, or contradict the sources.

{scale}

{json_instruction}

Example of a valid response:
{{"score": 4, "reason": "The answer correctly cites the context but omits one key detail."}}

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---
{answer}

JSON output:"""


def faithfulness_prompt(query: str, answer: str, context: str) -> str:
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
You are an impartial judge evaluating whether an AI answer addresses the user's question.

Task: Rate the ANSWER RELEVANCE (1-5).
Definition: A relevant answer directly responds to what was asked.
Penalize answers that are off-topic, answer a different question, or are only tangentially related.

{scale}

{json_instruction}

Example of a valid response:
{{"score": 5, "reason": "The answer directly and completely addresses the question asked."}}

--- QUESTION ---
{query}

--- ANSWER ---
{answer}

JSON output:"""


def answer_relevance_prompt(query: str, answer: str) -> str:
    return _ANSWER_RELEVANCE_TEMPLATE.format(
        scale=_SCALE,
        json_instruction=_JSON_INSTRUCTION,
        query=query,
        answer=answer or "[No answer generated]",
    )