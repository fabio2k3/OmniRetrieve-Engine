"""
test_prompts.py
===============
Tests de las plantillas de prompt del juez.

Verifica que las funciones devuelven strings no vacíos con los
elementos obligatorios (query, answer, escala, instrucción JSON).
Sin mocks — solo invocaciones directas de strings.
"""

import pytest
from backend.eval.rag.prompts import (
    faithfulness_prompt,
    answer_relevance_prompt,
)

QUERY   = "What is attention mechanism?"
ANSWER  = "Attention allows models to focus on relevant tokens."
CONTEXT = "[1] Transformer paper\nAttention is all you need."


class TestFaithfulnessPrompt:
    def test_returns_string(self):
        p = faithfulness_prompt(QUERY, ANSWER, CONTEXT)
        assert isinstance(p, str)

    def test_contains_query(self):
        p = faithfulness_prompt(QUERY, ANSWER, CONTEXT)
        assert QUERY in p

    def test_contains_answer(self):
        p = faithfulness_prompt(QUERY, ANSWER, CONTEXT)
        assert ANSWER in p

    def test_contains_context(self):
        p = faithfulness_prompt(QUERY, ANSWER, CONTEXT)
        assert CONTEXT in p

    def test_contains_json_instruction(self):
        p = faithfulness_prompt(QUERY, ANSWER, CONTEXT)
        assert "score" in p.lower()
        assert "json" in p.lower()

    def test_empty_context_handled(self):
        p = faithfulness_prompt(QUERY, ANSWER, "")
        assert isinstance(p, str) and len(p) > 0

    def test_empty_answer_handled(self):
        p = faithfulness_prompt(QUERY, "", CONTEXT)
        assert isinstance(p, str)


class TestAnswerRelevancePrompt:
    def test_returns_string(self):
        p = answer_relevance_prompt(QUERY, ANSWER)
        assert isinstance(p, str)

    def test_contains_query(self):
        assert QUERY in answer_relevance_prompt(QUERY, ANSWER)

    def test_contains_answer(self):
        assert ANSWER in answer_relevance_prompt(QUERY, ANSWER)

    def test_does_not_require_context(self):
        p = answer_relevance_prompt(QUERY, ANSWER)
        assert isinstance(p, str) and len(p) > 0

    def test_contains_scale(self):
        p = answer_relevance_prompt(QUERY, ANSWER)
        assert "1" in p and "5" in p