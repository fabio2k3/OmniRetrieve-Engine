"""
test_query_generator.py
=======================
Tests de QueryGenerator y la función _is_valid.
Sin llamadas reales a Ollama — todo mockeado.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.eval.query_generator import QueryGenerator, _is_valid, _jaccard


CHUNK = (
    "Attention mechanisms allow neural networks to focus on relevant parts "
    "of the input sequence when producing an output. The transformer architecture "
    "uses self-attention to model dependencies regardless of distance in sequences."
)


class TestIsValid:
    def test_valid_query(self):
        ok, _ = _is_valid("How do attention mechanisms work in transformers?", CHUNK)
        assert ok is True

    def test_too_short(self):
        ok, reason = _is_valid("What?", CHUNK)
        assert ok is False
        assert "corta" in reason

    def test_too_long(self):
        ok, reason = _is_valid("A" * 260 + "?", CHUNK)
        assert ok is False
        assert "larga" in reason

    def test_no_question_mark(self):
        ok, reason = _is_valid("How attention mechanisms work in transformers", CHUNK)
        assert ok is False
        assert "?" in reason

    def test_too_similar_to_chunk(self):
        # Query que copia casi literal del chunk
        similar = "Attention mechanisms allow networks to focus on relevant parts?"
        ok, reason = _is_valid(similar, CHUNK)
        assert ok is False
        assert "similar" in reason

    def test_empty_string(self):
        ok, _ = _is_valid("", CHUNK)
        assert ok is False


class TestQueryGenerator:
    def _mock_ollama(self, response: str):
        mock = MagicMock()
        mock.chat.return_value = {"message": {"content": response}}
        return mock

    def test_returns_valid_query(self):
        gen   = QueryGenerator(max_retries=1)
        ollama = self._mock_ollama("How does self-attention help process long sequences?")
        with patch.dict("sys.modules", {"ollama": ollama}):
            result = gen.generate(CHUNK)
        assert result is not None
        assert result.endswith("?")

    def test_returns_none_when_ollama_missing(self):
        gen = QueryGenerator()
        with patch.dict("sys.modules", {"ollama": None}):
            result = gen.generate(CHUNK)
        assert result is None

    def test_returns_none_when_query_always_invalid(self):
        """Si el LLM siempre devuelve texto sin '?', debe descartar."""
        gen    = QueryGenerator(max_retries=2)
        ollama = self._mock_ollama("This is not a question at all")
        with patch.dict("sys.modules", {"ollama": ollama}):
            result = gen.generate(CHUNK)
        assert result is None

    def test_retries_until_valid(self):
        """Primer intento inválido, segundo válido → devuelve el segundo."""
        gen    = QueryGenerator(max_retries=3)
        ollama = MagicMock()
        ollama.chat.side_effect = [
            {"message": {"content": "not a question"}},
            {"message": {"content": "What is the role of self-attention in transformers?"}},
        ]
        with patch.dict("sys.modules", {"ollama": ollama}):
            result = gen.generate(CHUNK)
        assert result is not None
        assert result.endswith("?")
        assert ollama.chat.call_count == 2

    def test_returns_none_when_ollama_raises(self):
        gen    = QueryGenerator()
        ollama = MagicMock()
        ollama.chat.side_effect = RuntimeError("Connection error")
        with patch.dict("sys.modules", {"ollama": ollama}):
            result = gen.generate(CHUNK)
        assert result is None

    def test_query_not_too_similar_to_chunk(self):
        """La query generada no debe copiar frases del chunk."""
        gen    = QueryGenerator(max_retries=1)
        valid_query = "What mechanisms enable neural networks to weigh input importance?"
        ollama = self._mock_ollama(valid_query)
        with patch.dict("sys.modules", {"ollama": ollama}):
            result = gen.generate(CHUNK)
        if result:
            from backend.eval.query_generator import _jaccard
            assert _jaccard(result, CHUNK) < 0.25

    def test_truncates_long_chunk(self):
        """Chunks muy largos deben truncarse antes de enviarse al LLM."""
        long_chunk = "word " * 1000
        gen        = QueryGenerator(max_retries=1)
        ollama     = self._mock_ollama("What does this text describe?")
        with patch.dict("sys.modules", {"ollama": ollama}):
            gen.generate(long_chunk)
        # El prompt enviado no debe superar el truncado de 1200 chars del chunk
        call_args = ollama.chat.call_args
        prompt    = call_args[1]["messages"][0]["content"]
        assert len(prompt) < 3000  # prompt = template + truncated chunk
