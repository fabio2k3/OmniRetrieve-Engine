"""
test_judge.py
=============
Tests de OllamaJudge y la función interna _extract_json.

Mockea Ollama completamente — no requiere servicios externos.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.eval.rag.judge import OllamaJudge, _extract_json
from backend.eval.rag._types import DimensionScore


# ---------------------------------------------------------------------------
# Tests de _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_parses_clean_json(self):
        result = _extract_json('{"score": 4, "reason": "Good answer"}')
        assert result == {"score": 4, "reason": "Good answer"}

    def test_parses_json_with_whitespace(self):
        result = _extract_json('  {"score": 3, "reason": "OK"}  ')
        assert result is not None
        assert result["score"] == 3

    def test_parses_markdown_json_block(self):
        text = '```json\n{"score": 5, "reason": "Excellent"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["score"] == 5

    def test_parses_plain_markdown_block(self):
        text = '```\n{"score": 2, "reason": "Poor"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["score"] == 2

    def test_extracts_json_from_mixed_text(self):
        text = 'Here is my evaluation: {"score": 3, "reason": "Average"} Hope that helps!'
        result = _extract_json(text)
        assert result is not None
        assert result["score"] == 3

    def test_returns_none_for_invalid_json(self):
        assert _extract_json("This is not JSON at all.") is None

    def test_returns_none_for_empty_string(self):
        assert _extract_json("") is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ollama(response_text: str):
    mock = MagicMock()
    mock.chat.return_value = {"message": {"content": response_text}}
    return mock


# ---------------------------------------------------------------------------
# Tests de OllamaJudge
# ---------------------------------------------------------------------------

class TestOllamaJudge:
    def test_returns_dimension_score_on_valid_response(self):
        judge = OllamaJudge(model="llama3.2:3b")
        mock  = _mock_ollama('{"score": 4, "reason": "Well grounded"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("some prompt", dimension="faithfulness")

        assert isinstance(result, DimensionScore)
        assert result.raw_score == 4
        assert result.dimension == "faithfulness"

    def test_normalizes_score_to_0_1(self):
        judge = OllamaJudge()
        mock  = _mock_ollama('{"score": 1, "reason": "Very poor"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="answer_relevance")

        assert result is not None
        assert result.score == pytest.approx(0.0)

    def test_score_5_normalizes_to_1(self):
        judge = OllamaJudge()
        mock  = _mock_ollama('{"score": 5, "reason": "Excellent"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="context_relevance")

        assert result.score == pytest.approx(1.0)

    def test_clamps_score_above_5(self):
        judge = OllamaJudge()
        mock  = _mock_ollama('{"score": 9, "reason": "Out of range"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="faithfulness")

        assert result is not None
        assert result.raw_score == 5

    def test_clamps_score_below_1(self):
        judge = OllamaJudge()
        mock  = _mock_ollama('{"score": -2, "reason": "Out of range"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="faithfulness")

        assert result is not None
        assert result.raw_score == 1

    def test_returns_none_when_json_unparseable(self):
        judge = OllamaJudge()
        mock  = _mock_ollama("I think this is a good answer!")  # sin JSON

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="faithfulness")

        assert result is None

    def test_returns_none_when_ollama_missing(self):
        judge = OllamaJudge()
        with patch.dict("sys.modules", {"ollama": None}):
            result = judge.evaluate("prompt", dimension="faithfulness")
        assert result is None

    def test_returns_none_when_ollama_raises(self):
        judge = OllamaJudge()
        mock  = MagicMock()
        mock.chat.side_effect = RuntimeError("Ollama crashed")

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="faithfulness")

        assert result is None

    def test_score_as_string_accepted(self):
        """Algunos modelos devuelven {"score": "4", ...} en lugar de int."""
        judge = OllamaJudge()
        mock  = _mock_ollama('{"score": "4", "reason": "Good"}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="answer_relevance")

        assert result is not None
        assert result.raw_score == 4

    def test_preserves_reason(self):
        judge  = OllamaJudge()
        reason = "The answer is directly supported by source [1]."
        mock   = _mock_ollama(f'{{"score": 5, "reason": "{reason}"}}')

        with patch.dict("sys.modules", {"ollama": mock}):
            result = judge.evaluate("prompt", dimension="faithfulness")

        assert result.reason == reason
