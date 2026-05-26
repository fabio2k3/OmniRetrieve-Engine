"""Tests de Paraphraser y la función _jaccard."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest

from backend.eval.paraphraser import _jaccard, Paraphraser


class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert _jaccard("apple orange", "banana pear") == 0.0

    def test_partial(self):
        j = _jaccard("the cat sat on the mat", "the cat chased a rat")
        assert 0.0 < j < 1.0

    def test_empty_strings(self):
        assert _jaccard("", "hello") == 0.0
        assert _jaccard("hello", "") == 0.0


class TestParaphraser:
    def test_returns_none_when_ollama_missing(self):
        p = Paraphraser(model="llama3.2:3b")
        with patch.dict("sys.modules", {"ollama": None}):
            assert p.paraphrase("Some text to paraphrase.") is None

    def test_rejects_too_similar_output(self):
        original = "Attention mechanisms allow models to focus on relevant parts."
        p = Paraphraser(max_retries=2)
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": original}}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert p.paraphrase(original) is None

    def test_accepts_valid_paraphrase(self):
        original = "Attention mechanisms allow models to focus on relevant parts."
        paraphrase = "Neural networks use focus strategies to identify important information."
        p = Paraphraser(max_retries=1)
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": paraphrase}}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            assert p.paraphrase(original) == paraphrase
