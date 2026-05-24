"""Tests de PromptBuilder — estructura del prompt grounded (en inglés)."""
from __future__ import annotations
from backend.rag.prompt_builder import PromptBuilder


def test_prompt_contains_required_sections():
    pb = PromptBuilder()
    prompt = pb.build("What is self-attention?", "[1] Attention Paper (2017)\n...")
    assert "Documents:" in prompt
    assert "Question:" in prompt
    assert "Answer with citations." in prompt