"""
prompt_builder.py
=================
Plantillas de prompt para respuestas RAG grounded.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class PromptBuilder:
    """Construye el prompt final desde query + contexto recuperado."""

    SYSTEM = (
        "You are a scientific assistant specialized in AI and ML research.\n"
        "Answer ONLY using the provided documents.\n"
        "Always respond in ENGLISH, regardless of the language of the query "
        "or the retrieved chunks.\n"
        "If the answer is not in the context, reply exactly: "
        "'Not found in sources.'\n"
        "Cite inline evidence using [1], [2], etc."
    )

    def build(self, query: str, context: str) -> str:
        return (
            f"Documents:\n{context or '[no context]'}\n\n"
            f"Question: {query}\n\n"
            "Answer with citations."
        )