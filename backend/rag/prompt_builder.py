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
        "Eres un asistente cientifico especializado en IA y ML.\n"
        "Responde SOLO usando los documentos proporcionados.\n"
        "Si la respuesta no aparece en el contexto, responde exactamente: 'No encontrado en fuentes.'.\n"
        "Cita evidencia en linea usando [1], [2], etc."
    )

    def build(self, query: str, context: str) -> str:
        prompt = (
            f"{self.SYSTEM}\n\n"
            f"Documentos:\n{context or '[sin contexto]'}\n\n"
            f"Pregunta: {query}\n\n"
            "Responde con citas."
        )
        log.debug("[prompt] prompt construido chars=%d", len(prompt))
        return prompt
