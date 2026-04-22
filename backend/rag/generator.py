"""
generator.py
============
Wrapper de generacion para la etapa final de RAG.

Soporta Ollama con import lazy.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Generator:
    """Generador basado en Ollama para respuestas RAG."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """Genera una respuesta de texto a partir de un prompt final."""
        try:
            return self._generate_ollama(prompt)
        except Exception as exc:
            log.error("[gen] error_ollama=%s", exc)
            return f"[Error de generacion: {exc}]"

    def _generate_ollama(self, prompt: str) -> str:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            return "[Paquete Ollama no instalado. Ejecuta: pip install ollama]"

        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return resp["message"]["content"]
