"""
generator.py
============
Wrapper de generación para la etapa final de RAG.

Soporta Ollama con import lazy.

Cambios
-------
- Se usa el rol ``system`` de la API de Ollama para inyectar la instrucción
  del sistema (idioma, restricción a fuentes, citas, etc.) en lugar de
  concatenarla al mensaje del usuario. Esto garantiza que el LLM la trate
  como directiva de sistema, no como parte de la query.
- El ``PromptBuilder`` ahora devuelve solo el cuerpo de usuario; la constante
  ``SYSTEM`` se pasa directamente aquí como system message.
"""

from __future__ import annotations

import logging

from .prompt_builder import PromptBuilder

log = logging.getLogger(__name__)

_prompt_builder = PromptBuilder()


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
        """Genera una respuesta de texto a partir del prompt de usuario."""
        try:
            return self._generate_ollama(prompt)
        except Exception as exc:
            log.error("[gen] error_ollama=%s", exc)
            return f"[Generation error: {exc}]"

    def _generate_ollama(self, prompt: str) -> str:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            return "[Ollama package not installed. Run: pip install ollama]"

        resp = ollama.chat(
            model=self.model,
            messages=[
                # Instrucción de sistema separada del contenido de usuario,
                # para que el LLM la trate con la prioridad adecuada.
                {"role": "system",  "content": _prompt_builder.SYSTEM},
                {"role": "user",    "content": prompt},
            ],
            options={"temperature": self.temperature},
        )
        return resp["message"]["content"]