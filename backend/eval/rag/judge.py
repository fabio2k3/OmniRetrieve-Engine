"""
judge.py
========
Juez LLM: envía un prompt al modelo Ollama y parsea la puntuación devuelta.

Responsabilidad única
---------------------
Comunicarse con Ollama y extraer un DimensionScore del texto de respuesta.
No conoce las dimensiones de evaluación ni cómo se construyen los prompts.

Protocolo esperado del LLM
---------------------------
El LLM debe responder con JSON puro (instruido por prompts.py):
    {"score": <1-5>, "reason": "<texto breve>"}

Si la respuesta no es parseable, se intenta extraer el JSON de un bloque
de código markdown (```json ... ```).  Si sigue fallando, se devuelve None
para que el llamador decida cómo manejar el error.
"""

from __future__ import annotations

import json
import logging
import re

from ._types import Dimension, DimensionScore

log = logging.getLogger(__name__)

# Patrón para extraer JSON de bloques markdown ```json ... ``` o ``` ... ```
_MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """
    Intenta parsear JSON de la respuesta del LLM.

    Estrategia
    ----------
    1. Parseo directo del texto completo (limpio).
    2. Extracción de bloque markdown ``` json ``` si el parseo directo falla.
    3. Búsqueda del primer { … } en el texto como último recurso.
    """
    text = text.strip()

    # Intento 1: parseo directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Intento 2: bloque markdown
    match = _MARKDOWN_JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Intento 3: primer objeto JSON en el texto
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


class OllamaJudge:
    """
    Juez basado en Ollama.

    Parámetros
    ----------
    model       : modelo Ollama a usar como juez.
    temperature : temperatura de muestreo (baja para determinismo).
    """

    def __init__(
        self,
        model:       str   = "llama3.2:3b",
        temperature: float = 0.0,
    ) -> None:
        self.model       = model
        self.temperature = temperature

    def evaluate(self, prompt: str, dimension: Dimension) -> DimensionScore | None:
        """
        Envía el prompt al LLM y parsea la respuesta como DimensionScore.

        Devuelve
        --------
        DimensionScore si el LLM respondió con JSON válido y puntuación 1-5.
        None si el LLM no está disponible o la respuesta no es parseable.
        """
        raw_text = self._call_ollama(prompt)
        if raw_text is None:
            return None

        return self._parse_score(raw_text, dimension)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str | None:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            log.error("[judge] Paquete ollama no instalado.")
            return None

        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]
        except Exception as exc:
            log.error("[judge] Error en llamada Ollama: %s", exc)
            return None

    def _parse_score(self, text: str, dimension: Dimension) -> DimensionScore | None:
        payload = _extract_json(text)
        if payload is None:
            log.warning("[judge] No se pudo parsear JSON de respuesta: %r…", text[:120])
            return None

        raw_score = payload.get("score")
        reason    = payload.get("reason", "")

        if not isinstance(raw_score, int):
            # Algunos modelos devuelven el score como string
            try:
                raw_score = int(raw_score)
            except (TypeError, ValueError):
                log.warning("[judge] Campo 'score' no es entero: %r", raw_score)
                return None

        return DimensionScore.from_raw(
            raw=raw_score,
            reason=str(reason),
            dimension=dimension,
        )
