"""
rag/judge.py
============
Juez LLM: envía un prompt al modelo Ollama y parsea la puntuación devuelta.

Cambio respecto a la versión anterior
--------------------------------------
Se activa ``format="json"`` en la llamada a Ollama. Esto fuerza al modelo
a generar JSON válido a nivel de tokens (grammar-constrained decoding),
eliminando la causa raíz de errores como ``"reason": texto sin comillas``.
El parseo de fallback se mantiene como red de seguridad para APIs que no
soporten el parámetro format.
"""

from __future__ import annotations

import json
import logging
import re

from ._types import Dimension, DimensionScore

log = logging.getLogger(__name__)

_MARKDOWN_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """
    Intenta parsear JSON de la respuesta del LLM.

    Estrategia (orden de intento)
    -----------------------------
    1. Parseo directo del texto completo.
    2. Extracción de bloque markdown ```json … ```.
    3. Primer objeto { … } encontrado en el texto.
    4. Reparación básica: añadir comillas al valor de 'reason' si falta.
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
        fragment = text[brace_start : brace_end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            # Intento 4: reparar "reason": texto sin comillas
            repaired = re.sub(
                r'"reason"\s*:\s*([^",\}][^,\}]*)',
                lambda m: f'"reason": "{m.group(1).strip()}"',
                fragment,
            )
            try:
                return json.loads(repaired)
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
                format="json",      # ← grammar-constrained JSON output
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]
        except Exception as exc:
            # Algunos clientes Ollama antiguos no soportan format="json"
            # → reintentar sin él como fallback
            log.warning(
                "[judge] format='json' no soportado (%s), reintentando sin él.", exc
            )
            try:
                resp = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature},
                )
                return resp["message"]["content"]
            except Exception as exc2:
                log.error("[judge] Error en llamada Ollama: %s", exc2)
                return None

    def _parse_score(self, text: str, dimension: Dimension) -> DimensionScore | None:
        payload = _extract_json(text)
        if payload is None:
            log.warning("[judge] No se pudo parsear JSON: %r…", text[:120])
            return None

        raw_score = payload.get("score")
        reason    = payload.get("reason", "")

        if not isinstance(raw_score, int):
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