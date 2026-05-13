"""
paraphraser.py
==============
Genera paráfrasis semánticamente equivalentes de fragmentos de texto
usando el mismo backend Ollama que el sistema RAG.

Objetivo
--------
Dado un fragmento de un chunk real, producir una query alternativa que
exprese el mismo concepto con palabras distintas.  El retriever debería
recuperar el chunk original aunque la query no comparta léxico con él.

Esto estresa específicamente:
  · la rama densa (FAISS / embeddings semánticos)
  · el reranker (CrossEncoder)
  · y expone debilidades del retriever léxico (LSI/BM25)

Diseño
------
- Prompt cuidadosamente construido para que el LLM NO copie palabras clave.
- Temperatura más alta que en generación RAG (0.5) para diversidad léxica.
- Validación simple: si el LLM devuelve el texto igual o casi igual se
  rechaza y se reintenta (hasta max_retries veces).
- Si todos los intentos fallan se devuelve None para que el generador de
  dataset omita ese caso en lugar de guardar un caso inútil.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Umbral de similitud léxica (Jaccard de unigramas) por encima del cual
# se considera que la paráfrasis es demasiado similar al original.
_MAX_JACCARD = 0.60


def _jaccard(a: str, b: str) -> float:
    """Similitud de Jaccard entre los conjuntos de palabras de a y b."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_PARAPHRASE_PROMPT = """\
Your task is to paraphrase the following text fragment.

Rules (follow all of them strictly):
1. Preserve the full meaning and all key technical concepts.
2. Use completely different wording — do NOT copy more than 2 consecutive words from the original.
3. You may change sentence structure, use synonyms, reorder ideas, and switch between active and passive voice.
4. Write in the same language as the input text.
5. Output ONLY the paraphrased text — no preamble, no explanation, no quotes.

Text to paraphrase:
\"\"\"
{text}
\"\"\"

Paraphrase:"""


class Paraphraser:
    """
    Genera paráfrasis usando Ollama.

    Parámetros
    ----------
    model       : modelo Ollama (mismo que Generator por defecto).
    temperature : temperatura de muestreo. Más alta → más diversidad léxica.
    max_retries : intentos antes de descartar el caso si la paráfrasis
                  resulta demasiado similar al original.
    """

    def __init__(
        self,
        model:       str   = "llama3.2:3b",
        temperature: float = 0.55,
        max_retries: int   = 3,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def paraphrase(self, text: str) -> str | None:
        """
        Genera una paráfrasis del texto dado.

        Devuelve
        --------
        str   → paráfrasis válida (Jaccard < _MAX_JACCARD).
        None  → todos los intentos produjeron paráfrasis demasiado similares
                o el LLM no está disponible.
        """
        for attempt in range(1, self.max_retries + 1):
            result = self._call_ollama(text)
            if result is None:
                log.warning("[paraphraser] Ollama no disponible; se omite el caso.")
                return None

            result = result.strip()
            similarity = _jaccard(text, result)

            if similarity <= _MAX_JACCARD:
                log.debug(
                    "[paraphraser] attempt=%d jaccard=%.2f OK",
                    attempt, similarity,
                )
                return result

            log.debug(
                "[paraphraser] attempt=%d jaccard=%.2f demasiado similar, reintentando…",
                attempt, similarity,
            )

        log.warning(
            "[paraphraser] Se agotaron %d intentos para texto: %r…",
            self.max_retries, text[:80],
        )
        return None

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _call_ollama(self, text: str) -> str | None:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            log.error("[paraphraser] Paquete ollama no instalado.")
            return None

        prompt = _PARAPHRASE_PROMPT.format(text=text)
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]
        except Exception as exc:
            log.error("[paraphraser] error_ollama=%s", exc)
            return None
