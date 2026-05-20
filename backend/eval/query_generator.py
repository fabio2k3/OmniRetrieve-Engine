"""
query_generator.py
==================
Genera queries realistas de usuario a partir del texto de un chunk.

Diferencia clave con paraphraser.py
-------------------------------------
El paráfraser toma un fragmento y lo reescribe con otras palabras.
El generador de queries toma el chunk completo y produce la *pregunta*
que un usuario haría si buscara esa información — algo que nunca
aparece textualmente en el chunk, sino que lo refleja conceptualmente.

Esto hace el dataset de evaluación mucho más representativo del uso
real del sistema RAG, donde los usuarios hacen preguntas, no pegan
trozos de texto de papers.

Validaciones aplicadas
-----------------------
· La query debe terminar en '?' (es una pregunta).
· Mínimo 15 caracteres — evita respuestas vacías o triviales.
· Máximo 250 caracteres — las queries reales son concisas.
· Similitud Jaccard con el chunk fuente < 0.25 — el LLM no debe
  copiar frases literales del texto original.
· Si el LLM falla ``max_retries`` veces, devuelve None y el caso
  se omite sin abortar la generación.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Umbral de similitud Jaccard máximo entre query generada y chunk fuente.
# Más estricto que en paraphraser (0.60) porque una query real no debe
# contener frases del paper — debe ser lo que el usuario escribiría.
_MAX_JACCARD    = 0.25
_MIN_QUERY_LEN  = 15
_MAX_QUERY_LEN  = 250


def _jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_valid(query: str, source_text: str) -> tuple[bool, str]:
    """
    Valida la query generada. Devuelve (válida, motivo_de_fallo).
    """
    q = query.strip()

    if len(q) < _MIN_QUERY_LEN:
        return False, f"demasiado corta ({len(q)} chars)"

    if len(q) > _MAX_QUERY_LEN:
        return False, f"demasiado larga ({len(q)} chars)"

    if not q.endswith("?"):
        return False, "no termina en '?'"

    j = _jaccard(q, source_text)
    if j > _MAX_JACCARD:
        return False, f"demasiado similar al chunk (Jaccard={j:.2f})"

    return True, ""


_QUERY_GEN_PROMPT = """\
You are evaluating a scientific paper retrieval system.

Read the excerpt below from a scientific paper and write ONE realistic question \
that a researcher or student would type into a search box to find this information.

Rules (follow all strictly):
1. Write a natural, concise question — as a real user would phrase it.
2. Do NOT copy phrases or sentences directly from the text.
3. The question must be specific enough that this excerpt is a relevant answer.
4. Output ONLY the question — no preamble, no explanation, no numbering.
5. The question MUST end with a question mark.
6. Write in the same language as the input text.
7. Keep it under 200 characters.

Excerpt:
\"\"\"
{text}
\"\"\"

Question:"""


class QueryGenerator:
    """
    Genera queries realistas de usuario a partir del texto de un chunk,
    usando el mismo backend Ollama que el sistema RAG.

    Parámetros
    ----------
    model       : modelo Ollama a usar.
    temperature : temperatura de muestreo. Algo de variedad es deseable.
    max_retries : intentos antes de descartar el caso si la query no es válida.
    """

    def __init__(
        self,
        model:       str   = "llama3.2:3b",
        temperature: float = 0.4,
        max_retries: int   = 3,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(self, chunk_text: str) -> str | None:
        """
        Genera una query realista de usuario para el chunk dado.

        Devuelve
        --------
        str   → query válida (pasa todas las validaciones).
        None  → todos los intentos fallaron o Ollama no disponible.
        """
        # Truncar el chunk si es muy largo — el prompt no necesita más
        context = chunk_text[:1200].strip()

        for attempt in range(1, self.max_retries + 1):
            raw = self._call_ollama(context)
            if raw is None:
                log.warning("[query_gen] Ollama no disponible.")
                return None

            query = raw.strip()
            valid, reason = _is_valid(query, chunk_text)

            if valid:
                log.debug(
                    "[query_gen] attempt=%d OK query=%r",
                    attempt, query[:80],
                )
                return query

            log.debug(
                "[query_gen] attempt=%d FAIL reason=%s query=%r",
                attempt, reason, query[:80],
            )

        log.warning(
            "[query_gen] Se agotaron %d intentos para chunk: %r…",
            self.max_retries, chunk_text[:60],
        )
        return None

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _call_ollama(self, text: str) -> str | None:
        try:
            import ollama  # type: ignore[import-not-found]
        except ImportError:
            log.error("[query_gen] Paquete ollama no instalado.")
            return None

        prompt = _QUERY_GEN_PROMPT.format(text=text)
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
            return resp["message"]["content"]
        except Exception as exc:
            log.error("[query_gen] error_ollama=%s", exc)
            return None
