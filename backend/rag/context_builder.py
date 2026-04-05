"""
context_builder.py
==================
Construccion de contexto para RAG a partir de resultados chunk-level.

Responsabilidad
---------------
- Seleccionar evidencia relevante de retrieval/reranking.
- Formatear contexto con citas numericas [1], [2], ...
- Exponer una estructura de "sources" para respuestas de API/UI.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.retrieval.protocols import RetrievalResult

log = logging.getLogger(__name__)


class ContextBuilder:
    """Selecciona y formatea evidencia para construir el prompt."""

    def build(
        self,
        results: list[RetrievalResult],
        max_chunks: int = 5,
        max_chars: int = 400,
    ) -> str:
        """Devuelve un bloque de contexto con citas inline."""
        if not results:
            return ""

        selected = results[:max_chunks]
        parts: list[str] = []

        for i, r in enumerate(selected, start=1):
            title = self._get_title(r)
            year = self._get_year(r)
            snippet = (r.text or "")[:max_chars].strip()
            if len(r.text or "") > max_chars:
                snippet += "..."

            header = f"[{i}] {title} ({year})"
            body = snippet if snippet else "[texto de chunk vacio]"
            parts.append(f"{header}\n{body}")

        context = "\n\n".join(parts)
        log.debug(
            "[context] contexto construido chunks=%d chars=%d",
            len(selected),
            len(context),
        )
        return context

    def build_sources(
        self,
        results: list[RetrievalResult],
        max_sources: int = 5,
    ) -> list[dict[str, Any]]:
        """Construye payload estructurado de fuentes para respuesta final."""
        out: list[dict[str, Any]] = []
        for i, r in enumerate(results[:max_sources], start=1):
            out.append(
                {
                    "citation": i,
                    "chunk_id": r.chunk_id,
                    "arxiv_id": r.arxiv_id,
                    "chunk_index": r.chunk_index,
                    "title": self._get_title(r),
                    "year": self._get_year(r),
                    "score": r.score,
                    "score_type": r.score_type,
                }
            )
        return out

    @staticmethod
    def _get_title(result: RetrievalResult) -> str:
        meta = result.metadata or {}
        return (
            str(meta.get("title") or meta.get("document_title") or meta.get("paper_title") or result.arxiv_id)
        )

    @staticmethod
    def _get_year(result: RetrievalResult) -> str:
        meta = result.metadata or {}
        year = meta.get("year")
        if year is not None:
            return str(year)

        published = meta.get("published")
        if isinstance(published, str) and len(published) >= 4:
            return published[:4]

        return "n/a"
