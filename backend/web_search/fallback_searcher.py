"""
fallback_searcher.py
====================
Buscador de respaldo usando DuckDuckGo (sin API key).

Se activa automáticamente cuando Tavily falla o no está disponible.
No requiere instalación adicional más allá de duckduckgo-search.

Instalación
-----------
    pip install duckduckgo-search

Uso directo
-----------
    from backend.web_search.fallback_searcher import DuckDuckGoSearcher

    searcher = DuckDuckGoSearcher()
    results  = searcher.search("transformer attention mechanisms")
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class DuckDuckGoSearcher:
    """
    Buscador de respaldo usando DuckDuckGo.

    No requiere API key. Se usa cuando Tavily no está disponible
    o falla durante una búsqueda.

    Parámetros
    ----------
    max_results : número máximo de resultados (default: 5)
    region      : región de búsqueda (default: "en-us")
    """

    def __init__(
        self,
        max_results: int = 5,
        region: str = "en-us",
    ) -> None:
        self.max_results = max_results
        self.region      = region

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Busca en DuckDuckGo y devuelve resultados normalizados.

        Devuelve el mismo formato que WebSearcher para ser
        intercambiable sin cambios en el pipeline.

        Parámetros
        ----------
        query       : consulta de búsqueda en texto libre
        max_results : sobreescribe el default si se especifica

        Devuelve
        --------
        Lista de dicts con keys: title, url, content, score, source.
        Lista vacía si la búsqueda falla.
        """
        n = max_results or self.max_results

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            log.error(
                "[DuckDuckGo] duckduckgo-search no está instalado. "
                "Instálalo con: pip install duckduckgo-search"
            )
            return []

        try:
            log.info("[DuckDuckGo] Buscando (fallback): '%s' (max=%d)", query, n)
            results = []

            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,
                    region=self.region,
                    max_results=n,
                ):
                    results.append({
                        "title":   r.get("title", "Sin título"),
                        "url":     r.get("href", ""),
                        "content": r.get("body", ""),
                        "score":   0.5,   # DuckDuckGo no devuelve score — valor neutro
                        "source":  "web_fallback",
                    })

            log.info("[DuckDuckGo] %d resultados obtenidos (fallback).", len(results))
            return results

        except Exception as exc:
            log.error("[DuckDuckGo] Error en búsqueda fallback: %s", exc)
            return []