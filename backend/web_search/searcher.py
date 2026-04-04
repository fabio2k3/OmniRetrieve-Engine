"""
searcher.py
===========
Cliente de búsqueda web usando la API de Tavily.

Responsabilidad única: recibir una query, llamar a Tavily
y devolver una lista de resultados normalizados.

Instalación
-----------
    pip install tavily-python python-dotenv

Configuración
-------------
Crea un archivo .env en la raíz del proyecto con:
    TAVILY_API_KEY=tvly-tu-key-aqui

Uso
---
    from backend.web_search.searcher import WebSearcher

    searcher = WebSearcher()  # lee la key del .env automáticamente
    results  = searcher.search("transformer attention mechanisms", max_results=5)
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)


class WebSearcher:
    """
    Wrapper sobre la API de Tavily para búsqueda web.

    Parámetros
    ----------
    api_key      : clave de API de Tavily (tvly-...).
                   Si no se pasa, se lee de TAVILY_API_KEY en el archivo .env.
    max_results  : número máximo de resultados por búsqueda (default: 5)
    search_depth : "basic" (más rápido) | "advanced" (más preciso, más créditos)
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ) -> None:
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass  # dotenv opcional — usa variables del sistema si no está
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError(
                    "TAVILY_API_KEY no encontrada. "
                    "Añádela al archivo .env o pásala como parámetro."
                )

        self.api_key      = api_key
        self.max_results  = max_results
        self.search_depth = search_depth
        self._client      = None

    def _get_client(self):
        """Inicializa el cliente Tavily de forma lazy."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "tavily-python no está instalado. "
                    "Instálalo con: pip install tavily-python"
                ) from e
        return self._client

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Busca en la web usando Tavily y devuelve resultados normalizados.

        Parámetros
        ----------
        query       : consulta de búsqueda en texto libre
        max_results : sobreescribe el default si se especifica

        Devuelve
        --------
        Lista de dicts con keys: title, url, content, score, source.
        Lista vacía si la búsqueda falla.
        """
        n      = max_results or self.max_results
        client = self._get_client()

        try:
            log.info("[WebSearcher] Buscando en Tavily: '%s' (max=%d)", query, n)
            response = client.search(
                query=query,
                max_results=n,
                search_depth=self.search_depth,
                include_answer=False,
                include_raw_content=False,
            )
            results = self._normalize(response.get("results", []))
            log.info("[WebSearcher] %d resultados obtenidos.", len(results))
            return results

        except Exception as exc:
            log.error("[WebSearcher] Error en búsqueda Tavily: %s", exc)
            return []

    def _normalize(self, raw_results: list[dict]) -> list[dict[str, Any]]:
        """Normaliza los resultados de Tavily al formato interno del sistema."""
        normalized = []
        for r in raw_results:
            normalized.append({
                "title":   r.get("title", "Sin título"),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
                "score":   float(r.get("score", 0.0)),
                "source":  "web",
            })
        return normalized