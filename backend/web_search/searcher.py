"""
web_search/searcher.py
======================
Cliente de búsqueda web con Tavily como primario y SiteSearcher como fallback.

Cambio respecto a la versión anterior
--------------------------------------
El fallback ya no es DuckDuckGo general sino SiteSearcher, que restringe
las búsquedas a una lista configurable de dominios académicos de confianza
(ver web_search/sites.py). Esto evita resultados de redes sociales,
foros u otras fuentes no relevantes para el proyecto.

Instalación
-----------
    pip install tavily-python   # primario (requiere API key)
    pip install ddgs            # fallback (sin API key)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from backend.web_search.site_searcher import SiteSearcher
from backend.web_search.sites import DEFAULT_SEED_DOMAINS

log = logging.getLogger(__name__)


class WebSearcher:
    """
    Wrapper sobre la API de Tavily con SiteSearcher como fallback.

    Si Tavily no está disponible o falla, las búsquedas se redirigen
    automáticamente a SiteSearcher, que usa DuckDuckGo restringido a
    dominios académicos de confianza.

    Parámetros
    ----------
    api_key      : clave de API de Tavily. Si no se pasa, se lee de
                   TAVILY_API_KEY en el .env. Si tampoco está, se
                   activa directamente SiteSearcher sin excepción.
    max_results  : número máximo de resultados por búsqueda (default: 5).
    search_depth : "basic" | "advanced" — solo para Tavily (default: "basic").
    use_fallback : si True, usa SiteSearcher cuando Tavily falla (default: True).
    seed_domains : dominios donde buscar en modo fallback. Si está vacío
                   usa DEFAULT_SEED_DOMAINS de sites.py.
    fetch_content: si True, SiteSearcher fetchea el contenido completo de
                   cada URL (default: True).
    """

    def __init__(
        self,
        api_key:      str | None  = None,
        max_results:  int         = 5,
        search_depth: str         = "basic",
        use_fallback: bool        = True,
        seed_domains: list[str] | None = None,
        fetch_content: bool       = True,
    ) -> None:
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            api_key = os.getenv("TAVILY_API_KEY")

        self.api_key           = api_key
        self.max_results       = max_results
        self.search_depth      = search_depth
        self.use_fallback      = use_fallback
        self._tavily_available = bool(api_key)
        self._client           = None
        self._fallback         = SiteSearcher(
            seed_domains  = seed_domains or DEFAULT_SEED_DOMAINS,
            max_results   = max_results,
            fetch_content = fetch_content,
        )

        if not self._tavily_available:
            log.warning(
                "[WebSearcher] TAVILY_API_KEY no encontrada. "
                "%s",
                "Se usará SiteSearcher (dominios académicos)."
                if use_fallback else
                "Fallback desactivado — búsqueda web no disponible.",
            )

    def _get_client(self):
        """Inicializa el cliente Tavily de forma lazy."""
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Busca en la web. Intenta Tavily primero; si falla usa SiteSearcher.

        Si no hay API key, va directamente a SiteSearcher.

        Parámetros
        ----------
        query       : consulta de búsqueda del usuario.
        max_results : sobreescribe el default si se especifica.

        Devuelve
        --------
        Lista de dicts con keys: title, url, content, score, source.
        Lista vacía solo si ambos métodos fallan.
        """
        n = max_results or self.max_results

        # Sin API key → ir directamente al fallback
        if not self._tavily_available:
            if self.use_fallback:
                log.info("[WebSearcher] Sin Tavily — usando SiteSearcher.")
                return self._fallback.search(query, max_results=n)
            log.warning("[WebSearcher] Sin Tavily y fallback desactivado.")
            return []

        # Con API key → intentar Tavily, fallback en cualquier error
        try:
            log.info("[WebSearcher] Buscando en Tavily: '%s' (max=%d)", query, n)
            client   = self._get_client()
            response = client.search(
                query=query,
                max_results=n,
                search_depth=self.search_depth,
                include_answer=False,
                include_raw_content=False,
            )
            results = self._normalize(response.get("results", []))
            log.info("[WebSearcher] %d resultados de Tavily.", len(results))
            return results

        except Exception as exc:
            log.warning(
                "[WebSearcher] Tavily falló (%s). %s",
                exc,
                "Activando SiteSearcher…" if self.use_fallback
                else "Sin fallback configurado.",
            )
            if self.use_fallback:
                return self._fallback.search(query, max_results=n)
            return []

    def _normalize(self, raw_results: list[dict]) -> list[dict[str, Any]]:
        """Normaliza los resultados de Tavily al formato interno."""
        return [
            {
                "title":   r.get("title", "Sin título"),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
                "score":   float(r.get("score", 0.0)),
                "source":  "web",
            }
            for r in raw_results
        ]