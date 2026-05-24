"""
web_search/site_searcher.py
============================
Buscador web dirigido a sitios académicos de confianza.

Reemplaza la búsqueda general de DuckDuckGo por búsquedas restringidas
a una lista configurable de dominios semilla usando el operador site: de
DuckDuckGo. Esto garantiza que los resultados sean relevantes para el
dominio del proyecto (IA, ML, ética de IA) y no procedan de redes
sociales, foros o fuentes no académicas.

Flujo
-----
1. Construir query restringida: "{query} (site:dominio1 OR site:dominio2 …)"
2. Buscar con ddgs / duckduckgo_search
3. Para cada URL obtenida, fetchear el contenido real de la página
4. Devolver resultados normalizados

Instalación
-----------
    pip install ddgs          # v8+
    # o: pip install duckduckgo-search  # v3-v7
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .sites import DEFAULT_SEED_DOMAINS, get_site_filter

log = logging.getLogger(__name__)

# Importación compatible con ddgs v8+ y duckduckgo_search v3-v7
_DDGS = None
_DDGS_VERSION = "not installed"

try:
    from ddgs import DDGS as _DDGS
    import ddgs as _mod
    _DDGS_VERSION = getattr(_mod, "__version__", "unknown")
except ImportError:
    pass

if _DDGS is None:
    try:
        from duckduckgo_search import DDGS as _DDGS
        import duckduckgo_search as _mod
        _DDGS_VERSION = getattr(_mod, "__version__", "unknown")
    except ImportError:
        pass

_MAX_PAGE_CHARS = 3_000


def _strip_html(html_bytes: bytes) -> str:
    """Extrae texto legible de HTML sin dependencias externas."""
    try:
        text = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
               .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_page(url: str, max_chars: int = _MAX_PAGE_CHARS) -> str:
    """Fetchea la URL y extrae texto limpio. Devuelve '' si falla."""
    try:
        from backend.crawler.http import fetch_bytes
        raw = fetch_bytes(url, timeout=10, accept="text/html")
        if not raw:
            return ""
        return _strip_html(raw)[:max_chars]
    except Exception as exc:
        log.debug("[SiteSearcher] No se pudo fetchear %s: %s", url, exc)
        return ""


def _ddgs_text(query: str, region: str, max_results: int) -> list[dict]:
    """Llama a DDGS().text() compatible con v3-v8+."""
    if _DDGS is None:
        return []
    ddgs = _DDGS()
    try:
        result = ddgs.text(query, region=region, max_results=max_results)
        if result is None:
            return []
        return result if isinstance(result, list) else list(result)
    except TypeError:
        try:
            result = ddgs.text(query, region=region)
            items = result if isinstance(result, list) else list(result)
            return items[:max_results]
        except Exception as exc:
            log.error("[SiteSearcher] Error en DDGS().text(): %s", exc)
            return []
    except Exception as exc:
        log.error("[SiteSearcher] Error en DDGS().text(): %s", exc)
        return []


class SiteSearcher:
    """
    Buscador web dirigido a sitios académicos de confianza.

    Parámetros
    ----------
    seed_domains  : lista de dominios donde buscar. Si está vacía usa
                    DEFAULT_SEED_DOMAINS de sites.py.
    max_results   : máximo de resultados por búsqueda (default: 5).
    region        : región de búsqueda (default: "en-us").
    fetch_content : si True, fetchea el contenido completo de cada URL
                    para dar al LLM contexto suficiente (default: True).
    """

    def __init__(
        self,
        seed_domains:  list[str] | None = None,
        max_results:   int  = 5,
        region:        str  = "en-us",
        fetch_content: bool = True,
    ) -> None:
        self.seed_domains  = seed_domains or DEFAULT_SEED_DOMAINS
        self.max_results   = max_results
        self.region        = region
        self.fetch_content = fetch_content

        log.info(
            "[SiteSearcher] Inicializado con %d dominios semilla (ddgs version=%s).",
            len(self.seed_domains), _DDGS_VERSION,
        )

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Busca en los dominios semilla y devuelve resultados normalizados.

        La query se restringe a los dominios semilla usando el operador
        site: de DuckDuckGo: "{query} (site:dom1 OR site:dom2 …)"

        Parámetros
        ----------
        query       : consulta de búsqueda del usuario.
        max_results : sobreescribe el default si se especifica.

        Devuelve
        --------
        Lista de dicts con keys: title, url, content, score, source.
        Lista vacía si la búsqueda falla o no hay resultados.
        """
        if _DDGS is None:
            log.error(
                "[SiteSearcher] Motor de búsqueda no instalado. "
                "Ejecuta: pip install ddgs"
            )
            return []

        n = max_results or self.max_results

        # Construir query restringida a los dominios semilla
        site_filter    = get_site_filter(self.seed_domains)
        restricted_q   = f"{query} ({site_filter})" if site_filter else query

        log.info(
            "[SiteSearcher] Buscando en %d dominios: '%s'",
            len(self.seed_domains), query,
        )
        log.debug("[SiteSearcher] Query restringida: %s", restricted_q[:120])

        raw = _ddgs_text(restricted_q, region=self.region, max_results=n)
        log.info("[SiteSearcher] %d URLs obtenidas.", len(raw))

        results = []
        for r in raw:
            title   = r.get("title", "Sin título")
            url     = r.get("href") or r.get("url", "")
            snippet = r.get("body") or r.get("snippet", "")

            content = ""
            if self.fetch_content and url:
                log.debug("[SiteSearcher] Fetcheando: %s", url)
                content = _fetch_page(url)

            # Fallback al snippet si el fetch fue vacío o muy corto
            if len(content) < 200:
                content = snippet

            if not content:
                log.debug("[SiteSearcher] Sin contenido para %s — omitiendo.", url)
                continue

            results.append({
                "title":   title,
                "url":     url,
                "content": content,
                "score":   0.5,
                "source":  "web_search",
            })

        log.info(
            "[SiteSearcher] %d resultados con contenido (fetch_content=%s).",
            len(results), self.fetch_content,
        )
        return results