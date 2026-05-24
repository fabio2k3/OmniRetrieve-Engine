"""
fallback_searcher.py
====================
Buscador de respaldo usando DuckDuckGo (sin API key).

Compatible con duckduckgo-search v3, v4, v5 y v6+.
En v6+ el context manager es opcional y text() devuelve lista directamente.

Instalación
-----------
    pip install duckduckgo-search
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

DDGS = None
_DDGS_VERSION = "not installed"

# v8+: paquete renombrado a 'ddgs'
try:
    from ddgs import DDGS
    import ddgs as _ddgs_mod
    _DDGS_VERSION = getattr(_ddgs_mod, "__version__", "unknown")
except ImportError:
    pass

# v3-v7: nombre antiguo 'duckduckgo_search'
if DDGS is None:
    try:
        from duckduckgo_search import DDGS
        import duckduckgo_search as _ddgs_mod
        _DDGS_VERSION = getattr(_ddgs_mod, "__version__", "unknown")
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


def _fetch_page_content(url: str, max_chars: int = _MAX_PAGE_CHARS) -> str:
    """Fetchea la URL y extrae texto limpio. Devuelve '' si falla."""
    try:
        from backend.crawler.http import fetch_bytes
        raw = fetch_bytes(url, timeout=8, accept="text/html")
        if not raw:
            return ""
        return _strip_html(raw)[:max_chars]
    except Exception as exc:
        log.debug("[DuckDuckGo] No se pudo fetchear %s: %s", url, exc)
        return ""


def _ddgs_text(query: str, region: str, max_results: int) -> list[dict]:
    """
    Llama a DDGS().text() de forma compatible con v3–v6+.

    En v6+ text() devuelve list[dict] directamente.
    En v3-v5 era un generador y requería context manager.
    """
    ddgs = DDGS()

    # Intentar llamada directa (v6+)
    try:
        result = ddgs.text(query, region=region, max_results=max_results)
        if result is None:
            return []
        if isinstance(result, list):
            return result
        # Es un generador — consumirlo
        return list(result)
    except TypeError:
        pass

    # Fallback: sin max_results (versiones que no lo soportan)
    try:
        result = ddgs.text(query, region=region)
        if result is None:
            return []
        items = list(result) if not isinstance(result, list) else result
        return items[:max_results]
    except Exception as exc:
        log.error("[DuckDuckGo] Error en DDGS().text(): %s", exc)
        return []


class DuckDuckGoSearcher:
    """
    Buscador de respaldo usando DuckDuckGo.

    Parámetros
    ----------
    max_results   : número máximo de resultados (default: 5).
    region        : región de búsqueda (default: "en-us").
    fetch_content : si True, fetchea el contenido completo de cada URL
                    para dar al LLM contexto suficiente (default: True).
    """

    def __init__(
        self,
        max_results:   int  = 5,
        region:        str  = "en-us",
        fetch_content: bool = True,
    ) -> None:
        self.max_results   = max_results
        self.region        = region
        self.fetch_content = fetch_content
        log.debug("[DuckDuckGo] duckduckgo-search version: %s", _DDGS_VERSION)

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Busca en DuckDuckGo y devuelve resultados normalizados.

        Fetchea el contenido real de cada URL si fetch_content=True,
        usando el snippet de DuckDuckGo como fallback.
        """
        n = max_results or self.max_results

        if DDGS is None:
            log.error(
                "[DuckDuckGo] duckduckgo-search no instalado. "
                "Ejecuta: pip install duckduckgo-search"
            )
            return []

        try:
            log.info("[DuckDuckGo] Buscando: '%s' (max=%d, version=%s)",
                     query, n, _DDGS_VERSION)
            raw = _ddgs_text(query, region=self.region, max_results=n)
            log.info("[DuckDuckGo] %d URLs obtenidas.", len(raw))

            results = []
            for r in raw:
                title   = r.get("title", "Sin título")
                url     = r.get("href") or r.get("url", "")
                snippet = r.get("body") or r.get("snippet", "")

                content = ""
                if self.fetch_content and url:
                    log.debug("[DuckDuckGo] Fetcheando: %s", url)
                    content = _fetch_page_content(url)

                # Fallback al snippet si el fetch fue vacío o muy corto
                if len(content) < 200:
                    content = snippet
                    if content:
                        log.debug("[DuckDuckGo] Usando snippet (%d chars) para %s",
                                  len(content), url)

                results.append({
                    "title":   title,
                    "url":     url,
                    "content": content,
                    "score":   0.5,
                    "source":  "web_fallback",
                })

            log.info("[DuckDuckGo] %d resultados listos (fetch_content=%s).",
                     len(results), self.fetch_content)
            return results

        except Exception as exc:
            log.error("[DuckDuckGo] Error inesperado: %s", exc, exc_info=True)
            return []