"""
clients/arxiv/extractors/html.py
=================================
Extractor de texto para HTML generado por LaTeXML (arXiv).

arXiv convierte los ficheros ``.tex`` originales a HTML usando LaTeXML,
que produce una estructura muy predecible con clases CSS ``ltx_*``.
Apuntar exactamente a esas clases da resultados mucho más limpios que
un parser HTML genérico.

API pública
-----------
extract(raw: bytes) -> str
    Punto de entrada. Recibe los bytes crudos del HTML y devuelve
    el texto limpio del artículo.
"""

from __future__ import annotations

import html as _html_std
import logging
import re
from html.parser import HTMLParser
from typing import List

from ....chunker import clean_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser HTML específico para LaTeXML
# ---------------------------------------------------------------------------

class _LaTeXMLParser(HTMLParser):
    """
    Extrae texto de HTML LaTeXML ignorando ruido tipográfico y secciones
    que ya están disponibles como metadatos (autores, referencias, etc.).

    Clases CSS omitidas
    -------------------
    ltx_authors, ltx_bibliography, ltx_figure, ltx_table,
    ltx_equation, ltx_equationgroup, ltx_pagination,
    ltx_page_footer, ltx_note, ltx_minipage.

    Tags omitidos
    -------------
    script, style, nav, math, svg.
    """

    _SKIP_CLASSES = frozenset({
        "ltx_authors",
        "ltx_bibliography",
        "ltx_figure",
        "ltx_table",
        "ltx_equation",
        "ltx_equationgroup",
        "ltx_pagination",
        "ltx_page_footer",
        "ltx_note",
        "ltx_minipage",
    })
    _SKIP_TAGS = frozenset({"script", "style", "nav", "math", "svg"})
    _HEADING_CLASSES = frozenset({
        "ltx_title_document", "ltx_title_section",
        "ltx_title_subsection", "ltx_title_subsubsection",
        "ltx_title_abstract", "ltx_title_appendix",
    })

    def __init__(self) -> None:
        super().__init__()
        self._in_doc     = False
        self._skip_depth = 0
        self._in_heading = False
        self._parts: List[str] = []

    def _classes(self, attrs) -> frozenset:
        return frozenset(dict(attrs).get("class", "").split())

    def handle_starttag(self, tag: str, attrs) -> None:
        classes = self._classes(attrs)
        if "ltx_document" in classes:
            self._in_doc = True
            return
        if not self._in_doc:
            return
        if tag in self._SKIP_TAGS or classes & self._SKIP_CLASSES:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if classes & self._HEADING_CLASSES:
            self._parts.append("\n\n")
            self._in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if not self._in_doc:
            return
        if tag in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            if tag in ("div", "section", "figure", "table", "span"):
                self._skip_depth -= 1
            return
        if self._in_heading and tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n\n")
            self._in_heading = False
        if tag == "p":
            self._parts.append("\n")
        if tag == "section":
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_doc and self._skip_depth == 0:
            self._parts.append(data)

    def result(self) -> str:
        return _html_std.unescape("".join(self._parts))


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def extract(raw: bytes) -> str:
    """
    Extrae texto limpio de los bytes crudos de un HTML LaTeXML de arXiv.

    Intenta primero el parser específico de LaTeXML (``ltx_document``).
    Si el documento no tiene esa estructura, cae a un extractor genérico
    (elimina tags y scripts con regex).

    Parámetros
    ----------
    raw : bytes crudos del HTML descargado.

    Returns
    -------
    str
        Texto normalizado del artículo (sin ruido tipográfico).

    Raises
    ------
    ValueError
        Si el texto resultante es demasiado corto para ser útil
        (< 500 caracteres incluso con el fallback).
    """
    decoded = raw.decode("utf-8", errors="replace")

    parser = _LaTeXMLParser()
    parser.feed(decoded)
    text = clean_text(parser.result())

    if len(text) < 500:
        logger.warning("[html] ltx_document no encontrado — usando fallback genérico.")
        stripped = re.sub(
            r"<(script|style)[^>]*>.*?</(script|style)>",
            "", decoded, flags=re.DOTALL | re.IGNORECASE,
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        text = clean_text(_html_std.unescape(stripped))

    if len(text) < 500:
        raise ValueError(f"HTML demasiado corto tras extracción ({len(text)} chars).")

    return text
