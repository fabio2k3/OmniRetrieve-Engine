"""
clients/arxiv_client.py
=======================
Cliente arXiv que implementa BaseClient.

Responsabilidades
-----------------
1. fetch_ids        — descubre IDs locales en la API Atom de arXiv.
2. fetch_documents  — descarga metadatos y devuelve Documents con ID compuesto.
3. download_text    — descarga el texto completo (HTML → PDF fallback)
                      y lo devuelve limpio, sin fragmentar.

El fragmentado (chunking) lo realiza el Crawler usando chunker.make_chunks()
una vez obtenido el texto, manteniendo la lógica separada del transporte.

Formato de ID compuesto: ``arxiv:{arxiv_local_id}``
Ejemplo: ``arxiv:2301.12345``
"""

from __future__ import annotations

import io
import html as _html_module
import logging
import re
import ssl
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import List, Optional

from ..document import Document
from ..robots import checker as _robots, USER_AGENT, _SSL_CTX
from .base_client import BaseClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes — API arXiv
# ---------------------------------------------------------------------------
BASE_URL  = "https://export.arxiv.org/api/query"
ATOM_NS   = "http://www.w3.org/2005/Atom"
ARXIV_NS  = "http://arxiv.org/schemas/atom"

ARXIV_HTML_URL = "https://arxiv.org/html/{local_id}"
ARXIV_PDF_URL  = "https://arxiv.org/pdf/{local_id}"

AI_ML_CATEGORIES = [
    "cs.AI",    # Artificial Intelligence
    "cs.LG",    # Machine Learning
    "cs.CV",    # Computer Vision
    "cs.CL",    # Computation & Language (NLP)
    "cs.NE",    # Neural and Evolutionary Computing
    "stat.ML",  # Statistics – Machine Learning
]
DEFAULT_SEARCH_QUERY = " OR ".join(f"cat:{c}" for c in AI_ML_CATEGORIES)

MIN_REQUEST_DELAY = 3.5   # arXiv pide ≥ 3 s entre llamadas
MAX_SIZE_MB       = 15
CHUNK_SIZE_BYTES  = 65_536
LOG_EVERY_KB      = 512


# ---------------------------------------------------------------------------
# Helpers XML
# ---------------------------------------------------------------------------
def _tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def _get_text(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return (element.text or "").strip()


# ---------------------------------------------------------------------------
# HTML parser (mismo que en pdf_extractor original)
# ---------------------------------------------------------------------------
_SKIP_TAGS = {
    "script", "style", "nav", "header", "footer",
    "figure", "figcaption", "table", "aside",
}
_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "section", "article", "div",
}


class _ArxivHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth: int = 0
        self._skip_stack: list = []
        self._parts: list = []

    def _cls(self, attrs) -> set:
        for name, val in attrs:
            if name == "class":
                return set((val or "").split())
        return set()

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        bad_classes = {
            "ltx_bibliography", "ltx_page_footer", "ltx_page_header",
            "ltx_authors", "ltx_dates",
        }
        if tag in _SKIP_TAGS or self._cls(attrs) & bad_classes:
            self._skip_stack.append(tag)
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _extract_text_from_html(raw_html: bytes) -> str:
    text = raw_html.decode("utf-8", errors="replace")
    text = _html_module.unescape(text)
    parser = _ArxivHTMLParser()
    parser.feed(text)
    raw = parser.get_text()
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    return raw.strip()


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()

    text = "\n\n".join(pages)
    # Basic cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# HTTP helper con rate limiting y robots.txt
# ---------------------------------------------------------------------------
class ArxivClient(BaseClient):
    """
    Cliente que extrae artículos de arXiv implementando BaseClient.

    Parámetros
    ----------
    search_query : str
        Consulta de búsqueda arXiv (por defecto categorías AI/ML).
    request_delay : float
        Pausa mínima entre peticiones HTTP (arXiv pide ≥ 3 s).
    timeout : int
        Timeout en segundos para cada petición HTTP.
    """

    def __init__(
        self,
        search_query: str = DEFAULT_SEARCH_QUERY,
        request_delay: float = MIN_REQUEST_DELAY,
        timeout: int = 30,
    ) -> None:
        self.search_query  = search_query
        self.request_delay = request_delay
        self.timeout       = timeout
        self._last_request: float = 0.0

    # ── BaseClient: identificador de fuente ───────────────────────────────────

    @property
    def source_name(self) -> str:
        return "arxiv"

    # ── HTTP interno ──────────────────────────────────────────────────────────

    def _get(self, url: str, timeout: Optional[int] = None,
             accept: str = "*/*") -> bytes:
        """GET con rate-limiting, robots.txt y límite de tamaño."""
        if not _robots.allowed(url):
            raise PermissionError(f"robots.txt disallows: {url}")

        robots_delay    = _robots.crawl_delay(url)
        effective_delay = max(self.request_delay, robots_delay)
        elapsed = time.monotonic() - self._last_request
        if elapsed < effective_delay:
            time.sleep(effective_delay - elapsed)

        to = timeout or self.timeout
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}
        )
        with urllib.request.urlopen(req, timeout=to, context=_SSL_CTX) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > MAX_SIZE_MB:
                    raise ValueError(
                        f"Archivo demasiado grande ({size_mb:.1f} MB). Saltando."
                    )
                logger.info("[ArxivClient] Conectado — %.0f KB.", int(content_length) / 1024)

            buf, downloaded, last_log = [], 0, 0
            start = time.monotonic()
            while True:
                chunk = resp.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                buf.append(chunk)
                downloaded += len(chunk)
                if downloaded > MAX_SIZE_MB * 1024 * 1024:
                    raise ValueError(f"Descarga supera {MAX_SIZE_MB} MB. Abortando.")
                if downloaded - last_log >= LOG_EVERY_KB * 1024:
                    t = time.monotonic() - start
                    logger.info("[ArxivClient] … %.0f KB (%.0f KB/s)",
                                downloaded / 1024, downloaded / 1024 / t if t else 0)
                    last_log = downloaded

        self._last_request = time.monotonic()
        return b"".join(buf)

    # ── BaseClient: descubrimiento de IDs ────────────────────────────────────

    def fetch_ids(
        self,
        max_results: int = 100,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> List[str]:
        """
        Devuelve IDs locales de arXiv (sin prefijo 'arxiv:').

        El Crawler añade el prefijo mediante make_doc_id().
        """
        params = urllib.parse.urlencode({
            "search_query": self.search_query,
            "start":        start,
            "max_results":  max_results,
            "sortBy":       sort_by,
            "sortOrder":    sort_order,
        })
        url = f"{BASE_URL}?{params}"
        try:
            xml_text = self._get(url).decode("utf-8")
        except Exception as exc:
            logger.error("[ArxivClient] fetch_ids falló: %s", exc)
            return []
        return self._parse_ids(xml_text)

    def _parse_ids(self, xml_text: str) -> List[str]:
        root = ET.fromstring(xml_text)
        ids: List[str] = []
        for entry in root.findall(_tag(ATOM_NS, "entry")):
            id_elem = entry.find(_tag(ATOM_NS, "id"))
            if id_elem is not None and id_elem.text:
                raw      = id_elem.text.strip().rstrip("/")
                local_id = raw.split("/abs/")[-1].split("v")[0]
                ids.append(local_id)
        return ids

    # ── BaseClient: descarga de metadatos ────────────────────────────────────

    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        """
        Descarga metadatos de los IDs locales y devuelve Documents.

        El campo ``arxiv_id`` de cada Document contiene el ID compuesto
        (``arxiv:{local_id}``), no el ID local.
        """
        if not local_ids:
            return []
        docs: List[Document] = []
        for i in range(0, len(local_ids), 20):
            docs.extend(self._fetch_chunk(local_ids[i: i + 20]))
        return docs

    def _fetch_chunk(self, local_ids: List[str]) -> List[Document]:
        params = urllib.parse.urlencode({
            "id_list":    ",".join(local_ids),
            "max_results": len(local_ids),
        })
        url = f"{BASE_URL}?{params}"
        try:
            xml_text = self._get(url).decode("utf-8")
        except Exception as exc:
            logger.error("[ArxivClient] fetch_documents chunk falló: %s", exc)
            return []
        return self._parse_entries(xml_text)

    def _parse_entries(self, xml_text: str) -> List[Document]:
        root = ET.fromstring(xml_text)
        docs: List[Document] = []
        for entry in root.findall(_tag(ATOM_NS, "entry")):
            try:
                doc = self._entry_to_document(entry)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                logger.warning("[ArxivClient] No se pudo parsear entry: %s", exc)
        return docs

    def _entry_to_document(self, entry: ET.Element) -> Optional[Document]:
        id_elem = entry.find(_tag(ATOM_NS, "id"))
        if id_elem is None or not id_elem.text:
            return None
        raw_local = id_elem.text.strip().rstrip("/").split("/abs/")[-1]
        local_id  = raw_local.split("v")[0]

        title    = _get_text(entry.find(_tag(ATOM_NS, "title"))).replace("\n", " ")
        authors  = ", ".join(
            _get_text(a.find(_tag(ATOM_NS, "name")))
            for a in entry.findall(_tag(ATOM_NS, "author"))
        )
        abstract   = _get_text(entry.find(_tag(ATOM_NS, "summary"))).replace("\n", " ")
        categories = ", ".join(filter(None, [
            c.get("term", "")
            for c in entry.findall(_tag(ATOM_NS, "category"))
        ]))
        published = _get_text(entry.find(_tag(ATOM_NS, "published")))
        updated   = _get_text(entry.find(_tag(ATOM_NS, "updated")))

        pdf_url = ""
        for link in entry.findall(_tag(ATOM_NS, "link")):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return Document(
            doc_id     = self.make_doc_id(local_id),   # "arxiv:2301.12345"
            title      = title,
            authors    = authors,
            abstract   = abstract,
            categories = categories,
            published  = published,
            updated    = updated,
            pdf_url    = pdf_url,
        )

    # ── BaseClient: descarga de texto ────────────────────────────────────────

    def download_text(self, local_id: str, **kwargs) -> str:
        """
        Descarga el texto completo del artículo (HTML → PDF fallback).

        Parámetros
        ----------
        local_id : str
            ID local arXiv (sin prefijo), p.ej. ``2301.12345``.
        pdf_url : str, opcional (en kwargs)
            URL directa al PDF; si no se proporciona se construye la
            URL estándar de arXiv.

        Returns
        -------
        str
            Texto limpio listo para ser fragmentado por el Crawler.
        """
        pdf_url = kwargs.get("pdf_url")

        # ── Método 1: HTML ───────────────────────────────────────────────────
        html_url = ARXIV_HTML_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Intentando HTML para %s …", local_id)
        html_exc: Optional[Exception] = None
        try:
            raw = self._get(html_url, timeout=20, accept="text/html")
            if b"HTML version not available" in raw or len(raw) < 5000:
                raise ValueError("HTML no disponible para este artículo")
            full_text = _extract_text_from_html(raw)
            if len(full_text) < 500:
                raise ValueError(f"Texto HTML demasiado corto ({len(full_text)} chars)")
            logger.info(
                "[ArxivClient] ✅ HTML extraído — %.1f KB → %d chars",
                len(raw) / 1024, len(full_text),
            )
            return full_text
        except Exception as exc:
            html_exc = exc
            logger.warning(
                "[ArxivClient] HTML falló para %s (%s). Intentando PDF …",
                local_id, exc,
            )

        # ── Método 2: PDF (fallback) ─────────────────────────────────────────
        url = pdf_url or ARXIV_PDF_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Descargando PDF %s …", url)
        try:
            pdf_bytes = self._get(url, timeout=15, accept="application/pdf")
            full_text = _extract_text_from_pdf(pdf_bytes)
            logger.info(
                "[ArxivClient] ✅ PDF extraído — %.1f KB → %d chars",
                len(pdf_bytes) / 1024, len(full_text),
            )
            return full_text
        except Exception as pdf_exc:
            raise RuntimeError(
                f"No se pudo extraer texto de {local_id}. "
                f"HTML: {html_exc} | PDF: {pdf_exc}"
            ) from pdf_exc
