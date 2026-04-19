"""
clients/arxiv_client.py
=======================
Cliente para arXiv — implementa BaseClient.

Política de crawling declarada en esta clase
--------------------------------------------
  request_delay   = 15.0 s
      Coincide con el Crawl-delay: 15 del robots.txt de arXiv para User-agent: *.
      El delay efectivo es max(request_delay, checker.crawl_delay(url)), por lo
      que si arXiv endurece su política en el futuro el nuevo valor toma precedencia
      automáticamente sin tocar este código.

  trusted_domains = {"arxiv.org", "export.arxiv.org"}
      La API Atom y la descarga de PDFs/HTML están explícitamente autorizadas
      por los ToS de arXiv, pero robots.txt incluye 'Disallow: /api'.
      Declarar estos dominios hace que RobotsChecker.allowed() devuelva True
      sin consultar el Disallow, mientras crawl_delay() sigue leyendo el real.

Descarga de texto
-----------------
  download_text() intenta primero HTML (más ligero, generado por LaTeXML),
  y cae a PDF con PyMuPDF si el HTML no está disponible.
  Toda la lógica de extracción es interna a este módulo.

Formato de ID compuesto
-----------------------
  "arxiv:{local_id}"   →   p.ej.  "arxiv:2301.12345"
"""

from __future__ import annotations

import html as _html_std
import io
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import FrozenSet, List, Optional

from ..document import Document
from ..robots import checker as _robots, USER_AGENT, _SSL_CTX
from .base_client import BaseClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes — API arXiv
# ---------------------------------------------------------------------------
BASE_URL       = "https://export.arxiv.org/api/query"
ATOM_NS        = "http://www.w3.org/2005/Atom"
ARXIV_HTML_URL = "https://arxiv.org/html/{local_id}"
ARXIV_PDF_URL  = "https://arxiv.org/pdf/{local_id}"

AI_ML_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML",
]
DEFAULT_SEARCH_QUERY = " OR ".join(f"cat:{c}" for c in AI_ML_CATEGORIES)

_MAX_SIZE_MB      = 15
_CHUNK_BYTES      = 65_536
_LOG_EVERY_KB     = 512


# ---------------------------------------------------------------------------
# Helpers XML
# ---------------------------------------------------------------------------
def _tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"

def _get_text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


# ---------------------------------------------------------------------------
# Extractor HTML — específico para LaTeXML (todos los papers arXiv)
# ---------------------------------------------------------------------------
# arXiv convierte los .tex originales a HTML usando LaTeXML, que produce
# una estructura muy predecible con clases CSS ltx_*. Apuntar exactamente
# a esas clases da resultados mucho más limpios que un parser genérico.

class _LaTeXMLExtractor(HTMLParser):
    """
    Extractor de texto para HTML generado por LaTeXML (arXiv).

    Incluye el contenido de secciones, párrafos y headings.
    Descarta autores, figuras, tablas, ecuaciones, bibliografía y pie de página.
    """

    SKIP_CLASSES = {
        "ltx_authors",        # autores/afiliaciones (ya en metadatos)
        "ltx_bibliography",   # sección de referencias
        "ltx_figure",         # figuras y pies de foto
        "ltx_table",          # tablas
        "ltx_equation",       # bloques de ecuación
        "ltx_equationgroup",  # grupos de ecuaciones
        "ltx_pagination",     # números de página
        "ltx_page_footer",    # footer
        "ltx_note",           # notas al pie
        "ltx_minipage",       # minipages (suelen ser figuras/tablas)
    }
    SKIP_TAGS = {"script", "style", "nav", "math", "svg"}
    HEADING_CLASSES = {
        "ltx_title_document", "ltx_title_section",
        "ltx_title_subsection", "ltx_title_subsubsection",
        "ltx_title_abstract", "ltx_title_appendix",
    }

    def __init__(self) -> None:
        super().__init__()
        self._in_doc     = False
        self._skip_depth = 0
        self._in_heading = False
        self._parts: List[str] = []

    def _cls(self, attrs) -> set:
        return set(dict(attrs).get("class", "").split())

    def handle_starttag(self, tag: str, attrs):
        classes = self._cls(attrs)
        if "ltx_document" in classes:
            self._in_doc = True
            return
        if not self._in_doc:
            return
        if tag in self.SKIP_TAGS or classes & self.SKIP_CLASSES:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if classes & self.HEADING_CLASSES:
            self._parts.append("\n\n")
            self._in_heading = True

    def handle_endtag(self, tag: str):
        if not self._in_doc:
            return
        if tag in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            if tag in ("div", "section", "figure", "table", "span"):
                self._skip_depth -= 1
            return
        if self._in_heading and tag in ("h1","h2","h3","h4","h5","h6"):
            self._parts.append("\n\n")
            self._in_heading = False
        if tag == "p":
            self._parts.append("\n")
        if tag == "section":
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._in_doc and self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return _html_std.unescape("".join(self._parts))


def _extract_text_from_html(raw: bytes) -> str:
    """Extrae texto de HTML LaTeXML de arXiv, con fallback genérico."""
    decoded = raw.decode("utf-8", errors="replace")

    parser = _LaTeXMLExtractor()
    parser.feed(decoded)
    text = _clean_text(parser.get_text())

    # Fallback si ltx_document no estaba presente (estructura inesperada)
    if len(text) < 500:
        logger.warning("[ArxivClient] ltx_document no encontrado — fallback genérico.")
        stripped = re.sub(
            r"<(script|style)[^>]*>.*?</(script|style)>",
            "", decoded, flags=re.DOTALL | re.IGNORECASE,
        )
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        text = _clean_text(_html_std.unescape(stripped))

    return text


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extrae texto de un PDF usando PyMuPDF (fitz)."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF es necesario para extraer PDFs. "
            "Instálalo con: pip install pymupdf"
        ) from exc
    doc   = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return _clean_text("\n\n".join(pages))


def _clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ArxivClient
# ---------------------------------------------------------------------------
class ArxivClient(BaseClient):
    """
    Cliente que extrae artículos de arXiv implementando BaseClient.

    La política de crawling, la lógica de descarga y la extracción de texto
    están completamente encapsuladas en esta clase.

    Rate-limiting a nivel de clase
    --------------------------------
    ``_rate_lock`` y ``_last_request`` son variables de clase, no de instancia.
    Esto garantiza que TODAS las instancias de ArxivClient comparten un único
    contador de tiempo, incluso si se crean varias (ej. en tests o en un
    Crawler con múltiples configuraciones).  Sin esto, dos instancias podrían
    hacer peticiones simultáneas violando el Crawl-delay de arXiv.
    """

    # ── Rate-limiting compartido entre todas las instancias ───────────────────
    _rate_lock:    threading.Lock = threading.Lock()
    _last_request: float          = 0.0

    # ── Política de crawling ──────────────────────────────────────────────────

    @property
    def request_delay(self) -> float:
        """
        15 s — igual al Crawl-delay: 15 del robots.txt de arXiv.
        El delay efectivo en _get() es max(request_delay, crawl_delay_robots).
        """
        return 15.0

    @property
    def trusted_domains(self) -> FrozenSet[str]:
        """
        Dominios arXiv con acceso autorizado por ToS.
        robots.txt dice 'Disallow: /api', pero la API Atom está permitida.
        Pasando estos dominios a checker.allowed() se evita el falso negativo.
        crawl_delay() sigue leyendo robots.txt de verdad (sin bypass).
        """
        return frozenset({"arxiv.org", "export.arxiv.org"})

    # ── Identificador de fuente ───────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return "arxiv"

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        search_query: str = DEFAULT_SEARCH_QUERY,
        timeout: int = 30,
    ) -> None:
        self.search_query = search_query
        self.timeout      = timeout
        # _rate_lock y _last_request son de clase — no se inicializan aquí

    # ── HTTP interno ──────────────────────────────────────────────────────────

    def _get(self, url: str, timeout: Optional[int] = None,
             accept: str = "*/*") -> bytes:
        """
        GET con rate-limiting thread-safe y comprobación de robots.txt.

        El Crawler ejecuta tres hilos concurrentes (discovery, downloader,
        text) que comparten la misma instancia de ArxivClient.  Para evitar
        que dos hilos lean _last_request al mismo tiempo y lancen peticiones
        simultáneas (violando el Crawl-delay), toda la sección de rate-limit
        está protegida por _rate_lock.  El lock se libera ANTES de hacer la
        petición HTTP para no bloquear a los otros hilos durante la descarga.
        """
        if not _robots.allowed(url, self.trusted_domains):
            raise PermissionError(f"robots.txt no permite: {url}")

        robots_delay    = _robots.crawl_delay(url)
        effective_delay = max(self.request_delay, robots_delay)

        # Sección crítica: leer y actualizar _last_request de forma atómica.
        # Usamos ArxivClient._rate_lock (clase) para que TODAS las instancias
        # compartan el mismo serializer, aunque haya varias instancias activas.
        with ArxivClient._rate_lock:
            elapsed = time.monotonic() - ArxivClient._last_request
            if elapsed < effective_delay:
                wait = effective_delay - elapsed
                logger.debug("[ArxivClient] Rate-limit: esperando %.1fs …", wait)
                time.sleep(wait)
            # Reservar el "turno" dentro del lock, antes de soltar,
            # para que el siguiente hilo espere a partir de este instante.
            ArxivClient._last_request = time.monotonic()

        to  = timeout or self.timeout
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}
        )
        with urllib.request.urlopen(req, timeout=to, context=_SSL_CTX) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                size_mb = int(cl) / (1024 * 1024)
                if size_mb > _MAX_SIZE_MB:
                    raise ValueError(f"Archivo demasiado grande ({size_mb:.1f} MB).")
                logger.info("[ArxivClient] %.0f KB — descargando …", int(cl) / 1024)

            buf, downloaded, last_log = [], 0, 0
            start = time.monotonic()
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                buf.append(chunk)
                downloaded += len(chunk)
                if downloaded > _MAX_SIZE_MB * 1024 * 1024:
                    raise ValueError(f"Descarga supera {_MAX_SIZE_MB} MB.")
                if downloaded - last_log >= _LOG_EVERY_KB * 1024:
                    t = time.monotonic() - start
                    logger.info("[ArxivClient] … %.0f KB (%.0f KB/s)",
                                downloaded / 1024, downloaded / 1024 / t if t else 0)
                    last_log = downloaded

        return b"".join(buf)

    def fetch_ids(self, max_results: int = 100, start: int = 0,
                  sort_by: str = "submittedDate",
                  sort_order: str = "descending") -> List[str]:
        """Devuelve IDs locales de arXiv (sin prefijo 'arxiv:')."""
        params = urllib.parse.urlencode({
            "search_query": self.search_query,
            "start":        start,
            "max_results":  max_results,
            "sortBy":       sort_by,
            "sortOrder":    sort_order,
        })
        try:
            return self._parse_ids(self._get(f"{BASE_URL}?{params}").decode("utf-8"))
        except Exception as exc:
            logger.error("[ArxivClient] fetch_ids falló: %s", exc)
            return []

    def _parse_ids(self, xml_text: str) -> List[str]:
        root = ET.fromstring(xml_text)
        ids: List[str] = []
        for entry in root.findall(_tag(ATOM_NS, "entry")):
            el = entry.find(_tag(ATOM_NS, "id"))
            if el is not None and el.text:
                raw = el.text.strip().rstrip("/")
                ids.append(raw.split("/abs/")[-1].split("v")[0])
        return ids

    # ── BaseClient: descarga de metadatos ────────────────────────────────────

    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        """Metadatos de los IDs locales con doc_id compuesto ("arxiv:…")."""
        if not local_ids:
            return []
        docs: List[Document] = []
        for i in range(0, len(local_ids), 20):
            docs.extend(self._fetch_chunk(local_ids[i: i + 20]))
        return docs

    def _fetch_chunk(self, local_ids: List[str]) -> List[Document]:
        params = urllib.parse.urlencode({
            "id_list":     ",".join(local_ids),
            "max_results": len(local_ids),
        })
        try:
            return self._parse_entries(self._get(f"{BASE_URL}?{params}").decode("utf-8"))
        except Exception as exc:
            logger.error("[ArxivClient] fetch_documents chunk falló: %s", exc)
            return []

    def _parse_entries(self, xml_text: str) -> List[Document]:
        root = ET.fromstring(xml_text)
        docs: List[Document] = []
        for entry in root.findall(_tag(ATOM_NS, "entry")):
            try:
                doc = self._entry_to_document(entry)
                if doc:
                    docs.append(doc)
            except Exception as exc:
                logger.warning("[ArxivClient] entry no parseado: %s", exc)
        return docs

    def _entry_to_document(self, entry: ET.Element) -> Optional[Document]:
        el = entry.find(_tag(ATOM_NS, "id"))
        if el is None or not el.text:
            return None
        local_id = el.text.strip().rstrip("/").split("/abs/")[-1].split("v")[0]

        authors = ", ".join(
            _get_text(a.find(_tag(ATOM_NS, "name")))
            for a in entry.findall(_tag(ATOM_NS, "author"))
        )
        pdf_url = ""
        for link in entry.findall(_tag(ATOM_NS, "link")):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break

        return Document(
            doc_id     = self.make_doc_id(local_id),
            title      = _get_text(entry.find(_tag(ATOM_NS, "title"))).replace("\n", " "),
            authors    = authors,
            abstract   = _get_text(entry.find(_tag(ATOM_NS, "summary"))).replace("\n", " "),
            categories = ", ".join(filter(None, [
                c.get("term", "") for c in entry.findall(_tag(ATOM_NS, "category"))
            ])),
            published  = _get_text(entry.find(_tag(ATOM_NS, "published"))),
            updated    = _get_text(entry.find(_tag(ATOM_NS, "updated"))),
            pdf_url    = pdf_url,
        )

    # ── BaseClient: descarga de texto ────────────────────────────────────────

    def download_text(self, local_id: str, **kwargs) -> str:
        """
        Descarga el texto del artículo: HTML (LaTeXML) → PDF (fallback).

        Método 1 — HTML: más ligero, sin dependencias extra, mejor estructura.
        Método 2 — PDF: fallback cuando el HTML no está disponible (papers antiguos).
        """
        pdf_url  = kwargs.get("pdf_url")
        html_exc: Optional[Exception] = None

        # Método 1: HTML
        html_url = ARXIV_HTML_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Intentando HTML para %s …", local_id)
        try:
            raw = self._get(html_url, timeout=20, accept="text/html")
            if b"HTML version not available" in raw or len(raw) < 5000:
                raise ValueError("HTML no disponible para este artículo")
            text = _extract_text_from_html(raw)
            if len(text) < 500:
                raise ValueError(f"HTML demasiado corto ({len(text)} chars)")
            logger.info("[ArxivClient] HTML — %.1f KB → %d chars", len(raw)/1024, len(text))
            return text
        except Exception as exc:
            html_exc = exc
            logger.warning("[ArxivClient] HTML falló para %s (%s). Probando PDF …",
                           local_id, exc)

        # Método 2: PDF
        url = pdf_url or ARXIV_PDF_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Descargando PDF %s …", url)
        try:
            pdf_bytes = self._get(url, timeout=15, accept="application/pdf")
            text      = _extract_text_from_pdf(pdf_bytes)
            logger.info("[ArxivClient] PDF — %.1f KB → %d chars",
                        len(pdf_bytes)/1024, len(text))
            return text
        except Exception as pdf_exc:
            raise RuntimeError(
                f"No se pudo extraer texto de {local_id}. "
                f"HTML: {html_exc} | PDF: {pdf_exc}"
            ) from pdf_exc
