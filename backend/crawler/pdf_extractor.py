"""
pdf_extractor.py
Extrae el texto de un artículo arXiv usando dos métodos por orden de preferencia:

  1. HTML  — arxiv.org/html/{id}   (~100-500 KB, sin dependencias extra)
  2. PDF   — arxiv.org/pdf/{id}    (~2-20 MB, requiere pymupdf)

El HTML es mucho más ligero y funciona bien para conexiones lentas.
El PDF se usa como fallback si el HTML no está disponible.
"""

from __future__ import annotations

import io
import html
import logging
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from .robots import checker as _robots, USER_AGENT, _SSL_CTX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
ARXIV_PDF_URL  = "https://arxiv.org/pdf/{arxiv_id}"

MIN_REQUEST_DELAY = 3.0   # arXiv pide >= 3s entre requests
MIN_CHUNK_CHARS   = 100

_last_request: float = 0.0


# ---------------------------------------------------------------------------
# HTTP GET sin socket.settimeout (compatible con SSL en Windows)
# ---------------------------------------------------------------------------
def _get(url: str, timeout: int = 15, accept: str = "*/*") -> bytes:
    global _last_request

    if not _robots.allowed(url):
        raise PermissionError(f"robots.txt disallows: {url}")

    elapsed = time.monotonic() - _last_request
    delay   = max(MIN_REQUEST_DELAY, _robots.crawl_delay(url))
    if elapsed < delay:
        time.sleep(delay - elapsed)

    MAX_SIZE_MB  = 15          # rechazar archivos mayores de 15 MB
    CHUNK_SIZE   = 65_536      # leer de 64 KB en 64 KB
    LOG_EVERY_KB = 512         # loguear progreso cada 512 KB

    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        content_length = resp.headers.get("Content-Length")
        if content_length:
            size_kb = int(content_length) / 1024
            size_mb = size_kb / 1024
            if size_mb > MAX_SIZE_MB:
                raise ValueError(
                    f"Archivo demasiado grande ({size_mb:.1f} MB > {MAX_SIZE_MB} MB). "
                    f"Saltando para no bloquear el crawler."
                )
            logger.info("[Extractor] Conectado — %.0f KB. Descargando …", size_kb)
        else:
            logger.info("[Extractor] Conectado — tamaño desconocido. Descargando …")

        # Descarga en chunks con progreso y límite de tamaño
        chunks_buf = []
        downloaded = 0
        last_log   = 0
        start      = time.monotonic()

        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            chunks_buf.append(chunk)
            downloaded += len(chunk)

            # Abortar si supera el límite (cuando Content-Length no estaba)
            if downloaded > MAX_SIZE_MB * 1024 * 1024:
                raise ValueError(
                    f"Archivo supera {MAX_SIZE_MB} MB durante la descarga. Abortando."
                )

            # Log de progreso cada LOG_EVERY_KB KB
            if downloaded - last_log >= LOG_EVERY_KB * 1024:
                elapsed = time.monotonic() - start
                speed   = (downloaded / 1024) / elapsed if elapsed > 0 else 0
                logger.info(
                    "[Extractor] Descargando … %.0f KB (%.0f KB/s)",
                    downloaded / 1024, speed,
                )
                last_log = downloaded

    elapsed = time.monotonic() - start
    logger.info(
        "[Extractor] Descarga completa — %.0f KB en %.1fs",
        downloaded / 1024, elapsed,
    )
    _last_request = time.monotonic()
    return b"".join(chunks_buf)


# ---------------------------------------------------------------------------
# Extractor de texto desde HTML (arXiv / LaTeXML)
# ---------------------------------------------------------------------------
# El HTML de arXiv lo genera LaTeXML desde el .tex original.
# La estructura es idéntica para TODOS los papers de arXiv:
#
#   <div class="ltx_document">
#     <h1  class="ltx_title">                  ← título del paper
#     <div class="ltx_authors">                ← autores (SKIP — ya en metadata)
#     <div class="ltx_abstract">               ← abstract
#       <p class="ltx_p">…</p>
#     <section class="ltx_section">            ← Introduction, Method…
#       <h2 class="ltx_title ltx_title_section">1 Introduction</h2>
#       <div class="ltx_para">
#         <p class="ltx_p">…</p>
#       </div>
#     <section class="ltx_bibliography">       ← referencias (SKIP)
#     <section class="ltx_appendix">           ← apéndices (incluir)
#
# SKIP: ltx_authors, ltx_bibliography, ltx_figure, ltx_table,
#       ltx_equation, ltx_pagination, ltx_note, ltx_minipage

class _LaTeXMLExtractor(HTMLParser):
    """
    Extractor específico para HTML generado por LaTeXML (todos los papers arXiv).
    Apunta exactamente a las clases ltx_* conocidas y salta el ruido.
    """

    # Clases ltx_* cuyo contenido completo se descarta
    SKIP_CLASSES = {
        "ltx_authors",        # autores y afiliaciones (ya en metadata)
        "ltx_bibliography",   # sección de referencias
        "ltx_figure",         # figuras y pies de foto
        "ltx_table",          # tablas
        "ltx_equation",       # bloques de ecuación
        "ltx_equationgroup",  # grupos de ecuaciones
        "ltx_pagination",     # números de página
        "ltx_page_footer",    # footer de la página
        "ltx_note",           # notas al pie
        "ltx_minipage",       # minipages (suelen contener figuras/tablas)
        "ltx_acknowledgements_false",  # agradecimientos (opcional)
    }

    # Tags HTML genéricos siempre ignorados
    SKIP_TAGS = {"script", "style", "nav", "math", "svg"}

    # Clases de headings → añadir doble salto de línea antes y después
    HEADING_CLASSES = {
        "ltx_title_document",       # título del paper
        "ltx_title_section",        # "1 Introduction"
        "ltx_title_subsection",     # "1.1 Background"
        "ltx_title_subsubsection",
        "ltx_title_abstract",       # "Abstract"
        "ltx_title_appendix",
    }

    def __init__(self) -> None:
        super().__init__()
        self._in_doc     = False  # dentro de ltx_document
        self._skip_depth = 0
        self._in_heading = False
        self._parts: List[str] = []

    def _cls(self, attrs) -> set:
        return set(dict(attrs).get("class", "").split())

    def handle_starttag(self, tag: str, attrs):
        classes = self._cls(attrs)

        # Punto de entrada: ltx_document
        if "ltx_document" in classes:
            self._in_doc = True
            return

        if not self._in_doc:
            return

        # Tags genéricos siempre ignorados
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        # Bloques ltx_* a ignorar
        if classes & self.SKIP_CLASSES:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        # Detectar headings de sección
        if classes & self.HEADING_CLASSES:
            self._parts.append("\n\n")
            self._in_heading = True

    def handle_endtag(self, tag: str):
        if not self._in_doc:
            return

        # Cerrar tags genéricos ignorados
        if tag in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if self._skip_depth > 0:
            if tag in ("div", "section", "figure", "table", "span"):
                self._skip_depth -= 1
            return

        # Cerrar heading
        if self._in_heading and tag in ("h1","h2","h3","h4","h5","h6"):
            self._parts.append("\n\n")
            self._in_heading = False
            return

        # Salto de línea tras párrafo
        if tag == "p":
            self._parts.append("\n")

        # Salto doble al cerrar una sección
        if tag == "section":
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._in_doc and self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return html.unescape("".join(self._parts))


def _extract_text_from_html(raw_html: bytes) -> str:
    decoded = raw_html.decode("utf-8", errors="replace")

    # Extractor principal: LaTeXML (arXiv)
    parser = _LaTeXMLExtractor()
    parser.feed(decoded)
    text = _clean_text(parser.get_text())

    # Fallback si ltx_document no se encontró (estructura inesperada)
    if len(text) < 500:
        logger.warning("[Extractor] ltx_document no encontrado — fallback genérico.")
        simple = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>",
                        "", decoded, flags=re.DOTALL | re.IGNORECASE)
        simple = re.sub(r"<[^>]+>", " ", simple)
        text   = _clean_text(html.unescape(simple))

    return text


# ---------------------------------------------------------------------------
# Extractor de texto desde PDF (requiere pymupdf)
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF es necesario para extraer PDFs. "
            "Instálalo con:  pip install pymupdf"
        ) from exc

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return _clean_text("\n\n".join(pages))


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _split_into_chunks(text: str, max_chars: int = 1000) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= max_chars:
                        buf = f"{buf} {sent}".strip()
                    else:
                        if len(buf) >= MIN_CHUNK_CHARS:
                            chunks.append(buf)
                        buf = sent
                if len(buf) >= MIN_CHUNK_CHARS:
                    chunks.append(buf)
                current = ""
            else:
                current = para

    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def download_and_extract(
    arxiv_id: str,
    pdf_url: Optional[str] = None,
    chunk_size: int = 1000,
) -> Tuple[str, List[str]]:
    """
    Descarga el contenido de un artículo arXiv y extrae su texto.

    Intenta primero HTML (ligero), luego PDF (pesado) como fallback.

    Returns
    -------
    full_text : texto completo extraído
    chunks    : fragmentos listos para embeddings
    """
    # ── Método 1: HTML ───────────────────────────────────────────────────────
    html_url = ARXIV_HTML_URL.format(arxiv_id=arxiv_id)
    logger.info("[Extractor] Intentando HTML para %s …", arxiv_id)
    try:
        raw = _get(html_url, timeout=20, accept="text/html")

        # arXiv devuelve 200 con página de error si el HTML no existe
        if b"HTML version not available" in raw or len(raw) < 5000:
            raise ValueError("HTML no disponible para este artículo")

        full_text = _extract_text_from_html(raw)
        if len(full_text) < 500:
            raise ValueError(f"Texto HTML demasiado corto ({len(full_text)} chars)")

        chunks = _split_into_chunks(full_text, max_chars=chunk_size)
        logger.info(
            "[Extractor] ✅ HTML extraído — %.1f KB descargados → "
            "%d chars texto, %d chunks",
            len(raw) / 1024, len(full_text), len(chunks),
        )
        return full_text, chunks

    except Exception as html_exc:
        logger.warning(
            "[Extractor] HTML falló para %s (%s). Intentando PDF …",
            arxiv_id, html_exc,
        )

    # ── Método 2: PDF (fallback) ─────────────────────────────────────────────
    url = pdf_url or ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    logger.info("[Extractor] Descargando PDF %s …", url)
    try:
        pdf_bytes = _get(url, timeout=15, accept="application/pdf")
        full_text = _extract_text_from_pdf(pdf_bytes)
        chunks    = _split_into_chunks(full_text, max_chars=chunk_size)
        logger.info(
            "[Extractor] ✅ PDF extraído — %.1f KB descargados → "
            "%d chars texto, %d chunks",
            len(pdf_bytes) / 1024, len(full_text), len(chunks),
        )
        return full_text, chunks

    except Exception as pdf_exc:
        raise RuntimeError(
            f"No se pudo extraer {arxiv_id}. "
            f"HTML: {html_exc} | PDF: {pdf_exc}"
        )