"""
clients/arxiv/client.py
========================
Cliente arXiv — implementa ``BaseClient``.

Responsabilidad única
---------------------
Gestionar la capa HTTP de arXiv: rate-limiting thread-safe, respeto a
robots.txt y descarga de texto vía HTML (LaTeXML) con fallback a PDF.

El parseo XML lo delega a ``api.py``.
La extracción de texto lo delega a ``extractors/html.py`` y ``extractors/pdf.py``.

Política de crawling declarada aquí
-------------------------------------
    request_delay   = 15.0 s
        Coincide con el ``Crawl-delay: 15`` del robots.txt de arXiv.

    trusted_domains = {"arxiv.org", "export.arxiv.org"}
        La API Atom y la descarga de PDFs/HTML están autorizadas por ToS,
        aunque robots.txt incluya ``Disallow: /api``.

Rate-limiting a nivel de clase
--------------------------------
``_rate_lock`` y ``_last_request`` son variables de clase, no de instancia,
para garantizar que todas las instancias de ``ArxivClient`` comparten un único
contador de tiempo incluso en entornos multi-instancia.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import FrozenSet, List, Optional

from ...document import Document
from ...robots   import checker as _robots
from ...http     import USER_AGENT, _SSL_CTX
from ..base_client import BaseClient
from .constants    import (
    BASE_URL, ARXIV_HTML_URL, ARXIV_PDF_URL,
    DEFAULT_SEARCH_QUERY, MAX_SIZE_MB, CHUNK_BYTES, LOG_EVERY_KB,
)
from . import api
from .extractors import extract_html, extract_pdf

logger = logging.getLogger(__name__)


class ArxivClient(BaseClient):
    """
    Cliente que extrae artículos de arXiv implementando ``BaseClient``.

    Parámetros del constructor
    --------------------------
    search_query : consulta Atom para ``fetch_ids`` (por defecto categorías AI/ML).
    timeout      : timeout HTTP global en segundos (por defecto 30).
    """

    # ── Rate-limiting compartido entre todas las instancias ───────────────────
    _rate_lock:    threading.Lock = threading.Lock()
    _last_request: float          = 0.0

    # ── Política de crawling ──────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        return "arxiv"

    @property
    def request_delay(self) -> float:
        """15 s — igual al ``Crawl-delay: 15`` del robots.txt de arXiv."""
        return 15.0

    @property
    def trusted_domains(self) -> FrozenSet[str]:
        """
        Dominios arXiv con acceso autorizado por ToS.
        robots.txt dice ``Disallow: /api``, pero la API Atom está permitida.
        """
        return frozenset({"arxiv.org", "export.arxiv.org"})

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        search_query: str = DEFAULT_SEARCH_QUERY,
        timeout: int = 30,
    ) -> None:
        self.search_query = search_query
        self.timeout      = timeout

    # ── HTTP interno con rate-limiting ────────────────────────────────────────

    def _get(
        self,
        url: str,
        timeout: Optional[int] = None,
        accept: str = "*/*",
    ) -> bytes:
        """
        GET con rate-limiting thread-safe y comprobación de robots.txt.

        El lock se libera antes de la petición HTTP para no bloquear a
        otros hilos durante la descarga. El ``_last_request`` se actualiza
        dentro del lock para serializar el acceso entre hilos.

        Parámetros
        ----------
        url     : URL a descargar.
        timeout : timeout en segundos (usa ``self.timeout`` si se omite).
        accept  : valor del header ``Accept``.

        Returns
        -------
        bytes
            Contenido completo de la respuesta.

        Raises
        ------
        PermissionError : si robots.txt prohíbe la URL.
        ValueError      : si la descarga supera ``MAX_SIZE_MB``.
        urllib.error.*  : ante errores de red o HTTP.
        """
        if not _robots.allowed(url, self.trusted_domains):
            raise PermissionError(f"robots.txt no permite: {url}")

        robots_delay    = _robots.crawl_delay(url)
        effective_delay = max(self.request_delay, robots_delay)

        with ArxivClient._rate_lock:
            elapsed = time.monotonic() - ArxivClient._last_request
            if elapsed < effective_delay:
                wait = effective_delay - elapsed
                logger.debug("[ArxivClient] Rate-limit: esperando %.1fs ...", wait)
                time.sleep(wait)
            ArxivClient._last_request = time.monotonic()

        to  = timeout or self.timeout
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": accept}
        )
        with urllib.request.urlopen(req, timeout=to, context=_SSL_CTX) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                size_mb = int(cl) / (1024 * 1024)
                if size_mb > MAX_SIZE_MB:
                    raise ValueError(f"Archivo demasiado grande ({size_mb:.1f} MB).")
                logger.info("[ArxivClient] %.0f KB — descargando ...", int(cl) / 1024)

            buf, downloaded, last_log = [], 0, 0
            start = time.monotonic()
            while True:
                chunk = resp.read(CHUNK_BYTES)
                if not chunk:
                    break
                buf.append(chunk)
                downloaded += len(chunk)
                if downloaded > MAX_SIZE_MB * 1024 * 1024:
                    raise ValueError(f"Descarga supera {MAX_SIZE_MB} MB.")
                if downloaded - last_log >= LOG_EVERY_KB * 1024:
                    t = time.monotonic() - start
                    logger.info(
                        "[ArxivClient] ... %.0f KB (%.0f KB/s)",
                        downloaded / 1024, downloaded / 1024 / t if t else 0,
                    )
                    last_log = downloaded

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
        Devuelve IDs locales de arXiv (sin prefijo ``arxiv:``).

        Parámetros
        ----------
        max_results : máximo de resultados a solicitar.
        start       : offset de paginación.
        sort_by     : campo de ordenación (``submittedDate``, ``relevance``, etc.).
        sort_order  : ``"descending"`` o ``"ascending"``.

        Returns
        -------
        List[str]
            IDs locales, p.ej. ``["2301.12345", "2302.00001"]``.
            Lista vacía si la petición falla.
        """
        params = urllib.parse.urlencode({
            "search_query": self.search_query,
            "start":        start,
            "max_results":  max_results,
            "sortBy":       sort_by,
            "sortOrder":    sort_order,
        })
        try:
            xml_text = self._get(f"{BASE_URL}?{params}").decode("utf-8")
            return api.parse_ids(xml_text)
        except Exception as exc:
            logger.error("[ArxivClient] fetch_ids falló: %s", exc)
            return []

    # ── BaseClient: descarga de metadatos ────────────────────────────────────

    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        """
        Descarga metadatos de los IDs locales indicados.

        Procesa los IDs en lotes de 20 para no superar los límites de la API.

        Parámetros
        ----------
        local_ids : IDs locales arXiv (sin prefijo).

        Returns
        -------
        List[Document]
            Documentos con ``doc_id`` compuesto (``"arxiv:…"``).
        """
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
            xml_text = self._get(f"{BASE_URL}?{params}").decode("utf-8")
            return api.parse_entries(xml_text, self.make_doc_id)
        except Exception as exc:
            logger.error("[ArxivClient] fetch_documents chunk falló: %s", exc)
            return []

    # ── BaseClient: descarga de texto ────────────────────────────────────────

    def download_text(self, local_id: str, **kwargs) -> str:
        """
        Descarga el texto completo del artículo: HTML (LaTeXML) → PDF (fallback).

        Intenta primero HTML (más ligero, mejor estructura). Si no está
        disponible para el artículo, cae a PDF usando PyMuPDF.

        Parámetros
        ----------
        local_id : ID local arXiv (sin prefijo ``arxiv:``).
        pdf_url  : URL alternativa del PDF (opcional, vía ``**kwargs``).

        Returns
        -------
        str
            Texto limpio del artículo listo para chunking.

        Raises
        ------
        RuntimeError
            Si tanto HTML como PDF fallan.
        """
        pdf_url  = kwargs.get("pdf_url")
        html_exc: Optional[Exception] = None

        # Intento 1: HTML (LaTeXML)
        html_url = ARXIV_HTML_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Intentando HTML para %s ...", local_id)
        try:
            raw = self._get(html_url, timeout=20, accept="text/html")
            if b"HTML version not available" in raw or len(raw) < 5000:
                raise ValueError("HTML no disponible para este artículo.")
            text = extract_html(raw)
            logger.info("[ArxivClient] HTML OK — %.1f KB.", len(raw) / 1024)
            return text
        except Exception as exc:
            html_exc = exc
            logger.warning("[ArxivClient] HTML falló (%s). Probando PDF ...", exc)

        # Intento 2: PDF (fallback)
        url = pdf_url or ARXIV_PDF_URL.format(local_id=local_id)
        logger.info("[ArxivClient] Descargando PDF %s ...", url)
        try:
            pdf_bytes = self._get(url, timeout=15, accept="application/pdf")
            text      = extract_pdf(pdf_bytes)
            logger.info("[ArxivClient] PDF OK — %.1f KB.", len(pdf_bytes) / 1024)
            return text
        except Exception as pdf_exc:
            raise RuntimeError(
                f"No se pudo extraer texto de {local_id}. "
                f"HTML: {html_exc} | PDF: {pdf_exc}"
            ) from pdf_exc
