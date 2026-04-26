"""
clients/arxiv/extractors/pdf.py
================================
Extractor de texto para PDFs de arXiv usando PyMuPDF.

API pública
-----------
extract(pdf_bytes: bytes) -> str
    Recibe los bytes crudos del PDF y devuelve el texto limpio.

Dependencia
-----------
``PyMuPDF`` (paquete ``pymupdf``).  Si no está instalado, ``extract``
lanza ``ImportError`` con instrucciones de instalación.
"""

from __future__ import annotations

import io
import logging

from ....chunker import clean_text

logger = logging.getLogger(__name__)


def extract(pdf_bytes: bytes) -> str:
    """
    Extrae texto de los bytes crudos de un PDF usando PyMuPDF (fitz).

    Parámetros
    ----------
    pdf_bytes : bytes crudos del PDF descargado.

    Returns
    -------
    str
        Texto normalizado concatenado de todas las páginas.

    Raises
    ------
    ImportError
        Si ``pymupdf`` no está instalado.
    RuntimeError
        Si PyMuPDF no puede abrir o procesar el fichero.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF es necesario para extraer texto de PDFs. "
            "Instálalo con: pip install pymupdf"
        ) from exc

    doc   = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()

    text = clean_text("\n\n".join(pages))
    logger.debug("[pdf] %d páginas extraídas → %d chars.", len(pages), len(text))
    return text
