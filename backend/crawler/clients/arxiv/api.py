"""
clients/arxiv/api.py
====================
Parseo de respuestas XML Atom de la API de arXiv.

Responsabilidad única
---------------------
Transformar el XML devuelto por ``export.arxiv.org/api/query``
en listas de IDs locales o de objetos ``Document``.

Este módulo no hace ninguna petición HTTP. Recibe texto XML ya
descargado y devuelve estructuras Python.

API pública
-----------
parse_ids(xml_text: str) -> List[str]
    Extrae los IDs locales (sin prefijo ``arxiv:``) de un feed Atom.

parse_entries(xml_text: str, make_doc_id: Callable) -> List[Document]
    Convierte las entradas de un feed Atom en objetos ``Document``.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Callable, List, Optional

from ...document import Document
from .constants  import ATOM_NS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers XML
# ---------------------------------------------------------------------------

def _tag(name: str) -> str:
    """Devuelve el nombre de tag con namespace Atom completo."""
    return f"{{{ATOM_NS}}}{name}"


def _text(el: Optional[ET.Element]) -> str:
    """Devuelve el texto de un elemento, o cadena vacía si es None."""
    return (el.text or "").strip() if el is not None else ""


def _local_id_from_url(url: str) -> str:
    """
    Extrae el ID local arXiv de una URL del tipo:
    ``https://arxiv.org/abs/2301.12345v2`` → ``2301.12345``.
    """
    return url.strip().rstrip("/").split("/abs/")[-1].split("v")[0]


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def parse_ids(xml_text: str) -> List[str]:
    """
    Extrae IDs locales de un feed Atom devuelto por la API de arXiv.

    Parámetros
    ----------
    xml_text : texto XML completo de la respuesta.

    Returns
    -------
    List[str]
        IDs locales arXiv (sin prefijo ``arxiv:``), p.ej. ``["2301.12345"]``.
        Lista vacía si el XML no contiene entradas o es inválido.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("[api] Error parseando XML de IDs: %s", exc)
        return []

    ids: List[str] = []
    for entry in root.findall(_tag("entry")):
        el = entry.find(_tag("id"))
        if el is not None and el.text:
            ids.append(_local_id_from_url(el.text))
    return ids


def parse_entries(
    xml_text: str,
    make_doc_id: Callable[[str], str],
) -> List[Document]:
    """
    Convierte las entradas de un feed Atom en objetos ``Document``.

    Parámetros
    ----------
    xml_text   : texto XML completo de la respuesta.
    make_doc_id : callable que transforma un ID local en ID compuesto
                  (habitualmente ``ArxivClient.make_doc_id``).

    Returns
    -------
    List[Document]
        Documentos con metadatos completos y ``doc_id`` compuesto.
        Las entradas que no puedan parsearse se omiten con un warning.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("[api] Error parseando XML de entradas: %s", exc)
        return []

    docs: List[Document] = []
    for entry in root.findall(_tag("entry")):
        try:
            doc = _entry_to_document(entry, make_doc_id)
            if doc:
                docs.append(doc)
        except Exception as exc:
            logger.warning("[api] Entrada no parseada: %s", exc)
    return docs


def _entry_to_document(
    entry: ET.Element,
    make_doc_id: Callable[[str], str],
) -> Optional[Document]:
    """
    Convierte un elemento ``<entry>`` de Atom en un ``Document``.

    Returns
    -------
    Document | None
        ``None`` si la entrada no tiene ID válido.
    """
    id_el = entry.find(_tag("id"))
    if id_el is None or not id_el.text:
        return None

    local_id = _local_id_from_url(id_el.text)

    authors = ", ".join(
        _text(a.find(_tag("name")))
        for a in entry.findall(_tag("author"))
    )

    pdf_url = ""
    for link in entry.findall(_tag("link")):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break

    categories = ", ".join(filter(None, [
        c.get("term", "") for c in entry.findall(_tag("category"))
    ]))

    return Document(
        doc_id     = make_doc_id(local_id),
        title      = _text(entry.find(_tag("title"))).replace("\n", " "),
        authors    = authors,
        abstract   = _text(entry.find(_tag("summary"))).replace("\n", " "),
        categories = categories,
        published  = _text(entry.find(_tag("published"))),
        updated    = _text(entry.find(_tag("updated"))),
        pdf_url    = pdf_url,
    )
