"""
_routing.py
===========
Funciones de enrutamiento de IDs compuestos a clientes.

Módulo interno del paquete (prefijo ``_``). No forma parte de la API pública.

Estas dos funciones se usaban antes como métodos privados de ``Crawler``
(``_client_for`` y ``_local_id``). Al extraerlas a un módulo propio se
evita duplicarlas en cada uno de los tres loops.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .clients.base_client import BaseClient

logger = logging.getLogger(__name__)


def client_for(
    doc_id: str,
    client_map: Dict[str, BaseClient],
) -> Optional[BaseClient]:
    """
    Devuelve el cliente correspondiente al ID compuesto, o ``None``.

    Parámetros
    ----------
    doc_id     : ID compuesto con formato ``"source:local_id"``.
    client_map : diccionario ``source_name → BaseClient``.

    Returns
    -------
    BaseClient | None
        El cliente registrado para la fuente del ID, o ``None`` si el
        ID tiene formato inválido o la fuente no está registrada.
    """
    try:
        source, _ = BaseClient.parse_doc_id(doc_id)
        return client_map.get(source)
    except ValueError:
        logger.warning("[routing] ID con formato inválido: %r", doc_id)
        return None


def local_id(doc_id: str) -> str:
    """
    Extrae la parte local del ID compuesto.

    Parámetros
    ----------
    doc_id : ID compuesto con formato ``"source:local_id"``.

    Returns
    -------
    str
        La parte local (sin el prefijo de fuente ni los dos puntos).

    Raises
    ------
    ValueError
        Si el formato del ID es inválido.
    """
    _, lid = BaseClient.parse_doc_id(doc_id)
    return lid
