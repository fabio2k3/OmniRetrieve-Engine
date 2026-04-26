"""
_sync.py
========
Sincronización entre el índice FAISS y los embeddings almacenados en BD.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Responsabilidad única
---------------------
Detectar y corregir desincronizaciones entre el índice FAISS en memoria
(o en disco) y los embeddings persistidos en SQLite, y gestionar el reset
completo de embeddings cuando se cambia de modelo.

Las desincronizaciones ocurren típicamente cuando el proceso se reinicia
después de haber generado embeddings pero antes de haber guardado el índice.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .faiss import FaissIndexManager

log = logging.getLogger(__name__)


def check_and_sync(
    faiss_mgr:        FaissIndexManager,
    already_embedded: int,
    db_path:          Path,
) -> None:
    """
    Comprueba si el índice FAISS refleja todos los embeddings de la BD
    y lo reconstruye si hay desincronización.

    Se ejecuta al inicio de ``EmbeddingPipeline.run()`` para garantizar que
    el índice cargado desde disco es coherente con el estado actual de la BD.

    Parámetros
    ----------
    faiss_mgr        : gestor del índice FAISS.
    already_embedded : número de chunks con embedding en la BD.
    db_path          : ruta a la BD SQLite.
    """
    faiss_vectors = faiss_mgr.total_vectors

    log.info(
        "[sync] Embeddings en BD: %d | vectores en FAISS: %d",
        already_embedded, faiss_vectors,
    )

    if already_embedded > 0 and faiss_vectors < already_embedded:
        log.warning(
            "[sync] Desincronización detectada: faltan %d vectores en el índice FAISS.",
            already_embedded - faiss_vectors,
        )
        log.info("[sync] Reconstruyendo índice FAISS desde la BD…")
        faiss_mgr.rebuild(db_path)
        log.info("[sync] Índice sincronizado — %d vectores listos.", faiss_mgr.total_vectors)
    else:
        log.info("[sync] Índice FAISS coherente con la BD.")


def reset_embeddings(db_path: Path) -> int:
    """
    Pone a NULL todos los embeddings de la tabla ``chunks``.

    Se llama cuando ``reembed=True`` para re-vectorizar todo el corpus
    desde cero (p.ej. al cambiar de modelo).

    Parámetros
    ----------
    db_path : ruta a la BD SQLite.

    Returns
    -------
    int
        Número de embeddings eliminados.
    """
    from backend.database.chunk_repository import reset_embeddings as _reset_db

    log.warning("[sync] reembed=True — reseteando embeddings existentes…")
    n = _reset_db(db_path)
    log.warning(
        "[sync] %d embeddings eliminados de la BD. Se re-vectorizará todo el corpus.", n
    )
    return n
