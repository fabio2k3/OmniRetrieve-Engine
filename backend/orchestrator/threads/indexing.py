"""
threads/indexing.py
===================
Hilo watcher de indexación BM25 incremental.

Responsabilidad única: sondear la BD cada ``index_poll_interval`` segundos
y disparar ``do_index()`` cuando hay suficientes PDFs sin indexar
(según ``cfg.pdf_threshold``).

La lógica de indexación en sí vive en ``_operations.do_index``.
Este módulo solo gestiona el timing y la condición de disparo.
"""

from __future__ import annotations

import logging
import threading

from backend.database.schema import get_connection

from ..config import OrchestratorConfig
from .._operations import do_index

log = logging.getLogger(__name__)


def run_indexing_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
) -> None:
    """
    Watcher de indexación BM25 incremental.

    Sondea la BD periódicamente. Cuando hay al menos ``cfg.pdf_threshold``
    documentos con PDF descargado pero sin indexar, lanza ``do_index()``.

    Parámetros
    ----------
    cfg      : configuración del orquestador.
    shutdown : evento de parada compartido.
    """
    log.info(
        "[indexing] Hilo iniciado (poll=%.0fs, threshold=%d docs).",
        cfg.index_poll_interval, cfg.pdf_threshold,
    )

    while not shutdown.is_set():
        try:
            pending = _count_pending(cfg)
            if pending >= cfg.pdf_threshold:
                log.info(
                    "[indexing] %d docs pendientes — lanzando indexación.", pending
                )
                do_index(cfg)
            else:
                log.debug(
                    "[indexing] %d/%d docs pendientes — sin acción.",
                    pending, cfg.pdf_threshold,
                )
        except Exception as exc:
            log.error("[indexing] Error inesperado: %s", exc, exc_info=True)

        shutdown.wait(timeout=cfg.index_poll_interval)

    log.info("[indexing] Hilo detenido.")


def _count_pending(cfg: OrchestratorConfig) -> int:
    """Cuenta docs con PDF descargado pero aún sin indexar en BM25."""
    try:
        conn = get_connection(cfg.db_path)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        log.warning("[indexing] No se pudo consultar pendientes: %s", exc)
        return 0
