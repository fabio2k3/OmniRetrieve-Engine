"""
threads/indexing.py
===================
Hilo watcher de indexación BM25.

Responsabilidad única: sondear la BD periódicamente y disparar ``do_index``
cuando hay suficientes PDFs nuevos sin indexar.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from backend.database.schema import get_connection
from ..config import OrchestratorConfig

log = logging.getLogger(__name__)


def run_indexing_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
    do_index: Callable[[], dict],
) -> None:
    """
    Sondea la BD cada ``index_poll_interval`` segundos y llama a ``do_index``
    cuando hay al menos ``pdf_threshold`` PDFs nuevos sin indexar.

    Parámetros
    ----------
    cfg      : configuración del orquestador.
    shutdown : evento de parada compartido.
    do_index : callable sin argumentos que ejecuta la indexación incremental.
               Se recibe inyectado para facilitar tests y mantener este módulo sin estado.
    """
    log.info(
        "[indexing] Watcher iniciado (umbral=%d PDFs, poll=%ds).",
        cfg.pdf_threshold, int(cfg.index_poll_interval),
    )

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.index_poll_interval)
        if shutdown.is_set():
            break
        _check_and_index(cfg, do_index)

    log.info("[indexing] Watcher detenido.")


def _check_and_index(cfg: OrchestratorConfig, do_index: Callable) -> None:
    """
    Consulta la BD y dispara ``do_index`` si se supera el umbral.
    Captura excepciones para que el hilo no muera ante fallos puntuales.
    """
    try:
        conn = get_connection(cfg.db_path)
        unindexed = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        log.warning("[indexing] Error al consultar BD: %s", exc)
        return

    if unindexed >= cfg.pdf_threshold:
        log.info(
            "[indexing] %d PDFs sin indexar ≥ umbral %d — indexando...",
            unindexed, cfg.pdf_threshold,
        )
        try:
            do_index()
        except Exception as exc:
            log.error("[indexing] Error durante indexación: %s", exc, exc_info=True)
    else:
        log.debug(
            "[indexing] %d PDFs pendientes (umbral=%d) — esperando.",
            unindexed, cfg.pdf_threshold,
        )
