"""
threads/embedding.py
====================
Hilo watcher de embedding y actualización del índice FAISS.

Responsabilidad única: sondear la BD periódicamente y disparar ``do_embed``
cuando hay suficientes chunks sin embedding. El primer chequeo ocurre
inmediatamente al arrancar para procesar chunks pendientes de sesiones anteriores.

La lógica del embedding en sí vive en ``_operations.do_embed``.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from backend.database.chunk_repository import get_chunk_stats
from ..config import OrchestratorConfig

log = logging.getLogger(__name__)


def run_embedding_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
    do_embed: Callable[[], dict],
) -> None:
    """
    Sondea la BD cada ``embed_poll_interval`` segundos y llama a ``do_embed``
    cuando hay al menos ``embed_threshold`` chunks sin embedding.

    Parámetros
    ----------
    cfg      : configuración del orquestador.
    shutdown : evento de parada compartido.
    do_embed : callable sin argumentos que ejecuta el pipeline de embedding.
               Se recibe inyectado para facilitar tests y mantener este módulo sin estado.
    """
    log.info(
        "[embedding] Watcher iniciado (umbral=%d chunks, poll=%ds).",
        cfg.embed_threshold, int(cfg.embed_poll_interval),
    )

    # Primer intento inmediato al arrancar
    _check_and_embed(cfg, do_embed)

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.embed_poll_interval)
        if shutdown.is_set():
            break
        _check_and_embed(cfg, do_embed)

    log.info("[embedding] Watcher detenido.")


def _check_and_embed(cfg: OrchestratorConfig, do_embed: Callable) -> None:
    """
    Consulta chunks pendientes y dispara ``do_embed`` si se supera el umbral.
    Captura excepciones para que el hilo no muera ante fallos puntuales.
    """
    try:
        stats   = get_chunk_stats(cfg.db_path)
        pending = stats["pending_chunks"]
    except Exception as exc:
        log.warning("[embedding] Error al consultar BD: %s", exc)
        return

    if pending >= cfg.embed_threshold:
        log.info(
            "[embedding] %d chunks sin embedding ≥ umbral %d — embediendo...",
            pending, cfg.embed_threshold,
        )
        try:
            do_embed()
        except Exception as exc:
            log.error("[embedding] Error durante embedding: %s", exc, exc_info=True)
    else:
        log.debug(
            "[embedding] %d chunks pendientes (umbral=%d) — esperando.",
            pending, cfg.embed_threshold,
        )
