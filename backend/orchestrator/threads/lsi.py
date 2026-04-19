"""
threads/lsi.py
==============
Hilo de reconstrucción periódica del modelo LSI.

Responsabilidad única: intentar un rebuild del modelo LSI al arrancar y
repetirlo cada ``lsi_rebuild_interval`` segundos.

La lógica de rebuild en sí vive en ``_operations.do_lsi_rebuild``.
Este módulo solo gestiona el timing y la comunicación con el orquestador.
"""

from __future__ import annotations

import logging
import threading

from .._operations import do_lsi_rebuild
from ..config import OrchestratorConfig

log = logging.getLogger(__name__)


def run_lsi_rebuild_thread(
    cfg:              OrchestratorConfig,
    shutdown:         threading.Event,
    lsi_lock:         threading.RLock,
    lsi_ready:        threading.Event,
    retriever_holder: list,
) -> None:
    """
    Reconstruye el modelo LSI cada ``lsi_rebuild_interval`` segundos.

    El primer intento ocurre inmediatamente al arrancar el hilo (sin esperar
    el intervalo) para que el sistema esté listo lo antes posible.

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    shutdown         : evento de parada compartido.
    lsi_lock         : RLock que protege ``retriever_holder`` durante el swap.
    lsi_ready        : Event que se activa cuando el primer modelo está listo.
    retriever_holder : ``list[LSIRetriever | None]`` de un solo elemento.
    """
    log.info(
        "[lsi] Hilo de rebuild iniciado (intervalo=%ds, k=%d).",
        int(cfg.lsi_rebuild_interval), cfg.lsi_k,
    )

    # Primer intento inmediato
    do_lsi_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.lsi_rebuild_interval)
        if shutdown.is_set():
            break
        do_lsi_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    log.info("[lsi] Hilo de rebuild detenido.")
