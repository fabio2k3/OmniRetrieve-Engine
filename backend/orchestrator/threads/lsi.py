"""
threads/lsi.py
==============
Hilo de reconstrucción periódica del modelo LSI.

Estrategia de arranque en dos pasos
------------------------------------
1. **Carga rápida** (segundos): si ya existe un .pkl guardado de una sesión
   anterior, se carga directamente. ``lsi_ready`` se activa de inmediato y
   las búsquedas LSI quedan disponibles sin esperar el rebuild completo.

2. **Rebuild completo** (puede tardar más): actualiza el modelo con todos los
   documentos indexados hasta ahora y vuelve a guardar el .pkl.
   Después se repite cada ``lsi_rebuild_interval`` segundos.

Si no hay .pkl previo, se salta el paso 1 y se va directamente al rebuild.
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
    Gestiona el ciclo de vida del modelo LSI.

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    shutdown         : evento de parada compartido.
    lsi_lock         : RLock que protege ``retriever_holder`` durante el swap.
    lsi_ready        : Event que se activa cuando el primer modelo está listo.
    retriever_holder : ``list[LSIRetriever | None]`` de un solo elemento.
    """
    log.info(
        "[lsi] Hilo iniciado (intervalo=%ds, k=%d).",
        int(cfg.lsi_rebuild_interval), cfg.lsi_k,
    )

    # ── Paso 1: carga rápida del modelo existente ────────────────────────────
    if cfg.model_path.exists():
        _fast_load(cfg, lsi_lock, lsi_ready, retriever_holder)
    else:
        log.info("[lsi] No hay modelo previo en disco — se hara rebuild completo.")

    # ── Paso 2: rebuild completo (actualiza con datos recientes) ─────────────
    do_lsi_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    # ── Ciclo periódico ───────────────────────────────────────────────────────
    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.lsi_rebuild_interval)
        if shutdown.is_set():
            break
        do_lsi_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    log.info("[lsi] Hilo detenido.")


def _fast_load(
    cfg:              OrchestratorConfig,
    lsi_lock:         threading.RLock,
    lsi_ready:        threading.Event,
    retriever_holder: list,
) -> None:
    """
    Carga el .pkl existente sin reconstruir el modelo.

    Es mucho más rápido que un rebuild (solo deserializa + consulta BD)
    y hace que ``lsi_ready`` se active en segundos en lugar de minutos.
    Si falla por cualquier razón, el rebuild completo lo compensará.
    """
    try:
        from backend.retrieval.lsi_retriever import LSIRetriever

        log.info("[lsi] Cargando modelo existente desde disco (carga rapida)...")
        retriever = LSIRetriever()
        retriever.load(model_path=cfg.model_path, db_path=cfg.db_path)

        with lsi_lock:
            retriever_holder[0] = retriever

        if not lsi_ready.is_set():
            lsi_ready.set()

        log.info(
            "[lsi] Modelo cargado rapidamente — %d docs | %d terminos. "
            "Rebuild en segundo plano...",
            len(retriever.model.doc_ids),
            len(retriever._word_index),
        )
    except Exception as exc:
        log.warning(
            "[lsi] Carga rapida fallida (%s) — el rebuild completo lo resolvera.",
            exc,
        )