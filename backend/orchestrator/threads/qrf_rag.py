"""
threads/qrf_rag.py
==================
Hilo de carga del pipeline unificado en dos fases solapadas.

Fase 1 — en cuanto FAISS está listo (no necesita LSI):
    CrossEncoderReranker  — carga el modelo cross-encoder (ms-marco).
    RAGPipeline           — construye EmbeddingRetriever sobre FAISS + Generator.
    → activa rag_ready

Fase 2 — en cuanto LSI también está listo:
    QueryPipeline (QRF)   — expansión LCE + BRF + MMR.
    HybridRetriever       — LSI sparse + FAISS dense, fusión RRF.
    → activa qrf_ready y pipeline_ready

Dentro de cada fase los componentes se cargan en paralelo mediante
ThreadPoolExecutor, reduciendo el tiempo de espera total.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TYPE_CHECKING

from .._operations import (
    build_qrf_pipeline,
    build_rag_pipeline,
    build_hybrid_retriever,
    build_cross_encoder,
)
from ..config import OrchestratorConfig

if TYPE_CHECKING:
    from backend.qrf.pipeline import QueryPipeline
    from backend.rag.pipeline import RAGPipeline
    from backend.retrieval.hybrid_retriever import HybridRetriever
    from backend.retrieval.reranker import CrossEncoderReranker
    from backend.embedding import FaissIndexManager

log = logging.getLogger(__name__)


def run_qrf_rag_loader_thread(
    cfg:                  OrchestratorConfig,
    shutdown:             threading.Event,
    faiss_ready:          threading.Event,
    lsi_ready:            threading.Event,
    faiss_mgr:            "FaissIndexManager",
    faiss_lock:           threading.RLock,
    retriever_holder:     list,
    lsi_lock:             threading.RLock,
    qrf_holder:           "list[Optional[QueryPipeline]]",
    qrf_lock:             threading.RLock,
    qrf_ready:            threading.Event,
    hybrid_holder:        "list[Optional[HybridRetriever]]",
    hybrid_lock:          threading.RLock,
    cross_encoder_holder: "list[Optional[CrossEncoderReranker]]",
    cross_encoder_lock:   threading.RLock,
    rag_holder:           "list[Optional[RAGPipeline]]",
    rag_lock:             threading.RLock,
    rag_ready:            threading.Event,
    pipeline_ready:       threading.Event,
) -> None:
    log.info("[qrf_rag] Hilo iniciado — esperando índice FAISS...")

    # ── Esperar FAISS ─────────────────────────────────────────────────────────
    while not shutdown.is_set():
        if faiss_ready.wait(timeout=5.0):
            break
    if shutdown.is_set():
        return

    log.info("[qrf_rag] FAISS listo — cargando Fase 1 (CrossEncoder + RAG) en paralelo...")

    # ── Fase 1: CrossEncoder + RAG en paralelo ────────────────────────────────
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qrf_rag_f1") as ex:
        fut_cross = ex.submit(_load_cross_encoder, cfg, cross_encoder_holder, cross_encoder_lock)
        fut_rag   = ex.submit(_load_rag, cfg, faiss_mgr, faiss_lock, rag_holder, rag_lock)

        for fut in as_completed((fut_cross, fut_rag)):
            try:
                fut.result()
            except Exception as exc:
                log.error("[qrf_rag] Error en Fase 1: %s", exc, exc_info=True)

    if rag_holder[0] is not None:
        rag_ready.set()
        log.info("[qrf_rag] Fase 1 completa — RAG standalone disponible.")

    # ── Esperar LSI (necesario para QRF y HybridRetriever) ───────────────────
    log.info("[qrf_rag] Esperando modelo LSI para Fase 2...")
    while not shutdown.is_set():
        if lsi_ready.wait(timeout=5.0):
            break
        log.debug("[qrf_rag] LSI aún no disponible...")
    if shutdown.is_set():
        return

    log.info("[qrf_rag] LSI listo — cargando Fase 2 (QRF + Hybrid) en paralelo...")

    # ── Fase 2: QRF + HybridRetriever en paralelo ────────────────────────────
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qrf_rag_f2") as ex:
        fut_qrf    = ex.submit(_load_qrf, cfg, qrf_holder, qrf_lock)
        fut_hybrid = ex.submit(
            _load_hybrid, cfg, retriever_holder, lsi_lock, faiss_mgr,
            hybrid_holder, hybrid_lock,
        )

        for fut in as_completed((fut_qrf, fut_hybrid)):
            try:
                fut.result()
            except Exception as exc:
                log.error("[qrf_rag] Error en Fase 2: %s", exc, exc_info=True)

    if qrf_holder[0] is not None:
        qrf_ready.set()

    # pipeline_ready solo si los 4 componentes están disponibles
    with qrf_lock, hybrid_lock, cross_encoder_lock, rag_lock:
        all_ok = all(
            h[0] is not None
            for h in (qrf_holder, hybrid_holder, cross_encoder_holder, rag_holder)
        )

    if all_ok:
        pipeline_ready.set()
        log.info("[qrf_rag] Pipeline unificado listo — pipeline_ask() disponible.")
    else:
        log.warning("[qrf_rag] Algún componente no cargó. pipeline_ask() puede fallar.")

    log.info("[qrf_rag] Hilo terminado.")


# ---------------------------------------------------------------------------
# Helpers de carga (uno por componente, ejecutados en el pool)
# ---------------------------------------------------------------------------

def _load_cross_encoder(cfg, holder, lock):
    try:
        ce = build_cross_encoder(cfg)
        with lock:
            holder[0] = ce
        log.info("[qrf_rag] CrossEncoderReranker listo: %s.", cfg.rag_reranker_model)
    except Exception as exc:
        log.error("[qrf_rag] CrossEncoder falló: %s", exc, exc_info=True)


def _load_rag(cfg, faiss_mgr, faiss_lock, holder, lock):
    try:
        rag = build_rag_pipeline(cfg, faiss_mgr, faiss_lock)
        with lock:
            holder[0] = rag
        log.info("[qrf_rag] RAGPipeline listo.")
    except Exception as exc:
        log.error("[qrf_rag] RAGPipeline falló: %s", exc, exc_info=True)


def _load_qrf(cfg, holder, lock):
    try:
        qrf = build_qrf_pipeline(cfg)
        with lock:
            holder[0] = qrf
        log.info("[qrf_rag] QueryPipeline (QRF) listo.")
    except Exception as exc:
        log.error("[qrf_rag] QRF falló: %s", exc, exc_info=True)


def _load_hybrid(cfg, retriever_holder, lsi_lock, faiss_mgr, holder, lock):
    try:
        hybrid = build_hybrid_retriever(cfg, retriever_holder, lsi_lock, faiss_mgr)
        with lock:
            holder[0] = hybrid
        log.info("[qrf_rag] HybridRetriever listo.")
    except Exception as exc:
        log.error("[qrf_rag] HybridRetriever falló: %s", exc, exc_info=True)