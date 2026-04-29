"""
threads/qrf_rag.py
==================
Hilo de carga de los pipelines QRF y RAG.

Responsabilidad única: esperar a que el índice FAISS esté disponible y,
a continuación, construir y cargar ``QueryPipeline`` (QRF) y ``RAGPipeline``
(RAG) utilizando la configuración del orquestador.

Ambos pipelines se almacenan en sus respectivos holders de un único elemento.
Una vez listos, se activan los Events correspondientes para que el
``Orchestrator`` pueda exponerlos a través de su API pública.

La carga se realiza una sola vez. El hilo termina tras ella o si se recibe
la señal de shutdown antes de que FAISS esté listo.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

from .._operations import build_qrf_pipeline, build_rag_pipeline
from ..config import OrchestratorConfig

if TYPE_CHECKING:
    from backend.qrf.pipeline import QueryPipeline
    from backend.rag.pipeline import RAGPipeline
    from backend.embedding import FaissIndexManager

log = logging.getLogger(__name__)


def run_qrf_rag_loader_thread(
    cfg:        OrchestratorConfig,
    shutdown:   threading.Event,
    faiss_ready: threading.Event,
    faiss_mgr:  "FaissIndexManager",
    faiss_lock: threading.RLock,
    qrf_holder: "list[Optional[QueryPipeline]]",
    qrf_lock:   threading.RLock,
    qrf_ready:  threading.Event,
    rag_holder: "list[Optional[RAGPipeline]]",
    rag_lock:   threading.RLock,
    rag_ready:  threading.Event,
) -> None:
    """
    Espera a que el índice FAISS esté listo y luego carga QRF y RAG.

    El pipeline QRF requiere FAISS para las búsquedas y, opcionalmente,
    el modelo LSI para la expansión LCE (si no está disponible, el
    QueryPipeline desactiva la expansión automáticamente).

    El pipeline RAG reutiliza el ``FaissIndexManager`` compartido del
    orquestador a través de ``EmbeddingRetriever``.

    Parámetros
    ----------
    cfg        : configuración del orquestador.
    shutdown   : evento de parada compartido.
    faiss_ready : evento que señala que el índice FAISS tiene vectores.
    faiss_mgr  : gestor FAISS compartido (ya inicializado en el orquestador).
    faiss_lock : RLock que protege el acceso a ``faiss_mgr``.
    qrf_holder : ``list[QueryPipeline | None]`` de un solo elemento.
    qrf_lock   : RLock que protege ``qrf_holder`` durante el swap.
    qrf_ready  : Event que se activa cuando QueryPipeline está listo.
    rag_holder : ``list[RAGPipeline | None]`` de un solo elemento.
    rag_lock   : RLock que protege ``rag_holder`` durante el swap.
    rag_ready  : Event que se activa cuando RAGPipeline está listo.
    """
    log.info("[qrf_rag] Hilo de carga iniciado — esperando índice FAISS...")

    # Esperar FAISS con chequeos periódicos de shutdown para no bloquear indefinidamente.
    while not shutdown.is_set():
        if faiss_ready.wait(timeout=10.0):
            break
    if shutdown.is_set():
        log.info("[qrf_rag] Shutdown antes de que FAISS esté listo — cancelando.")
        return

    log.info("[qrf_rag] FAISS listo — cargando QueryPipeline (QRF)...")

    # ── QRF ──────────────────────────────────────────────────────────────────
    try:
        qrf_pipeline = build_qrf_pipeline(cfg)
        with qrf_lock:
            qrf_holder[0] = qrf_pipeline
        qrf_ready.set()
        log.info("[qrf_rag] QueryPipeline cargado correctamente.")
    except FileNotFoundError as exc:
        log.warning(
            "[qrf_rag] Índice FAISS no encontrado al cargar QRF: %s. "
            "El pipeline QRF no estará disponible hasta el primer embedding.",
            exc,
        )
    except Exception as exc:
        log.error("[qrf_rag] Error al cargar QueryPipeline: %s", exc, exc_info=True)

    log.info("[qrf_rag] Cargando RAGPipeline...")

    # ── RAG ──────────────────────────────────────────────────────────────────
    try:
        rag_pipeline = build_rag_pipeline(cfg, faiss_mgr, faiss_lock)
        with rag_lock:
            rag_holder[0] = rag_pipeline
        rag_ready.set()
        log.info(
            "[qrf_rag] RAGPipeline cargado correctamente (reranker=%s).",
            "activo" if cfg.rag_use_reranker else "desactivado",
        )
    except Exception as exc:
        log.error("[qrf_rag] Error al cargar RAGPipeline: %s", exc, exc_info=True)

    log.info("[qrf_rag] Hilo de carga terminado.")
