"""
_status.py
==========
Construcción del snapshot de estado del sistema.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Extrae ``build_status()`` de ``Orchestrator.status()`` para mantener
``orchestrator.py`` centrado en coordinación y ciclo de vida.

---------------------------------------
- ``start_time``     : float (time.time()) para calcular uptime.
- ``thread_alive``   : dict[str, bool] con is_alive() de cada hilo daemon.
- ``uptime_seconds`` : segundos transcurridos desde el inicio del orquestador.
- ``thread_statuses``: estado detallado por hilo (alive/stopped).
- ``recent_docs_24h``: docs descubiertos en las últimas 24 h.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from backend.database import get_index_stats, get_document_counts, get_chunk_stats, get_connection
from backend.embedding import FaissIndexManager
from backend.retrieval import LSIRetriever

from .config import OrchestratorConfig

log = logging.getLogger(__name__)

# Nombres canónicos de los hilos daemon del orquestador
THREAD_LABELS: dict[str, str] = {
    "crawler":     "Crawler",
    "indexing":    "Indexing",
    "lsi_rebuild": "LSI Rebuild",
    "embedding":   "Embedding",
    "qrf_rag":     "QRF / RAG Loader",
}


def _query_recent_docs(db_path, hours: int = 24) -> int:
    """Documentos descubiertos en las últimas ``hours`` horas."""
    try:
        conn = get_connection(db_path)
        # Filtra por prefijo de ISO timestamp (los primeros 13 chars = "YYYY-MM-DDTHH")
        cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE fetched_at >= ?", (cutoff,)
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return -1


def build_status(
    cfg:              OrchestratorConfig,
    lsi_lock:         threading.RLock,
    retriever_holder: list,
    faiss_lock:       threading.RLock,
    faiss_mgr:        Optional[FaissIndexManager],
    lsi_ready:        threading.Event,
    faiss_ready:      threading.Event,
    qrf_ready:        threading.Event,
    rag_ready:        threading.Event,
    pipeline_ready:   threading.Event,
    # ── nuevos parámetros ────────────────────────────────────────────────────
    start_time:       float = 0.0,
    thread_alive:     Optional[dict[str, bool]] = None,
) -> dict:
    """
    Devuelve un snapshot del estado actual del sistema.

    Parámetros heredados
    --------------------
    (mismos que en la versión original)

    Parámetros nuevos
    -----------------
    start_time   : timestamp float (time.time()) del momento de init.
                   Usado para calcular el uptime.
    thread_alive : dict {nombre_hilo: is_alive()}. Si se omite, los estados
                   de hilo no se incluyen en el snapshot.

    Returns
    -------
    dict
        Snapshot completo. Claves nuevas:
        ``uptime_seconds``, ``thread_statuses``, ``recent_docs_24h``.
    """

    # ── BD: documentos ───────────────────────────────────────────────────────
    try:
        idx = get_index_stats(db_path=cfg.db_path)
    except Exception:
        idx = {}

    try:
        counts  = get_document_counts(cfg.db_path)
        total   = counts["total"]
        indexed = counts["indexed"]
        pending = counts["pending"]
    except Exception:
        total = indexed = pending = -1

    # ── Documentos recientes ─────────────────────────────────────────────────
    recent_docs = _query_recent_docs(cfg.db_path, hours=24)

    # ── LSI ──────────────────────────────────────────────────────────────────
    # Bugs encontrados en el código original:
    #   1. El atributo es `_model` (privado), no `model`.
    #   2. `LSIModel.doc_ids` arranca como None hasta que load()/build()
    #      terminan; si status() se llama antes de que el modelo esté listo,
    #      len(None) lanzaría TypeError.
    # La solución correcta es verificar ambas condiciones explícitamente,
    # no suprimir la excepción con try/except.
    with lsi_lock:
        retriever: Optional[LSIRetriever] = retriever_holder[0]
        lsi_model  = retriever._model if retriever is not None else None
        doc_ids    = getattr(lsi_model, "doc_ids", None)   # None si aún no cargado
        lsi_docs   = len(doc_ids) if doc_ids is not None else 0

    # ── Chunks ───────────────────────────────────────────────────────────────
    try:
        chunk_stats = get_chunk_stats(cfg.db_path)
    except Exception:
        chunk_stats = {"total_chunks": -1, "embedded_chunks": -1, "pending_chunks": -1}

    # ── FAISS ────────────────────────────────────────────────────────────────
    with faiss_lock:
        faiss_vectors = faiss_mgr.total_vectors if faiss_mgr else 0
        faiss_type    = faiss_mgr.index_type    if faiss_mgr else "none"

    # ── Uptime ───────────────────────────────────────────────────────────────
    uptime_seconds = int(time.time() - start_time) if start_time else 0

    # ── Estado de hilos daemon ───────────────────────────────────────────────
    thread_statuses: dict[str, dict] = {}
    if thread_alive:
        for raw_name, alive in thread_alive.items():
            label = THREAD_LABELS.get(raw_name, raw_name)
            thread_statuses[raw_name] = {
                "label":  label,
                "alive":  alive,
                "status": "running" if alive else "stopped",
            }

    return {
        # Documentos
        "docs_total":        total,
        "docs_pdf_indexed":  indexed,
        "docs_pdf_pending":  pending,
        "recent_docs_24h":   recent_docs,
        # Índice BM25
        "vocab_size":        idx.get("vocab_size", 0),
        "total_postings":    idx.get("total_postings", 0),
        # LSI
        "lsi_docs_in_model": lsi_docs,
        "lsi_model_ready":   lsi_ready.is_set(),
        # Chunks
        "total_chunks":      chunk_stats["total_chunks"],
        "embedded_chunks":   chunk_stats["embedded_chunks"],
        "pending_chunks":    chunk_stats["pending_chunks"],
        # FAISS
        "faiss_vectors":     faiss_vectors,
        "faiss_index_type":  faiss_type,
        "faiss_ready":       faiss_ready.is_set(),
        # Pipelines
        "qrf_ready":         qrf_ready.is_set(),
        "rag_ready":         rag_ready.is_set(),
        "pipeline_ready":    pipeline_ready.is_set(),
        # Configuración
        "embed_model":       cfg.embed_model,
        "web_threshold":     cfg.web_threshold,
        "web_min_docs":      cfg.web_min_docs,
        # Campos nuevos
        "uptime_seconds":    uptime_seconds,
        "thread_statuses":   thread_statuses,
        # Timestamp
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }