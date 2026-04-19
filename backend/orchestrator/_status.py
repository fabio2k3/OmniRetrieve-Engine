"""
_status.py
==========
Construcción del snapshot de estado del sistema.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Extrae ``build_status()`` de ``Orchestrator.status()`` para mantener
``orchestrator.py`` centrado en coordinación y ciclo de vida.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from backend.database.schema import get_connection
from backend.database.index_repository import get_index_stats
from backend.database.chunk_repository import get_chunk_stats
from backend.embedding.faiss_index import FaissIndexManager
from backend.retrieval.lsi_retriever import LSIRetriever

from .config import OrchestratorConfig

log = logging.getLogger(__name__)


def build_status(
    cfg:              OrchestratorConfig,
    lsi_lock:         threading.RLock,
    retriever_holder: list,
    faiss_lock:       threading.RLock,
    faiss_mgr:        Optional[FaissIndexManager],
    lsi_ready:        threading.Event,
    faiss_ready:      threading.Event,
) -> dict:
    """
    Devuelve un snapshot del estado actual del sistema.

    Consulta SQLite, el modelo LSI y el índice FAISS para obtener
    las métricas en tiempo real. Los errores de BD se capturan y devuelven
    como ``-1`` para no interrumpir la respuesta.

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    lsi_lock         : RLock que protege ``retriever_holder``.
    retriever_holder : ``list[LSIRetriever | None]`` de un elemento.
    faiss_lock       : RLock que protege ``faiss_mgr``.
    faiss_mgr        : gestor del índice FAISS compartido.
    lsi_ready        : Event — True si el modelo LSI está disponible.
    faiss_ready      : Event — True si el índice FAISS tiene vectores.

    Returns
    -------
    dict
        Snapshot con claves:
        ``docs_total``, ``docs_pdf_indexed``, ``docs_pdf_pending``,
        ``docs_not_in_index``, ``vocab_size``, ``total_postings``,
        ``lsi_docs_in_model``, ``lsi_model_ready``, ``total_chunks``,
        ``embedded_chunks``, ``pending_chunks``, ``faiss_vectors``,
        ``faiss_index_type``, ``faiss_ready``, ``embed_model``,
        ``web_threshold``, ``web_min_docs``, ``timestamp``.
    """
    # ── BD: documentos ───────────────────────────────────────────────────────
    try:
        idx = get_index_stats(db_path=cfg.db_path)
    except Exception:
        idx = {}

    try:
        conn      = get_connection(cfg.db_path)
        total     = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed   = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1"
        ).fetchone()[0]
        pending   = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=0"
        ).fetchone()[0]
        unindexed = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
        ).fetchone()[0]
        conn.close()
    except Exception:
        total = indexed = pending = unindexed = -1

    # ── LSI ──────────────────────────────────────────────────────────────────
    with lsi_lock:
        retriever: Optional[LSIRetriever] = retriever_holder[0]
        lsi_docs = len(retriever.model.doc_ids) if retriever else 0

    # ── Chunks ───────────────────────────────────────────────────────────────
    try:
        chunk_stats = get_chunk_stats(cfg.db_path)
    except Exception:
        chunk_stats = {"total_chunks": -1, "embedded_chunks": -1, "pending_chunks": -1}

    # ── FAISS ────────────────────────────────────────────────────────────────
    with faiss_lock:
        faiss_vectors = faiss_mgr.total_vectors if faiss_mgr else 0
        faiss_type    = faiss_mgr.index_type    if faiss_mgr else "none"

    return {
        "docs_total":        total,
        "docs_pdf_indexed":  indexed,
        "docs_pdf_pending":  pending,
        "docs_not_in_index": unindexed,
        "vocab_size":        idx.get("vocab_size", 0),
        "total_postings":    idx.get("total_postings", 0),
        "lsi_docs_in_model": lsi_docs,
        "lsi_model_ready":   lsi_ready.is_set(),
        "total_chunks":      chunk_stats["total_chunks"],
        "embedded_chunks":   chunk_stats["embedded_chunks"],
        "pending_chunks":    chunk_stats["pending_chunks"],
        "faiss_vectors":     faiss_vectors,
        "faiss_index_type":  faiss_type,
        "faiss_ready":       faiss_ready.is_set(),
        "embed_model":       cfg.embed_model,
        "web_threshold":     cfg.web_threshold,
        "web_min_docs":      cfg.web_min_docs,
        "timestamp":         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
