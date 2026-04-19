"""
_operations.py
==============
Operaciones de negocio del orquestador.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Cada función encapsula una operación completa que el ``Orchestrator``
delega cuando un hilo watcher o un comando CLI lo solicitan.
Reciben sus dependencias por parámetro para mantenerse sin estado propio
y facilitar los tests unitarios.

Operaciones
-----------
do_index          — ejecuta IndexingPipeline BM25 incremental.
do_lsi_rebuild    — reconstruye el modelo LSI y actualiza el retriever compartido.
do_embed          — ejecuta EmbeddingPipeline y recarga el índice FAISS compartido.
do_web_search     — evalúa suficiencia local y, si procede, busca en la web.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.database.schema import get_connection
from backend.indexing.pipeline import IndexingPipeline
from backend.embedding.pipeline import EmbeddingPipeline
from backend.embedding.faiss_index import FaissIndexManager
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever

from .config import OrchestratorConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexación BM25
# ---------------------------------------------------------------------------

def do_index(cfg: OrchestratorConfig) -> dict:
    """
    Ejecuta el ``IndexingPipeline`` BM25 en modo incremental.

    Usa ``new_indexing.IndexingPipeline`` (BM25) en lugar del pipeline
    TF-IDF anterior.

    Parámetros
    ----------
    cfg : configuración del orquestador.

    Returns
    -------
    dict
        Stats devueltas por el pipeline:
        ``docs_processed``, ``terms_added``, ``postings_added``.
    """
    pipeline = IndexingPipeline(
        db_path       = cfg.db_path,
        field         = cfg.index_field,
        batch_size    = cfg.index_batch_size,
        use_stemming  = cfg.index_use_stemming,
        min_token_len = cfg.index_min_token_len,
    )
    stats = pipeline.run(reindex=False)
    log.info(
        "[indexing] Completado — docs=%d términos=%d postings=%d",
        stats["docs_processed"], stats["terms_added"], stats["postings_added"],
    )
    return stats


# ---------------------------------------------------------------------------
# Rebuild LSI
# ---------------------------------------------------------------------------

def do_lsi_rebuild(
    cfg:              OrchestratorConfig,
    lsi_lock:         threading.RLock,
    lsi_ready:        threading.Event,
    retriever_holder: list,
) -> Optional[dict]:
    """
    Construye un nuevo ``LSIModel``, lo guarda y actualiza el retriever compartido.

    Si no hay suficientes documentos indexados o el build falla, devuelve ``None``
    sin propagar la excepción (el hilo de rebuild no debe morir ante fallos puntuales).

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    lsi_lock         : RLock que protege ``retriever_holder`` durante el swap.
    lsi_ready        : Event que se activa cuando el primer modelo está listo.
    retriever_holder : ``list[LSIRetriever | None]`` de un solo elemento.

    Returns
    -------
    dict | None
        Stats del build (``n_docs``, ``n_terms``, ``var_explained``),
        o ``None`` si no fue posible construir el modelo.
    """
    try:
        conn = get_connection(cfg.db_path)
        n_indexed = conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM postings"
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        log.warning("[lsi] No se pudo consultar la BD: %s", exc)
        return None

    if n_indexed < cfg.lsi_min_docs:
        log.info(
            "[lsi] Solo %d docs indexados (mínimo %d) — omitiendo rebuild.",
            n_indexed, cfg.lsi_min_docs,
        )
        return None

    k = min(cfg.lsi_k, n_indexed - 1)
    log.info("[lsi] Reconstruyendo modelo (n_indexed=%d, k=%d)…", n_indexed, k)

    try:
        model = LSIModel(k=k)
        stats = model.build(db_path=cfg.db_path)
        model.save(path=cfg.model_path)

        new_retriever = LSIRetriever(model=model)
        new_retriever.load(model_path=cfg.model_path, db_path=cfg.db_path)

        with lsi_lock:
            retriever_holder[0] = new_retriever

        lsi_ready.set()
        log.info(
            "[lsi] Modelo actualizado — n_docs=%d varianza=%.1f%%",
            stats["n_docs"], stats["var_explained"] * 100,
        )
        return stats

    except Exception as exc:
        log.error("[lsi] Error durante rebuild: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Embedding + FAISS
# ---------------------------------------------------------------------------

def do_embed(
    cfg:         OrchestratorConfig,
    faiss_lock:  threading.RLock,
    faiss_mgr:   Optional[FaissIndexManager],
    faiss_ready: threading.Event,
) -> dict:
    """
    Ejecuta ``EmbeddingPipeline`` incremental y recarga el índice FAISS compartido.

    El ``FaissIndexManager`` gestiona internamente la política de rebuild
    (cada ``embed_rebuild_every`` chunks). Al terminar, si el índice tiene
    vectores, activa ``faiss_ready`` para habilitar ``semantic_query``.

    Parámetros
    ----------
    cfg         : configuración del orquestador.
    faiss_lock  : RLock que protege ``faiss_mgr`` durante el reload.
    faiss_mgr   : gestor del índice FAISS compartido.
    faiss_ready : Event que se activa cuando el índice tiene vectores.

    Returns
    -------
    dict
        Stats del pipeline: ``chunks_processed``, ``batches_processed``,
        ``rebuilds_triggered``.
    """
    pipeline = EmbeddingPipeline(
        db_path       = cfg.db_path,
        model_name    = cfg.embed_model,
        batch_size    = cfg.embed_batch_size,
        rebuild_every = cfg.embed_rebuild_every,
        nlist         = cfg.embed_nlist,
        m             = cfg.embed_m,
        nbits         = cfg.embed_nbits,
        nprobe        = cfg.embed_nprobe,
        index_path    = cfg.faiss_index_path,
        id_map_path   = cfg.faiss_id_map_path,
    )
    stats = pipeline.run(reembed=False)

    if cfg.faiss_index_path.exists() and faiss_mgr is not None:
        with faiss_lock:
            loaded = faiss_mgr.load()
            if loaded:
                faiss_ready.set()
                log.info(
                    "[embedding] Índice FAISS actualizado — tipo=%s vectores=%d",
                    faiss_mgr.index_type, faiss_mgr.total_vectors,
                )

    log.info(
        "[embedding] Completado — chunks=%d lotes=%d rebuilds=%d",
        stats["chunks_processed"], stats["batches_processed"], stats["rebuilds_triggered"],
    )
    return stats


# ---------------------------------------------------------------------------
# Búsqueda web con evaluación de suficiencia
# ---------------------------------------------------------------------------

def do_web_search(
    query:             str,
    retriever_results: list[dict],
    cfg:               OrchestratorConfig,
) -> dict:
    """
    Evalúa si los resultados locales son suficientes y, si no, busca en la web.

    Delega en ``WebSearchPipeline``, que internamente usa Tavily (con
    DuckDuckGo como fallback) y guarda e indexa automáticamente los
    resultados web nuevos en la BD.

    Parámetros
    ----------
    query             : consulta original del usuario.
    retriever_results : resultados del LSIRetriever (lista de dicts con 'score').
    cfg               : configuración del orquestador.

    Returns
    -------
    dict con:
        ``results``       — lista combinada (local + web si aplica).
        ``web_activated`` — True si se activó la búsqueda web.
        ``web_results``   — solo los resultados web.
        ``reason``        — explicación de la decisión de suficiencia.
        ``query``         — query original.
        ``indexed``       — docs web indexados (0 si ``web_auto_index=False``).
    """
    try:
        from backend.web_search.pipeline import WebSearchPipeline
    except ImportError as exc:
        log.error("[web_search] Módulo web_search no disponible: %s", exc)
        return {
            "results": retriever_results, "web_activated": False,
            "web_results": [], "reason": "Módulo no disponible.",
            "query": query, "indexed": 0,
        }

    pipeline = WebSearchPipeline(
        threshold    = cfg.web_threshold,
        min_docs     = cfg.web_min_docs,
        max_results  = cfg.web_max_results,
        search_depth = cfg.web_search_depth,
        use_fallback = cfg.web_use_fallback,
        auto_index   = cfg.web_auto_index,
        db_path      = cfg.db_path,
    )
    return pipeline.run(query=query, retriever_results=retriever_results)
