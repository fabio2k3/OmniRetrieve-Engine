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
do_index              — ejecuta IndexingPipeline BM25 incremental.
do_lsi_rebuild        — reconstruye el modelo LSI y actualiza el retriever compartido.
do_embed              — ejecuta EmbeddingPipeline y recarga el índice FAISS compartido.
do_web_search         — evalúa suficiencia local y, si procede, busca en la web.
build_qrf_pipeline    — construye y carga el QueryPipeline (QRF completo).
do_qrf_search         — ejecuta búsqueda QRF (expand + BRF + MMR).
do_qrf_search_with_session — igual que do_qrf_search pero devuelve también un session_id.
build_rag_pipeline    — construye el RAGPipeline sobre EmbeddingRetriever.
do_rag_search         — ejecuta retrieval + reranking sin generación LLM.
do_rag_ask            — ejecuta el pipeline RAG completo (retrieval → LLM).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

from backend.database.schema import get_connection
from backend.indexing.pipeline import IndexingPipeline
from backend.embedding.pipeline import EmbeddingPipeline
from backend.embedding import FaissIndexManager          # ← import correcto
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever

from .config import OrchestratorConfig

if TYPE_CHECKING:
    from backend.qrf.pipeline import QueryPipeline
    from backend.rag.pipeline import RAGPipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexación BM25
# ---------------------------------------------------------------------------

def do_index(cfg: OrchestratorConfig) -> dict:
    """
    Ejecuta el ``IndexingPipeline`` BM25 en modo incremental.

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


# ---------------------------------------------------------------------------
# QRF (Query Refinement Framework)
# ---------------------------------------------------------------------------

def build_qrf_pipeline(cfg: OrchestratorConfig) -> "QueryPipeline":
    """
    Construye y carga el ``QueryPipeline`` completo (expand + BRF + MMR).

    El pipeline carga el modelo de embedding, el índice FAISS y el modelo
    LSI necesario para la expansión LCE. Puede tardar varios segundos.

    Parámetros
    ----------
    cfg : configuración del orquestador.

    Returns
    -------
    QueryPipeline listo para ejecutar búsquedas.

    Lanza
    -----
    FileNotFoundError si el índice FAISS no existe todavía.
    """
    from backend.qrf.pipeline import QueryPipeline

    pipeline = QueryPipeline(
        model_name       = cfg.embed_model,
        lsi_model_path   = cfg.model_path,
        index_path       = cfg.faiss_index_path,
        id_map_path      = cfg.faiss_id_map_path,
        db_path          = cfg.db_path,
        top_k_initial    = cfg.qrf_top_k_initial,
        expand           = cfg.qrf_expand,
        expand_top_dims  = cfg.qrf_expand_top_dims,
        expand_min_corr  = cfg.qrf_expand_min_corr,
        expand_max_terms = cfg.qrf_expand_max_terms,
        brf_alpha        = cfg.qrf_brf_alpha,
        brf_top_k        = cfg.qrf_brf_top_k,
        mmr_lambda       = cfg.qrf_mmr_lambda,
    )
    pipeline.load()
    return pipeline


def do_qrf_search(
    query:    str,
    pipeline: "QueryPipeline",
    top_k:    int = 10,
) -> list[dict]:
    """
    Ejecuta el pipeline QRF completo: expansión LCE + embedding + BRF + MMR.

    Parámetros
    ----------
    query    : consulta en lenguaje natural.
    pipeline : instancia cargada de QueryPipeline.
    top_k    : número de resultados finales a devolver.

    Returns
    -------
    list[dict]
        Resultados enriquecidos con claves: ``score``, ``mmr_score``,
        ``chunk_id``, ``arxiv_id``, ``chunk_index``, ``text``, ``title``,
        ``authors``, ``abstract``, ``pdf_url``, ``expanded_terms``.
        Lista vacía si el pipeline falla.
    """
    try:
        return pipeline.search(query, top_k=top_k)
    except Exception as exc:
        log.error("[qrf] Error durante búsqueda QRF: %s", exc, exc_info=True)
        return []


def do_qrf_search_with_session(
    query:    str,
    pipeline: "QueryPipeline",
    top_k:    int = 10,
) -> tuple[list[dict], str]:
    """
    Igual que ``do_qrf_search`` pero devuelve también un ``session_id``.

    El ``session_id`` puede usarse para futuras rondas de refinamiento.
    Nota: el método ``refine()`` (Rocchio) no está integrado de momento.

    Returns
    -------
    tuple[list[dict], str]
        ``(results, session_id)``. En caso de error: ``([], "")``.
    """
    try:
        return pipeline.search_with_session(query, top_k=top_k)
    except Exception as exc:
        log.error("[qrf] Error durante búsqueda QRF con sesión: %s", exc, exc_info=True)
        return [], ""


# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------

def build_rag_pipeline(
    cfg:        OrchestratorConfig,
    faiss_mgr:  FaissIndexManager,
    faiss_lock: threading.RLock,
) -> "RAGPipeline":
    """
    Construye el ``RAGPipeline`` usando ``EmbeddingRetriever`` sobre el
    ``FaissIndexManager`` compartido del orquestador.

    El retriever es denso (FAISS + sentence-transformers). Si
    ``cfg.rag_use_reranker`` está activo, añade un ``CrossEncoderReranker``
    de segunda etapa.

    Parámetros
    ----------
    cfg        : configuración del orquestador.
    faiss_mgr  : gestor FAISS compartido (ya inicializado).
    faiss_lock : RLock que protege el acceso a ``faiss_mgr`` (documentado
                 por consistencia con el patrón del orquestador; el acceso
                 concurrente lo gestiona internamente FaissIndexManager).

    Returns
    -------
    RAGPipeline listo para ejecutar búsquedas y generación.
    """
    from backend.retrieval.embedding_retriever import EmbeddingRetriever
    from backend.rag.pipeline import RAGPipeline

    dense_retriever = EmbeddingRetriever(
        faiss_mgr  = faiss_mgr,
        db_path    = cfg.db_path,
        model_name = cfg.embed_model,
    )

    reranker = None
    if cfg.rag_use_reranker:
        from backend.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(model_name=cfg.rag_reranker_model)
        log.info("[rag] CrossEncoderReranker activado: %s", cfg.rag_reranker_model)

    return RAGPipeline(
        retriever = dense_retriever,
        reranker  = reranker,
    )


def do_rag_search(
    query:       str,
    pipeline:    "RAGPipeline",
    top_k:       int = 10,
    candidate_k: int = 50,
) -> list[dict]:
    """
    Ejecuta retrieval + reranking (opcional) sin generación LLM.

    Parámetros
    ----------
    query       : consulta en lenguaje natural.
    pipeline    : instancia de RAGPipeline lista.
    top_k       : resultados finales a devolver.
    candidate_k : candidatos recuperados antes del reranking.

    Returns
    -------
    list[dict]
        Lista con claves: ``chunk_id``, ``arxiv_id``, ``chunk_index``,
        ``title``, ``text``, ``score``, ``score_type``.
        Lista vacía si el pipeline falla.
    """
    try:
        return pipeline.search(query, top_k=top_k, candidate_k=candidate_k)
    except Exception as exc:
        log.error("[rag] Error durante búsqueda RAG: %s", exc, exc_info=True)
        return []


def do_rag_ask(
    query:       str,
    pipeline:    "RAGPipeline",
    top_k:       int = 10,
    candidate_k: int = 50,
    max_chunks:  int = 5,
    max_chars:   int = 400,
) -> dict:
    """
    Ejecuta el pipeline RAG completo: retrieval → contexto → prompt → respuesta LLM.

    Parámetros
    ----------
    query       : pregunta del usuario.
    pipeline    : instancia de RAGPipeline lista.
    top_k       : chunks candidatos recuperados.
    candidate_k : candidatos antes del reranking.
    max_chunks  : máximo de chunks inyectados en el contexto del LLM.
    max_chars   : caracteres máximos por chunk en el contexto.

    Returns
    -------
    dict con claves:
        ``query``   — pregunta original.
        ``answer``  — respuesta generada por el LLM.
        ``sources`` — lista de fuentes usadas en el contexto.
        ``error``   — presente solo si ocurrió un error.
    """
    try:
        return pipeline.ask(
            query       = query,
            top_k       = top_k,
            candidate_k = candidate_k,
            max_chunks  = max_chunks,
            max_chars   = max_chars,
        )
    except Exception as exc:
        log.error("[rag] Error durante ask RAG: %s", exc, exc_info=True)
        return {"query": query, "answer": "", "sources": [], "error": str(exc)}
