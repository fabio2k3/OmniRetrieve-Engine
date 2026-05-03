"""
_operations.py
==============
Operaciones de negocio del orquestador.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Operaciones
-----------
do_lsi_rebuild           — reconstruye el modelo LSI y actualiza el retriever compartido.
do_embed                 — ejecuta EmbeddingPipeline y recarga el índice FAISS compartido.
do_web_search            — evalúa suficiencia local y, si procede, busca en la web.
build_qrf_pipeline       — construye y carga el QueryPipeline (QRF completo).
do_qrf_search            — ejecuta búsqueda QRF standalone (expand + BRF + MMR).
do_qrf_search_with_session — igual que do_qrf_search pero devuelve session_id.
build_rag_pipeline       — construye RAGPipeline sobre EmbeddingRetriever.
do_rag_search            — ejecuta retrieval denso sin generación LLM.
do_rag_ask               — ejecuta pipeline RAG completo (retrieval → LLM).
build_hybrid_retriever   — construye HybridRetriever (LSI sparse + FAISS dense, RRF).
build_cross_encoder      — construye CrossEncoderReranker.
do_pipeline_ask          — pipeline unificado: QRF expand → Hybrid → Web → Rerank → RAG.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

from backend.database.schema import get_connection
from backend.indexing.pipeline import IndexingPipeline
from backend.embedding.pipeline import EmbeddingPipeline
from backend.embedding import FaissIndexManager
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever

from .config import OrchestratorConfig

if TYPE_CHECKING:
    from backend.qrf.pipeline import QueryPipeline
    from backend.rag.pipeline import RAGPipeline
    from backend.retrieval.hybrid_retriever import HybridRetriever
    from backend.retrieval.reranker import CrossEncoderReranker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexación BM25 (terms + postings)
# ---------------------------------------------------------------------------

def do_index(cfg: OrchestratorConfig) -> dict:
    """
    Ejecuta el ``IndexingPipeline`` en modo incremental.

    Solo procesa documentos con PDF descargado cuyo ``indexed_tfidf_at``
    sea NULL. Al terminar marca los documentos como indexados.

    Returns
    -------
    dict con: ``docs_processed``, ``terms_added``, ``postings_added``.
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

    Returns ``dict`` con stats o ``None`` si no hay suficientes docs o falla.
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
    Activa ``faiss_ready`` cuando el índice tiene vectores.
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
# Búsqueda web standalone (para uso desde CLI sin pipeline unificado)
# ---------------------------------------------------------------------------

def do_web_search(
    query:             str,
    retriever_results: list,
    cfg:               OrchestratorConfig,
) -> dict:
    """
    Evalúa suficiencia de resultados locales y activa búsqueda web si procede.
    Acepta tanto ``list[dict]`` como ``list[RetrievalResult]``.
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
# QRF (QueryPipeline standalone)
# ---------------------------------------------------------------------------

def build_qrf_pipeline(cfg: OrchestratorConfig) -> "QueryPipeline":
    """Construye y carga el ``QueryPipeline`` completo."""
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
    """Ejecuta búsqueda QRF standalone (expand + BRF + MMR sobre FAISS)."""
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
    """Igual que ``do_qrf_search`` pero devuelve también un ``session_id``."""
    try:
        return pipeline.search_with_session(query, top_k=top_k)
    except Exception as exc:
        log.error("[qrf] Error durante búsqueda QRF con sesión: %s", exc, exc_info=True)
        return [], ""


# ---------------------------------------------------------------------------
# RAG (standalone — hace su propio retrieval denso)
# ---------------------------------------------------------------------------

def build_rag_pipeline(
    cfg:        OrchestratorConfig,
    faiss_mgr:  FaissIndexManager,
    faiss_lock: threading.RLock,
) -> "RAGPipeline":
    """
    Construye el ``RAGPipeline`` con ``EmbeddingRetriever`` sobre el FAISS
    compartido. Usado solo para el modo RAG standalone (``rag_ask``).
    En el pipeline unificado el RAG recibe resultados ya recuperados.
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

    return RAGPipeline(retriever=dense_retriever, reranker=reranker)


def do_rag_search(
    query:       str,
    pipeline:    "RAGPipeline",
    top_k:       int = 10,
    candidate_k: int = 50,
) -> list[dict]:
    """Recuperación densa sin generación LLM."""
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
    """Pipeline RAG standalone: retrieval denso → contexto → respuesta LLM."""
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


# ---------------------------------------------------------------------------
# HybridRetriever (LSI sparse + FAISS dense, fusión RRF)
# ---------------------------------------------------------------------------

def build_hybrid_retriever(
    cfg:              OrchestratorConfig,
    retriever_holder: list,      # list[LSIRetriever | None]
    lsi_lock:         threading.RLock,
    faiss_mgr:        FaissIndexManager,
) -> "HybridRetriever":
    """
    Construye el ``HybridRetriever`` combinando LSI (sparse) y FAISS (dense)
    con fusión RRF.

    El ``LSIRetriever`` del orquestador se envuelve en ``LSIRetrieverAdapter``
    para que sea compatible con ``RetrieverProtocol`` (devuelve RetrievalResult).
    El retriever denso usa el mismo ``FaissIndexManager`` compartido del orquestador.

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    retriever_holder : holder del LSIRetriever compartido (lista de un elemento).
    lsi_lock         : RLock que protege el acceso al holder.
    faiss_mgr        : FaissIndexManager compartido.
    """
    from backend.retrieval.lsi_retriever import LSIRetrieverAdapter
    from backend.retrieval.embedding_retriever import EmbeddingRetriever
    from backend.retrieval.hybrid_retriever import HybridRetriever

    with lsi_lock:
        lsi_retriever = retriever_holder[0]

    if lsi_retriever is None:
        raise RuntimeError(
            "LSIRetriever no está disponible aún. "
            "Espera a que lsi_ready esté activo antes de construir el HybridRetriever."
        )

    sparse = LSIRetrieverAdapter(lsi_retriever)
    dense  = EmbeddingRetriever(
        faiss_mgr  = faiss_mgr,
        db_path    = cfg.db_path,
        model_name = cfg.embed_model,
    )

    return HybridRetriever(
        sparse      = sparse,
        dense       = dense,
        candidate_k = cfg.hybrid_candidate_k,
        rrf_k       = cfg.hybrid_rrf_k,
        parallel    = cfg.hybrid_parallel,
    )


def build_cross_encoder(cfg: OrchestratorConfig) -> "CrossEncoderReranker":
    """Construye el ``CrossEncoderReranker`` para la etapa final del pipeline."""
    from backend.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_name=cfg.rag_reranker_model)


# ---------------------------------------------------------------------------
# Pipeline unificado: QRF expand → Hybrid → Web → CrossEncoder → RAG
# ---------------------------------------------------------------------------

def do_pipeline_ask(
    query:           str,
    qrf_pipeline:    "QueryPipeline",
    hybrid_retriever: "HybridRetriever",
    cross_encoder:   "CrossEncoderReranker",
    rag_pipeline:    "RAGPipeline",
    cfg:             OrchestratorConfig,
) -> dict:
    """
    Pipeline unificado de respuesta a consultas.

    Flujo
    -----
    1. **QRF expand** — expande la query con LCE (Latent Concept Expansion)
       usando el modelo LSI. Devuelve ``(expanded_query, new_terms)``.

    2. **HybridRetriever** — recupera los ``pipeline_top_k`` chunks más
       relevantes usando la query expandida. Combina LSI sparse + FAISS dense
       con fusión RRF. Devuelve ``list[RetrievalResult]``.

    3. **WebSearchPipeline** — evalúa si los resultados locales son suficientes.
       Si no superan el umbral (``web_threshold``), activa la búsqueda web y
       añade los resultados web al pool. Devuelve ``list[RetrievalResult]``
       combinada (local + web si aplica).

    4. **CrossEncoderReranker** — reordena todos los candidatos usando un
       cross-encoder (ms-marco). Selecciona los ``pipeline_rerank_k`` mejores
       chunks. Devuelve ``list[RetrievalResult]`` ordenados por relevancia real.

    5. **RAGPipeline.generate_from_results** — construye el contexto con los
       chunks rerankeados, arma el prompt y llama al LLM para generar la
       respuesta final.

    Parámetros
    ----------
    query            : pregunta del usuario en lenguaje natural.
    qrf_pipeline     : QueryPipeline cargado (para la expansión LCE).
    hybrid_retriever : HybridRetriever listo (LSI + FAISS + RRF).
    cross_encoder    : CrossEncoderReranker listo.
    rag_pipeline     : RAGPipeline listo (solo se usa para generación).
    cfg              : configuración del orquestador.

    Returns
    -------
    dict con:
        ``query``          — pregunta original.
        ``expanded_query`` — query tras expansión LCE.
        ``expanded_terms`` — términos añadidos por LCE.
        ``answer``         — respuesta generada por el LLM.
        ``sources``        — fuentes usadas en el contexto.
        ``web_activated``  — True si se activó la búsqueda web.
        ``error``          — solo si ocurrió un error crítico.
    """
    from backend.web_search.pipeline import WebSearchPipeline

    # ── 1. Expansión LCE ──────────────────────────────────────────────────────
    try:
        expanded_query, expanded_terms = qrf_pipeline.expand_query(query)
        if expanded_terms:
            log.info(
                "[pipeline] LCE expandió la query con %d términos: %s",
                len(expanded_terms), expanded_terms,
            )
    except Exception as exc:
        log.warning("[pipeline] Expansión LCE falló, usando query original: %s", exc)
        expanded_query, expanded_terms = query, []

    # ── 2. HybridRetriever ────────────────────────────────────────────────────
    try:
        hybrid_results = hybrid_retriever.retrieve(
            expanded_query, top_n=cfg.pipeline_top_k
        )
        log.info("[pipeline] HybridRetriever devolvió %d candidatos.", len(hybrid_results))
        if hybrid_results:
            log.info("[pipeline] Top-%d Hybrid (score / arxiv_id / titulo):", len(hybrid_results))
            for i, r in enumerate(hybrid_results, 1):
                title = (r.metadata.get("title") or r.arxiv_id or "")[:55]
                log.info(
                    "  %2d. [%s]  score=%.4f  %s",
                    i, r.score_type, r.score, title,
                )
    except Exception as exc:
        log.error("[pipeline] HybridRetriever falló: %s", exc, exc_info=True)
        return {
            "query": query, "expanded_query": expanded_query,
            "expanded_terms": expanded_terms, "answer": "", "sources": [],
            "web_activated": False, "error": str(exc),
        }

    # ── 3. WebSearch (si procede) ─────────────────────────────────────────────
    try:
        web_pipeline = WebSearchPipeline(
            threshold    = cfg.web_threshold,
            min_docs     = cfg.web_min_docs,
            max_results  = cfg.web_max_results,
            search_depth = cfg.web_search_depth,
            use_fallback = cfg.web_use_fallback,
            auto_index   = cfg.web_auto_index,
            db_path      = cfg.db_path,
        )
        web_activated, all_results = web_pipeline.run_with_retrieval_results(
            query=query,
            retriever_results=hybrid_results,
        )
        log.info(
            "[pipeline] WebSearch: activada=%s candidatos_totales=%d",
            web_activated, len(all_results),
        )
    except Exception as exc:
        log.warning("[pipeline] WebSearch falló, continuando sin web: %s", exc)
        web_activated = False
        all_results   = hybrid_results

    # ── 4. CrossEncoder reranking ─────────────────────────────────────────────
    try:
        reranked = cross_encoder.rerank(
            query      = query,       # query original para el cross-encoder
            candidates = all_results,
            top_k      = cfg.pipeline_rerank_k,
        )
        log.info(
            "[pipeline] CrossEncoder: %d candidatos → %d chunks seleccionados.",
            len(all_results), len(reranked),
        )
        if reranked:
            log.info("[pipeline] Top-%d tras reranking (score / arxiv_id / titulo):", len(reranked))
            for i, r in enumerate(reranked, 1):
                title = (r.metadata.get("title") or r.arxiv_id or "")[:55]
                log.info(
                    "  %2d. [rerank]  score=%.4f  %s",
                    i, r.score, title,
                )
    except Exception as exc:
        log.warning("[pipeline] CrossEncoder falló, usando resultados sin reranking: %s", exc)
        reranked = all_results[:cfg.pipeline_rerank_k]

    # ── 5. RAG generation ─────────────────────────────────────────────────────
    try:
        result = rag_pipeline.generate_from_results(
            query       = query,
            results     = reranked,
            max_chunks  = cfg.pipeline_max_chunks,
            max_chars   = cfg.pipeline_max_chars,
        )
    except Exception as exc:
        log.error("[pipeline] RAG generation falló: %s", exc, exc_info=True)
        result = {"query": query, "answer": "", "sources": [], "error": str(exc)}

    result["expanded_query"] = expanded_query
    result["expanded_terms"] = expanded_terms
    result["web_activated"]  = web_activated
    return result