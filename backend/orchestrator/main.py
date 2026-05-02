"""
main.py
=======
Entrypoint del orquestador OmniRetrieve-Engine.

Los valores por defecto de todos los argumentos CLI provienen directamente
de OrchestratorConfig, garantizando que config.py es la única fuente de
verdad. Cualquier argumento CLI sobreescribe solo el parámetro concreto.

Uso
---
    python -m backend.orchestrator
    python -m backend.orchestrator --lsi-interval 1800 --web-threshold 0.25
    python -m backend.orchestrator --pipeline-top-k 20 --pipeline-rerank-k 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.retrieval.lsi_model import MODEL_PATH
from .config import OrchestratorConfig
from .orchestrator import Orchestrator

# ── Logging ───────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_DATA_DIR / "orchestrator.log", encoding="utf-8"),
    ],
)
logging.getLogger("backend.retrieval.lsi_model").setLevel(logging.WARNING)

# Instancia de referencia para leer defaults sin efectos secundarios
_D = OrchestratorConfig()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — Orquestador",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Retrieval general
    p.add_argument("--retrieval-top-k", type=int, default=_D.retrieval_top_k,
                   help="Resultados devueltos por query/qrf/rag standalone.")

    # Rutas
    p.add_argument("--db",    type=Path, default=DB_PATH,    help="Ruta a la BD SQLite.")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="Ruta al modelo LSI .pkl.")

    # Crawler
    g = p.add_argument_group("crawler")
    g.add_argument("--ids-per-discovery",  type=int,   default=_D.ids_per_discovery,  help="IDs por ciclo de discovery.")
    g.add_argument("--batch-size",         type=int,   default=_D.batch_size,         help="Metadatos por ciclo.")
    g.add_argument("--pdf-batch",          type=int,   default=_D.pdf_batch_size,     help="PDFs por ciclo.")
    g.add_argument("--discovery-interval", type=float, default=_D.discovery_interval, help="Segundos entre ciclos de discovery.")
    g.add_argument("--download-interval",  type=float, default=_D.download_interval,  help="Segundos entre ciclos de metadatos.")
    g.add_argument("--pdf-interval",       type=float, default=_D.pdf_interval,       help="Segundos entre ciclos de PDF.")

    # Indexing BM25
    g = p.add_argument_group("indexing BM25")
    g.add_argument("--pdf-threshold",    type=int,   default=_D.pdf_threshold,       help="PDFs sin indexar para disparar hilo.")
    g.add_argument("--index-poll",       type=float, default=_D.index_poll_interval, help="Segundos entre sondeos del watcher BM25.")
    g.add_argument("--index-field",      type=str,   default=_D.index_field,
                   choices=["full_text", "abstract", "both"],                        help="Campo a indexar.")
    g.add_argument("--index-batch",      type=int,   default=_D.index_batch_size,    help="Docs por lote en IndexingPipeline.")
    g.add_argument("--stemming",         action="store_true",                        help="Activar stemming en BM25.")
    g.add_argument("--min-token-len",    type=int,   default=_D.index_min_token_len, help="Longitud mínima de token BM25.")

    # LSI
    g = p.add_argument_group("lsi")
    g.add_argument("--lsi-interval", type=float, default=_D.lsi_rebuild_interval, help="Segundos entre rebuilds LSI.")
    g.add_argument("--lsi-k",        type=int,   default=_D.lsi_k,                help="Componentes latentes del SVD.")
    g.add_argument("--lsi-min-docs", type=int,   default=_D.lsi_min_docs,         help="Mínimo de docs para construir el modelo.")

    # Embedding / FAISS
    g = p.add_argument_group("embedding / FAISS")
    g.add_argument("--embed-model",        type=str,   default=_D.embed_model,         help="Modelo sentence-transformers.")
    g.add_argument("--embed-batch",        type=int,   default=_D.embed_batch_size,    help="Chunks por lote en el embedder.")
    g.add_argument("--embed-poll",         type=float, default=_D.embed_poll_interval, help="Segundos entre sondeos del watcher de embedding.")
    g.add_argument("--embed-threshold",    type=int,   default=_D.embed_threshold,     help="Chunks sin embedding para disparar el pipeline.")
    g.add_argument("--embed-rebuild-every",type=int,   default=_D.embed_rebuild_every, help="Chunks añadidos entre rebuilds FAISS.")
    g.add_argument("--embed-nlist",        type=int,   default=_D.embed_nlist,         help="Celdas Voronoi para IndexIVFPQ.")
    g.add_argument("--embed-m",            type=int,   default=_D.embed_m,             help="Subvectores PQ.")
    g.add_argument("--embed-nbits",        type=int,   default=_D.embed_nbits,         help="Bits por código PQ.")
    g.add_argument("--embed-nprobe",       type=int,   default=_D.embed_nprobe,        help="Celdas inspeccionadas en búsqueda FAISS.")

    # Web search
    g = p.add_argument_group("web search")
    g.add_argument("--web-threshold",   type=float, default=_D.web_threshold,    help="Score mínimo para no activar web.")
    g.add_argument("--web-min-docs",    type=int,   default=_D.web_min_docs,     help="Docs mínimos que deben superar el umbral.")
    g.add_argument("--web-max-results", type=int,   default=_D.web_max_results,  help="Máximo de resultados web.")
    g.add_argument("--web-depth",       type=str,   default=_D.web_search_depth,
                   choices=["basic", "advanced"],                                  help="Profundidad de búsqueda Tavily.")
    g.add_argument("--no-web-fallback", action="store_true",                      help="Desactivar fallback DuckDuckGo.")
    g.add_argument("--no-web-index",    action="store_true",                      help="No indexar docs web automáticamente.")

    # QRF
    g = p.add_argument_group("QRF (query refinement)")
    g.add_argument("--qrf-top-k-initial",    type=int,   default=_D.qrf_top_k_initial,    help="Candidatos en búsqueda inicial BRF.")
    g.add_argument("--no-qrf-expand",        action="store_true",                          help="Desactivar expansión LCE.")
    g.add_argument("--qrf-expand-top-dims",  type=int,   default=_D.qrf_expand_top_dims,  help="Dimensiones latentes en LCE.")
    g.add_argument("--qrf-expand-min-corr",  type=float, default=_D.qrf_expand_min_corr,  help="Correlación mínima para añadir término LCE.")
    g.add_argument("--qrf-expand-max-terms", type=int,   default=_D.qrf_expand_max_terms, help="Máximo de términos añadidos por LCE.")
    g.add_argument("--qrf-brf-alpha",        type=float, default=_D.qrf_brf_alpha,        help="Peso del vector original en BRF.")
    g.add_argument("--qrf-brf-top-k",        type=int,   default=_D.qrf_brf_top_k,        help="Resultados usados para centroide BRF.")
    g.add_argument("--qrf-mmr-lambda",       type=float, default=_D.qrf_mmr_lambda,       help="Balance relevancia/diversidad MMR.")

    # Hybrid retriever
    g = p.add_argument_group("hybrid retriever")
    g.add_argument("--hybrid-candidate-k", type=int,  default=_D.hybrid_candidate_k, help="Candidatos por rama antes de RRF.")
    g.add_argument("--hybrid-rrf-k",       type=int,  default=_D.hybrid_rrf_k,       help="Constante RRF.")
    g.add_argument("--no-hybrid-parallel", action="store_true",                       help="Desactivar paralelismo en HybridRetriever.")

    # Pipeline unificado
    g = p.add_argument_group("pipeline unificado")
    g.add_argument("--pipeline-top-k",     type=int, default=_D.pipeline_top_k,     help="Candidatos del HybridRetriever al web_search.")
    g.add_argument("--pipeline-rerank-k",  type=int, default=_D.pipeline_rerank_k,  help="Chunks tras CrossEncoder.")
    g.add_argument("--pipeline-max-chunks",type=int, default=_D.pipeline_max_chunks, help="Chunks máximos en contexto LLM.")
    g.add_argument("--pipeline-max-chars", type=int, default=_D.pipeline_max_chars,  help="Caracteres máximos por chunk en contexto LLM.")

    # RAG
    g = p.add_argument_group("RAG")
    g.add_argument("--reranker-model",  type=str, default=_D.rag_reranker_model, help="Modelo cross-encoder para reranking.")
    g.add_argument("--rag-candidate-k", type=int, default=_D.rag_candidate_k,    help="Candidatos en RAG standalone.")
    g.add_argument("--rag-max-chunks",  type=int, default=_D.rag_max_chunks,     help="Chunks en contexto RAG standalone.")
    g.add_argument("--rag-max-chars",   type=int, default=_D.rag_max_chars,      help="Chars por chunk en RAG standalone.")

    return p.parse_args()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    cfg = OrchestratorConfig(
        db_path    = args.db,
        model_path = args.model,
        retrieval_top_k = args.retrieval_top_k,
        # crawler
        ids_per_discovery  = args.ids_per_discovery,
        batch_size         = args.batch_size,
        pdf_batch_size     = args.pdf_batch,
        discovery_interval = args.discovery_interval,
        download_interval  = args.download_interval,
        pdf_interval       = args.pdf_interval,
        # indexing BM25
        pdf_threshold       = args.pdf_threshold,
        index_poll_interval = args.index_poll,
        index_field         = args.index_field,
        index_batch_size    = args.index_batch,
        index_use_stemming  = args.stemming,
        index_min_token_len = args.min_token_len,
        # lsi
        lsi_rebuild_interval = args.lsi_interval,
        lsi_k                = args.lsi_k,
        lsi_min_docs         = args.lsi_min_docs,
        # embedding / FAISS
        embed_model         = args.embed_model,
        embed_batch_size    = args.embed_batch,
        embed_poll_interval = args.embed_poll,
        embed_threshold     = args.embed_threshold,
        embed_rebuild_every = args.embed_rebuild_every,
        embed_nlist         = args.embed_nlist,
        embed_m             = args.embed_m,
        embed_nbits         = args.embed_nbits,
        embed_nprobe        = args.embed_nprobe,
        # web search
        web_threshold    = args.web_threshold,
        web_min_docs     = args.web_min_docs,
        web_max_results  = args.web_max_results,
        web_search_depth = args.web_depth,
        web_use_fallback = not args.no_web_fallback,
        web_auto_index   = not args.no_web_index,
        # QRF
        qrf_top_k_initial    = args.qrf_top_k_initial,
        qrf_expand           = not args.no_qrf_expand,
        qrf_expand_top_dims  = args.qrf_expand_top_dims,
        qrf_expand_min_corr  = args.qrf_expand_min_corr,
        qrf_expand_max_terms = args.qrf_expand_max_terms,
        qrf_brf_alpha        = args.qrf_brf_alpha,
        qrf_brf_top_k        = args.qrf_brf_top_k,
        qrf_mmr_lambda       = args.qrf_mmr_lambda,
        # hybrid retriever
        hybrid_candidate_k = args.hybrid_candidate_k,
        hybrid_rrf_k       = args.hybrid_rrf_k,
        hybrid_parallel    = not args.no_hybrid_parallel,
        # pipeline unificado
        pipeline_top_k      = args.pipeline_top_k,
        pipeline_rerank_k   = args.pipeline_rerank_k,
        pipeline_max_chunks = args.pipeline_max_chunks,
        pipeline_max_chars  = args.pipeline_max_chars,
        # RAG
        rag_reranker_model = args.reranker_model,
        rag_candidate_k    = args.rag_candidate_k,
        rag_max_chunks     = args.rag_max_chunks,
        rag_max_chars      = args.rag_max_chars,
    )

    orc = Orchestrator(cfg)
    orc.start()
    orc.run_cli()
    orc.stop()


if __name__ == "__main__":
    main()