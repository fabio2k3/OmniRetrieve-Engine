"""
main.py
=======
Entrypoint del orquestador OmniRetrieve-Engine.

Serializa la configuración en la variable de entorno OMNIRETRIEVE_CONFIG
y lanza Streamlit apuntando a frontend/frontend/app.py.
El Orchestrator (con todos sus hilos daemon) se crea dentro de la sesión
Streamlit mediante @st.cache_resource en app.py.

Uso
---
    python -m backend.orchestrator          (configuración por defecto)
    python -m backend.orchestrator --lsi-interval 1800 --web-threshold 0.25
    streamlit run frontend/frontend/app.py  (alternativa directa, config por defecto)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.retrieval.lsi_model import MODEL_PATH
from .config import OrchestratorConfig

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

_D = OrchestratorConfig()


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — lanzador Streamlit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Retrieval general
    p.add_argument("--retrieval-top-k", type=int, default=_D.retrieval_top_k)

    # Rutas
    p.add_argument("--db",    type=Path, default=DB_PATH)
    p.add_argument("--model", type=Path, default=MODEL_PATH)

    # Crawler
    g = p.add_argument_group("crawler")
    g.add_argument("--ids-per-discovery",  type=int,   default=_D.ids_per_discovery)
    g.add_argument("--batch-size",         type=int,   default=_D.batch_size)
    g.add_argument("--pdf-batch",          type=int,   default=_D.pdf_batch_size)
    g.add_argument("--discovery-interval", type=float, default=_D.discovery_interval)
    g.add_argument("--download-interval",  type=float, default=_D.download_interval)
    g.add_argument("--pdf-interval",       type=float, default=_D.pdf_interval)

    # Indexing BM25
    g = p.add_argument_group("indexing BM25")
    g.add_argument("--pdf-threshold",    type=int,   default=_D.pdf_threshold)
    g.add_argument("--index-poll",       type=float, default=_D.index_poll_interval)
    g.add_argument("--index-field",      type=str,   default=_D.index_field,
                   choices=["full_text", "abstract", "both"])
    g.add_argument("--index-batch",      type=int,   default=_D.index_batch_size)
    g.add_argument("--stemming",         action="store_true")
    g.add_argument("--min-token-len",    type=int,   default=_D.index_min_token_len)

    # LSI
    g = p.add_argument_group("lsi")
    g.add_argument("--lsi-interval", type=float, default=_D.lsi_rebuild_interval)
    g.add_argument("--lsi-k",        type=int,   default=_D.lsi_k)
    g.add_argument("--lsi-min-docs", type=int,   default=_D.lsi_min_docs)

    # Embedding / FAISS
    g = p.add_argument_group("embedding / FAISS")
    g.add_argument("--embed-model",         type=str,   default=_D.embed_model)
    g.add_argument("--embed-batch",         type=int,   default=_D.embed_batch_size)
    g.add_argument("--embed-poll",          type=float, default=_D.embed_poll_interval)
    g.add_argument("--embed-threshold",     type=int,   default=_D.embed_threshold)
    g.add_argument("--embed-rebuild-every", type=int,   default=_D.embed_rebuild_every)
    g.add_argument("--embed-nlist",         type=int,   default=_D.embed_nlist)
    g.add_argument("--embed-m",             type=int,   default=_D.embed_m)
    g.add_argument("--embed-nbits",         type=int,   default=_D.embed_nbits)
    g.add_argument("--embed-nprobe",        type=int,   default=_D.embed_nprobe)

    # Web search
    g = p.add_argument_group("web search")
    g.add_argument("--web-threshold",   type=float, default=_D.web_threshold)
    g.add_argument("--web-min-docs",    type=int,   default=_D.web_min_docs)
    g.add_argument("--web-max-results", type=int,   default=_D.web_max_results)
    g.add_argument("--web-depth",       type=str,   default=_D.web_search_depth,
                   choices=["basic", "advanced"])
    g.add_argument("--no-web-fallback", action="store_true")
    g.add_argument("--no-web-index",    action="store_true")

    # QRF
    g = p.add_argument_group("QRF")
    g.add_argument("--qrf-top-k-initial",    type=int,   default=_D.qrf_top_k_initial)
    g.add_argument("--no-qrf-expand",        action="store_true")
    g.add_argument("--qrf-expand-top-dims",  type=int,   default=_D.qrf_expand_top_dims)
    g.add_argument("--qrf-expand-min-corr",  type=float, default=_D.qrf_expand_min_corr)
    g.add_argument("--qrf-expand-max-terms", type=int,   default=_D.qrf_expand_max_terms)
    g.add_argument("--qrf-brf-alpha",        type=float, default=_D.qrf_brf_alpha)
    g.add_argument("--qrf-brf-top-k",        type=int,   default=_D.qrf_brf_top_k)
    g.add_argument("--qrf-mmr-lambda",       type=float, default=_D.qrf_mmr_lambda)

    # Hybrid
    g = p.add_argument_group("hybrid retriever")
    g.add_argument("--hybrid-candidate-k", type=int,  default=_D.hybrid_candidate_k)
    g.add_argument("--hybrid-rrf-k",       type=int,  default=_D.hybrid_rrf_k)
    g.add_argument("--no-hybrid-parallel", action="store_true")

    # Pipeline unificado
    g = p.add_argument_group("pipeline unificado")
    g.add_argument("--pipeline-top-k",      type=int, default=_D.pipeline_top_k)
    g.add_argument("--pipeline-rerank-k",   type=int, default=_D.pipeline_rerank_k)
    g.add_argument("--pipeline-max-chunks", type=int, default=_D.pipeline_max_chunks)
    g.add_argument("--pipeline-max-chars",  type=int, default=_D.pipeline_max_chars)

    # RAG
    g = p.add_argument_group("RAG")
    g.add_argument("--reranker-model",  type=str, default=_D.rag_reranker_model)
    g.add_argument("--rag-candidate-k", type=int, default=_D.rag_candidate_k)
    g.add_argument("--rag-max-chunks",  type=int, default=_D.rag_max_chunks)
    g.add_argument("--rag-max-chars",   type=int, default=_D.rag_max_chars)

    # Streamlit
    g = p.add_argument_group("streamlit")
    g.add_argument("--port",    type=int, default=8501, help="Puerto de Streamlit.")
    g.add_argument("--host",    type=str, default="localhost", help="Host de Streamlit.")
    g.add_argument("--no-browser", action="store_true", help="No abrir navegador automáticamente.")

    return p.parse_args()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Serializar parámetros clave como JSON para que app.py los lea
    cfg_override = {
        "retrieval_top_k":      args.retrieval_top_k,
        "ids_per_discovery":    args.ids_per_discovery,
        "batch_size":           args.batch_size,
        "pdf_batch_size":       args.pdf_batch,
        "discovery_interval":   args.discovery_interval,
        "download_interval":    args.download_interval,
        "pdf_interval":         args.pdf_interval,
        "pdf_threshold":        args.pdf_threshold,
        "index_poll_interval":  args.index_poll,
        "index_field":          args.index_field,
        "index_batch_size":     args.index_batch,
        "index_use_stemming":   args.stemming,
        "index_min_token_len":  args.min_token_len,
        "lsi_rebuild_interval": args.lsi_interval,
        "lsi_k":                args.lsi_k,
        "lsi_min_docs":         args.lsi_min_docs,
        "embed_model":          args.embed_model,
        "embed_batch_size":     args.embed_batch,
        "embed_poll_interval":  args.embed_poll,
        "embed_threshold":      args.embed_threshold,
        "embed_rebuild_every":  args.embed_rebuild_every,
        "embed_nlist":          args.embed_nlist,
        "embed_m":              args.embed_m,
        "embed_nbits":          args.embed_nbits,
        "embed_nprobe":         args.embed_nprobe,
        "web_threshold":        args.web_threshold,
        "web_min_docs":         args.web_min_docs,
        "web_max_results":      args.web_max_results,
        "web_search_depth":     args.web_depth,
        "web_use_fallback":     not args.no_web_fallback,
        "web_auto_index":       not args.no_web_index,
        "qrf_top_k_initial":    args.qrf_top_k_initial,
        "qrf_expand":           not args.no_qrf_expand,
        "qrf_expand_top_dims":  args.qrf_expand_top_dims,
        "qrf_expand_min_corr":  args.qrf_expand_min_corr,
        "qrf_expand_max_terms": args.qrf_expand_max_terms,
        "qrf_brf_alpha":        args.qrf_brf_alpha,
        "qrf_brf_top_k":        args.qrf_brf_top_k,
        "qrf_mmr_lambda":       args.qrf_mmr_lambda,
        "hybrid_candidate_k":   args.hybrid_candidate_k,
        "hybrid_rrf_k":         args.hybrid_rrf_k,
        "hybrid_parallel":      not args.no_hybrid_parallel,
        "pipeline_top_k":       args.pipeline_top_k,
        "pipeline_rerank_k":    args.pipeline_rerank_k,
        "pipeline_max_chunks":  args.pipeline_max_chunks,
        "pipeline_max_chars":   args.pipeline_max_chars,
        "rag_reranker_model":   args.reranker_model,
        "rag_candidate_k":      args.rag_candidate_k,
        "rag_max_chunks":       args.rag_max_chunks,
        "rag_max_chars":        args.rag_max_chars,
    }

    env = os.environ.copy()
    env["OMNIRETRIEVE_CONFIG"] = json.dumps(cfg_override)

    # Ruta al frontend
    _ROOT = Path(__file__).resolve().parent.parent.parent
    app_path = _ROOT / "frontend" / "frontend" / "app.py"
    if not app_path.exists():
        # fallback: buscar app.py relativo al proyecto
        alt = _ROOT / "frontend" / "app.py"
        app_path = alt if alt.exists() else app_path

    logging.info("[main] Lanzando Streamlit: %s", app_path)
    logging.info("[main] Host=%s  Puerto=%d", args.host, args.port)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port",    str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.no_browser else "false",
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        logging.info("[main] Detenido por el usuario.")
    except subprocess.CalledProcessError as exc:
        logging.error("[main] Streamlit terminó con código %d.", exc.returncode)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()