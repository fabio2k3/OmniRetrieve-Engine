"""
main.py
=======
Entrypoint del orquestador OmniRetrieve-Engine.

Uso
---
    python -m backend.orchestrator

    # Umbral bajo y rebuild cada 30 min (pruebas)
    python -m backend.orchestrator --pdf-threshold 3 --lsi-interval 1800

    # Búsqueda web con umbral más exigente
    python -m backend.orchestrator --web-threshold 0.25 --web-max-results 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.retrieval.lsi_model import MODEL_PATH
from .orchestrator import Orchestrator
from .config import OrchestratorConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_DATA_DIR / "orchestrator.log", encoding="utf-8"),
    ],
)
logging.getLogger("backend.new_indexing.pipeline").setLevel(logging.WARNING)
logging.getLogger("backend.retrieval.lsi_model").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — Orquestador completo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    p.add_argument("--db",    type=Path, default=DB_PATH,    help="Ruta a la BD SQLite.")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="Ruta al modelo LSI .pkl.")

    # Crawler
    g = p.add_argument_group("crawler")
    g.add_argument("--ids-per-discovery",  type=int,   default=500,   help="IDs por ciclo de discovery.")
    g.add_argument("--batch-size",         type=int,   default=50,    help="Metadatos por ciclo.")
    g.add_argument("--pdf-batch",          type=int,   default=10,    help="PDFs por ciclo.")
    g.add_argument("--discovery-interval", type=float, default=600.0, help="Segundos entre ciclos de discovery.")
    g.add_argument("--download-interval",  type=float, default=60.0,  help="Segundos entre ciclos de metadatos.")
    g.add_argument("--pdf-interval",       type=float, default=1.0,   help="Segundos entre ciclos de PDF.")

    # Indexing BM25
    g = p.add_argument_group("indexing (BM25)")
    g.add_argument("--pdf-threshold",  type=int,   default=10,      help="PDFs nuevos para disparar indexación.")
    g.add_argument("--index-poll",     type=float, default=30.0,    help="Segundos entre sondeos del watcher.")
    g.add_argument("--index-field",    type=str,   default="both",
                   choices=["full_text", "abstract", "both"],       help="Campo a indexar.")
    g.add_argument("--index-batch",    type=int,   default=100,     help="Docs por lote en BM25.")
    g.add_argument("--stemming",       action="store_true",         help="Activar stemming en BM25.")
    g.add_argument("--min-token-len",  type=int,   default=3,       help="Longitud mínima de token BM25.")

    # LSI
    g = p.add_argument_group("lsi")
    g.add_argument("--lsi-interval", type=float, default=3600.0, help="Segundos entre rebuilds LSI.")
    g.add_argument("--lsi-k",        type=int,   default=100,    help="Componentes latentes del SVD.")
    g.add_argument("--lsi-min-docs", type=int,   default=10,     help="Mínimo de docs para construir el modelo.")

    # Web search
    g = p.add_argument_group("web search")
    g.add_argument("--web-threshold",   type=float, default=0.15,    help="Score mínimo para no activar web.")
    g.add_argument("--web-min-docs",    type=int,   default=1,       help="Docs mínimos que deben superar el umbral.")
    g.add_argument("--web-max-results", type=int,   default=5,       help="Máximo de resultados web.")
    g.add_argument("--web-depth",       type=str,   default="basic",
                   choices=["basic", "advanced"],                    help="Profundidad de búsqueda Tavily.")
    g.add_argument("--no-web-fallback", action="store_true",         help="Desactivar fallback DuckDuckGo.")
    g.add_argument("--no-web-index",    action="store_true",         help="No indexar docs web automáticamente.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    cfg = OrchestratorConfig(
        db_path    = args.db,
        model_path = args.model,
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
        # web search
        web_threshold    = args.web_threshold,
        web_min_docs     = args.web_min_docs,
        web_max_results  = args.web_max_results,
        web_search_depth = args.web_depth,
        web_use_fallback = not args.no_web_fallback,
        web_auto_index   = not args.no_web_index,
    )

    orc = Orchestrator(cfg)
    orc.start()
    orc.run_cli()   # bloquea hasta 'quit'
    orc.stop()


if __name__ == "__main__":
    main()
