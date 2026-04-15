"""
main.py
=======
Entrypoint del módulo orquestador de OmniRetrieve-Engine.

Arranca el sistema completo: crawler + indexing + LSI en hilos de fondo
y una CLI interactiva en el hilo principal para lanzar queries en tiempo real.

Uso
---
    python -m backend.orchestrator

    # Umbral más bajo y rebuild cada 30 min (pruebas)
    python -m backend.orchestrator --pdf-threshold 3 --lsi-interval 1800

    # Solo 50 componentes latentes y corpus pequeño
    python -m backend.orchestrator --lsi-k 50 --lsi-min-docs 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.retrieval.lsi_model import MODEL_PATH
from .orchestrator import Orchestrator, OrchestratorConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "orchestrator.log", encoding="utf-8"),
    ],
)
# Silenciar logs de bajo nivel de módulos verbosos en la CLI
logging.getLogger("backend.indexing.pipeline").setLevel(logging.WARNING)
logging.getLogger("backend.retrieval.lsi_model").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — Orquestador completo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    p.add_argument("--db",    type=Path, default=DB_PATH,    help="Ruta a la base de datos SQLite.")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="Ruta al modelo LSI .pkl.")

    # Crawler
    g = p.add_argument_group("crawler")
    g.add_argument("--ids-per-discovery",  type=int,   default=500,   help="IDs por ciclo de discovery.")
    g.add_argument("--batch-size",         type=int,   default=50,    help="Metadatos por ciclo.")
    g.add_argument("--pdf-batch",          type=int,   default=10,     help="PDFs por ciclo.")
    g.add_argument("--discovery-interval", type=float, default=600.0, help="Segundos entre ciclos de discovery.")
    g.add_argument("--download-interval",  type=float, default=60.0,  help="Segundos entre ciclos de metadatos.")
    g.add_argument("--pdf-interval",       type=float, default=1.0,  help="Segundos entre ciclos de PDF.")

    # Indexing
    g = p.add_argument_group("indexing")
    g.add_argument("--pdf-threshold",    type=int,   default=10,   help="PDFs nuevos sin indexar para disparar indexación.")
    g.add_argument("--index-poll",       type=float, default=30.0, help="Segundos entre sondeos del watcher.")
    g.add_argument("--index-field",      type=str,   default="both",
                   choices=["full_text", "abstract", "both"],      help="Campo a indexar.")

    # LSI
    g = p.add_argument_group("lsi")
    g.add_argument("--lsi-interval", type=float, default=3600.0, help="Segundos entre rebuilds del modelo LSI.")
    g.add_argument("--lsi-k",        type=int,   default=100,    help="Componentes latentes del SVD.")
    g.add_argument("--lsi-min-docs", type=int,   default=10,     help="Mínimo de docs indexados para construir el modelo.")

    return p.parse_args()


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
        # indexing
        pdf_threshold       = args.pdf_threshold,
        index_poll_interval = args.index_poll,
        index_field         = args.index_field,
        # lsi
        lsi_rebuild_interval = args.lsi_interval,
        lsi_k                = args.lsi_k,
        lsi_min_docs         = args.lsi_min_docs,
    )

    orc = Orchestrator(cfg)
    orc.start()
    orc.run_cli()   # bloquea hasta 'quit'
    orc.stop()


if __name__ == "__main__":
    main()
