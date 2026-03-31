"""
main.py
=======
Entrypoint del orchestrador OmniRetrieve-Engine.

Arranca el sistema completo: crawler + indexing + LSI + embeddings
en hilos de fondo, y una CLI interactiva en el hilo principal.

OrchestratorConfig es la unica fuente de verdad para los valores
por defecto. Los argumentos de linea de comandos solo sobreescriben
lo que el usuario pasa explicitamente; si no pasa nada, se usa el
valor que ya tiene OrchestratorConfig en su dataclass.

Uso
---
    # arrancar con todos los defaults
    python -m backend.orchestrator

    # sobreescribir solo lo que interesa
    python -m backend.orchestrator --lsi-k 50 --embed-model BAAI/bge-small-en-v1.5
    python -m backend.orchestrator --pdf-threshold 3 --lsi-interval 1800
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import OrchestratorConfig
from .orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Logging global — antes de instanciar cualquier cosa
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_DATA_DIR / "orchestrator.log", encoding="utf-8"),
    ],
)
# Silenciar modulos muy verbosos que no aportan info util en la CLI
logging.getLogger("backend.indexing.pipeline").setLevel(logging.WARNING)
logging.getLogger("backend.retrieval.lsi_model").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argparse — defaults tomados directamente de OrchestratorConfig
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Construye el parser usando OrchestratorConfig como unica fuente de
    defaults. Asi no hay valores duplicados ni desincronizados.
    """
    _d = OrchestratorConfig()   # instancia solo para leer defaults

    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — Orquestador completo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Rutas ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("rutas")
    g.add_argument("--db",     type=Path, default=_d.db_path,
                   help="Base de datos SQLite.")
    g.add_argument("--model",  type=Path, default=_d.model_path,
                   help="Modelo LSI (.pkl).")
    g.add_argument("--chroma", type=Path, default=_d.chroma_path,
                   help="Directorio de ChromaDB.")

    # ── Crawler ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("crawler")
    g.add_argument("--ids-per-discovery",  type=int,   default=_d.ids_per_discovery,
                   help="IDs descubiertos por ciclo.")
    g.add_argument("--batch-size",         type=int,   default=_d.batch_size,
                   help="Metadatos descargados por ciclo.")
    g.add_argument("--pdf-batch",          type=int,   default=_d.pdf_batch_size,
                   help="PDFs descargados por ciclo.")
    g.add_argument("--discovery-interval", type=float, default=_d.discovery_interval,
                   help="Segundos entre ciclos de discovery.")
    g.add_argument("--download-interval",  type=float, default=_d.download_interval,
                   help="Segundos entre descargas de metadatos.")
    g.add_argument("--pdf-interval",       type=float, default=_d.pdf_interval,
                   help="Segundos entre descargas de PDF.")

    # ── Indexing ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("indexing")
    g.add_argument("--pdf-threshold",  type=int,   default=_d.pdf_threshold,
                   help="PDFs sin indexar para disparar IndexingPipeline.")
    g.add_argument("--index-poll",     type=float, default=_d.index_poll_interval,
                   help="Segundos entre sondeos del watcher de indexing.")
    g.add_argument("--index-field",    type=str,   default=_d.index_field,
                   choices=["full_text", "abstract", "both"],
                   help="Campo de texto a indexar.")

    # ── LSI ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("lsi")
    g.add_argument("--lsi-interval", type=float, default=_d.lsi_rebuild_interval,
                   help="Segundos entre rebuilds del modelo LSI.")
    g.add_argument("--lsi-k",        type=int,   default=_d.lsi_k,
                   help="Componentes latentes del SVD.")
    g.add_argument("--lsi-min-docs", type=int,   default=_d.lsi_min_docs,
                   help="Minimo de docs indexados para construir LSI.")

    # ── Embeddings ───────────────────────────────────────────────────────────
    g = p.add_argument_group("embeddings")
    g.add_argument("--embed-model",     type=str,   default=_d.embed_model,
                   help="Modelo sentence-transformers para embeddings.")
    g.add_argument("--embed-batch",     type=int,   default=_d.embed_batch_size,
                   help="Chunks codificados por lote.")
    g.add_argument("--embed-poll",      type=float, default=_d.embed_poll_interval,
                   help="Segundos entre sondeos del watcher de embeddings.")
    g.add_argument("--embed-threshold", type=int,   default=_d.embed_threshold,
                   help="Chunks sin embeber para disparar EmbeddingPipeline.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    cfg = OrchestratorConfig(
        # rutas
        db_path     = args.db,
        model_path  = args.model,
        chroma_path = args.chroma,
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
        # embeddings
        embed_model         = args.embed_model,
        embed_batch_size    = args.embed_batch,
        embed_poll_interval = args.embed_poll,
        embed_threshold     = args.embed_threshold,
    )

    orc = Orchestrator(cfg)
    orc.start()
    orc.run_cli()   # bloquea en el hilo principal hasta 'quit'
    orc.stop()


if __name__ == "__main__":
    main()