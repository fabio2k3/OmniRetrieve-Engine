"""
main.py
=======
Entrypoint del módulo embedding.

Uso
---
    python -m backend.embedding
    python -m backend.embedding --model all-mpnet-base-v2 --batch-size 128
    python -m backend.embedding --reembed
    python -m backend.embedding --stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH, DATA_DIR
from .embedder  import DEFAULT_MODEL
from .pipeline  import EmbeddingPipeline
from ._meta     import print_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

_FAISS_DIR   = DATA_DIR / "faiss"
_INDEX_PATH  = _FAISS_DIR / "index.faiss"
_ID_MAP_PATH = _FAISS_DIR / "id_map.npy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db",           type=Path, default=DB_PATH,      help="Ruta a la BD SQLite.")
    parser.add_argument("--model",        default=DEFAULT_MODEL,            help="Modelo sentence-transformers.")
    parser.add_argument("--device",       default=None,                     help="cpu | cuda | mps | None (auto).")
    parser.add_argument("--batch-size",   type=int,  default=256,           help="Chunks por lote.")
    parser.add_argument("--rebuild-every",type=int,  default=10_000,        help="Reconstruir FAISS cada N chunks.")
    parser.add_argument("--nlist",        type=int,  default=100,           help="Celdas Voronoi para IVFPQ.")
    parser.add_argument("--m",            type=int,  default=8,             help="Subvectores PQ.")
    parser.add_argument("--nbits",        type=int,  default=8,             help="Bits por código PQ.")
    parser.add_argument("--nprobe",       type=int,  default=10,            help="Celdas inspeccionadas en búsqueda.")
    parser.add_argument("--index-path",   type=Path, default=_INDEX_PATH,   help="Ruta del fichero .faiss.")
    parser.add_argument("--id-map-path",  type=Path, default=_ID_MAP_PATH,  help="Ruta del fichero .npy de IDs.")
    parser.add_argument("--reembed",      action="store_true",              help="Resetear y re-vectorizar todo.")
    parser.add_argument("--stats",        action="store_true",              help="Mostrar estadísticas y salir.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.db.exists():
        logging.error("Base de datos no encontrada: %s", args.db)
        sys.exit(1)

    if args.stats:
        print_stats(args.db)
        return

    EmbeddingPipeline(
        db_path       = args.db,
        model_name    = args.model,
        device        = args.device,
        batch_size    = args.batch_size,
        rebuild_every = args.rebuild_every,
        nlist         = args.nlist,
        m             = args.m,
        nbits         = args.nbits,
        nprobe        = args.nprobe,
        index_path    = args.index_path,
        id_map_path   = args.id_map_path,
    ).run(reembed=args.reembed)


if __name__ == "__main__":
    main()
