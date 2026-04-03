"""
embed_chunks.py
===============
Embediza todos los chunks pendientes de la base de datos usando el pipeline
de embedding de OmniRetrieve-Engine.

Uso
---
    python -m backend.tools.embed_chunks
    python -m backend.tools.embed_chunks --model all-mpnet-base-v2 --m 16
    python -m backend.tools.embed_chunks --reembed
    python -m backend.tools.embed_chunks --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, DATA_DIR
from backend.embedding.pipeline import EmbeddingPipeline
from backend.embedding.embedder import DEFAULT_MODEL

_FAISS_DIR    = DATA_DIR / "faiss"
_INDEX_PATH   = _FAISS_DIR / "index.faiss"
_ID_MAP_PATH  = _FAISS_DIR / "id_map.npy"


def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Embediza todos los chunks pendientes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",
                   type=Path, default=DB_PATH,
                   help="Ruta a la base de datos SQLite.")
    p.add_argument("--model",
                   default=DEFAULT_MODEL,
                   help="Modelo sentence-transformers a usar.")
    p.add_argument("--device",
                   default=None,
                   help="Dispositivo de inferencia: cpu, cuda, mps.")
    p.add_argument("--batch-size",
                   type=int, default=256, metavar="N",
                   help="Chunks procesados por lote.")
    p.add_argument("--rebuild-every",
                   type=int, default=10_000, metavar="N",
                   help="Reconstruir el índice FAISS cada N chunks añadidos.")
    p.add_argument("--nlist",
                   type=int, default=100,
                   help="Celdas Voronoi para IndexIVFPQ.")
    p.add_argument("--m",
                   type=int, default=8,
                   help="Subvectores PQ (debe dividir la dimensión del modelo).")
    p.add_argument("--nbits",
                   type=int, default=8,
                   help="Bits por código PQ.")
    p.add_argument("--nprobe",
                   type=int, default=10,
                   help="Celdas inspeccionadas en búsqueda.")
    p.add_argument("--index-path",
                   type=Path, default=_INDEX_PATH,
                   help="Ruta del archivo .faiss.")
    p.add_argument("--id-map-path",
                   type=Path, default=_ID_MAP_PATH,
                   help="Ruta del archivo .npy con el mapa de IDs.")
    p.add_argument("--reembed",
                   action="store_true",
                   help="Resetea embeddings existentes y re-vectoriza todo.")
    p.add_argument("--stats",
                   action="store_true",
                   help="Muestra el estado actual y sale sin procesar nada.")
    args = p.parse_args()

    if not args.db.exists():
        print(f"\n  La base de datos no existe: {args.db}\n")
        sys.exit(1)

    if args.stats:
        from backend.database.embedding_repository import (
            init_embedding_schema,
            get_embedding_stats,
        )
        init_embedding_schema(args.db)
        s = get_embedding_stats(args.db)
        print(f"\n  Total chunks     : {s['total_chunks']:,}")
        print(f"  Embedidos        : {s['embedded_chunks']:,}")
        print(f"  Pendientes       : {s['pending_chunks']:,}")
        print(f"  Último modelo    : {s.get('last_index_type') or '—'}")
        print(f"  Última build     : {s.get('last_build_at') or '—'}")
        print(f"  Vectores FAISS   : {s.get('last_n_vectors', 0):,}\n")
        return

    EmbeddingPipeline(
        db_path=args.db,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        rebuild_every=args.rebuild_every,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        nprobe=args.nprobe,
        index_path=args.index_path,
        id_map_path=args.id_map_path,
    ).run(reembed=args.reembed)


if __name__ == "__main__":
    main()