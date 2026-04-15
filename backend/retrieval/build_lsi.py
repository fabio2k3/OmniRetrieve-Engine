"""
build_lsi.py
============
Entrypoint del módulo de recuperación (fase offline).

Uso
---
    python -m backend.retrieval.build_lsi
    python -m backend.retrieval.build_lsi --k 200
    python -m backend.retrieval.build_lsi --k 150 --out data/models/custom.pkl
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent / "data" / "lsi_build.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)

from backend.database.schema import DB_PATH
from .lsi_model import LSIModel, MODEL_PATH


def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Construcción del índice LSI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k",        type=int,  default=100,    help="Dimensiones latentes (SVD)")
    p.add_argument("--max-docs", type=int,  default=None,   help="Límite de documentos (debug)")
    p.add_argument("--db",       type=Path, default=DB_PATH, help="Ruta a la base de datos")
    p.add_argument("--out",      type=Path, default=MODEL_PATH, help="Ruta de salida del modelo .pkl")
    args = p.parse_args()

    if not args.db.exists():
        log.error("Base de datos no encontrada: %s", args.db)
        sys.exit(1)

    log.info("=" * 55)
    log.info("OmniRetrieve — Construcción del índice LSI")
    log.info("k=%d  db=%s  out=%s", args.k, args.db, args.out)
    log.info("=" * 55)

    model = LSIModel(k=args.k)
    stats = model.build(db_path=args.db, max_docs=args.max_docs)
    model.save(path=args.out)

    log.info("-" * 55)
    log.info(
        "Completado: n_docs=%d  k=%d  varianza=%.2f%%  tiempo=%.1fs",
        stats["n_docs"], stats["k"],
        stats["var_explained"] * 100, stats["elapsed_s"],
    )
    log.info("=" * 55)


if __name__ == "__main__":
    main()
