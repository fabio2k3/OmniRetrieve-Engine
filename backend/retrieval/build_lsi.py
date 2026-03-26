"""
build_lsi.py
============
Entrypoint del Modulo C (fase offline).
Configura el logging, lee argumentos CLI, construye el modelo LSI
y lo guarda en data/models/lsi_model.pkl.

Uso:
    python -m backend.retrieval.build_lsi
    python -m backend.retrieval.build_lsi --k 200
    python -m backend.retrieval.build_lsi --k 100 --out custom.pkl
"""
import argparse
import logging
import sys
from pathlib import Path

# Configurar logging UNA VEZ en el entrypoint.
# Los modulos (LSIModel, etc.) usan getLogger(__name__) y heredan esto.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # pantalla
        logging.FileHandler(  # archivo
            Path(__file__).resolve().parent.parent /
            "data" / "lsi_build.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)  # "retrieval.build_lsi"

from .lsi_model import LSIModel, MODEL_PATH


def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve -- Construye el indice LSI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--k", type=int, default=100,
                   help="Numero de conceptos latentes")
    p.add_argument("--out", type=Path, default=MODEL_PATH,
                   help="Ruta de salida del modelo .pkl")
    args = p.parse_args()

    log.info("=" * 55)
    log.info("OmniRetrieve -- Construccion del indice LSI")
    log.info("k=%d salida=%s", args.k, args.out)
    log.info("=" * 55)

    model = LSIModel(k=args.k)
    stats = model.build()
    model.save(path=args.out)

    log.info("-" * 55)
    log.info("Completado: n_docs=%d k=%d varianza=%.2f%% tiempo=%.1fs",
             stats["n_docs"], stats["k"],
             stats["var_explained"] * 100, stats["elapsed_s"])
    log.info("=" * 55)


if __name__ == "__main__":
    main()
