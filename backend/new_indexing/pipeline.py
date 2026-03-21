"""
pipeline.py
===========
Coordinador del módulo de indexación BM25. Punto de entrada principal.

Combina TextPreprocessor + BM25Indexer y presenta:
  1-) La clase IndexingPipeline para uso programático desde otros módulos.
  2-) Una CLI completa para ejecución directa desde terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.new_indexing.preprocessor import TextPreprocessor
from backend.new_indexing.bm25 import BM25Indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class IndexingPipeline:
    """
    Orquestador del módulo de indexación BM25.

    Parámetros
    ----------
    db_path       : ruta a documents.db
    field         : "full_text" | "abstract" | "both"
    batch_size    : documentos procesados por lote
    use_stemming  : activa SnowballStemmer (requiere NLTK)
    min_token_len : longitud mínima de token
    k1            : parámetro BM25 de saturación de TF (default: 1.5)
    b             : parámetro BM25 de normalización por longitud (default: 0.75)
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        field: str = "both",
        batch_size: int = 100,
        use_stemming: bool = False,
        min_token_len: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            min_token_len=min_token_len,
        )
        self.indexer = BM25Indexer(
            db_path=db_path,
            preprocessor=self.preprocessor,
            field=field,
            batch_size=batch_size,
            k1=k1,
            b=b,
        )

    def run(self, reindex: bool = False) -> dict:
        """
        Ejecuta el pipeline BM25 completo.

        Parámetros
        ----------
        reindex : borra y reconstruye el índice desde cero
        """
        log.info("=" * 60)
        log.info("OmniRetrieve — Módulo de Indexación BM25")
        log.info("=" * 60)
        log.info(
            "DB: %s | Campo: %s | Lote: %d | Stemming: %s | k1=%.2f | b=%.2f",
            self.indexer.db_path,
            self.indexer.field,
            self.indexer.batch_size,
            self.preprocessor.use_stemming,
            self.indexer.k1,
            self.indexer.b,
        )
        log.info("-" * 60)

        self.indexer.init_schema()
        stats = self.indexer.build(reindex=reindex)

        log.info("=" * 60)
        return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Indexación BM25",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    parser.add_argument(
        "--field", choices=["full_text", "abstract", "both"], default="both",
        help="Campo a indexar. 'both' usa full_text con fallback a abstract.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, metavar="N",
        help="Número de documentos procesados por lote.",
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Borrar el índice existente y reconstruirlo desde cero.",
    )
    parser.add_argument(
        "--stemming", action="store_true",
        help="Aplicar stemming (SnowballStemmer). Requiere NLTK.",
    )
    parser.add_argument(
        "--min-len", type=int, default=3, metavar="N",
        help="Longitud mínima de token.",
    )
    parser.add_argument(
        "--k1", type=float, default=1.5,
        help="Parámetro BM25 de saturación de TF.",
    )
    parser.add_argument(
        "--b", type=float, default=0.75,
        help="Parámetro BM25 de normalización por longitud de documento.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.db.exists():
        log.error("Base de datos no encontrada: %s", args.db)
        sys.exit(1)

    IndexingPipeline(
        db_path=args.db,
        field=args.field,
        batch_size=args.batch_size,
        use_stemming=args.stemming,
        min_token_len=args.min_len,
        k1=args.k1,
        b=args.b,
    ).run(reindex=args.reindex)


if __name__ == "__main__":
    main()
