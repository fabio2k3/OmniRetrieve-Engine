"""
pipeline.py
===========
Coordinador del módulo de indexación. Punto de entrada principal.

Combina TextPreprocessor + TFIDFIndexer y presenta:
  1-) La clase IndexingPipeline para uso programático desde otros módulos.
  2-) Una CLI completa para ejecución directa desde terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.indexing.preprocessor import TextPreprocessor
from backend.indexing.tfidf import TFIDFIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(self, db_path: Path = DB_PATH, field: str = "both", batch_size: int = 100, use_stemming: bool = False, min_token_len: int = 3,) -> None:
        self.preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            min_token_len=min_token_len,
        )
        self.indexer = TFIDFIndexer(
            db_path=db_path,
            preprocessor=self.preprocessor,
            field=field,
            batch_size=batch_size,
        )

    def run(self, reindex: bool = False, recalculate_idf: bool = True) -> dict:
        """
        Ejecuta el proceso de indexación completo.

        Parámetros:
            reindex (bool): si True borra el índice existente y reconstruye desde cero.
            recalculate_idf (bool): si True recalcula los valores IDF al finalizar
                                   (solo si se procesaron documentos).

        Retorna:
            dict: estadísticas devueltas por self.indexer.build() que describen
                  el trabajo realizado (por ejemplo 'docs_processed', etc.).

        Comportamiento:
            - Inicializa el esquema de la base de datos (indexer.init_schema()).
            - Construye el índice (indexer.build()).
            - Si recalculate_idf es True y se procesaron documentos, llama a
              indexer.recalculate_idf().
            - Emite logs informativos sobre el proceso.
        """
        log.info("=" * 60)
        log.info("OmniRetrieve — Módulo de Indexación TF-IDF")
        log.info("=" * 60)
        log.info(
            "DB: %s | Campo: %s | Lote: %d | Stemming: %s",
            self.indexer.db_path,
            self.indexer.field,
            self.indexer.batch_size,
            self.preprocessor.use_stemming,
        )
        log.info("-" * 60)

        self.indexer.init_schema()
        stats = self.indexer.build(reindex=reindex)

        if recalculate_idf and stats["docs_processed"] > 0:
            self.indexer.recalculate_idf()

        log.info("=" * 60)
        return stats

def _parse_args() -> argparse.Namespace:
    """
    Construye y parsea los argumentos de la línea de comandos.

    Retorna:
        argparse.Namespace con los valores de los argumentos:
            --db: Path a la base de datos.
            --field: campo a indexar ('full_text', 'abstract', 'both').
            --batch-size: tamaño de lote (N).
            --reindex: flag para reindexar desde cero.
            --stemming: flag para aplicar stemming.
            --min-len: longitud mínima de token.
            --no-recalc-idf: flag para omitir el recálculo de IDF al final.
    """
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Indexación TF-IDF",
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
        "--no-recalc-idf", action="store_true",
        help="Omitir el recalculo de IDF al final.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Punto de entrada de la CLI: valida argumentos y lanza la indexación.

    Comportamiento:
        - Llama a _parse_args() para obtener la configuración.
        - Verifica que la base de datos exista; si no existe, registra un error y
          finaliza con código de salida 1.
        - Crea una instancia de IndexingPipeline con los parámetros recibidos y
          ejecuta .run() con las opciones adecuadas (reindex y recalculate_idf).
    """
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
    ).run(
        reindex=args.reindex,
        recalculate_idf=not args.no_recalc_idf,
    )


if __name__ == "__main__":
    main()