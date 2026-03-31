"""
pipeline.py
===========
Coordinador del modulo de embeddings. Punto de entrada principal.

Responsabilidad unica
---------------------
Lee chunks de SQLite con embedded_at IS NULL, los codifica en lotes
con Embedder, los inserta en ChromaDB y marca cada chunk como
vectorizado actualizando embedded_at en SQLite.

Diseno
------
- Criterio canonico de pendiente: embedded_at IS NULL en SQLite.
  Simple, con indice, sin cargar IDs en RAM.
- Un chunk se considera vectorizado si y solo si embedded_at IS NOT NULL.
  Eso significa que su vector ya esta en ChromaDB.
- Atomicidad por chunk: primero se inserta en Chroma, luego se marca
  en SQLite. Si el proceso se interrumpe entre los dos pasos, el chunk
  se reintentara (upsert en Chroma es idempotente).
- Sin estado: cada llamada a run() es independiente e idempotente.
- Sigue el mismo patron que IndexingPipeline y LSIModel.build().

Uso
---
    python -m backend.embedding.pipeline
    python -m backend.embedding.pipeline --batch-size 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.database.schema import DB_PATH, get_connection
from backend.database.crawler_repository import (
    get_unembedded_chunks,
    mark_chunk_embedded,
)
from backend.embedding import chroma_store
from backend.embedding.chroma_store import CHROMA_PATH
from .embedder import Embedder

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingPipeline:
    """
    Genera y persiste embeddings en ChromaDB para los chunks pendientes.

    Un chunk es pendiente si y solo si su campo embedded_at en SQLite
    es NULL. Tras vectorizarlo, se actualiza embedded_at con el timestamp
    actual. Eso lo excluye de futuras ejecuciones del pipeline.

    Parametros
    ----------
    db_path     : ruta a la base de datos SQLite (fuente del texto).
    chroma_path : directorio de ChromaDB (destino de los vectores).
    embedder    : instancia de Embedder (lazy si no se pasa ninguna).
    batch_size  : chunks codificados por lote.
    """

    def __init__(
        self,
        db_path:     Path            = DB_PATH,
        chroma_path: Path            = CHROMA_PATH,
        embedder:    Embedder | None = None,
        batch_size:  int             = 64,
    ) -> None:
        self.db_path     = Path(db_path)
        self.chroma_path = Path(chroma_path)
        self.embedder    = embedder or Embedder()
        self.batch_size  = batch_size

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Procesa todos los chunks con embedded_at IS NULL.

        Flujo por lote
        --------------
        1. SELECT chunks WHERE embedded_at IS NULL LIMIT batch_size.
        2. Embedder.encode() -> vectores float32 normalizados.
        3. chroma_store.add_chunks() -> upsert en ChromaDB.
        4. mark_chunk_embedded() -> UPDATE embedded_at = now() en SQLite.
        5. Repetir hasta que no queden pendientes.

        Devuelve
        --------
        dict con: chunks_embedded, started_at, finished_at.
        """
        stats: dict = {
            "chunks_embedded": 0,
            "started_at":      _now(),
        }

        log.info("=" * 60)
        log.info("OmniRetrieve — Modulo de Embeddings (ChromaDB)")
        log.info("=" * 60)
        log.info(
            "SQLite: %s | ChromaDB: %s | Modelo: %s | Lote: %d",
            self.db_path, self.chroma_path,
            self.embedder.model_name, self.batch_size,
        )
        log.info("-" * 60)

        while True:
            rows = get_unembedded_chunks(
                limit   = self.batch_size,
                db_path = self.db_path,
            )
            if not rows:
                break

            texts       = [r["text"]        for r in rows]
            chunk_ids   = [r["id"]          for r in rows]
            arxiv_ids   = [r["arxiv_id"]    for r in rows]
            chunk_idxs  = [r["chunk_index"] for r in rows]
            char_counts = [r["char_count"] or 0 for r in rows]

            log.debug(
                "[EmbeddingPipeline] Codificando lote de %d chunks...",
                len(texts),
            )
            vectors = self.embedder.encode(texts, batch_size=self.batch_size)

            # 1. Insertar en ChromaDB (upsert: seguro ante reintento)
            chroma_store.add_chunks(
                chunk_ids   = chunk_ids,
                arxiv_ids   = arxiv_ids,
                chunk_idxs  = chunk_idxs,
                texts       = texts,
                char_counts = char_counts,
                embeddings  = vectors,
                chroma_path = self.chroma_path,
            )

            # 2. Marcar como embebidos en SQLite (embedded_at = now)
            for chunk_id in chunk_ids:
                mark_chunk_embedded(chunk_id, db_path=self.db_path)

            stats["chunks_embedded"] += len(rows)
            log.info(
                "[EmbeddingPipeline] %d chunks embebidos acumulados.",
                stats["chunks_embedded"],
            )

        stats["finished_at"] = _now()
        log.info("=" * 60)
        log.info(
            "[EmbeddingPipeline] Completado — %d chunks nuevos en ChromaDB.",
            stats["chunks_embedded"],
        )
        return stats

    def pending_count(self) -> int:
        """
        Devuelve el numero de chunks con embedded_at IS NULL.
        Criterio exacto: no depende de contar Chroma ni de conjuntos en RAM.
        """
        conn = get_connection(self.db_path)
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Modulo de Embeddings (ChromaDB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    parser.add_argument(
        "--chroma", type=Path, default=CHROMA_PATH,
        help="Directorio de ChromaDB.",
    )
    parser.add_argument(
        "--model", type=str, default="all-MiniLM-L6-v2",
        help="Nombre del modelo sentence-transformers.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N",
        help="Chunks procesados por lote.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.db.exists():
        log.error("Base de datos no encontrada: %s", args.db)
        sys.exit(1)

    EmbeddingPipeline(
        db_path     = args.db,
        chroma_path = args.chroma,
        embedder    = Embedder(model_name=args.model),
        batch_size  = args.batch_size,
    ).run()


if __name__ == "__main__":
    main()