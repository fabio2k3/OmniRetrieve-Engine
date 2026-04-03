"""
pipeline.py
===========
Punto de entrada del módulo embedding.

Combina ChunkEmbedder + FaissIndexManager + embedding_repository y expone:
  1. La clase EmbeddingPipeline para uso programático desde el orquestador.
  2. Una CLI completa para ejecución directa desde terminal.

Flujo principal (EmbeddingPipeline.run)
----------------------------------------
1. Inicializar esquema de la BD (tablas faiss_log, embedding_meta).
2. Intentar cargar índice FAISS desde disco (reanudación tras reinicio).
2b. Comprobar sincronización FAISS vs BD y reconstruir si hay vectores huérfanos.
3. Leer chunks sin embedding en lotes de `batch_size`.
4. Para cada lote:
   a. Vectorizar con ChunkEmbedder.encode().
   b. Serializar embeddings y persistir en la BD (save_chunk_embeddings_batch).
   c. Añadir vectores al índice FAISS (FaissIndexManager.add).
   d. Llamar a maybe_rebuild(): si se alcanzaron 10 000 chunks desde el
      último rebuild, reconstruir el índice completo y guardarlo en disco.
5. Al terminar todos los lotes, forzar un guardado final del índice.
6. Devolver estadísticas de la ejecución.

Uso
---
    python -m backend.embedding.pipeline
    python -m backend.embedding.pipeline --model all-mpnet-base-v2 --batch-size 128
    python -m backend.embedding.pipeline --reembed   # re-vectoriza todo
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.database.schema import DB_PATH, DATA_DIR
from backend.embedding.embedder import ChunkEmbedder, DEFAULT_MODEL
from backend.embedding.faiss_index import FaissIndexManager
from backend.database.embedding_repository import (
    init_embedding_schema,
    log_faiss_build,
    save_embedding_meta,
    get_embedding_stats,
)
from backend.database.chunk_repository import (
    get_unembedded_chunks_iter,
    save_chunk_embeddings_batch,
    reset_embeddings as _reset_embeddings_db,
    get_chunk_stats,
    get_embedded_count,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas por defecto del índice FAISS
# ---------------------------------------------------------------------------

_FAISS_DIR     = DATA_DIR / "faiss"
_INDEX_PATH    = _FAISS_DIR / "index.faiss"
_ID_MAP_PATH   = _FAISS_DIR / "id_map.npy"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddingPipeline:
    """
    Coordina la vectorización de chunks y la gestión del índice FAISS.

    Parámetros
    ----------
    db_path       : ruta a la BD SQLite.
    model_name    : nombre del modelo sentence-transformers.
    device        : dispositivo para inferencia ('cpu', 'cuda', None).
    batch_size    : chunks procesados en cada lote de embedding.
    rebuild_every : chunks añadidos entre reconstrucciones completas del índice.
    nlist         : celdas Voronoi para IndexIVFPQ.
    m             : subvectores PQ (debe dividir a la dimensión del modelo).
    nbits         : bits por código PQ.
    nprobe        : celdas inspeccionadas durante la búsqueda.
    index_path    : ruta del archivo .faiss.
    id_map_path   : ruta del archivo .npy con el mapa chunk_id.
    """

    def __init__(
        self,
        db_path:       Path = DB_PATH,
        model_name:    str  = DEFAULT_MODEL,
        device:        str | None = None,
        batch_size:    int  = 256,
        rebuild_every: int  = 10_000,
        nlist:         int  = 100,
        m:             int  = 8,
        nbits:         int  = 8,
        nprobe:        int  = 10,
        index_path:    Path = _INDEX_PATH,
        id_map_path:   Path = _ID_MAP_PATH,
    ) -> None:
        self.db_path       = Path(db_path)
        self.model_name    = model_name
        self.batch_size    = batch_size
        self.rebuild_every = rebuild_every
        self.index_path    = Path(index_path)
        self.id_map_path   = Path(id_map_path)

        # Inicialización perezosa: el modelo se carga en run()
        self._embedder: ChunkEmbedder | None = None
        self._faiss_mgr: FaissIndexManager | None = None

        self._faiss_params = dict(
            nlist=nlist, m=m, nbits=nbits, nprobe=nprobe,
            rebuild_every=rebuild_every,
            index_path=index_path,
            id_map_path=id_map_path,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, reembed: bool = False) -> dict:
        """
        Ejecuta la vectorización de todos los chunks pendientes.

        Parámetros
        ----------
        reembed : si True, resetea los embeddings existentes y re-vectoriza
                  todo el corpus desde cero (útil al cambiar de modelo).

        Devuelve
        --------
        dict con: chunks_processed, chunks_skipped, batches_processed,
                  rebuilds_triggered, model_name, started_at, finished_at.
        """
        stats = {
            "chunks_processed":  0,
            "chunks_skipped":    0,
            "batches_processed": 0,
            "rebuilds_triggered": 0,
            "model_name":        self.model_name,
            "started_at":        _now(),
        }

        log.info("=" * 60)
        log.info("OmniRetrieve — Módulo de Embedding")
        log.info("=" * 60)
        log.info(
            "DB: %s | Modelo: %s | Lote: %d | Rebuild cada: %d",
            self.db_path, self.model_name, self.batch_size, self.rebuild_every,
        )
        log.info("-" * 60)

        # 1. Esquema de BD
        init_embedding_schema(self.db_path)

        # 2. Opcional: resetear embeddings previos
        if reembed:
            self._reset_embeddings()

        # 3. Cargar modelo e índice
        self._embedder = ChunkEmbedder(
            model_name=self.model_name,
            device=None,
            batch_size=min(self.batch_size, 128),
        )
        self._faiss_mgr = FaissIndexManager(
            dim=self._embedder.dim,
            **self._faiss_params,
        )

        # Intentar cargar índice previo desde disco (reanudación)
        loaded = self._faiss_mgr.load()
        if not loaded:
            log.info("[Pipeline] No hay índice previo en disco; se creará uno nuevo.")

        # Sincronización: comprobar que el índice refleja todos los embeddings de la BD
        already_embedded = get_embedded_count(self.db_path)
        faiss_vectors    = self._faiss_mgr.total_vectors
        log.info(
            "[Pipeline] Sincronización — embeddings en BD: %d | vectores en FAISS: %d",
            already_embedded, faiss_vectors,
        )
        if already_embedded > 0 and faiss_vectors < already_embedded:
            log.warning(
                "[Pipeline] Desincronización detectada: faltan %d vectores en el índice FAISS.",
                already_embedded - faiss_vectors,
            )
            log.info("[Pipeline] Reconstruyendo índice FAISS desde la BD antes de continuar…")
            self._faiss_mgr.rebuild(self.db_path)
            self._log_faiss_build(self._faiss_mgr)
            log.info(
                "[Pipeline] Índice sincronizado — %d vectores listos.",
                self._faiss_mgr.total_vectors,
            )

        # Guardar nombre del modelo en metadata
        save_embedding_meta("model_name", self.model_name, self.db_path)

        # 4. Contar chunks pendientes antes de empezar
        chunk_info    = get_chunk_stats(self.db_path)
        pending_total = chunk_info["pending_chunks"]

        log.info(
            "[Pipeline] Estado BD — total: %d chunks | embedidos: %d | pendientes: %d",
            chunk_info["total_chunks"],
            chunk_info["embedded_chunks"],
            pending_total,
        )

        if pending_total == 0:
            log.info("[Pipeline] Nada que procesar — todos los chunks ya están embedidos.")
        else:
            log.info(
                "[Pipeline] Iniciando embedding de %d chunks en lotes de %d…",
                pending_total, self.batch_size,
            )

        # 5. Procesar chunks pendientes
        for batch_rows in get_unembedded_chunks_iter(
            batch_size=self.batch_size,
            db_path=self.db_path,
        ):
            processed, skipped = self._process_batch(batch_rows)
            stats["chunks_processed"]  += processed
            stats["chunks_skipped"]    += skipped
            stats["batches_processed"] += 1

            pct = (stats["chunks_processed"] / pending_total * 100) if pending_total else 0
            log.info(
                "[Pipeline] Lote %d | +%d embedidos%s | Progreso: %d/%d (%.1f%%) | FAISS: %d vectores",
                stats["batches_processed"],
                processed,
                f" | omitidos: {skipped}" if skipped else "",
                stats["chunks_processed"],
                pending_total,
                pct,
                self._faiss_mgr.total_vectors,
            )

            # ¿Reconstruir índice?
            if self._faiss_mgr.maybe_rebuild(self.db_path):
                stats["rebuilds_triggered"] += 1
                self._log_faiss_build(self._faiss_mgr)
                log.info(
                    "[Pipeline] Índice reconstruido (rebuild #%d) — tipo: %s | vectores: %d",
                    stats["rebuilds_triggered"],
                    self._faiss_mgr.index_type,
                    self._faiss_mgr.total_vectors,
                )

        # 6. Guardado final del índice
        if self._faiss_mgr.total_vectors > 0:
            log.info(
                "[Pipeline] Guardando índice en disco — tipo: %s | vectores: %d",
                self._faiss_mgr.index_type,
                self._faiss_mgr.total_vectors,
            )
            self._faiss_mgr.save()
            self._log_faiss_build(self._faiss_mgr)
            log.info("[Pipeline] Índice guardado correctamente.")
        else:
            log.info("[Pipeline] No se generaron vectores; índice no guardado.")

        stats["finished_at"] = _now()
        self._save_run_meta(stats)

        log.info("=" * 60)
        log.info(
            "Embedding completado — chunks: %d | lotes: %d | rebuilds: %d",
            stats["chunks_processed"],
            stats["batches_processed"],
            stats["rebuilds_triggered"],
        )
        log.info("=" * 60)

        return stats

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _process_batch(self, rows: list) -> tuple[int, int]:
        """
        Vectoriza un lote de rows y persiste los embeddings en la BD.

        Parámetros
        ----------
        rows : lista de sqlite3.Row con columnas id, arxiv_id, chunk_index, text.

        Devuelve
        --------
        (n_processed, n_skipped) — skipped cuando el texto está vacío.
        """
        texts     = [row["text"] or "" for row in rows]
        chunk_ids = [row["id"] for row in rows]

        # Filtrar chunks sin texto
        valid_mask  = [bool(t.strip()) for t in texts]
        valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
        valid_ids   = [cid for cid, ok in zip(chunk_ids, valid_mask) if ok]
        n_skipped   = sum(1 for ok in valid_mask if not ok)

        if not valid_texts:
            log.debug("[Pipeline] Lote omitido — todos los chunks tienen texto vacío (%d).", n_skipped)
            return 0, n_skipped

        # Vectorizar
        vectors = self._embedder.encode(valid_texts)  # (N, dim)

        # Serializar y persistir en BD
        ts    = _now()
        db_batch = [
            (vec.astype(np.float32).tobytes(), ts, cid)
            for vec, cid in zip(vectors, valid_ids)
        ]
        save_chunk_embeddings_batch(db_batch, self.db_path)

        # Añadir al índice FAISS
        self._faiss_mgr.add(vectors, valid_ids)

        return len(valid_ids), n_skipped

    def _reset_embeddings(self) -> None:
        """Pone a NULL todos los embeddings de la tabla chunks."""
        log.warning("[Pipeline] reembed=True — reseteando embeddings existentes…")
        n = _reset_embeddings_db(self.db_path)
        log.warning("[Pipeline] %d embeddings eliminados de la BD. Se re-vectorizará todo el corpus.", n)

    def _log_faiss_build(self, mgr: FaissIndexManager) -> None:
        """Registra la construcción del índice en faiss_log."""
        stats = mgr._build_stats(mgr.total_vectors)
        stats["model_name"] = self.model_name
        log_faiss_build(stats, self.db_path)

    def _save_run_meta(self, stats: dict) -> None:
        """Persiste metadatos del último run en embedding_meta."""
        save_embedding_meta("last_run_at",         stats["finished_at"],            self.db_path)
        save_embedding_meta("last_chunks_embedded", str(stats["chunks_processed"]), self.db_path)
        save_embedding_meta("last_model",           stats["model_name"],             self.db_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Nombre del modelo sentence-transformers.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Dispositivo de inferencia: cpu, cuda, mps. None = autodetección.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="N",
        help="Chunks procesados por lote.",
    )
    parser.add_argument(
        "--rebuild-every", type=int, default=10_000, metavar="N",
        help="Reconstruir el índice FAISS cada N chunks añadidos.",
    )
    parser.add_argument(
        "--nlist", type=int, default=100,
        help="Número de celdas Voronoi para IndexIVFPQ.",
    )
    parser.add_argument(
        "--m", type=int, default=8,
        help="Subvectores PQ (debe dividir a la dimensión del modelo).",
    )
    parser.add_argument(
        "--nbits", type=int, default=8,
        help="Bits por código PQ.",
    )
    parser.add_argument(
        "--nprobe", type=int, default=10,
        help="Celdas inspeccionadas en la búsqueda.",
    )
    parser.add_argument(
        "--index-path", type=Path, default=_INDEX_PATH,
        help="Ruta del archivo .faiss.",
    )
    parser.add_argument(
        "--id-map-path", type=Path, default=_ID_MAP_PATH,
        help="Ruta del archivo .npy con el mapa de IDs.",
    )
    parser.add_argument(
        "--reembed", action="store_true",
        help="Resetear embeddings existentes y re-vectorizar todo el corpus.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Mostrar estadísticas del estado actual y salir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.db.exists():
        log.error("Base de datos no encontrada: %s", args.db)
        sys.exit(1)

    if args.stats:
        _print_stats(args.db)
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


def _print_stats(db_path: Path) -> None:
    """Imprime un resumen del estado del módulo embedding."""
    try:
        init_embedding_schema(db_path)
        s = get_embedding_stats(db_path)
    except Exception as exc:
        log.error("Error al obtener estadísticas: %s", exc)
        sys.exit(1)

    print("\n── Embedding Stats ──────────────────────────")
    print(f"  Chunks totales     : {s['total_chunks']}")
    print(f"  Embedidos          : {s['embedded_chunks']}")
    print(f"  Pendientes         : {s['pending_chunks']}")
    print(f"  Último build FAISS : {s['last_build_at'] or 'nunca'}")
    print(f"  Tipo de índice     : {s['last_index_type'] or 'N/A'}")
    print(f"  Vectores en índice : {s['last_n_vectors']}")
    print("─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()