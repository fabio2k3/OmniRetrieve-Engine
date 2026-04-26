"""
pipeline.py
===========
Punto de entrada del módulo embedding.

Responsabilidad única
---------------------
Coordinar la vectorización de chunks y la gestión del índice FAISS delegando
cada responsabilidad específica a los módulos internos:

    _sync.py   → sincronización FAISS-BD y reset de embeddings.
    _batch.py  → vectorización y persistencia de un lote.
    _meta.py   → registro de metadatos y logs de build.

Flujo de ``EmbeddingPipeline.run()``
--------------------------------------
1. Inicializar esquema de BD.
2. (Opcional) resetear embeddings previos si ``reembed=True``.
3. Cargar modelo de embedding e índice FAISS desde disco.
4. Sincronizar FAISS con BD (reconstruir si hay desincronización).
5. Procesar cada lote de chunks pendientes:
   a. Vectorizar y persistir en BD + FAISS.
   b. Disparar rebuild del índice si se alcanza el umbral.
6. Guardar el índice final en disco.
7. Persistir metadatos del run.
8. Devolver estadísticas.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.database.schema import DB_PATH, DATA_DIR
from backend.database.embedding_repository import (
    init_embedding_schema,
    save_embedding_meta,
)
from backend.database.chunk_repository import (
    get_unembedded_chunks_iter,
    get_chunk_stats,
    get_embedded_count,
)

from .embedder import ChunkEmbedder, DEFAULT_MODEL
from .faiss    import FaissIndexManager
from ._sync    import check_and_sync, reset_embeddings
from ._batch   import process_batch
from ._meta    import log_faiss_build, save_run_meta

log = logging.getLogger(__name__)

_FAISS_DIR   = DATA_DIR / "faiss"
_INDEX_PATH  = _FAISS_DIR / "index.faiss"
_ID_MAP_PATH = _FAISS_DIR / "id_map.npy"


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
    batch_size    : chunks procesados en cada lote.
    rebuild_every : chunks añadidos entre reconstrucciones completas del índice.
    nlist         : celdas Voronoi para IndexIVFPQ.
    m             : subvectores PQ (debe dividir a la dimensión del modelo).
    nbits         : bits por código PQ.
    nprobe        : celdas inspeccionadas durante la búsqueda.
    index_path    : ruta del fichero .faiss.
    id_map_path   : ruta del fichero .npy con el mapa chunk_id.
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
        self.device        = device
        self.batch_size    = batch_size
        self.rebuild_every = rebuild_every
        self._faiss_params = dict(
            nlist=nlist, m=m, nbits=nbits, nprobe=nprobe,
            rebuild_every=rebuild_every,
            index_path=index_path,
            id_map_path=id_map_path,
        )

    def run(self, reembed: bool = False) -> dict:
        """
        Ejecuta la vectorización de todos los chunks pendientes.

        Parámetros
        ----------
        reembed : si ``True``, resetea los embeddings existentes y re-vectoriza
                  todo el corpus desde cero (útil al cambiar de modelo).

        Returns
        -------
        dict con: ``chunks_processed``, ``chunks_skipped``, ``batches_processed``,
        ``rebuilds_triggered``, ``model_name``, ``started_at``, ``finished_at``.
        """
        stats = {
            "chunks_processed":   0,
            "chunks_skipped":     0,
            "batches_processed":  0,
            "rebuilds_triggered": 0,
            "model_name":         self.model_name,
            "started_at":         _now(),
        }

        log.info("=" * 60)
        log.info("OmniRetrieve — Módulo de Embedding")
        log.info("=" * 60)
        log.info(
            "DB: %s | Modelo: %s | Lote: %d | Rebuild cada: %d",
            self.db_path, self.model_name, self.batch_size, self.rebuild_every,
        )

        # 1. Esquema de BD
        init_embedding_schema(self.db_path)

        # 2. Reset opcional
        if reembed:
            reset_embeddings(self.db_path)

        # 3. Cargar modelo e índice
        embedder = ChunkEmbedder(
            model_name=self.model_name,
            device=self.device,
            batch_size=min(self.batch_size, 128),
        )
        faiss_mgr = FaissIndexManager(dim=embedder.dim, **self._faiss_params)
        faiss_mgr.load()

        # 4. Sincronización FAISS ↔ BD
        check_and_sync(faiss_mgr, get_embedded_count(self.db_path), self.db_path)
        if faiss_mgr.total_vectors > 0:
            log_faiss_build(faiss_mgr, self.model_name, self.db_path)

        save_embedding_meta("model_name", self.model_name, self.db_path)

        # Estado previo
        chunk_info    = get_chunk_stats(self.db_path)
        pending_total = chunk_info["pending_chunks"]
        log.info(
            "[Pipeline] BD — total: %d | embedidos: %d | pendientes: %d",
            chunk_info["total_chunks"], chunk_info["embedded_chunks"], pending_total,
        )

        if pending_total == 0:
            log.info("[Pipeline] Nada que procesar — todos los chunks ya están embedidos.")
        else:
            log.info("[Pipeline] Iniciando embedding de %d chunks en lotes de %d…",
                     pending_total, self.batch_size)

        # 5. Procesar lotes
        for batch_rows in get_unembedded_chunks_iter(
            batch_size=self.batch_size, db_path=self.db_path
        ):
            processed, skipped = process_batch(batch_rows, embedder, faiss_mgr, self.db_path)
            stats["chunks_processed"]  += processed
            stats["chunks_skipped"]    += skipped
            stats["batches_processed"] += 1

            pct = (stats["chunks_processed"] / pending_total * 100) if pending_total else 0
            log.info(
                "[Pipeline] Lote %d | +%d embedidos%s | %d/%d (%.1f%%) | FAISS: %d",
                stats["batches_processed"], processed,
                f" | omitidos: {skipped}" if skipped else "",
                stats["chunks_processed"], pending_total, pct,
                faiss_mgr.total_vectors,
            )

            if faiss_mgr.maybe_rebuild(self.db_path):
                stats["rebuilds_triggered"] += 1
                log_faiss_build(faiss_mgr, self.model_name, self.db_path)
                log.info(
                    "[Pipeline] Rebuild #%d — tipo: %s | vectores: %d",
                    stats["rebuilds_triggered"], faiss_mgr.index_type, faiss_mgr.total_vectors,
                )

        # 6. Guardado final
        if faiss_mgr.total_vectors > 0:
            log.info("[Pipeline] Guardando índice — tipo: %s | vectores: %d",
                     faiss_mgr.index_type, faiss_mgr.total_vectors)
            faiss_mgr.save()
            log_faiss_build(faiss_mgr, self.model_name, self.db_path)
        else:
            log.info("[Pipeline] No se generaron vectores; índice no guardado.")

        stats["finished_at"] = _now()
        save_run_meta(stats, self.db_path)

        log.info("=" * 60)
        log.info(
            "Embedding completado — chunks: %d | lotes: %d | rebuilds: %d",
            stats["chunks_processed"], stats["batches_processed"], stats["rebuilds_triggered"],
        )
        log.info("=" * 60)

        return stats
