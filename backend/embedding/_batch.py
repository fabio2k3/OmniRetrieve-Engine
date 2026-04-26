"""
_batch.py
=========
Procesamiento de lotes de chunks en el pipeline de embedding.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Responsabilidad única
---------------------
Recibir un lote de filas de la BD, vectorizarlas, serializar los resultados
y persistirlos tanto en SQLite como en el índice FAISS.
Ningún detalle de coordinación del pipeline (bucles, logs de progreso,
reconstrucción del índice) vive aquí.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embedder import ChunkEmbedder
from .faiss    import FaissIndexManager

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_batch(
    rows:      list,
    embedder:  ChunkEmbedder,
    faiss_mgr: FaissIndexManager,
    db_path:   Path,
) -> tuple[int, int]:
    """
    Vectoriza un lote de chunks y persiste los embeddings en BD e índice FAISS.

    Parámetros
    ----------
    rows      : lista de ``sqlite3.Row`` con columnas ``id``, ``arxiv_id``,
                ``chunk_index``, ``text``.
    embedder  : instancia de ``ChunkEmbedder`` ya inicializada.
    faiss_mgr : gestor del índice FAISS compartido.
    db_path   : ruta a la BD SQLite.

    Returns
    -------
    tuple[int, int]
        ``(n_processed, n_skipped)`` donde ``n_skipped`` cuenta los chunks
        con texto vacío que se omitieron sin vectorizar.
    """
    from backend.database.chunk_repository import save_chunk_embeddings_batch

    texts     = [row["text"] or "" for row in rows]
    chunk_ids = [row["id"] for row in rows]

    # Filtrar chunks con texto vacío
    valid_mask  = [bool(t.strip()) for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]
    valid_ids   = [cid for cid, ok in zip(chunk_ids, valid_mask) if ok]
    n_skipped   = sum(1 for ok in valid_mask if not ok)

    if not valid_texts:
        log.debug("[batch] Lote omitido — todos los chunks tienen texto vacío (%d).", n_skipped)
        return 0, n_skipped

    # Vectorizar
    vectors = embedder.encode(valid_texts)  # (N, dim) float32

    # Serializar y persistir en BD
    ts       = _now()
    db_batch = [
        (vec.astype(np.float32).tobytes(), ts, cid)
        for vec, cid in zip(vectors, valid_ids)
    ]
    save_chunk_embeddings_batch(db_batch, db_path)

    # Añadir al índice FAISS
    faiss_mgr.add(vectors, valid_ids)

    return len(valid_ids), n_skipped
