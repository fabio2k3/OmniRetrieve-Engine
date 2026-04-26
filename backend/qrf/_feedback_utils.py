"""
_feedback_utils.py
==================
Utilidades internas compartidas por los módulos de retroalimentación.

No forma parte de la API pública del módulo — usar los imports
desde brf.py, rocchio.py o mmr.py directamente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def get_embeddings_by_chunk_ids(
    chunk_ids: list[int],
    db_path:   Path,
) -> dict[int, np.ndarray]:
    """
    Recupera los vectores de embedding para una lista de chunk_ids desde la BD.

    Devuelve dict {chunk_id: vector float32 (dim,)}.
    Chunks sin embedding almacenado se omiten silenciosamente.

    Los vectores se leen desde la columna chunks.embedding (BLOB float32)
    en lugar de reconstruirse desde FAISS, evitando el error de aproximación
    de la cuantización PQ de IndexIVFPQ.
    """
    from backend.database.schema import get_connection

    if not chunk_ids:
        return {}

    ph   = ",".join("?" * len(chunk_ids))
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            f"SELECT id, embedding FROM chunks "
            f"WHERE id IN ({ph}) AND embedding IS NOT NULL",
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()

    return {
        row["id"]: np.frombuffer(row["embedding"], dtype=np.float32)
        for row in rows
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores 1-D."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normaliza un vector en L2. Devuelve el vector sin cambios si la norma es ~0."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-10 else v