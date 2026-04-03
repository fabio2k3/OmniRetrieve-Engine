"""
embedding_repository.py
=======================
Capa de acceso a datos para el módulo de embedding.

Responsabilidades
-----------------
- Extender el esquema con tablas propias (faiss_log, embedding_meta).
- Registrar cada reconstrucción del índice FAISS en faiss_log.
- Leer y escribir metadatos del módulo (modelo usado, última ejecución).

Las operaciones sobre la tabla chunks (leer chunks sin embedding, persistir
embeddings, leer embeddings para reconstruir el índice) se encuentran en
chunk_repository, que es el repositorio canónico de esa tabla.

Diseño
------
- Funciones standalone, stateless: cada una gestiona su propia conexión.
- Inserts idempotentes o ON CONFLICT.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import DB_PATH, get_connection
from .chunk_repository import get_chunk_stats

# ---------------------------------------------------------------------------
# DDL propio del módulo embedding
# ---------------------------------------------------------------------------

_EMBEDDING_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS faiss_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    built_at    TEXT    NOT NULL,
    n_vectors   INTEGER NOT NULL,
    index_type  TEXT    NOT NULL,      -- 'IndexIVFPQ' | 'IndexFlatL2'
    model_name  TEXT,
    nlist       INTEGER,
    m           INTEGER,
    nbits       INTEGER,
    index_path  TEXT,
    id_map_path TEXT,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS embedding_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_embedding_schema(db_path: Path = DB_PATH) -> None:
    """
    Crea las tablas faiss_log y embedding_meta si aún no existen.

    Llamar una vez al arrancar EmbeddingPipeline. Es idempotente.
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(_EMBEDDING_DDL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Registro de construcciones FAISS
# ---------------------------------------------------------------------------

def log_faiss_build(stats: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """
    Inserta una fila en faiss_log con los metadatos de la última construcción.

    Claves esperadas en stats
    -------------------------
    n_vectors, index_type, model_name, nlist, m, nbits,
    index_path, id_map_path, notes (todas opcionales excepto n_vectors).
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO faiss_log
                (built_at, n_vectors, index_type, model_name,
                 nlist, m, nbits, index_path, id_map_path, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                stats.get("n_vectors", 0),
                stats.get("index_type", "unknown"),
                stats.get("model_name"),
                stats.get("nlist"),
                stats.get("m"),
                stats.get("nbits"),
                stats.get("index_path"),
                stats.get("id_map_path"),
                stats.get("notes"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Metadatos del módulo embedding (clave/valor)
# ---------------------------------------------------------------------------

def save_embedding_meta(key: str, value: str, db_path: Path = DB_PATH) -> None:
    """Persiste o actualiza un par clave/valor en embedding_meta."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO embedding_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_embedding_meta(key: str, db_path: Path = DB_PATH) -> str | None:
    """Recupera el valor asociado a una clave en embedding_meta."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM embedding_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def get_embedding_stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    """
    Resumen del estado del módulo embedding.

    Claves devueltas
    ----------------
    total_chunks, embedded_chunks, pending_chunks,
    last_build_at, last_index_type, last_n_vectors.
    """
    conn = get_connection(db_path)
    try:
        last_log = conn.execute(
            "SELECT built_at, index_type, n_vectors FROM faiss_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    return {
        **get_chunk_stats(db_path),
        "last_build_at":   last_log["built_at"]    if last_log else None,
        "last_index_type": last_log["index_type"]  if last_log else None,
        "last_n_vectors":  last_log["n_vectors"]   if last_log else 0,
    }