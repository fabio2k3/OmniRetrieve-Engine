"""
chunk_repository.py
===================
Capa de acceso a datos para la tabla `chunks`.

Centraliza todas las operaciones SQL sobre chunks de forma que ningún
módulo externo al paquete database necesite escribir queries directamente.

Responsabilidades
-----------------
Escritura
    save_chunks()                — inserta/reemplaza chunks de un documento.
    save_chunk_embedding()       — persiste un embedding individual.
    save_chunk_embeddings_batch()— persiste un lote de embeddings.
    reset_embeddings()           — pone a NULL todos los embeddings (re-embed).

Lectura por documento
    get_chunks()                 — todos los chunks de un arxiv_id.

Lectura para el pipeline de embedding
    get_unembedded_chunks()      — lista de chunks sin embedding (fetch simple).
    get_unembedded_chunks_iter() — generador por lotes (corpus grande).
    get_all_embeddings_iter()    — generador de (id, embedding) para reconstruir FAISS.

Estadísticas
    get_chunk_count()            — total de chunks en la tabla.
    get_embedded_count()         — chunks que ya tienen embedding.
    get_chunk_stats()            — resumen {total, embedded, pending}.

Diseño
------
- Funciones standalone, stateless: cada una gestiona su propia conexión,
  salvo save_chunks() que acepta una conexión opcional para participar
  en transacciones externas.
- Inserts idempotentes: DELETE + INSERT en lugar de upsert para chunks,
  ya que el conjunto completo se reemplaza por documento.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .schema import DB_PATH, get_connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def save_chunks(
    arxiv_id: str,
    texts:    List[str],
    conn:     Optional[sqlite3.Connection] = None,
    db_path:  Path = DB_PATH,
) -> None:
    """
    Reemplaza todos los chunks de *arxiv_id* con los textos proporcionados.

    Elimina los chunks previos del documento y vuelve a insertar la lista
    completa. Asigna chunk_index secuencial empezando en 0.

    Parámetros
    ----------
    arxiv_id : identificador del documento.
    texts    : lista de strings, uno por chunk.
    conn     : conexión abierta opcional; si se pasa, el commit es
               responsabilidad del llamador (útil para transacciones mayores).
    db_path  : ruta a la BD (ignorado si se pasa conn).
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM chunks WHERE arxiv_id = ?", (arxiv_id,))
        now = _now()
        conn.executemany(
            """
            INSERT INTO chunks (arxiv_id, chunk_index, text, char_count, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(arxiv_id, i, text, len(text), now) for i, text in enumerate(texts)],
        )
        if own_conn:
            conn.commit()
    finally:
        if own_conn:
            conn.close()


def save_chunk_embedding(
    chunk_id:  int,
    embedding: bytes,
    db_path:   Path = DB_PATH,
) -> None:
    """
    Persiste el vector de embedding serializado para un único chunk.

    Parámetros
    ----------
    chunk_id  : id de la fila en chunks.
    embedding : numpy_array.astype('float32').tobytes()
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE chunks SET embedding = ?, embedded_at = ? WHERE id = ?",
            (embedding, _now(), chunk_id),
        )
        conn.commit()
    finally:
        conn.close()


def save_chunk_embeddings_batch(
    batch:   list[tuple[bytes, str, int]],
    db_path: Path = DB_PATH,
) -> int:
    """
    Persiste un lote de embeddings en la tabla chunks.

    Parámetros
    ----------
    batch : lista de (embedding_bytes, timestamp_iso, chunk_id).

    Devuelve
    --------
    Número de filas actualizadas.
    """
    if not batch:
        return 0
    conn = get_connection(db_path)
    try:
        conn.executemany(
            "UPDATE chunks SET embedding = ?, embedded_at = ? WHERE id = ?",
            batch,
        )
        conn.commit()
        return len(batch)
    finally:
        conn.close()


def reset_embeddings(db_path: Path = DB_PATH) -> int:
    """
    Pone a NULL los campos embedding y embedded_at de todos los chunks.

    Útil cuando se cambia de modelo y hay que re-vectorizar el corpus completo.

    Devuelve
    --------
    Número de filas afectadas.
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "UPDATE chunks SET embedding = NULL, embedded_at = NULL"
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lectura por documento
# ---------------------------------------------------------------------------

def get_chunks(
    arxiv_id: str,
    db_path:  Path = DB_PATH,
) -> List[sqlite3.Row]:
    """
    Devuelve todos los chunks de un documento ordenados por chunk_index.

    Parámetros
    ----------
    arxiv_id : identificador del documento.

    Devuelve
    --------
    Lista de sqlite3.Row con todas las columnas de la tabla chunks.
    Lista vacía si el documento no tiene chunks.
    """
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY chunk_index",
            (arxiv_id,),
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lectura para el pipeline de embedding
# ---------------------------------------------------------------------------

def get_unembedded_chunks(
    limit:   int  = 100,
    db_path: Path = DB_PATH,
) -> List[sqlite3.Row]:
    """
    Devuelve hasta *limit* chunks que aún no tienen embedding.

    Para corpus grandes, preferir get_unembedded_chunks_iter().

    Devuelve
    --------
    Lista de sqlite3.Row con columnas id, arxiv_id, chunk_index, text.
    """
    conn = get_connection(db_path)
    try:
        return conn.execute(
            """
            SELECT id, arxiv_id, chunk_index, text
            FROM   chunks
            WHERE  embedding IS NULL
            ORDER  BY id
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()


def get_unembedded_chunks_iter(
    batch_size: int  = 256,
    db_path:    Path = DB_PATH,
) -> Iterator[list[sqlite3.Row]]:
    """
    Generador que devuelve lotes de chunks sin embedding.

    Itera en orden de id, abriendo un cursor único hasta agotar los
    resultados. Adecuado para corpus con millones de chunks.

    Parámetros
    ----------
    batch_size : filas por lote.

    Yields
    ------
    list[sqlite3.Row] con columnas id, arxiv_id, chunk_index, text.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, arxiv_id, chunk_index, text
            FROM   chunks
            WHERE  embedding IS NULL
            ORDER  BY id
            """
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield rows
    finally:
        conn.close()


def get_all_embeddings_iter(
    batch_size: int  = 1_000,
    db_path:    Path = DB_PATH,
) -> Iterator[list[sqlite3.Row]]:
    """
    Generador que devuelve lotes de (id, embedding) de todos los chunks
    que ya tienen embedding, ordenados por id.

    Usado por FaissIndexManager.rebuild() para reconstruir el índice
    completo sin cargar todos los vectores en memoria a la vez.

    Yields
    ------
    list[sqlite3.Row] con columnas id, embedding (BLOB).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, embedding
            FROM   chunks
            WHERE  embedding IS NOT NULL
            ORDER  BY id
            """
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Estadísticas
# ---------------------------------------------------------------------------

def get_chunks_by_ids(
    chunk_ids: list[int],
    db_path:   Path = DB_PATH,
) -> list[dict]:
    """
    Devuelve texto y metadatos del documento para una lista de chunk_ids.

    Hace un JOIN con documents para incluir titulo, autores, abstract y url
    en una sola consulta. Usado por EmbeddingRetriever para enriquecer los
    resultados de busqueda de FAISS sin que el modulo retrieval escriba SQL.

    Parametros
    ----------
    chunk_ids : lista de IDs de la tabla chunks (devueltos por FAISS).

    Devuelve
    --------
    Lista de dicts con claves:
        chunk_id, arxiv_id, chunk_index, text, char_count,
        title, authors, abstract, pdf_url.
    El orden sigue el de chunk_ids.
    """
    if not chunk_ids:
        return []

    _CHUNK = 900
    conn = get_connection(db_path)
    try:
        rows = []
        for i in range(0, len(chunk_ids), _CHUNK):
            batch = chunk_ids[i : i + _CHUNK]
            ph = ",".join("?" * len(batch))
            rows.extend(conn.execute(
                f"""
                SELECT c.id        AS chunk_id,
                       c.arxiv_id,
                       c.chunk_index,
                       c.text,
                       c.char_count,
                       d.title,
                       d.authors,
                       d.abstract,
                       d.pdf_url
                FROM   chunks    c
                JOIN   documents d ON d.arxiv_id = c.arxiv_id
                WHERE  c.id IN ({ph})
                """,
                batch,
            ).fetchall())
    finally:
        conn.close()

    # Preservar el orden original de chunk_ids
    row_by_id = {r["chunk_id"]: dict(r) for r in rows}
    return [row_by_id[cid] for cid in chunk_ids if cid in row_by_id]


def get_chunk_count(db_path: Path = DB_PATH) -> int:
    """Devuelve el número total de chunks en la tabla."""
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()


def get_embedded_count(db_path: Path = DB_PATH) -> int:
    """Devuelve el número de chunks que ya tienen embedding almacenado."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()


def get_chunk_stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    """
    Devuelve un resumen del estado de los chunks.

    Claves
    ------
    total_chunks    : total de chunks en la tabla.
    embedded_chunks : chunks con embedding ya almacenado.
    pending_chunks  : chunks que aún no tienen embedding.
    """
    conn = get_connection(db_path)
    try:
        total    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "total_chunks":    total,
        "embedded_chunks": embedded,
        "pending_chunks":  total - embedded,
    }