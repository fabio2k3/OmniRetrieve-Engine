"""
repository.py
All SQLite read/write operations for the SRI project.

Design
------
* Every public function opens and closes its own connection (stateless).
* Bulk operations accept an optional open connection to batch in one tx.
* All inserts are idempotent (INSERT OR IGNORE / ON CONFLICT DO UPDATE).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .schema import DB_PATH, get_connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------------

def upsert_document(
    arxiv_id: str,
    title: str,
    authors: str,
    abstract: str,
    categories: str,
    published: str,
    updated: str,
    pdf_url: str,
    fetched_at: str,
    db_path: Path = DB_PATH,
) -> None:
    """
    Insert or update a document's metadata.
    PDF content and status fields are NOT touched on update,
    so calling this repeatedly is safe.
    """
    sql = """
        INSERT INTO documents
            (arxiv_id, title, authors, abstract, categories,
             published, updated, pdf_url, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            title      = excluded.title,
            authors    = excluded.authors,
            abstract   = excluded.abstract,
            categories = excluded.categories,
            published  = excluded.published,
            updated    = excluded.updated,
            pdf_url    = excluded.pdf_url,
            fetched_at = excluded.fetched_at
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (arxiv_id, title, authors, abstract, categories,
                           published, updated, pdf_url, fetched_at))
        conn.commit()
    finally:
        conn.close()


def save_pdf_text(
    arxiv_id: str,
    full_text: str,
    db_path: Path = DB_PATH,
) -> None:
    """Persist extracted PDF text and mark as successfully indexed."""
    sql = """
        UPDATE documents
        SET full_text      = ?,
            text_length    = ?,
            pdf_downloaded = 1,
            indexed_at     = ?,
            index_error    = NULL
        WHERE arxiv_id = ?
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (full_text, len(full_text), _now(), arxiv_id))
        conn.commit()
    finally:
        conn.close()


def save_pdf_error(
    arxiv_id: str,
    error: str,
    db_path: Path = DB_PATH,
) -> None:
    """Record a failed PDF download/extraction attempt."""
    sql = """
        UPDATE documents
        SET pdf_downloaded = 2,
            indexed_at     = ?,
            index_error    = ?
        WHERE arxiv_id = ?
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (_now(), error, arxiv_id))
        conn.commit()
    finally:
        conn.close()


def get_pending_pdf_ids(
    limit: int = 10,
    db_path: Path = DB_PATH,
) -> List[str]:
    """Return IDs of documents whose PDF has not been downloaded yet (status=0)."""
    sql = """
        SELECT arxiv_id FROM documents
        WHERE pdf_downloaded = 0
        ORDER BY published DESC
        LIMIT ?
    """
    conn = get_connection(db_path)
    try:
        return [row["arxiv_id"] for row in conn.execute(sql, (limit,)).fetchall()]
    finally:
        conn.close()


def get_document(
    arxiv_id: str,
    db_path: Path = DB_PATH,
) -> Optional[sqlite3.Row]:
    """Return a single document row or None."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT * FROM documents WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    finally:
        conn.close()


def document_exists(arxiv_id: str, db_path: Path = DB_PATH) -> bool:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Chunk operations
# ---------------------------------------------------------------------------

def save_chunks(
    arxiv_id: str,
    texts: List[str],
    conn: Optional[sqlite3.Connection] = None,
    db_path: Path = DB_PATH,
) -> None:
    """
    Replace all chunks for *arxiv_id* with *texts*.
    Pass an open *conn* to include this in a larger transaction.
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
    chunk_id: int,
    embedding: bytes,
    db_path: Path = DB_PATH,
) -> None:
    """
    Persist a serialised embedding vector for a single chunk.
    Called by the embedder (Phase 2).
    embedding should be numpy_array.astype('float32').tobytes()
    """
    sql = """
        UPDATE chunks
        SET embedding   = ?,
            embedded_at = ?
        WHERE id = ?
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (embedding, _now(), chunk_id))
        conn.commit()
    finally:
        conn.close()


def get_chunks(
    arxiv_id: str,
    db_path: Path = DB_PATH,
) -> List[sqlite3.Row]:
    """Return all chunks for a document ordered by chunk_index."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT * FROM chunks WHERE arxiv_id = ? ORDER BY chunk_index",
            (arxiv_id,),
        ).fetchall()
    finally:
        conn.close()


def get_unembedded_chunks(
    limit: int = 100,
    db_path: Path = DB_PATH,
) -> List[sqlite3.Row]:
    """Return chunks that do not have an embedding yet (for the embedder)."""
    conn = get_connection(db_path)
    try:
        return conn.execute(
            "SELECT * FROM chunks WHERE embedding IS NULL LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Crawl log
# ---------------------------------------------------------------------------

def log_crawl_start(db_path: Path = DB_PATH) -> int:
    """Insert a new crawl_log row and return its id."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO crawl_log (started_at) VALUES (?)", (_now(),)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_crawl_end(
    log_id: int,
    ids_discovered: int = 0,
    docs_downloaded: int = 0,
    pdfs_indexed: int = 0,
    errors: int = 0,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> None:
    sql = """
        UPDATE crawl_log
        SET finished_at     = ?,
            ids_discovered  = ?,
            docs_downloaded = ?,
            pdfs_indexed    = ?,
            errors          = ?,
            notes           = ?
        WHERE id = ?
    """
    conn = get_connection(db_path)
    try:
        conn.execute(sql, (_now(), ids_discovered, docs_downloaded,
                           pdfs_indexed, errors, notes, log_id))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats(db_path: Path = DB_PATH) -> dict:
    conn = get_connection(db_path)
    try:
        total    = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1").fetchone()[0]
        errors   = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0").fetchone()[0]
        chunks   = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        return {
            "total_documents" : total,
            "pdf_indexed"     : indexed,
            "pdf_pending"     : pending,
            "pdf_errors"      : errors,
            "total_chunks"    : chunks,
            "embedded_chunks" : embedded,
        }
    finally:
        conn.close()