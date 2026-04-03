"""
crawler_repository.py
=====================
Operaciones de lectura/escritura en SQLite para el módulo de adquisición.

Diseño
------
- Cada función pública abre y cierra su propia conexión (stateless).
- Las operaciones bulk aceptan una conexión abierta opcional para
  agruparlas en una única transacción.
- Todos los inserts son idempotentes (INSERT OR IGNORE / ON CONFLICT DO UPDATE).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .schema import DB_PATH, get_connection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------------------------------------------------------------------
# Operaciones sobre documentos
# ---------------------------------------------------------------------------

def upsert_document(
    arxiv_id:   str,
    title:      str,
    authors:    str,
    abstract:   str,
    categories: str,
    published:  str,
    updated:    str,
    pdf_url:    str,
    fetched_at: str,
    db_path:    Path = DB_PATH,
) -> None:
    """
    Inserta o actualiza los metadatos de un documento.
    Los campos de contenido PDF y estado no se tocan en el UPDATE,
    por lo que llamar a esta función repetidamente es seguro.
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
    arxiv_id:  str,
    full_text: str,
    db_path:   Path = DB_PATH,
) -> None:
    """Persiste el texto extraído del PDF y marca el documento como descargado."""
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
    error:    str,
    db_path:  Path = DB_PATH,
) -> None:
    """Registra un intento fallido de descarga o extracción de PDF."""
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
    limit:   int  = 10,
    db_path: Path = DB_PATH,
) -> List[str]:
    """Devuelve IDs de documentos cuyo PDF aún no se ha descargado (estado 0)."""
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
    db_path:  Path = DB_PATH,
) -> Optional[sqlite3.Row]:
    """Devuelve una fila de documento o None si no existe."""
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
# Crawl log
# ---------------------------------------------------------------------------

def log_crawl_start(db_path: Path = DB_PATH) -> int:
    """Inserta una nueva fila en crawl_log y devuelve su id."""
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
    log_id:         int,
    ids_discovered: int  = 0,
    docs_downloaded: int = 0,
    pdfs_indexed:   int  = 0,
    errors:         int  = 0,
    notes:          str  = "",
    db_path:        Path = DB_PATH,
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
# Estadísticas
# ---------------------------------------------------------------------------

def get_stats(db_path: Path = DB_PATH) -> dict:
    from .chunk_repository import get_chunk_stats
    conn = get_connection(db_path)
    try:
        total   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1").fetchone()[0]
        errors  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0").fetchone()[0]
    finally:
        conn.close()
    return {
        "total_documents": total,
        "pdf_indexed":     indexed,
        "pdf_pending":     pending,
        "pdf_errors":      errors,
        **get_chunk_stats(db_path),
    }