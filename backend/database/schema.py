"""
schema.py
SQLite schema and database initialisation.

Tables
------
documents     — article metadata + extracted PDF text + indexing status
chunks        — paragraph-level text fragments (embedding column reserved)
crawl_log     — one row per crawler run, for monitoring
terms         — [indexing] vocabulary: one row per unique token
postings      — [indexing] inverted index: raw term frequencies per (term, document)
index_meta    — [indexing] key/value store for indexing audit metadata

Diseño del índice
-----------------
El indexador guarda ÚNICAMENTE frecuencias crudas (freq).
El cálculo de pesos (BM25, TF-IDF, etc.) es responsabilidad
del módulo recuperador, que puede aplicar el modelo que considere.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH  = DATA_DIR / "db" / "documents.db"

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Módulo de adquisición (crawler) ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT    NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    categories      TEXT,
    published       TEXT,
    updated         TEXT,
    pdf_url         TEXT,
    fetched_at      TEXT,
    full_text       TEXT,
    text_length     INTEGER,
    pdf_downloaded  INTEGER NOT NULL DEFAULT 0,
    indexed_at      TEXT,
    index_error     TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id        TEXT    NOT NULL
                        REFERENCES documents(arxiv_id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    char_count      INTEGER,
    embedding       BLOB,
    embedded_at     TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(arxiv_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS crawl_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    ids_discovered  INTEGER DEFAULT 0,
    docs_downloaded INTEGER DEFAULT 0,
    pdfs_indexed    INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    notes           TEXT
);

-- ── Módulo de indexación (frecuencias crudas) ──────────────────────────────
-- El indexador solo guarda frecuencias. Los pesos (BM25, TF-IDF, etc.)
-- los calcula el módulo recuperador según el modelo que implemente.

CREATE TABLE IF NOT EXISTS terms (
    term_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    word     TEXT    NOT NULL UNIQUE,
    df       INTEGER NOT NULL DEFAULT 0   -- document frequency: en cuántos docs aparece
);

CREATE TABLE IF NOT EXISTS postings (
    term_id  INTEGER NOT NULL REFERENCES terms(term_id)      ON DELETE CASCADE,
    doc_id   TEXT    NOT NULL REFERENCES documents(arxiv_id) ON DELETE CASCADE,
    freq     INTEGER NOT NULL DEFAULT 0,   -- frecuencia cruda del término en el doc
    PRIMARY KEY (term_id, doc_id)
);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ── Índices ────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_doc_categories   ON documents(categories);
CREATE INDEX IF NOT EXISTS idx_doc_published    ON documents(published);
CREATE INDEX IF NOT EXISTS idx_doc_pdf_status   ON documents(pdf_downloaded);
CREATE INDEX IF NOT EXISTS idx_chunks_arxiv     ON chunks(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded  ON chunks(embedded_at);
CREATE INDEX IF NOT EXISTS idx_postings_doc     ON postings(doc_id);
CREATE INDEX IF NOT EXISTS idx_postings_term    ON postings(term_id);
CREATE INDEX IF NOT EXISTS idx_terms_word       ON terms(word);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(_DDL)
        conn.commit()
    finally:
        conn.close()