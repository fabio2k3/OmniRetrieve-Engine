"""
index_repository.py
===================
Capa de acceso a datos para el índice invertido (módulos indexing y retrieval).

Responsabilidades
-----------------
- Escritura (usada por indexing/): persistir términos y postings con
  frecuencias crudas; guardar metadatos de la indexación.
- Lectura (usada por retrieval/): exponer los datos necesarios para
  construir la matriz de pesos sin aplicar ninguna fórmula TF-IDF.
  El cálculo de pesos es responsabilidad del módulo retrieval.

Diseño
------
- Funciones standalone, stateless: cada una abre y cierra su conexión.
- Inserts idempotentes (INSERT OR IGNORE / ON CONFLICT DO UPDATE).
- Ningún método de lectura calcula pesos; solo devuelve freq y df crudos.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .schema import DB_PATH, get_connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Escritura — usada por indexing/
# ---------------------------------------------------------------------------

def clear_index(db_path: Path = DB_PATH) -> None:
    """Borra completamente las tablas postings y terms (modo reindex)."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM postings")
        conn.execute("DELETE FROM terms")
        conn.commit()
    finally:
        conn.close()


def upsert_terms(
    df_map:  dict[str, int],
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    """
    Inserta términos nuevos y acumula el df de los ya existentes.

    Parámetros
    ----------
    df_map : {word: df_delta}
        Frecuencias de documento del lote actual.

    Devuelve
    --------
    dict {word: term_id} con todos los términos del df_map.
    """
    if not df_map:
        return {}

    words = list(df_map)
    placeholders = ",".join("?" * len(words))
    conn = get_connection(db_path)
    try:
        existing: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute(
                f"SELECT term_id, word FROM terms WHERE word IN ({placeholders})", words
            )
        }

        new_terms = [(w, df_map[w]) for w in words if w not in existing]
        if new_terms:
            conn.executemany(
                "INSERT OR IGNORE INTO terms (word, df) VALUES (?, ?)", new_terms
            )
            conn.commit()

        update_df = [(df_map[w], w) for w in words if w in existing]
        if update_df:
            conn.executemany(
                "UPDATE terms SET df = df + ? WHERE word = ?", update_df
            )
            conn.commit()

        all_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute(
                f"SELECT term_id, word FROM terms WHERE word IN ({placeholders})", words
            )
        }
        return all_terms
    finally:
        conn.close()


def flush_postings(
    batch:   list[tuple[int, str, int]],
    db_path: Path = DB_PATH,
) -> int:
    """
    Inserta o actualiza un lote de postings (term_id, doc_id, freq).

    Devuelve el número de registros procesados.
    """
    if not batch:
        return 0
    conn = get_connection(db_path)
    try:
        conn.executemany(
            """
            INSERT INTO postings (term_id, doc_id, freq)
            VALUES (?, ?, ?)
            ON CONFLICT(term_id, doc_id) DO UPDATE SET freq = excluded.freq
            """,
            batch,
        )
        conn.commit()
        return len(batch)
    finally:
        conn.close()


def save_index_meta(stats: dict, db_path: Path = DB_PATH) -> None:
    """Persiste metadatos de la última ejecución del indexador en index_meta."""
    conn = get_connection(db_path)
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [
                ("last_run_at",         stats.get("finished_at", _now())),
                ("last_docs_indexed",   str(stats.get("docs_processed", 0))),
                ("last_terms_added",    str(stats.get("terms_added",    0))),
                ("last_postings_added", str(stats.get("postings_added", 0))),
            ],
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lectura — generador de documentos (compartido por indexing/ y retrieval/)
# ---------------------------------------------------------------------------

def get_unindexed_documents(
    field:      str  = "full_text",
    batch_size: int  = 100,
    reindex:    bool = False,
    db_path:    Path = DB_PATH,
) -> Iterator[tuple[str, str]]:
    """
    Generador que devuelve (arxiv_id, texto) de documentos listos para indexar.

    Solo devuelve documentos con pdf_downloaded = 1, independientemente
    del campo elegido. Garantiza que el índice solo contiene documentos
    con contenido completo descargado por el crawler.

    Parámetros
    ----------
    field : 'full_text' | 'abstract' | 'both'
        Campo de texto a usar. Siempre requiere pdf_downloaded = 1.
    batch_size : int
        Filas a recuperar por fetchmany.
    reindex : bool
        True  -> todos los docs con PDF (ignora indexed_tfidf_at).
        False -> solo docs con PDF cuyo indexed_tfidf_at sea NULL.
    """
    if field == "abstract":
        text_expr = "COALESCE(abstract, '')"
    elif field == "both":
        text_expr = "COALESCE(full_text, abstract, '')"
    else:
        text_expr = "COALESCE(full_text, '')"

    where_indexed = (
        "1=1"
        if reindex
        else "indexed_tfidf_at IS NULL"
    )

    sql = f"""
        SELECT arxiv_id, {text_expr} AS texto
        FROM   documents
        WHERE  pdf_downloaded = 1
          AND  {where_indexed}
          AND  {text_expr} != ''
        ORDER  BY published DESC
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(sql)
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield row["arxiv_id"], row["texto"]
    finally:
        conn.close()


def mark_documents_indexed(
    arxiv_ids: list[str],
    db_path:   Path = DB_PATH,
) -> None:
    """
    Marca un conjunto de documentos como indexados (indexed_tfidf_at = now).

    Llamado por TFIndexer tras persistir los postings de cada lote.
    El watcher del orquestador y get_unindexed_documents no volverán
    a procesar estos documentos mientras indexed_tfidf_at no sea NULL.

    Parámetros
    ----------
    arxiv_ids : arxiv_ids que acaban de ser indexados.
    """
    if not arxiv_ids:
        return
    conn = get_connection(db_path)
    try:
        placeholders = ",".join("?" * len(arxiv_ids))
        conn.execute(
            f"UPDATE documents SET indexed_tfidf_at = ? "
            f"WHERE arxiv_id IN ({placeholders})",
            (_now(), *arxiv_ids),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lectura — estadísticas e inspección
# ---------------------------------------------------------------------------

def get_index_stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    """
    Devuelve estadísticas básicas del índice invertido.

    Claves: vocab_size, total_postings, total_docs, meta.
    """
    conn = get_connection(db_path)
    try:
        vocab_size     = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        total_postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        total_docs     = conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM postings"
        ).fetchone()[0]
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM index_meta")}
        return {
            "vocab_size":     vocab_size,
            "total_postings": total_postings,
            "total_docs":     total_docs,
            "meta":           meta,
        }
    finally:
        conn.close()


def get_top_terms(
    arxiv_id: str,
    n:        int  = 20,
    db_path:  Path = DB_PATH,
) -> list[dict]:
    """
    Devuelve los n términos con mayor frecuencia en un documento.

    Cada elemento: {word, freq, df}.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT t.word, p.freq, t.df
            FROM   postings p
            JOIN   terms    t ON t.term_id = p.term_id
            WHERE  p.doc_id = ?
            ORDER  BY p.freq DESC
            LIMIT  ?
            """,
            (arxiv_id, n),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_postings_for_term(
    word:    str,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """
    Devuelve todas las entradas de postings para una palabra.

    Cada elemento: {doc_id, freq}. Ordenado por freq desc.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT p.doc_id, p.freq
            FROM   postings p
            JOIN   terms    t ON t.term_id = p.term_id
            WHERE  t.word = ?
            ORDER  BY p.freq DESC
            """,
            (word,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lectura para retrieval/ — datos crudos para construir la matriz
# ---------------------------------------------------------------------------

def get_postings_for_matrix(
    max_docs: int | None = None,
    db_path:  Path       = DB_PATH,
) -> tuple[list[tuple[int, str, int]], dict[int, int], list[str], list[int], int]:
    """
    Devuelve todos los datos crudos necesarios para construir la matriz
    TF-IDF en el módulo retrieval, sin aplicar ninguna fórmula de pesado.

    Parámetros
    ----------
    max_docs : int | None
        Si se indica, limita el número de documentos.

    Devuelve (tupla de 5 elementos)
    --------------------------------
    postings  : list[(term_id, doc_id, freq)]
    df_map    : dict{term_id: df}
    doc_ids   : list[str]   — columnas de la matriz (orden estable)
    term_ids  : list[int]   — filas de la matriz (orden estable)
    n_docs    : int         — total de docs distintos en postings
    """
    conn = get_connection(db_path)
    try:
        n_docs: int = conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM postings"
        ).fetchone()[0]

        sql_docs = "SELECT DISTINCT doc_id FROM postings ORDER BY doc_id"
        if max_docs:
            sql_docs += f" LIMIT {max_docs}"
        doc_ids: list[str] = [r["doc_id"] for r in conn.execute(sql_docs)]

        term_ids: list[int] = [
            r["term_id"]
            for r in conn.execute("SELECT term_id FROM terms ORDER BY term_id")
        ]

        df_map: dict[int, int] = {
            r["term_id"]: r["df"]
            for r in conn.execute("SELECT term_id, df FROM terms")
        }

        if not doc_ids:
            return [], df_map, doc_ids, term_ids, n_docs

        placeholders = ",".join("?" * len(doc_ids))
        postings: list[tuple[int, str, int]] = [
            (r["term_id"], r["doc_id"], r["freq"])
            for r in conn.execute(
                f"SELECT term_id, doc_id, freq FROM postings "
                f"WHERE doc_id IN ({placeholders})",
                doc_ids,
            )
        ]
        return postings, df_map, doc_ids, term_ids, n_docs
    finally:
        conn.close()


def get_document_metadata(
    arxiv_ids: list[str] | None = None,
    db_path:   Path             = DB_PATH,
) -> dict[str, dict]:
    """
    Devuelve metadatos de documentos para mostrar en resultados de búsqueda.

    Parámetros
    ----------
    arxiv_ids : list[str] | None
        Si None, devuelve todos los docs con pdf_downloaded = 1.

    Devuelve
    --------
    dict {arxiv_id: {title, authors, abstract, pdf_url}}
    """
    conn = get_connection(db_path)
    try:
        if arxiv_ids:
            placeholders = ",".join("?" * len(arxiv_ids))
            rows = conn.execute(
                f"SELECT arxiv_id, title, authors, abstract, pdf_url "
                f"FROM documents WHERE arxiv_id IN ({placeholders})",
                arxiv_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT arxiv_id, title, authors, abstract, pdf_url "
                "FROM documents WHERE pdf_downloaded = 1"
            ).fetchall()
        return {r["arxiv_id"]: dict(r) for r in rows}
    finally:
        conn.close()