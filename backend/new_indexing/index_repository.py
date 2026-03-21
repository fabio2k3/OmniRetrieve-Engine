"""
index_repository.py
===================
Capa de acceso a datos para el índice BM25.

"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from backend.database.schema import DB_PATH


class IndexRepository:
    """Interfaz de consulta sobre el índice BM25 almacenado en la BD."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    # Conexión 
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # Estadísticas del índice 
    def get_index_stats(self) -> dict[str, Any]:
        """Devuelve métricas generales del índice."""
        conn = self._connect()
        try:
            vocab_size     = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            total_postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
            total_docs     = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM postings"
            ).fetchone()[0]
            meta = {
                r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM index_meta")
            }
            return {
                "vocab_size":     vocab_size,
                "total_postings": total_postings,
                "total_docs":     total_docs,
                "meta":           meta,
            }
        finally:
            conn.close()

    # Consultas por documento 
    def get_document_vector(self, arxiv_id: str) -> dict[str, float]:
        """Devuelve el vector BM25 de un documento como {término: peso}."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT t.word, p.tfidf_weight AS bm25_score
                FROM   postings p
                JOIN   terms    t ON t.term_id = p.term_id
                WHERE  p.doc_id = ?
                ORDER  BY p.tfidf_weight DESC
                """,
                (arxiv_id,),
            ).fetchall()
            return {r["word"]: r["bm25_score"] for r in rows}
        finally:
            conn.close()

    def get_top_terms(self, arxiv_id: str, n: int = 20) -> list[dict]:
        """Devuelve los N términos más relevantes de un documento por BM25."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT t.word, p.tf, p.tfidf_weight AS bm25_score
                FROM   postings p
                JOIN   terms    t ON t.term_id = p.term_id
                WHERE  p.doc_id = ?
                ORDER  BY p.tfidf_weight DESC
                LIMIT  ?
                """,
                (arxiv_id, n),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # Consultas por término 
    def get_postings_for_term(self, word: str) -> list[dict]:
        """
        Devuelve la posting list de un término:
        [{doc_id, tf, bm25_score}, …] ordenada por peso descendente.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT p.doc_id, p.tf, p.tfidf_weight AS bm25_score
                FROM   postings p
                JOIN   terms    t ON t.term_id = p.term_id
                WHERE  t.word = ?
                ORDER  BY p.tfidf_weight DESC
                """,
                (word,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_term_df(self, word: str) -> int | None:
        """Devuelve el document frequency de un término, o None si no existe."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT df FROM terms WHERE word = ?", (word,)
            ).fetchone()
            return row["df"] if row else None
        finally:
            conn.close()

    # Exportación de la matriz BM25 (para LSI / SVD) 
    def get_tfidf_matrix(
        self, max_docs: int | None = None
    ) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Construye y devuelve la matriz BM25 lista para SVD.

        El nombre get_tfidf_matrix se mantiene por compatibilidad
        con el módulo LSI de Alina — misma interfaz, mejores pesos.

        Retorna
        -------
        matrix   : np.ndarray de forma (n_terms × n_docs)
        doc_ids  : lista de arxiv_id en el orden de las columnas
        term_ids : lista de term_id en el orden de las filas
        """
        conn = self._connect()
        try:
            sql_docs = "SELECT DISTINCT doc_id FROM postings ORDER BY doc_id"
            if max_docs:
                sql_docs += f" LIMIT {max_docs}"
            doc_ids: list[str]  = [r["doc_id"]  for r in conn.execute(sql_docs)]
            term_ids: list[int] = [
                r["term_id"]
                for r in conn.execute("SELECT term_id FROM terms ORDER BY term_id")
            ]

            doc_idx  = {d: i for i, d in enumerate(doc_ids)}
            term_idx = {t: i for i, t in enumerate(term_ids)}
            matrix   = np.zeros((len(term_ids), len(doc_ids)), dtype=np.float32)

            placeholders = ",".join("?" * len(doc_ids))
            for row in conn.execute(
                f"SELECT term_id, doc_id, tfidf_weight FROM postings "
                f"WHERE doc_id IN ({placeholders})",
                doc_ids,
            ):
                ti = term_idx.get(row["term_id"])
                di = doc_idx.get(row["doc_id"])
                if ti is not None and di is not None:
                    matrix[ti, di] = row["tfidf_weight"]

            return matrix, doc_ids, term_ids
        finally:
            conn.close()

    # Estadísticas del corpus
    def get_corpus_stats(self) -> dict[str, Any]:
        """Estadísticas del corpus para la documentación del Corte 1."""
        conn = self._connect()
        try:
            corpus = dict(conn.execute("""
                SELECT
                    COUNT(*)         AS total_docs,
                    AVG(text_length) AS avg_chars,
                    MIN(text_length) AS min_chars,
                    MAX(text_length) AS max_chars,
                    SUM(text_length) AS total_chars
                FROM documents
                WHERE pdf_downloaded = 1
            """).fetchone())

            categories = conn.execute("""
                SELECT categories, COUNT(*) AS n
                FROM documents
                WHERE pdf_downloaded = 1
                GROUP BY categories
                ORDER BY n DESC
                LIMIT 10
            """).fetchall()

            corpus["top_categories"] = [dict(r) for r in categories]
            return corpus
        finally:
            conn.close()

    # Búsqueda por consulta 
    def search(self, query_tokens: list[str], top_k: int = 10) -> list[dict]:
        """
        Búsqueda BM25: suma de pesos BM25 de los tokens de la consulta.
        Devuelve los top_k documentos con mayor puntuación.
        """
        if not query_tokens:
            return []

        placeholders = ",".join("?" * len(query_tokens))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT p.doc_id,
                       SUM(p.tfidf_weight) AS score,
                       d.title,
                       d.categories,
                       d.published
                FROM   postings  p
                JOIN   terms     t ON t.term_id  = p.term_id
                JOIN   documents d ON d.arxiv_id = p.doc_id
                WHERE  t.word IN ({placeholders})
                GROUP  BY p.doc_id
                ORDER  BY score DESC
                LIMIT  ?
                """,
                (*query_tokens, top_k),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()