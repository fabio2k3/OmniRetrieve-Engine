"""
index_repository.py
=============
Capa de acceso a datos para el índice TF-IDF.

Proveer consultas de solo-lectura limpias sobre las
tablas `terms` y `postings` para que otros módulos (LSI/SVD, recuperación,
evaluación) puedan operar sin escribir SQL directamente.

"""

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any
import numpy as np
from backend.database.schema import DB_PATH


class IndexRepository:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get_index_stats(self) -> dict[str, Any]:
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


    def get_document_vector(self, arxiv_id: str) -> dict[str, float]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT t.word, p.tfidf_weight
                FROM   postings p
                JOIN   terms    t ON t.term_id = p.term_id
                WHERE  p.doc_id = ?
                ORDER  BY p.tfidf_weight DESC
                """,
                (arxiv_id,),
            ).fetchall()
            return {r["word"]: r["tfidf_weight"] for r in rows}
        finally:
            conn.close()

    def get_top_terms(self, arxiv_id: str, n: int = 20) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT t.word, p.tf, p.tfidf_weight
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

    def get_postings_for_term(self, word: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT p.doc_id, p.tf, p.tfidf_weight
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
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT df FROM terms WHERE word = ?", (word,)
            ).fetchone()
            return row["df"] if row else None
        finally:
            conn.close()

    def get_tfidf_matrix(
        self, max_docs: int | None = None
    ) -> tuple[np.ndarray, list[str], list[int]]:
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

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[dict]:
        if not query_tokens:
            return []

        placeholders = ",".join("?" * len(query_tokens))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT p.doc_id,
                       SUM(p.tfidf_weight) AS score,
                       d.title
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