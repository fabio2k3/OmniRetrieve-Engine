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
        """
        Crea y devuelve una conexión a la base de datos SQLite.

        Configuraciones:
            - check_same_thread=False para permitir uso desde hilos diferentes.
            - row_factory=sqlite3.Row para acceder a columnas por nombre.

        Retorna:
            sqlite3.Connection listo para ejecutar consultas.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


    def get_index_stats(self) -> dict[str, Any]:
        """
        Recupera estadísticas básicas del índice TF-IDF.

        Retorna un diccionario con:
            - vocab_size: tamaño del vocabulario (nº de términos).
            - total_postings: número total de entradas en postings.
            - total_docs: número de documentos distintos en postings.
            - meta: diccionario con pares clave/valor desde la tabla index_meta.

        Asegura el cierre de la conexión en el bloque finally.
        """
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
        """
        Obtiene el vector TF-IDF de un documento identificado por arxiv_id.

        Parámetros:
            - arxiv_id (str): identificador del documento.

        Retorna:
            - dict mapeando palabra -> peso tfidf (float).

        Nota:
            La consulta ordena por tfidf descendente; el dict resultante no mantiene orden.
        """
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
        """
        Devuelve los 'n' términos más importantes de un documento.

        Parámetros:
            - arxiv_id (str): id del documento.
            - n (int): número máximo de términos a devolver (por defecto 20).

        Retorna:
            - lista de diccionarios con keys: 'word', 'tf', 'tfidf_weight'.
        """
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
        """
        Recupera todas las entradas (postings) asociadas a una palabra.

        Parámetros:
            - word (str): término a buscar en la tabla terms.

        Retorna:
            - lista de diccionarios con 'doc_id', 'tf' y 'tfidf_weight', ordenada por tfidf desc.
        """
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
        """
        Devuelve la frecuencia de documento (df) para un término dado.

        Parámetros:
            - word (str): término cuyo df se consulta.

        Retorna:
            - int con el df si existe, o None si el término no está en la tabla.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT df FROM terms WHERE word = ?", (word,)
            ).fetchone()
            return row["df"] if row else None
        finally:
            conn.close()


    def get_tfidf_matrix(self, max_docs: int | None = None) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Construye y devuelve la matriz TF-IDF completa (términos x documentos).

        Parámetros:
            - max_docs (int | None): si se proporciona, limita el número de documentos considerados.

        Retorna una tupla:
            - matrix: np.ndarray de forma (n_terms, n_docs) con dtype float32.
            - doc_ids: lista de ids de documentos (orden de columnas).
            - term_ids: lista de term_ids (orden de filas).
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


    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Devuelve estadísticas agregadas del corpus filtrado por pdf_downloaded = 1.

        Retorna un diccionario con:
            - total_docs, avg_chars, min_chars, max_chars, total_chars
            - top_categories: lista (hasta 10) de diccionarios {'categories', 'n'} ordenadas por frecuencia.
        """
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


    def search(self, query_tokens: list[str], top_k: int = 10) -> list[dict]:
        """
        Realiza una búsqueda simple por tokens: suma los pesos tfidf de los términos
        en cada documento para calcular una puntuación (score) y devuelve los top_k documentos.

        Parámetros:
            - query_tokens: lista de tokens/ palabras a buscar.
            - top_k: número máximo de resultados a devolver (por defecto 10).

        Retorna:
            - lista de diccionarios con 'doc_id', 'score' y 'title', ordenada por score descendente.

        Comportamiento:
            - Si query_tokens está vacío, devuelve lista vacía.
            - Usa una consulta SQL que suma los tfidf_weight de los postings para los términos dados.
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