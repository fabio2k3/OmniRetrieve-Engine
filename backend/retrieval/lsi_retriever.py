"""
lsi_retriever.py
================
Fase online del módulo LSI.

Recibe una query en texto libre, la convierte a un vector TF-IDF
en el vocabulario del modelo, la proyecta al espacio latente y devuelve
los K artículos más relevantes por similitud coseno.
"""

from __future__ import annotations
import logging
import math
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.database.schema import DB_PATH, get_connection
from backend.indexing.preprocessor import TextPreprocessor
from .lsi_model import LSIModel, MODEL_PATH

log = logging.getLogger(__name__)


class LSIRetriever:
    """
    Motor de recuperación LSI. Carga el modelo y responde consultas.

    Uso
    ---
        retriever = LSIRetriever()
        retriever.load()
        results = retriever.retrieve("attention transformer mechanisms", top_n=10)
    """

    def __init__(self, model: LSIModel | None = None) -> None:
        self.model        = model or LSIModel()
        self._meta:       dict[str, dict] | None = None
        # word → (row_idx, df)  — construido al cargar el modelo
        self._word_index: dict[str, tuple[int, int]] = {}
        self._preprocessor = TextPreprocessor()

    def load(self, model_path: Path = MODEL_PATH, db_path: Path = DB_PATH) -> None:
        """
        Carga el modelo .pkl y prepara las estructuras de consulta.

        Parámetros
        ----------
        model_path : ruta al .pkl generado en la fase offline.
        db_path    : ruta a la BD para cargar metadatos y vocabulario.
        """
        self.model.load(model_path)
        self._word_index = self._build_word_index(db_path)
        self._meta       = self._load_meta(db_path)
        log.info(
            "[LSIRetriever] Listo. %d documentos | %d términos en vocabulario.",
            len(self.model.doc_ids), len(self._word_index),
        )

    def _build_word_index(self, db_path: Path) -> dict[str, tuple[int, int]]:
        """
        Construye el mapa word → (row_idx, df) para vectorizar queries.

        row_idx es la posición del term_id en self.model.term_ids,
        que es exactamente el orden de filas de la matriz usada en build().
        df es la frecuencia de documento del corpus, guardada en df_map.
        """
        if not self.model.term_ids:
            return {}

        # SQLite tiene un límite de 999 variables por consulta (SQLITE_LIMIT_VARIABLE_NUMBER).
        # Si el vocabulario es grande, la consulta se parte en lotes para no superarlo.
        _CHUNK_SIZE = 900
        term_ids = self.model.term_ids
        rows = []
        conn = get_connection(db_path)
        try:
            for offset in range(0, len(term_ids), _CHUNK_SIZE):
                chunk = term_ids[offset : offset + _CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                rows.extend(
                    conn.execute(
                        f"SELECT term_id, word FROM terms WHERE term_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
        finally:
            conn.close()

        term_id_to_row = {tid: i for i, tid in enumerate(self.model.term_ids)}
        index: dict[str, tuple[int, int]] = {}
        for r in rows:
            tid = r["term_id"]
            row_idx = term_id_to_row.get(tid)
            df      = self.model.df_map.get(tid, 1)
            if row_idx is not None:
                index[r["word"]] = (row_idx, df)
        return index

    def _vectorize_query(self, query: str) -> np.ndarray:
        """
        Convierte la query a un vector TF-IDF alineado con el vocabulario
        del modelo (mismo orden de filas que la matriz usada en build()).

        Usa TextPreprocessor para tokenizar de forma consistente con
        la fase de indexación.

        Para cada token presente en el vocabulario:
            TF  = log(1 + freq_en_query)
            IDF = log((N + 1) / (df_corpus + 1))   ← df real del corpus
        """
        if self.model.docs_latent is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        n_terms = len(self.model.term_ids)
        n_docs  = len(self.model.doc_ids)
        vec     = np.zeros(n_terms, dtype=np.float32)

        tokens = self._preprocessor.process(query)
        # contar frecuencias en la query
        freq_in_query: dict[str, int] = {}
        for t in tokens:
            freq_in_query[t] = freq_in_query.get(t, 0) + 1

        for token, freq in freq_in_query.items():
            entry = self._word_index.get(token)
            if entry is None:
                continue
            row_idx, df = entry
            tf  = math.log(1 + freq)
            idf = math.log((n_docs + 1) / (df + 1))
            vec[row_idx] = tf * idf

        return vec

    def retrieve(self, query: str, top_n: int = 10) -> list[dict]:
        """
        Devuelve los top_n documentos más relevantes para la query.

        Pasos
        -----
        1. Tokeniza y vectoriza la query en el espacio TF-IDF del vocabulario.
        2. Proyecta al espacio latente con model.project_query().
        3. Calcula similitud coseno contra todos los documentos indexados.
        4. Devuelve los top_n con metadatos (title, authors, abstract, url).

        Lanza RuntimeError si el modelo no ha sido cargado.
        """
        if self.model.docs_latent is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        t0 = time.monotonic()

        q_tfidf  = self._vectorize_query(query)
        q_latent = self.model.project_query(q_tfidf)
        scores   = cosine_similarity(
            q_latent.reshape(1, -1), self.model.docs_latent
        ).flatten()
        top_idx  = np.argsort(scores)[::-1][:top_n]

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.debug(
            "[LSIRetriever] query='%s…' top_score=%.4f tiempo=%.1fms",
            query[:40], scores[top_idx[0]] if len(top_idx) else 0.0, elapsed_ms,
        )

        results = []
        for i in top_idx:
            aid      = self.model.doc_ids[i]
            meta     = self._meta.get(aid, {}) if self._meta else {}
            abstract = meta.get("abstract") or ""
            results.append({
                "score":    float(scores[i]),
                "arxiv_id": aid,
                "title":    meta.get("title", ""),
                "authors":  meta.get("authors", ""),
                "abstract": abstract[:300] + ("…" if len(abstract) > 300 else ""),
                "url":      meta.get("pdf_url", ""),
            })
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_meta(self, db_path: Path) -> dict[str, dict]:
        """Carga metadatos de documentos indexados desde la BD."""
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT arxiv_id, title, authors, abstract, pdf_url "
                "FROM documents WHERE pdf_downloaded = 1"
            ).fetchall()
        finally:
            conn.close()
        return {r["arxiv_id"]: dict(r) for r in rows}