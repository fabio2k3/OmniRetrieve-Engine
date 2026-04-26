"""
query_expander.py
=================
Expansión "ciega" de consulta usando el espacio latente LSI.

Técnica: Latent Concept Expansion (LCE)
----------------------------------------
Antes de tocar el índice FAISS, enriquece la consulta corta del usuario
con términos del corpus que comparten los mismos conceptos latentes.

Flujo
-----
1. Tokeniza y vectoriza la consulta al espacio TF-IDF del vocabulario LSI.
2. Proyecta al espacio latente k-dimensional: q_latent = SVD.transform(q_tfidf).
3. Identifica las dimensiones latentes más activadas (conceptos dominantes).
4. Para cada dimensión activada, extrae los términos con mayor peso en
   svd.components_[dim] (la fila del vocabulario para ese concepto).
5. Filtra términos por umbral de correlación para evitar Query Drift.
6. Devuelve la consulta enriquecida como string para re-encodear con
   SentenceTransformer.

Por qué LCE antes de FAISS
---------------------------
SentenceTransformer es preciso pero sensible a la longitud y vocabulario
de la query. Una consulta de 3 palabras puede no activar el embedding
correcto si el corpus usa sinónimos o jerga específica del dominio.
LCE añade 5-10 términos del corpus que estadísticamente co-ocurren en
los mismos conceptos, mejorando la cobertura semántica del embedding.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from backend.indexing.preprocessor import TextPreprocessor
from backend.retrieval.lsi_model import LSIModel, MODEL_PATH

log = logging.getLogger(__name__)


class QueryExpander:
    """
    Expande una consulta usando los conceptos latentes del modelo LSI.

    Parámetros
    ----------
    lsi_model         : modelo LSI ya cargado. Si es None, se carga de disco.
    top_dims          : número de dimensiones latentes a examinar.
    top_terms_per_dim : términos candidatos por dimensión.
    min_correlation   : umbral mínimo de peso para añadir un término.
                        Evita el Query Drift filtrando términos marginales.
    max_expansion     : número máximo de términos nuevos a añadir.
    """

    def __init__(
        self,
        lsi_model:         LSIModel | None = None,
        top_dims:          int   = 3,
        top_terms_per_dim: int   = 10,
        min_correlation:   float = 0.4,
        max_expansion:     int   = 8,
    ) -> None:
        self._model            = lsi_model
        self.top_dims          = top_dims
        self.top_terms_per_dim = top_terms_per_dim
        self.min_correlation   = min_correlation
        self.max_expansion     = max_expansion
        self._preprocessor     = TextPreprocessor()
        # word -> (row_idx, df) — construido al cargar el modelo
        self._word_index: dict[str, tuple[int, int]] = {}
        # row_idx -> word — para recuperar palabras desde componentes SVD
        self._idx_to_word: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load(self, model_path: Path = MODEL_PATH, db_path: "Path | None" = None) -> None:
        """
        Carga el modelo LSI y construye los índices necesarios.

        Parámetros
        ----------
        model_path : ruta al .pkl del modelo LSI.
        db_path    : ruta a la BD para construir el índice de palabras.
                     Si es None, usa la DB por defecto definida en schema.
        """
        from backend.database.schema import DB_PATH
        from backend.database.schema import get_connection

        db = db_path or DB_PATH

        if self._model is None:
            self._model = LSIModel()
            self._model.load(model_path)

        self._word_index, self._idx_to_word = self._build_indices(db)
        log.info(
            "[QueryExpander] Listo. %d términos en vocabulario | k=%d",
            len(self._word_index), self._model.k,
        )

    def _build_indices(
        self, db_path: "Path"
    ) -> tuple[dict[str, tuple[int, int]], dict[int, str]]:
        """
        Construye dos índices desde la BD:
          word_index   : word -> (row_idx_en_SVD, df)
          idx_to_word  : row_idx_en_SVD -> word
        """
        from backend.database.schema import get_connection

        if not self._model or not self._model.term_ids:
            return {}, {}

        _CHUNK_SIZE = 900
        term_ids    = self._model.term_ids
        rows        = []
        conn        = get_connection(db_path)
        try:
            for offset in range(0, len(term_ids), _CHUNK_SIZE):
                chunk = term_ids[offset: offset + _CHUNK_SIZE]
                ph    = ",".join("?" * len(chunk))
                rows.extend(
                    conn.execute(
                        f"SELECT term_id, word FROM terms WHERE term_id IN ({ph})",
                        chunk,
                    ).fetchall()
                )
        finally:
            conn.close()

        term_id_to_row = {tid: i for i, tid in enumerate(self._model.term_ids)}
        word_index:   dict[str, tuple[int, int]] = {}
        idx_to_word:  dict[int, str]             = {}

        for r in rows:
            tid     = r["term_id"]
            row_idx = term_id_to_row.get(tid)
            df      = self._model.df_map.get(tid, 1)
            if row_idx is not None:
                word_index[r["word"]]  = (row_idx, df)
                idx_to_word[row_idx]   = r["word"]

        return word_index, idx_to_word

    # ------------------------------------------------------------------
    # Expansión
    # ------------------------------------------------------------------

    def expand(self, query: str) -> tuple[str, list[str]]:
        """
        Expande la consulta con términos del espacio latente LSI.

        Parámetros
        ----------
        query : consulta original del usuario.

        Devuelve
        --------
        (query_expandida, terminos_añadidos)
            query_expandida : query original + términos nuevos como string.
            terminos_añadidos : lista de los términos que se añadieron.

        Si el modelo no está cargado o la consulta no activa conceptos
        suficientes, devuelve la consulta original sin cambios.
        """
        if not self._model or self._model.docs_latent is None:
            log.warning("[QueryExpander] Modelo no cargado — devolviendo query sin expandir.")
            return query, []

        # 1. Vectorizar query al espacio TF-IDF
        q_tfidf = self._vectorize(query)
        if q_tfidf.sum() == 0:
            log.debug("[QueryExpander] Query sin términos en vocabulario: '%s'", query[:50])
            return query, []

        # 2. Proyectar al espacio latente
        q_latent = self._model.project_query(q_tfidf)   # (k,)

        # 3. Identificar dimensiones más activadas
        top_dim_indices = np.argsort(np.abs(q_latent))[::-1][: self.top_dims]

        # 4. Recoger términos candidatos de los componentes SVD
        # components_[dim] tiene shape (n_terms,) — peso de cada término en el concepto
        components = self._model.svd.components_   # (k, n_terms)

        query_tokens = set(self._preprocessor.process(query))
        candidates:  dict[str, float] = {}   # word -> max correlation

        for dim in top_dim_indices:
            dim_activation = float(np.abs(q_latent[dim]))
            if dim_activation < 1e-6:
                continue

            comp_row = components[dim]   # (n_terms,)
            # Normalizar para obtener correlaciones entre 0 y 1
            max_abs  = float(np.abs(comp_row).max()) or 1.0
            top_term_indices = np.argsort(np.abs(comp_row))[::-1][: self.top_terms_per_dim]

            for idx in top_term_indices:
                correlation = float(np.abs(comp_row[idx])) / max_abs
                if correlation < self.min_correlation:
                    break   # ya están ordenados descendente

                word = self._idx_to_word.get(int(idx))
                if word and word not in query_tokens:
                    # Ponderar por la activación de la dimensión
                    score = correlation * dim_activation
                    if word not in candidates or candidates[word] < score:
                        candidates[word] = score

        if not candidates:
            log.debug("[QueryExpander] No se encontraron términos de expansión.")
            return query, []

        # 5. Seleccionar los mejores términos por score combinado
        sorted_terms = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        new_terms    = [w for w, _ in sorted_terms[: self.max_expansion]]

        expanded = query.strip() + " " + " ".join(new_terms)

        log.info(
            "[QueryExpander] '%s…' → +%d términos: %s",
            query[:40], len(new_terms), new_terms,
        )
        return expanded.strip(), new_terms

    # ------------------------------------------------------------------
    # Helper: vectorización TF-IDF
    # ------------------------------------------------------------------

    def _vectorize(self, query: str) -> np.ndarray:
        """Vectoriza la query al espacio TF-IDF del vocabulario LSI."""
        n_terms = len(self._model.term_ids)
        n_docs  = len(self._model.doc_ids)
        vec     = np.zeros(n_terms, dtype=np.float32)

        tokens        = self._preprocessor.process(query)
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