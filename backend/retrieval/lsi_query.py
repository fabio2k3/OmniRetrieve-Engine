"""
lsi_query.py
============
Vectorización de queries al espacio TF-IDF del modelo LSI.

Responsabilidad única
---------------------
Convertir una query de texto libre en un vector TF-IDF alineado
con el vocabulario del modelo LSI cargado.

Separado de lsi_retriever.py para que la lógica de vectorización
pueda ser testeada y reutilizada de forma independiente.

Fórmulas
--------
    TF(t, q)  = log(1 + freq(t, q))
    IDF(t)    = log((N + 1) / (df(t) + 1))   ← df real del corpus
    W(t, q)   = TF * IDF
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from backend.database.schema import DB_PATH
from backend.database.index_repository import get_term_words_by_ids
from backend.indexing.preprocessor import TextPreprocessor
from .lsi_model import LSIModel

log = logging.getLogger(__name__)

def build_word_index(
    model: LSIModel,
    db_path: Path = DB_PATH,
) -> dict[str, tuple[int, int]]:
    """
    Construye el mapa word → (row_idx, df) para vectorizar queries.

    row_idx es la posición del term_id en model.term_ids,
    que coincide con el orden de filas de la matriz TF-IDF usada en build().
    df es la frecuencia de documento guardada en model.df_map.

    Parámetros
    ----------
    model   : LSIModel ya cargado.
    db_path : ruta a la BD SQLite para leer la tabla terms.

    Devuelve
    --------
    dict word → (row_idx, df).  Vacío si el modelo no tiene term_ids.
    """
    if not model.term_ids:
        return {}

    rows = get_term_words_by_ids(model.term_ids, db_path=db_path)

    term_id_to_row = {tid: i for i, tid in enumerate(model.term_ids)}
    index: dict[str, tuple[int, int]] = {}
    for r in rows:
        tid     = r["term_id"]
        row_idx = term_id_to_row.get(tid)
        df      = model.df_map.get(tid, 1)
        if row_idx is not None:
            index[r["word"]] = (row_idx, df)

    log.debug("[lsi_query] word_index construido: %d términos", len(index))
    return index


class QueryVectorizer:
    """
    Convierte una query de texto en un vector TF-IDF alineado con el
    vocabulario del modelo LSI.

    Parámetros
    ----------
    model       : LSIModel ya cargado (necesita term_ids, df_map, docs_latent).
    word_index  : dict generado por build_word_index().
    preprocessor: tokenizador consistente con la fase de indexación.
    """

    def __init__(
        self,
        model:        LSIModel,
        word_index:   dict[str, tuple[int, int]],
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        self._model       = model
        self._word_index  = word_index
        self._preprocessor = preprocessor or TextPreprocessor()

    def vectorize(self, query: str) -> np.ndarray:
        """
        Convierte la query en un vector TF-IDF (n_terms,).

        Tokens no presentes en el vocabulario del modelo se ignoran
        silenciosamente — no lanzan error.

        Lanza
        -----
        RuntimeError si el modelo no ha sido cargado (docs_latent is None).
        """
        if self._model.docs_latent is None:
            raise RuntimeError("LSIModel no cargado. Llama a load() primero.")

        n_terms = len(self._model.term_ids)
        n_docs  = len(self._model.doc_ids)
        vec     = np.zeros(n_terms, dtype=np.float32)

        tokens = self._preprocessor.process(query)
        freq_in_query: dict[str, int] = {}
        for t in tokens:
            freq_in_query[t] = freq_in_query.get(t, 0) + 1

        oov = 0
        for token, freq in freq_in_query.items():
            entry = self._word_index.get(token)
            if entry is None:
                oov += 1
                continue
            row_idx, df = entry
            tf  = math.log(1 + freq)
            idf = math.log((n_docs + 1) / (df + 1))
            vec[row_idx] = tf * idf

        if oov:
            log.debug(
                "[lsi_query] query='%s…' tokens_oov=%d/%d",
                query[:40], oov, len(freq_in_query),
            )

        return vec