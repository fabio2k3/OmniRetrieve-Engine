"""
lsi_retriever.py
================
Retriever LSI que implementa RetrieverProtocol correctamente.

Cambios respecto a la versión anterior
---------------------------------------
· retrieve() devuelve list[RetrievalResult] con chunk_ids enteros reales
  (no strings sintéticos), eliminando la necesidad de LSIRetrieverAdapter.
· La vectorización de queries se delega a QueryVectorizer (lsi_query.py).
· La expansión documento → chunks reales se hace contra la tabla chunks de
  la BD, convirtiendo LSI de nivel-documento a nivel-chunk.

Diseño de la expansión documento → chunk
-----------------------------------------
LSI calcula similitud coseno a nivel de documento (arxiv_id). Para ser
compatible con el RRF del HybridRetriever (que opera a nivel de chunk_id
entero), cada documento recuperado se expande a sus chunks reales:

    1. Top-N documentos por similitud coseno.
    2. Consulta a la BD: todos los chunks de esos documentos.
    3. Cada chunk hereda el score coseno del documento al que pertenece.
    4. Se ordenan por (score DESC, chunk_index ASC) y se devuelven top_n.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.database.chunk_repository import get_chunks_with_metadata_by_arxiv_ids
from backend.database.schema import DB_PATH
from backend.indexing.preprocessor import TextPreprocessor
from .lsi_model import LSIModel, MODEL_PATH
from .lsi_query import QueryVectorizer, build_word_index
from .protocols import RetrievalResult, RetrieverProtocol

log = logging.getLogger(__name__)

# Cuántos documentos candidatos recuperar internamente antes de expandir a chunks.
_DEFAULT_DOC_CANDIDATES = 20


class LSIRetriever(RetrieverProtocol):
    """
    Retriever LSI compatible con RetrieverProtocol.

    Devuelve list[RetrievalResult] con chunk_ids enteros reales,
    listo para fusionarse con EmbeddingRetriever en HybridRetriever.

    Parámetros
    ----------
    model         : LSIModel (si None, se crea uno vacío para cargar con load()).
    doc_candidates: documentos LSI a expandir a chunks antes de devolver top_n.
    """

    def __init__(
        self,
        model:          LSIModel | None = None,
        doc_candidates: int = _DEFAULT_DOC_CANDIDATES,
    ) -> None:
        self._model          = model or LSIModel()
        self._doc_candidates = doc_candidates
        self._vectorizer:    QueryVectorizer | None = None
        self._db_path:       Path = DB_PATH

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load(
        self,
        model_path: Path = MODEL_PATH,
        db_path:    Path = DB_PATH,
    ) -> None:
        """
        Carga el modelo LSI y construye las estructuras de consulta.

        Parámetros
        ----------
        model_path : ruta al .pkl generado en la fase offline.
        db_path    : BD SQLite para leer vocabulario y chunks.
        """
        self._db_path = Path(db_path)
        self._model.load(model_path)
        word_index       = build_word_index(self._model, db_path=self._db_path)
        self._vectorizer = QueryVectorizer(
            model=self._model,
            word_index=word_index,
            preprocessor=TextPreprocessor(),
        )
        log.info(
            "[LSIRetriever] Listo. %d documentos | %d términos.",
            len(self._model.doc_ids),
            len(word_index),
        )

    # ------------------------------------------------------------------
    # RetrieverProtocol
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """
        Recupera los top_n chunks más relevantes para la query.

        Implementa RetrieverProtocol: devuelve list[RetrievalResult] con
        chunk_ids enteros reales obtenidos de la tabla chunks de la BD.

        Pasos
        -----
        1. Vectoriza la query (TF-IDF en vocabulario LSI).
        2. Proyecta al espacio latente.
        3. Calcula similitud coseno contra documentos indexados.
        4. Toma los top doc_candidates documentos.
        5. Expande cada documento a sus chunks reales.
        6. Devuelve top_n ordenados por (score DESC, chunk_index ASC).

        Lanza RuntimeError si load() no fue llamado.
        """
        if self._vectorizer is None:
            raise RuntimeError("LSIRetriever no inicializado. Llama a load() primero.")

        if not query or not query.strip():
            log.debug("[LSIRetriever] Query vacía; se devuelve lista vacía.")
            return []

        t0 = time.monotonic()

        q_tfidf  = self._vectorizer.vectorize(query)
        q_latent = self._model.project_query(q_tfidf)
        scores   = cosine_similarity(
            q_latent.reshape(1, -1), self._model.docs_latent
        ).flatten()

        n_candidates = min(self._doc_candidates, len(scores))
        top_idx      = np.argsort(scores)[::-1][:n_candidates]
        scored_docs  = [
            (self._model.doc_ids[i], float(scores[i]))
            for i in top_idx
            if float(scores[i]) > 0.0
        ]

        log.debug(
            "[LSIRetriever] query='%s…' doc_candidates=%d top_score=%.4f time=%.1fms",
            query[:40], len(scored_docs),
            scored_docs[0][1] if scored_docs else 0.0,
            (time.monotonic() - t0) * 1000,
        )

        if not scored_docs:
            return []

        return self._expand_to_chunks(scored_docs, top_n=top_n)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _expand_to_chunks(
        self,
        scored_docs: list[tuple[str, float]],
        top_n: int,
    ) -> list[RetrievalResult]:
        """
        Expande (arxiv_id, score) a RetrievalResult con chunk_ids reales.
        Cada chunk hereda el score coseno del documento padre.
        """
        arxiv_ids = [aid for aid, _ in scored_docs]
        score_map = {aid: score for aid, score in scored_docs}

        rows = get_chunks_with_metadata_by_arxiv_ids(arxiv_ids, db_path=self._db_path)

        results: list[RetrievalResult] = []
        for row in rows:
            aid = row["arxiv_id"]
            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    arxiv_id=aid,
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=score_map.get(aid, 0.0),
                    score_type="cosine_lsi",
                    metadata={
                        "title":   row["title"]   or "",
                        "authors": row["authors"] or "",
                        "pdf_url": row["pdf_url"] or "",
                    },
                )
            )

        results.sort(key=lambda r: (-r.score, r.chunk_index))
        return results[:top_n]