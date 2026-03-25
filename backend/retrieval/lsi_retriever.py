"""
lsi_retriever.py
================
Fase online del modulo LSI.
Proyecta la consulta al espacio latente y devuelve los K articulos
mas relevantes por similitud coseno.

Referencia: Manning et al. (2008), Cap. 18, sec. 18.3 "Queries in LSI"
"""
from __future__ import annotations
import logging
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.database.schema import DB_PATH, get_connection
from .lsi_model import LSIModel, MODEL_PATH

log = logging.getLogger(__name__)  # "retrieval.lsi_retriever"


class LSIRetriever:
    """ Motor de recuperacion LSI. Carga el modelo y responde consultas. """

    def __init__(self, model: LSIModel | None = None):
        self.model = model or LSIModel()
        self._meta: dict[str, dict] | None = None

    def load(self, path=MODEL_PATH, db_path=DB_PATH) -> None:
        """ Carga modelo y cachea metadatos de la BD. """
        self.model.load(path)
        self._meta = self._load_meta(db_path)
        log.info("[LSIRetriever] Listo. %d documentos indexados.",
                 len(self.model.doc_ids))

    def _load_meta(self, db_path=DB_PATH) -> dict[str, dict]:
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT arxiv_id, title, authors, abstract, pdf_url "
                "FROM documents WHERE pdf_downloaded = 1"
            ).fetchall()
        finally:
            conn.close()
        return {r["arxiv_id"]: dict(r) for r in rows}

    def retrieve(self, query: str, top_n: int = 10) -> list[dict]:
        """
        Devuelve los top_n documentos mas relevantes para la consulta.

        1. Vectoriza la consulta con el MISMO Pipeline (transform, no fit).
        2. Calcula similitud coseno contra todos los documentos.
        3. Devuelve los top_n con sus metadatos.
        """
        if self.model.docs_latent is None:
            raise RuntimeError("Modelo no cargado. Llama a .load() primero.")

        t0 = time.monotonic()

        # Opcion A garantizada: transform usa el vocabulario aprendido en build()
        q_latent = self.model.pipeline.transform([query])  # (1, k)
        scores = cosine_similarity(q_latent, self.model.docs_latent).flatten()
        top_idx = np.argsort(scores)[::-1][:top_n]

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.debug("[LSIRetriever] query = '%s...' top_score=%.4f tiempo=%.1fms",
                  query[:40], scores[top_idx[0]], elapsed_ms)

        results = []
        for i in top_idx:
            aid = self.model.doc_ids[i]
            meta = self._meta.get(aid, {}) if self._meta else {}
            results.append({
                "score": float(scores[i]),
                "arxiv_id": aid,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "abstract": meta.get("abstract", "")[:300] + "...",
                "url": meta.get("pdf_url", ""),
            })
        return results
