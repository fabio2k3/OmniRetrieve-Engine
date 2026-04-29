"""
embedding_retriever.py
======================
Adaptador de recuperación densa sobre FAISS para OmniRetrieve-Engine.

Responsabilidad
---------------
- Vectorizar una query con el mismo modelo de embedding del pipeline.
- Consultar el índice FAISS (FaissIndexManager.search).
- Enriquecer los IDs recuperados con texto/metadatos desde SQLite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.database.chunk_repository import get_chunks_by_ids
from backend.database.schema import DB_PATH
from backend.embedding.embedder import DEFAULT_MODEL
from backend.embedding import FaissIndexManager
from .protocols import RetrievalResult, RetrieverProtocol

log = logging.getLogger(__name__)


class EmbeddingRetriever(RetrieverProtocol):
    """
    Retriever denso basado en embeddings + FAISS.

    Parámetros
    ----------
    faiss_mgr   : instancia compartida de FaissIndexManager.
    db_path     : ruta de la BD SQLite para enriquecer resultados.
    model_name  : modelo sentence-transformers usado para vectorizar queries.
    """

    def __init__(
        self,
        faiss_mgr: FaissIndexManager,
        db_path: Path = DB_PATH,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._faiss_mgr = faiss_mgr
        self._db_path = Path(db_path)
        self._model_name = model_name
        self._embedder = None

    def _get_embedder(self):
        """Inicializa el embedder de forma perezosa para evitar costo al arrancar."""
        if self._embedder is None:
            from backend.embedding.embedder import ChunkEmbedder

            log.info("[dense] Inicializando embedder de query: model=%s", self._model_name)
            self._embedder = ChunkEmbedder(model_name=self._model_name)
        return self._embedder

    def retrieve(self, query: str, top_n: int = 10) -> list[RetrievalResult]:
        """
        Ejecuta búsqueda densa y devuelve chunks enriquecidos.

        Devuelve
        --------
        Lista de RetrievalResult (chunk-level).
        """
        if not query or not query.strip():
            log.debug("[dense] Query vacía; se devuelve lista vacía.")
            return []

        embedder = self._get_embedder()
        query_vec = embedder.encode_single(query)

        raw_results = self._faiss_mgr.search(query_vec, top_k=top_n)
        if not raw_results:
            log.debug("[dense] FAISS no devolvió candidatos para query='%s…'", query[:40])
            return []

        chunk_ids = [r["chunk_id"] for r in raw_results]
        score_map = {r["chunk_id"]: r["score"] for r in raw_results}
        rows = get_chunks_by_ids(chunk_ids, db_path=self._db_path)

        results: list[RetrievalResult] = []
        for row in rows:
            meta: dict[str, Any] = {
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "abstract": row.get("abstract", ""),
                "pdf_url": row.get("pdf_url", ""),
            }
            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    arxiv_id=row["arxiv_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=score_map.get(row["chunk_id"], float("inf")),
                    score_type="l2",
                    metadata=meta,
                )
            )

        # Distancia L2: menor valor implica mayor similitud.
        results.sort(key=lambda r: r.score)

        best = results[0].score if results else float("inf")
        log.debug(
            "[dense] query='%s…' top_n=%d candidatos=%d best_l2=%.6f",
            query[:40],
            top_n,
            len(results),
            best,
        )
        return results
