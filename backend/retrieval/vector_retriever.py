"""
vector_retriever.py
===================
Recuperacion vectorial usando ChromaDB como base de datos vectorial.

Diseno
------
ChromaDB gestiona la persistencia, los vectores y la busqueda por
similitud coseno nativamente (hnsw:space=cosine en la coleccion).
Este modulo es una capa delgada que:
  1. Codifica la query con Embedder.
  2. Delega la busqueda a chroma_store.query().
  3. Deduplica por arxiv_id (un documento puede tener muchos chunks).
  4. Enriquece los resultados con metadatos de SQLite (titulo, autores...).

La similitud coseno de ChromaDB se expresa como distancia (0=identico,
2=opuesto). Se convierte a score = 1 - distance para mantener la
convencion score-alto=mejor del resto del sistema.

Interfaz publica identica a LSIRetriever para que ambos motores sean
intercambiables.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from backend.database.schema import DB_PATH, get_connection
from backend.embedding.chroma_store import CHROMA_PATH, query as chroma_query, count
from backend.embedding.embedder import Embedder

log = logging.getLogger(__name__)


class VectorRetriever:
    """
    Motor de recuperacion por similitud coseno sobre ChromaDB.

    Uso
    ---
        retriever = VectorRetriever()
        retriever.load(db_path, chroma_path)
        results   = retriever.retrieve("attention transformer", top_n=10)
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder    = embedder or Embedder()
        self._db_path:     Path | None = None
        self._chroma_path: Path | None = None
        self._meta:        dict[str, dict] = {}
        self._ready:       bool = False

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load(
        self,
        db_path:     Path = DB_PATH,
        chroma_path: Path = CHROMA_PATH,
    ) -> None:
        """
        Prepara el retriever: verifica que ChromaDB tiene vectores y
        carga los metadatos de documentos desde SQLite.

        No carga nada en RAM: la busqueda vectorial la hace ChromaDB
        directamente desde disco en cada consulta.

        Parametros
        ----------
        db_path     : BD SQLite con metadatos de documentos.
        chroma_path : directorio de ChromaDB.
        """
        self._db_path     = Path(db_path)
        self._chroma_path = Path(chroma_path)
        self._meta        = self._load_meta(db_path)

        n_vectors = count(chroma_path)
        n_docs    = len(self._meta)

        if n_vectors == 0:
            log.warning(
                "[VectorRetriever] ChromaDB vacia en %s. "
                "Ejecuta EmbeddingPipeline primero.",
                chroma_path,
            )
            self._ready = False
        else:
            self._ready = True
            log.info(
                "[VectorRetriever] Listo. %d vectores en ChromaDB | %d docs en SQLite.",
                n_vectors, n_docs,
            )

    def is_ready(self) -> bool:
        """True si ChromaDB tiene vectores y load() ha sido llamado."""
        return self._ready

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_n: int = 10) -> list[dict]:
        """
        Devuelve los top_n documentos mas relevantes para la query.

        Pasos
        -----
        1. Codifica la query con Embedder (float32, norma L2=1).
        2. Busca los top_n*4 chunks mas cercanos en ChromaDB.
        3. Convierte distancia coseno a score: score = 1 - distance.
        4. Deduplica por arxiv_id, conservando el score maximo por doc.
        5. Devuelve los top_n con metadatos de SQLite.

        Lanza RuntimeError si load() no ha sido llamado.
        """
        if not self.is_ready():
            raise RuntimeError(
                "VectorRetriever no cargado. Llama a load() primero."
            )

        t0 = time.monotonic()

        q_vec = self.embedder.encode_one(query)
        hits  = chroma_query(
            query_embedding = q_vec,
            n_results       = top_n * 4,    # margen para deduplicar
            chroma_path     = self._chroma_path,
        )

        # Deduplicar: score maximo por arxiv_id
        # ChromaDB con coseno devuelve distancia [0, 2]; score = 1 - distance
        best: dict[str, float] = {}
        for h in hits:
            score    = 1.0 - h["distance"]
            arxiv_id = h["arxiv_id"]
            if score > best.get(arxiv_id, float("-inf")):
                best[arxiv_id] = score

        sorted_docs = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.debug(
            "[VectorRetriever] query='%s...' top_score=%.4f tiempo=%.1fms",
            query[:40],
            sorted_docs[0][1] if sorted_docs else 0.0,
            elapsed_ms,
        )

        results = []
        for arxiv_id, score in sorted_docs:
            meta     = self._meta.get(arxiv_id, {})
            abstract = meta.get("abstract") or ""
            results.append({
                "score":    score,
                "arxiv_id": arxiv_id,
                "title":    meta.get("title", ""),
                "authors":  meta.get("authors", ""),
                "abstract": abstract[:300] + ("..." if len(abstract) > 300 else ""),
                "url":      meta.get("pdf_url", ""),
            })
        return results

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _load_meta(self, db_path: Path) -> dict[str, dict]:
        """Carga metadatos de documentos con PDF descargado desde SQLite."""
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                "SELECT arxiv_id, title, authors, abstract, pdf_url "
                "FROM   documents "
                "WHERE  pdf_downloaded = 1"
            ).fetchall()
        finally:
            conn.close()
        return {r["arxiv_id"]: dict(r) for r in rows}