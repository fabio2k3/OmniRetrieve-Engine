"""
mmr.py
======
Diversificación de resultados con Maximal Marginal Relevance (MMR).

MMR selecciona iterativamente el documento que maximiza:

    MMR(d) = lambda * sim(d, query) - (1 - lambda) * max_sim(d, D_seleccionados)

Equilibra relevancia (similitud con la query) y diversidad (disimilitud
con los documentos ya seleccionados), evitando que el módulo RAG reciba
N chunks redundantes que dicen lo mismo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._feedback_utils import get_embeddings_by_chunk_ids, cosine_similarity

log = logging.getLogger(__name__)


class MMRReranker:
    """
    Reordena resultados usando Maximal Marginal Relevance (MMR).

    Parámetros
    ----------
    lambda_ : peso entre relevancia (1.0) y diversidad (0.0).
              Recomendado: 0.5-0.7 para recuperación en RAG.
    """

    def __init__(self, lambda_: float = 0.6) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ debe estar en [0, 1], recibido: {lambda_}")
        self.lambda_ = lambda_

    def rerank(
        self,
        results:      list[dict],
        query_vector: np.ndarray,
        top_n:        int,
        db_path:      Path,
    ) -> list[dict]:
        """
        Aplica MMR sobre una lista de resultados de FAISS.

        Parámetros
        ----------
        results      : lista de dicts con clave 'chunk_id' y 'score' (FAISS).
        query_vector : vector de query L2-normalizado, shape (dim,).
        top_n        : número de resultados a devolver tras el reranking.
        db_path      : ruta a la BD para recuperar los embeddings.

        Devuelve
        --------
        Lista de hasta top_n dicts reordenados. Cada dict conserva los campos
        originales y añade 'mmr_score' (float, mayor = más relevante y diverso).

        Si no se pueden recuperar embeddings, devuelve los primeros top_n
        resultados originales sin modificar (fallback).
        """
        if not results:
            return []

        top_n     = min(top_n, len(results))
        chunk_ids = [r["chunk_id"] for r in results]
        vecs_map  = get_embeddings_by_chunk_ids(chunk_ids, db_path)

        if not vecs_map:
            log.warning("[MMR] No se encontraron embeddings — devolviendo sin reordenar.")
            return results[:top_n]

        sim_to_query: dict[int, float] = {
            cid: cosine_similarity(query_vector, vecs_map[cid])
            for cid in chunk_ids
            if cid in vecs_map
        }

        candidates   = [r for r in results if r["chunk_id"] in sim_to_query]
        selected:     list[dict] = []
        selected_ids: list[int]  = []

        for _ in range(top_n):
            if not candidates:
                break

            best_score = float("-inf")
            best_doc   = None

            for doc in candidates:
                cid = doc["chunk_id"]
                rel = sim_to_query[cid]
                red = (
                    max(
                        cosine_similarity(vecs_map[cid], vecs_map[sid])
                        for sid in selected_ids
                        if sid in vecs_map
                    )
                    if selected_ids else 0.0
                )
                mmr_score = self.lambda_ * rel - (1 - self.lambda_) * red
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc   = doc

            if best_doc is None:
                break

            selected.append({**best_doc, "mmr_score": round(best_score, 4)})
            selected_ids.append(best_doc["chunk_id"])
            candidates = [d for d in candidates if d["chunk_id"] != best_doc["chunk_id"]]

        log.info(
            "[MMR] Reranking: %d candidatos → %d seleccionados (lambda=%.2f)",
            len(results), len(selected), self.lambda_,
        )
        return selected