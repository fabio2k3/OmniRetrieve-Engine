"""
brf.py
======
Pseudo-Retroalimentación de Relevancia Ciega (Blind Relevance Feedback).

Asume que los primeros resultados de FAISS son relevantes y ajusta el
vector de query hacia su centroide.

Técnica: Vector Mean Shift con interpolación alpha.

    v_new = alpha * v_orig + (1 - alpha) * centroide(top_k)

Los embeddings se recuperan directamente de la BD (columna chunks.embedding)
para evitar el error de aproximación de la cuantización PQ de IndexIVFPQ.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._feedback_utils import get_embeddings_by_chunk_ids, cosine_similarity, l2_normalize

log = logging.getLogger(__name__)


class BlindRelevanceFeedback:
    """
    Ajusta el vector de query hacia el centroide de los top resultados de FAISS.

    Recupera los embeddings de los mejores chunks directamente de la BD para
    evitar el error de aproximación de la cuantización PQ del índice FAISS.

    Parámetros
    ----------
    alpha    : peso del vector original (0-1). Mayor alpha = menos ajuste.
               Recomendado: 0.7-0.8 para corpus científico denso.
    top_k_rf : número de resultados usados para calcular el centroide.
    """

    def __init__(self, alpha: float = 0.75, top_k_rf: int = 5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha debe estar en [0, 1], recibido: {alpha}")
        self.alpha    = alpha
        self.top_k_rf = top_k_rf

    def adjust(
        self,
        query_vector: np.ndarray,
        top_results:  list[dict],
        db_path:      Path,
    ) -> np.ndarray:
        """
        Ajusta el vector de query hacia el centroide de los top resultados.

        Parámetros
        ----------
        query_vector : vector de query original, shape (dim,), float32.
        top_results  : lista de dicts con clave 'chunk_id' (resultado de FAISS).
        db_path      : ruta a la BD para recuperar los embeddings.

        Devuelve
        --------
        Vector ajustado, shape (dim,), L2-normalizado.
        Si no hay embeddings disponibles, devuelve el vector original sin cambios.
        """
        chunk_ids = [r["chunk_id"] for r in top_results[: self.top_k_rf]]
        vecs_map  = get_embeddings_by_chunk_ids(chunk_ids, db_path)

        if not vecs_map:
            log.warning("[BRF] No se encontraron embeddings para los top resultados.")
            return query_vector

        vecs     = np.stack(list(vecs_map.values()))   # (N, dim)
        centroid = vecs.mean(axis=0)                   # (dim,)

        adjusted = self.alpha * query_vector + (1 - self.alpha) * centroid
        adjusted = l2_normalize(adjusted.astype(np.float32))

        log.info(
            "[BRF] Vector ajustado con centroide de %d chunks (alpha=%.2f). "
            "Similitud original→ajustado: %.3f",
            len(vecs_map), self.alpha,
            cosine_similarity(query_vector, adjusted),
        )
        return adjusted