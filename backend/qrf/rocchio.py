"""
rocchio.py
==========
Retroalimentación explícita de relevancia usando el algoritmo de Rocchio.

El usuario marca chunks como relevantes (D_r) o irrelevantes (D_n) tras
ver los resultados. El vector de query se desplaza en el espacio de embeddings:

    v_new = alpha * v_orig
          + beta  * (1/|D_r|) * sum(D_r)
          - gamma * (1/|D_n|) * sum(D_n)

Los vectores ajustados se guardan en caché en memoria por query_id para
que múltiples rondas de refinamiento en la misma sesión sean acumulativas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._feedback_utils import get_embeddings_by_chunk_ids, cosine_similarity, l2_normalize

log = logging.getLogger(__name__)


class RocchioFeedback:
    """
    Retroalimentación explícita de relevancia usando el algoritmo de Rocchio.

    Parámetros
    ----------
    alpha : peso del vector original.
    beta  : peso de los documentos relevantes.
    gamma : penalización de los documentos irrelevantes.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta:  float = 0.4,
        gamma: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        # Caché en sesión: query_id -> vector ajustado
        self._cache: dict[str, np.ndarray] = {}

    def adjust(
        self,
        query_id:       str,
        query_vector:   np.ndarray,
        relevant_ids:   list[int],
        irrelevant_ids: list[int],
        db_path:        Path,
    ) -> np.ndarray:
        """
        Ajusta el vector de query según el feedback explícito del usuario.

        Parámetros
        ----------
        query_id       : identificador de sesión (para caché acumulativo).
        query_vector   : vector de query actual, shape (dim,), float32.
        relevant_ids   : chunk_ids marcados como relevantes.
        irrelevant_ids : chunk_ids marcados como irrelevantes.
        db_path        : ruta a la BD para recuperar los embeddings.

        Devuelve
        --------
        Vector ajustado, shape (dim,), L2-normalizado.
        Devuelve el vector original si no se proporciona ningún feedback.
        """
        if not relevant_ids and not irrelevant_ids:
            return query_vector

        all_ids  = list(set(relevant_ids + irrelevant_ids))
        vecs_map = get_embeddings_by_chunk_ids(all_ids, db_path)

        rel_vecs = [vecs_map[cid] for cid in relevant_ids if cid in vecs_map]
        rel_term = (
            self.beta * np.stack(rel_vecs).mean(axis=0)
            if rel_vecs else np.zeros_like(query_vector)
        )

        irrel_vecs = [vecs_map[cid] for cid in irrelevant_ids if cid in vecs_map]
        irrel_term = (
            self.gamma * np.stack(irrel_vecs).mean(axis=0)
            if irrel_vecs else np.zeros_like(query_vector)
        )

        adjusted = self.alpha * query_vector + rel_term - irrel_term
        adjusted = l2_normalize(adjusted.astype(np.float32))

        self._cache[query_id] = adjusted

        log.info(
            "[Rocchio] query_id='%s' | relevantes=%d | irrelevantes=%d | "
            "similitud original→ajustado: %.3f",
            query_id, len(rel_vecs), len(irrel_vecs),
            cosine_similarity(query_vector, adjusted),
        )
        return adjusted

    def get_cached(self, query_id: str) -> np.ndarray | None:
        """
        Devuelve el vector ajustado en caché para una query_id, o None si no existe.
        Útil para reutilizar el vector calibrado en búsquedas de seguimiento.
        """
        return self._cache.get(query_id)

    def clear_cache(self, query_id: str | None = None) -> None:
        """Limpia la caché completa o solo la entrada de una query_id."""
        if query_id:
            self._cache.pop(query_id, None)
        else:
            self._cache.clear()

    @property
    def cached_queries(self) -> list[str]:
        """Lista de query_ids con vector calibrado en caché."""
        return list(self._cache.keys())