"""
reranker.py
===========
Segunda etapa de recuperación: reranking con Cross-Encoder.

Diseño
------
- NO hace retrieval. Recibe candidatos (chunk-level) y los reordena.
- Usa sentence-transformers CrossEncoder con modelo MS MARCO por defecto.
- Devuelve nuevos RetrievalResult con score_type='rerank' y score del reranker.
"""

from __future__ import annotations

import logging
from typing import Any

from .protocols import RetrievalResult, RerankerProtocol

log = logging.getLogger(__name__)


class CrossEncoderReranker(RerankerProtocol):
    """Reranker basado en cross-encoder para segunda etapa de precisión."""

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._model = None

    def _get_model(self):
        """Carga perezosa del modelo para evitar coste al inicio del proceso."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info(
                "[rerank] Cargando cross-encoder model=%s max_length=%d",
                self._model_name,
                self._max_length,
            )
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """
        Reordena candidatos por relevancia con cross-encoder.

        Parámetros
        ----------
        query      : consulta textual.
        candidates : lista de candidatos recuperados previamente.
        top_k      : número máximo de resultados a devolver.
        """
        if not query or not query.strip():
            log.debug("[rerank] Query vacía; se devuelve lista vacía.")
            return []
        if not candidates:
            log.debug("[rerank] Sin candidatos de entrada; se devuelve lista vacía.")
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in candidates]

        scores = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        out: list[RetrievalResult] = []
        for score, item in scored[:top_k]:
            meta: dict[str, Any] = {
                **item.metadata,
                "retrieval_score": item.score,
                "retrieval_score_type": item.score_type,
                "rerank_model": self._model_name,
                "rerank_score": float(score),
            }
            out.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    arxiv_id=item.arxiv_id,
                    chunk_index=item.chunk_index,
                    text=item.text,
                    score=float(score),
                    score_type="rerank",
                    metadata=meta,
                )
            )

        best = out[0].score if out else float("-inf")
        log.debug(
            "[rerank] query='%s…' in=%d out=%d best=%.6f",
            query[:40],
            len(candidates),
            len(out),
            best,
        )
        return out
