"""
hybrid_retriever.py
===================
Hybrid retriever con fusión RRF (Reciprocal Rank Fusion).

Combina cualquier retriever sparse + dense que cumpla RetrieverProtocol.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging

from .protocols import RetrievalResult, RerankerProtocol, RetrieverProtocol

log = logging.getLogger(__name__)


class HybridRetriever:
    """
    Fusión híbrida basada en RRF.

    Parámetros
    ----------
    sparse      : retriever léxico/sparse.
    dense       : retriever semántico/denso.
    candidate_k : candidatos recuperados por cada rama antes de fusionar.
    rrf_k       : constante del RRF (k más alto => fusión más suave).
    parallel    : si True, ejecuta sparse+dense en paralelo con TPE.
    """

    def __init__(
        self,
        sparse: RetrieverProtocol,
        dense: RetrieverProtocol,
        candidate_k: int = 50,
        rrf_k: int = 60,
        parallel: bool = True,
        reranker: RerankerProtocol | None = None,
        rerank_k: int | None = None,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.parallel = parallel
        self.reranker = reranker
        self.rerank_k = rerank_k

    def retrieve(self, query: str, top_n: int = 10) -> list[RetrievalResult]:
        """Recupera y fusiona resultados sparse+dense con RRF."""
        if not query or not query.strip():
            log.debug("[hybrid] Query vacía; se devuelve lista vacía.")
            return []

        mode = "parallel" if self.parallel else "sequential"
        log.debug(
            "[hybrid] Iniciando retrieve mode=%s candidate_k=%d rrf_k=%d",
            mode,
            self.candidate_k,
            self.rrf_k,
        )

        if self.parallel:
            # Para consultas, TPE suele funcionar bien: ambas ramas son mayormente
            # cómputo en código nativo (NumPy/FAISS/torch) + lecturas I/O.
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_sparse = ex.submit(self.sparse.retrieve, query, self.candidate_k)
                fut_dense = ex.submit(self.dense.retrieve, query, self.candidate_k)
                sparse_res = fut_sparse.result()
                dense_res = fut_dense.result()
        else:
            sparse_res = self.sparse.retrieve(query, self.candidate_k)
            dense_res = self.dense.retrieve(query, self.candidate_k)

        merged = self._rrf(sparse_res, dense_res, top_n)

        if self.reranker is not None and merged:
            rk = self.rerank_k if self.rerank_k is not None else top_n
            log.debug("[hybrid] Aplicando reranker top_k=%d", rk)
            merged = self.reranker.rerank(query, merged, top_k=rk)

        log.debug(
            "[hybrid] sparse=%d dense=%d merged=%d",
            len(sparse_res),
            len(dense_res),
            len(merged),
        )
        return merged

    def _rrf(
        self,
        sparse: list[RetrievalResult],
        dense: list[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        """Fusiona dos rankings con Reciprocal Rank Fusion."""
        scores: dict[int | str, float] = {}
        base: dict[int | str, RetrievalResult] = {}

        for rank, item in enumerate(sparse, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            base.setdefault(item.chunk_id, item)

        for rank, item in enumerate(dense, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            base.setdefault(item.chunk_id, item)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        out: list[RetrievalResult] = []
        for chunk_id, rrf_score in ranked[:top_n]:
            ref = base[chunk_id]
            out.append(
                RetrievalResult(
                    chunk_id=ref.chunk_id,
                    arxiv_id=ref.arxiv_id,
                    chunk_index=ref.chunk_index,
                    text=ref.text,
                    score=rrf_score,
                    score_type="rrf",
                    metadata={
                        **ref.metadata,
                        "rrf_k": self.rrf_k,
                        "candidate_k": self.candidate_k,
                    },
                )
            )
        return out
