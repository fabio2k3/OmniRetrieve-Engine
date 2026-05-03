"""
pipeline.py
===========
Orquestador RAG: retrieval -> rerank -> contexto -> prompt -> generacion.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.retrieval.protocols import RetrievalResult, RetrieverProtocol, RerankerProtocol
from .context_builder import ContextBuilder
from .prompt_builder import PromptBuilder
from .generator import Generator

log = logging.getLogger(__name__)


class RAGPipeline:
    """Coordina etapas RAG manteniendo responsabilidades separadas."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.generator = generator or Generator()

    def search(self, query: str, top_k: int = 10, candidate_k: int = 50) -> list[dict[str, Any]]:
        """Ejecucion de retrieval (+ rerank opcional) sin generacion LLM."""
        ranked = self._retrieve_and_rank(query=query, top_k=top_k, candidate_k=candidate_k)
        out: list[dict[str, Any]] = []
        for r in ranked:
            out.append(
                {
                    "chunk_id": r.chunk_id,
                    "arxiv_id": r.arxiv_id,
                    "chunk_index": r.chunk_index,
                    "title": r.metadata.get("title", r.arxiv_id),
                    "text": r.text[:300],
                    "score": r.score,
                    "score_type": r.score_type,
                }
            )
        return out

    def ask(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        max_chunks: int = 5,
        max_chars: int = 400,
        include_debug: bool = False,
    ) -> dict[str, Any]:
        """Ruta RAG completa: retrieval, rerank, contexto, prompt y respuesta."""
        ranked = self._retrieve_and_rank(query=query, top_k=top_k, candidate_k=candidate_k)

        context = self.context_builder.build(
            ranked,
            max_chunks=max_chunks,
            max_chars=max_chars,
        )
        prompt = self.prompt_builder.build(query=query, context=context)
        answer = self.generator.generate(prompt)

        payload: dict[str, Any] = {
            "query": query,
            "answer": answer,
            "sources": self.context_builder.build_sources(ranked, max_sources=max_chunks),
        }

        if include_debug:
            payload["context"] = context
            payload["prompt"] = prompt

        return payload


    def generate_from_results(
        self,
        query:      str,
        results:    list,
        max_chunks: int = 5,
        max_chars:  int = 400,
    ) -> dict:
        """
        Genera una respuesta LLM a partir de resultados ya recuperados y
        rerankeados (sin hacer retrieval propio).

        Usado por el pipeline unificado de do_pipeline_ask(), donde el
        HybridRetriever + CrossEncoder ya produjeron los chunks finales.

        Parametros
        ----------
        query      : pregunta original del usuario.
        results    : lista de RetrievalResult ya ordenados por relevancia.
        max_chunks : maximo de chunks a incluir en el contexto.
        max_chars  : maximo de caracteres por chunk en el contexto.

        Returns
        -------
        dict con claves: query, answer, sources.
        """
        context = self.context_builder.build(
            results,
            max_chunks=max_chunks,
            max_chars=max_chars,
        )
        prompt = self.prompt_builder.build(query=query, context=context)
        answer = self.generator.generate(prompt)

        return {
            "query":   query,
            "answer":  answer,
            "sources": self.context_builder.build_sources(results, max_sources=max_chunks),
        }

    def _retrieve_and_rank(self, query: str, top_k: int, candidate_k: int) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []

        n_candidates = max(top_k, candidate_k)
        retrieved = self.retriever.retrieve(query, top_n=n_candidates)

        if self.reranker is None:
            return retrieved[:top_k]

        reranked = self.reranker.rerank(query, retrieved, top_k=top_k)
        log.debug(
            "[rag] query='%s…' retrieved=%d reranked=%d",
            query[:40],
            len(retrieved),
            len(reranked),
        )
        return reranked