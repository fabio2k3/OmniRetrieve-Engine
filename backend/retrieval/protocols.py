"""
protocols.py
============
Contratos de recuperación para arquitectura híbrida en OmniRetrieve-Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RetrievalResult:
    """Unidad de resultado de cualquier retriever (nivel chunk)."""

    chunk_id: int | str
    arxiv_id: str
    chunk_index: int
    text: str
    score: float
    score_type: str = "bm25"  # bm25 | cosine | l2 | rrf | rerank
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Contrato mínimo para cualquier retriever del sistema."""

    def retrieve(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        """Devuelve los top_n chunks más relevantes para la query."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Contrato mínimo para rerankers de segunda etapa."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Reordena candidatos y devuelve los top_k más relevantes."""
        ...
