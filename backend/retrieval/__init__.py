"""API pública del módulo de retrieval."""

from .protocols import RetrievalResult, RetrieverProtocol, RerankerProtocol
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .lsi_model import LSIModel
from .lsi_retriever import LSIRetriever

__all__ = [
    "RetrievalResult",
    "RetrieverProtocol",
    "RerankerProtocol",
    "EmbeddingRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "LSIModel",
    "LSIRetriever",
]
