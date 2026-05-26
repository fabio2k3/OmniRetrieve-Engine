"""
retrieval
=========
Módulo de recuperación híbrida: LSI (sparse) + FAISS (dense) + CrossEncoder (rerank).

API pública
-----------
Tipos y protocolos:
    RetrievalResult     — resultado unificado a nivel chunk.
    RetrieverProtocol   — contrato de cualquier retriever.
    RerankerProtocol    — contrato de cualquier reranker.

Retrievers:
    LSIRetriever        — sparse, implementa RetrieverProtocol (chunk_ids reales).
    EmbeddingRetriever  — dense, FAISS + sentence-transformers.
    HybridRetriever     — fusión RRF de sparse + dense.

Reranker:
    CrossEncoderReranker — segunda etapa con cross-encoder MS MARCO.

Modelo LSI (fase offline):
    LSIModel            — construcción y persistencia del espacio latente.

Factory (construcción con configuración real del sistema):
    build_lsi_retriever()       — LSIRetriever cargado desde disco.
    build_embedding_retriever() — EmbeddingRetriever con FAISS real.
    build_hybrid_retriever()    — pipeline completo listo para usar.
    build_faiss_manager()       — FaissIndexManager cargado desde disco.
"""

from .protocols import RetrievalResult, RetrieverProtocol, RerankerProtocol
from .lsi_model import LSIModel, MODEL_PATH, MODEL_DIR
from .lsi_retriever import LSIRetriever
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker
from .factory import (
    build_faiss_manager,
    build_lsi_retriever,
    build_embedding_retriever,
    build_hybrid_retriever,
)

__all__ = [
    #Paths
    "MODEL_PATH", "MODEL_DIR",
    # Protocolos y tipos
    "RetrievalResult",
    "RetrieverProtocol",
    "RerankerProtocol",
    # Retrievers
    "LSIRetriever",
    "EmbeddingRetriever",
    "HybridRetriever",
    # Reranker
    "CrossEncoderReranker",
    # Modelo offline
    "LSIModel",
    # Factory
    "build_faiss_manager",
    "build_lsi_retriever",
    "build_embedding_retriever",
    "build_hybrid_retriever",
]
