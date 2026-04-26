"""backend/embedding — vectorización de chunks e índice FAISS."""
from .embedder  import ChunkEmbedder
from .faiss     import FaissIndexManager
from .pipeline  import EmbeddingPipeline

__all__ = ["ChunkEmbedder", "FaissIndexManager", "EmbeddingPipeline"]