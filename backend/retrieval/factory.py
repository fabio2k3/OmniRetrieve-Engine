"""
factory.py
==========
Construcción centralizada de retrievers del sistema.

Responsabilidad única
---------------------
Ser el único lugar donde se ensamblan retrievers con sus dependencias
reales (FaissIndexManager, rutas de modelos, etc.).

Las importaciones pesadas (sentence-transformers, FAISS) están al nivel
de módulo para que patch() en tests pueda interceptarlas correctamente.

API pública
-----------
build_faiss_manager()       → FaissIndexManager cargado desde disco.
build_lsi_retriever()       → LSIRetriever cargado e inicializado.
build_embedding_retriever() → EmbeddingRetriever con FAISS real.
build_hybrid_retriever()    → HybridRetriever (LSI + FAISS [+ reranker]).
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.embedding.faiss.index_manager import FaissIndexManager
from backend.embedding.embedder import DEFAULT_MODEL
from backend.retrieval.lsi_model import MODEL_PATH
from backend.retrieval.lsi_retriever import LSIRetriever
from backend.retrieval.embedding_retriever import EmbeddingRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker

log = logging.getLogger(__name__)


def build_faiss_manager() -> FaissIndexManager:
    """
    Construye y carga un FaissIndexManager con las rutas por defecto.

    Lanza RuntimeError si los ficheros de índice no existen en disco.
    """
    from sentence_transformers import SentenceTransformer
    from backend.embedding.pipeline import _INDEX_PATH, _ID_MAP_PATH

    dim = SentenceTransformer(DEFAULT_MODEL).get_sentence_embedding_dimension()
    mgr = FaissIndexManager(dim=dim, index_path=_INDEX_PATH, id_map_path=_ID_MAP_PATH)
    loaded = mgr.load()

    if not loaded:
        from backend.embedding.pipeline import _INDEX_PATH, _ID_MAP_PATH
        raise RuntimeError(
            f"No se encontró el índice FAISS en disco.\n"
            f"  index  : {_INDEX_PATH}\n"
            f"  id_map : {_ID_MAP_PATH}\n"
            f"Ejecuta el pipeline de embedding antes de evaluar."
        )

    log.info("[factory] FaissIndexManager cargado: %d vectores", mgr.total_vectors)
    return mgr


def build_lsi_retriever(doc_candidates: int = 20) -> LSIRetriever:
    """
    Construye y carga un LSIRetriever con las rutas por defecto.

    Lanza RuntimeError si el modelo LSI no existe en disco.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No se encontró el modelo LSI en disco: {MODEL_PATH}\n"
            f"Ejecuta build_lsi antes de evaluar."
        )

    r = LSIRetriever(doc_candidates=doc_candidates)
    r.load()
    log.info("[factory] LSIRetriever cargado")
    return r


def build_embedding_retriever(faiss_mgr: FaissIndexManager | None = None) -> EmbeddingRetriever:
    """
    Construye un EmbeddingRetriever.

    Parámetros
    ----------
    faiss_mgr : FaissIndexManager ya construido (se reutiliza si se comparte).
                Si None, se construye uno nuevo.
    """
    if faiss_mgr is None:
        faiss_mgr = build_faiss_manager()

    r = EmbeddingRetriever(faiss_mgr=faiss_mgr)
    log.info("[factory] EmbeddingRetriever listo")
    return r


def build_hybrid_retriever(
    with_reranker:  bool = True,
    candidate_k:   int  = 50,
    rrf_k:         int  = 60,
    doc_candidates: int  = 20,
) -> HybridRetriever:
    """
    Construye el HybridRetriever completo (LSI + FAISS + reranker opcional).

    FaissIndexManager se construye una vez y se comparte entre componentes.

    Parámetros
    ----------
    with_reranker  : si True, añade CrossEncoderReranker como segunda etapa.
    candidate_k    : candidatos por rama antes del RRF.
    rrf_k          : constante de suavizado RRF.
    doc_candidates : documentos LSI a expandir a chunks por query.
    """
    faiss_mgr = build_faiss_manager()
    lsi       = build_lsi_retriever(doc_candidates=doc_candidates)
    dense     = build_embedding_retriever(faiss_mgr=faiss_mgr)
    reranker  = CrossEncoderReranker() if with_reranker else None

    retriever = HybridRetriever(
        sparse=lsi,
        dense=dense,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        reranker=reranker,
        rerank_k=None,
    )

    log.info(
        "[factory] HybridRetriever listo (candidate_k=%d rrf_k=%d reranker=%s)",
        candidate_k, rrf_k, "sí" if with_reranker else "no",
    )
    return retriever
