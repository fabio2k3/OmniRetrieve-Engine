"""
config.py
=========
Configuración del orquestador OmniRetrieve-Engine.

Todos los parámetros ajustables del sistema viven aquí agrupados por módulo.
Ningún otro fichero del paquete debe hardcodear valores de comportamiento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.database.schema import DB_PATH, DATA_DIR
from backend.retrieval.lsi_model import MODEL_PATH

# Rutas por defecto del índice FAISS
_FAISS_DIR    = DATA_DIR / "faiss"
_FAISS_INDEX  = _FAISS_DIR / "index.faiss"
_FAISS_ID_MAP = _FAISS_DIR / "id_map.npy"


@dataclass
class OrchestratorConfig:
    """
    Parámetros de configuración del orquestador.

    Rutas
    -----
    db_path    : base de datos SQLite compartida por todos los módulos.
    model_path : fichero .pkl donde se guarda el modelo LSI.

    Crawler
    -------
    ids_per_discovery  : IDs a descubrir por ciclo.
    batch_size         : metadatos de artículos a descargar por ciclo.
    pdf_batch_size     : PDFs a descargar por ciclo.
    discovery_interval : segundos entre ciclos de discovery.
    download_interval  : segundos entre ciclos de descarga de metadatos.
    pdf_interval       : segundos entre ciclos de descarga de PDF.

    Indexing (BM25)
    ---------------
    pdf_threshold       : mínimo de PDFs nuevos sin indexar para disparar indexación.
    index_poll_interval : segundos entre sondeos del watcher.
    index_field         : campo a indexar: 'full_text', 'abstract' o 'both'.
    index_batch_size    : documentos por lote en IndexingPipeline BM25.
    index_use_stemming  : activa SnowballStemmer en el preprocesado BM25.
    index_min_token_len : longitud mínima de token para BM25.

    LSI rebuild
    -----------
    lsi_rebuild_interval : segundos entre reconstrucciones del modelo LSI.
    lsi_k                : número de componentes latentes del SVD.
    lsi_min_docs         : mínimo de documentos indexados para construir el modelo.

    Embedding / FAISS
    -----------------
    embed_model         : nombre del modelo sentence-transformers.
    embed_batch_size    : chunks por lote en cada llamada al embedder.
    embed_poll_interval : segundos entre sondeos del watcher de embedding.
    embed_threshold     : mínimo de chunks sin embedding para disparar el pipeline.
    embed_rebuild_every : chunks añadidos entre reconstrucciones completas de FAISS.
    embed_nlist         : celdas Voronoi para IndexIVFPQ.
    embed_m             : subvectores PQ (debe dividir la dimensión del modelo).
    embed_nbits         : bits por código PQ.
    embed_nprobe        : celdas inspeccionadas durante la búsqueda semántica.
    faiss_index_path    : ruta del fichero .faiss serializado.
    faiss_id_map_path   : ruta del fichero .npy con el mapa posición → chunk_id.

    Web Search
    ----------
    web_threshold     : score mínimo del retriever LSI para no activar búsqueda web.
    web_min_docs      : docs mínimos que deben superar web_threshold.
    web_max_results   : máximo de resultados a pedir a Tavily / DuckDuckGo.
    web_search_depth  : profundidad de búsqueda Tavily ("basic" | "advanced").
    web_use_fallback  : usar DuckDuckGo si Tavily falla.
    web_auto_index    : indexar automáticamente los docs web guardados.

    QRF (Query Refinement Framework)
    ---------------------------------
    qrf_top_k_initial    : candidatos recuperados en la búsqueda inicial para BRF.
    qrf_expand           : activar expansión LCE (Latent Concept Expansion) vía LSI.
    qrf_expand_top_dims  : dimensiones latentes inspeccionadas en la expansión LCE.
    qrf_expand_min_corr  : correlación mínima para añadir un término de expansión.
    qrf_expand_max_terms : máximo de términos nuevos añadidos por LCE.
    qrf_brf_alpha        : peso del vector original en BRF (Blind Relevance Feedback).
    qrf_brf_top_k        : top resultados usados para calcular el centroide BRF.
    qrf_mmr_lambda       : balance relevancia/diversidad en MMR (1.0 = solo relevancia).

    RAG (Retrieval-Augmented Generation)
    -------------------------------------
    rag_use_reranker   : activar CrossEncoderReranker de segunda etapa.
    rag_reranker_model : nombre del modelo cross-encoder de sentence-transformers.
    rag_candidate_k    : candidatos recuperados por EmbeddingRetriever antes del reranking.
    rag_max_chunks     : chunks máximos inyectados en el contexto enviado al LLM.
    rag_max_chars      : caracteres máximos por chunk dentro del contexto LLM.
    """

    # ── rutas ────────────────────────────────────────────────────────────────
    db_path:    Path = field(default_factory=lambda: DB_PATH)
    model_path: Path = field(default_factory=lambda: MODEL_PATH)

    # ── crawler ──────────────────────────────────────────────────────────────
    ids_per_discovery:  int   = 100
    batch_size:         int   = 10
    pdf_batch_size:     int   = 5
    discovery_interval: float = 120.0
    download_interval:  float = 30.0
    pdf_interval:       float = 60.0

    # ── indexing (BM25) ──────────────────────────────────────────────────────
    pdf_threshold:       int   = 10
    index_poll_interval: float = 30.0
    index_field:         str   = "full_text"
    index_batch_size:    int   = 100
    index_use_stemming:  bool  = False
    index_min_token_len: int   = 3

    # ── LSI rebuild ──────────────────────────────────────────────────────────
    lsi_rebuild_interval: float = 3600.0
    lsi_k:                int   = 100
    lsi_min_docs:         int   = 10

    # ── embedding / FAISS ────────────────────────────────────────────────────
    embed_model:         str   = "all-MiniLM-L6-v2"
    embed_batch_size:    int   = 256
    embed_poll_interval: float = 60.0
    embed_threshold:     int   = 50
    embed_rebuild_every: int   = 10_000
    embed_nlist:         int   = 100
    embed_m:             int   = 8
    embed_nbits:         int   = 8
    embed_nprobe:        int   = 10
    faiss_index_path:    Path  = field(default_factory=lambda: _FAISS_INDEX)
    faiss_id_map_path:   Path  = field(default_factory=lambda: _FAISS_ID_MAP)

    # ── web search ───────────────────────────────────────────────────────────
    web_threshold:    float = 0.15
    web_min_docs:     int   = 1
    web_max_results:  int   = 5
    web_search_depth: str   = "basic"
    web_use_fallback: bool  = True
    web_auto_index:   bool  = True

    # ── QRF (Query Refinement Framework) ─────────────────────────────────────
    # Pipeline completo: expansión LCE + embedding + BRF + MMR.
    qrf_top_k_initial:    int   = 20     # candidatos en la búsqueda inicial (BRF)
    qrf_expand:           bool  = True   # activar expansión LCE vía LSI
    qrf_expand_top_dims:  int   = 3      # dimensiones latentes a examinar en LCE
    qrf_expand_min_corr:  float = 0.4   # umbral mínimo de correlación para añadir término
    qrf_expand_max_terms: int   = 8      # máximo de términos nuevos añadidos por LCE
    qrf_brf_alpha:        float = 0.75  # peso del vector original en BRF
    qrf_brf_top_k:        int   = 5      # resultados usados para el centroide BRF
    qrf_mmr_lambda:       float = 0.6   # balance relevancia/diversidad en MMR (1=solo relevancia)

    # ── RAG (Retrieval-Augmented Generation) ──────────────────────────────────
    # Recuperación densa + generación LLM. El retriever usa FAISS (EmbeddingRetriever).
    rag_use_reranker:   bool  = False                                     # activar CrossEncoderReranker
    rag_reranker_model: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # modelo del reranker
    rag_candidate_k:    int   = 50   # candidatos recuperados antes de reranking
    rag_max_chunks:     int   = 5    # chunks máximos inyectados en el contexto LLM
    rag_max_chars:      int   = 400  # caracteres máximos por chunk en el contexto LLM
