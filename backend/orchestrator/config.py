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
