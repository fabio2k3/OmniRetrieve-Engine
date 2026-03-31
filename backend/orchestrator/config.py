"""
config.py
=========
Configuración del orquestador OmniRetrieve-Engine.

Contiene únicamente el dataclass OrchestratorConfig con todos los
parámetros ajustables del sistema: rutas, tiempos de ciclo del crawler,
umbral de indexación y parámetros del modelo LSI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.embedding.chroma_store import CHROMA_PATH
from backend.retrieval.lsi_model import MODEL_PATH


@dataclass
class OrchestratorConfig:
    """
    Parámetros de configuración del orquestador.

    Rutas
    -----
    db_path : Path
        Base de datos SQLite compartida por todos los módulos.
    model_path : Path
        Archivo .pkl donde se guarda el modelo LSI.

    Crawler
    -------
    ids_per_discovery : int
        IDs de arXiv a descubrir por ciclo.
    batch_size : int
        Metadatos de artículos a descargar por ciclo.
    pdf_batch_size : int
        PDFs a descargar por ciclo.
    discovery_interval : float
        Segundos entre ciclos de discovery.
    download_interval : float
        Segundos entre ciclos de descarga de metadatos.
    pdf_interval : float
        Segundos entre ciclos de descarga de PDF.

    Indexing watcher
    ----------------
    pdf_threshold : int
        Mínimo de PDFs nuevos sin indexar para disparar IndexingPipeline.
    index_poll_interval : float
        Segundos entre sondeos del watcher.
    index_field : str
        Campo a indexar: 'full_text', 'abstract' o 'both'.

    LSI rebuild
    -----------
    lsi_rebuild_interval : float
        Segundos entre reconstrucciones del modelo LSI.
    lsi_k : int
        Número de componentes latentes del SVD.
    lsi_min_docs : int
        Mínimo de documentos indexados para intentar construir el modelo.
        Debe ser > lsi_k para que SVD tenga sentido.
    """

    # rutas
    db_path:     Path = field(default_factory=lambda: DB_PATH)
    model_path:  Path = field(default_factory=lambda: MODEL_PATH)
    chroma_path: Path = field(default_factory=lambda: CHROMA_PATH)

    # crawler
    ids_per_discovery:  int   = 500
    batch_size:         int   = 50
    pdf_batch_size:     int   = 10
    discovery_interval: float = 120.0
    download_interval:  float = 30.0
    pdf_interval:       float = 5.0

    # indexing watcher
    pdf_threshold:       int   = 10
    index_poll_interval: float = 30.0
    index_field:         str   = "full_text"

    # LSI rebuild
    lsi_rebuild_interval: float = 3600.0
    lsi_k:                int   = 100
    lsi_min_docs:         int   = 10

    # Embedding vectorial
    embed_model:            str   = "all-MiniLM-L6-v2"
    embed_batch_size:       int   = 64
    embed_poll_interval:    float = 60.0
    embed_threshold:        int   = 10      # chunks sin embedding para disparar pipeline
    vector_reload_interval: float = 300.0   # segundos entre recargas de la matriz vectorial