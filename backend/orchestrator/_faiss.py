"""
_faiss.py
=========
Inicialización del índice FAISS.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Extrae dos responsabilidades que antes vivían mezcladas en ``Orchestrator.__init__``:

    resolve_embedding_dim  — obtiene la dimensión del modelo sin cargar sus pesos.
    init_faiss_mgr         — crea el FaissIndexManager y carga el índice previo.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.embedding import FaissIndexManager          # ← import correcto
from .config import OrchestratorConfig

log = logging.getLogger(__name__)

# Mapa de dimensiones conocidas para los modelos sentence-transformers más comunes.
# Evita cargar los pesos completos del modelo solo para conocer su dimensión.
_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2":          384,
    "all-MiniLM-L12-v2":         384,
    "all-mpnet-base-v2":         768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "allenai-specter":            768,
    "paraphrase-MiniLM-L6-v2":   384,
}


def resolve_embedding_dim(model_name: str) -> int:
    """
    Devuelve la dimensión del modelo de embedding sin cargar sus pesos.

    Consulta ``_KNOWN_DIMS``; si el modelo no está en el mapa, devuelve
    384 como valor por defecto (dimensión de ``all-MiniLM-L6-v2``).

    Parámetros
    ----------
    model_name : nombre del modelo sentence-transformers.

    Returns
    -------
    int
        Dimensión del espacio de embedding.
    """
    return _KNOWN_DIMS.get(model_name, 384)


def init_faiss_mgr(cfg: OrchestratorConfig) -> tuple[FaissIndexManager, bool]:
    """
    Crea el ``FaissIndexManager`` e intenta cargar un índice previo desde disco.

    Parámetros
    ----------
    cfg : configuración del orquestador.

    Returns
    -------
    tuple[FaissIndexManager, bool]
        ``(manager, loaded)`` donde ``loaded`` indica si se cargó un índice
        existente desde disco. Si ``loaded`` es ``True``, el índice está
        listo para búsquedas inmediatamente.
    """
    dim = resolve_embedding_dim(cfg.embed_model)

    mgr = FaissIndexManager(
        dim           = dim,
        nlist         = cfg.embed_nlist,
        m             = cfg.embed_m,
        nbits         = cfg.embed_nbits,
        nprobe        = cfg.embed_nprobe,
        rebuild_every = cfg.embed_rebuild_every,
        index_path    = cfg.faiss_index_path,
        id_map_path   = cfg.faiss_id_map_path,
    )

    loaded = mgr.load()
    if loaded:
        log.info(
            "[faiss] Índice cargado desde disco — tipo=%s vectores=%d",
            mgr.index_type, mgr.total_vectors,
        )
    else:
        log.info("[faiss] No hay índice previo en disco. Se creará al primer embedding.")

    return mgr, loaded
