"""
_meta.py
========
Persistencia de metadatos y presentación de estadísticas del módulo embedding.

Módulo interno (prefijo ``_``). No forma parte de la API pública del paquete.

Responsabilidad única
---------------------
Registrar en la BD los metadatos de cada ejecución del pipeline
(modelo usado, timestamps, número de chunks procesados, builds de FAISS)
y mostrar un resumen del estado por pantalla cuando se invoca desde CLI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .faiss import FaissIndexManager

log = logging.getLogger(__name__)


def log_faiss_build(
    faiss_mgr:  FaissIndexManager,
    model_name: str,
    db_path:    Path,
) -> None:
    """
    Registra la construcción del índice FAISS en la tabla ``faiss_log`` de la BD.

    Parámetros
    ----------
    faiss_mgr  : gestor del índice FAISS (para obtener las stats del build).
    model_name : nombre del modelo de embedding activo.
    db_path    : ruta a la BD SQLite.
    """
    from backend.database.embedding_repository import log_faiss_build as _log

    stats = faiss_mgr.build_stats()
    stats["model_name"] = model_name
    _log(stats, db_path)


def save_run_meta(stats: dict, db_path: Path) -> None:
    """
    Persiste los metadatos del último run en la tabla ``embedding_meta``.

    Parámetros
    ----------
    stats   : dict devuelto por ``EmbeddingPipeline.run()``.
    db_path : ruta a la BD SQLite.
    """
    from backend.database.embedding_repository import save_embedding_meta as _save

    _save("last_run_at",          stats["finished_at"],             db_path)
    _save("last_chunks_embedded", str(stats["chunks_processed"]),   db_path)
    _save("last_model",           stats["model_name"],               db_path)


def print_stats(db_path: Path) -> None:
    """
    Imprime un resumen del estado del módulo embedding por stdout.

    Usado por el flag ``--stats`` de la CLI.

    Parámetros
    ----------
    db_path : ruta a la BD SQLite.
    """
    from backend.database.embedding_repository import (
        init_embedding_schema,
        get_embedding_stats,
    )

    try:
        init_embedding_schema(db_path)
        s = get_embedding_stats(db_path)
    except Exception as exc:
        log.error("Error al obtener estadísticas: %s", exc)
        sys.exit(1)

    print("\n── Embedding Stats ──────────────────────────")
    print(f"  Chunks totales     : {s['total_chunks']}")
    print(f"  Embedidos          : {s['embedded_chunks']}")
    print(f"  Pendientes         : {s['pending_chunks']}")
    print(f"  Último build FAISS : {s['last_build_at'] or 'nunca'}")
    print(f"  Tipo de índice     : {s['last_index_type'] or 'N/A'}")
    print(f"  Vectores en índice : {s['last_n_vectors']}")
    print("─────────────────────────────────────────────\n")
