"""
chroma_store.py
===============
Capa de acceso a ChromaDB para el modulo de embeddings.

Responsabilidad unica
---------------------
Encapsula toda interaccion con ChromaDB: apertura del cliente persistente,
acceso a la coleccion de chunks, y las operaciones de escritura/lectura
que usan el resto de modulos.

Diseno
------
- Cache de clientes por ruta: evita abrir multiples clientes sobre el mismo
  directorio, lo que causa errores en ChromaDB.
- close_client() / close_all_clients(): liberan los ficheros antes de que
  el SO intente borrar el directorio (critico en Windows).
- Coleccion unica: COLLECTION_NAME = "chunks".
- Los IDs de Chroma tienen la forma "chunk_{id}" donde id es el INTEGER
  PRIMARY KEY de la tabla chunks en SQLite.
- Los metadatos almacenados por vector: arxiv_id, chunk_index, char_count.
  El texto se guarda como document para posibles busquedas hibridas futuras.
- Stateless excepto el cache de clientes: todas las funciones reciben
  la ruta de Chroma como parametro.

Rutas por defecto
-----------------
  data/chromadb/   -- directorio persistente de ChromaDB
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, List

import numpy as np

log = logging.getLogger(__name__)

_DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
CHROMA_PATH     = _DATA_DIR / "chromadb"
COLLECTION_NAME = "chunks"

# Cache de clientes indexado por ruta absoluta como string.
# Evita crear multiples PersistentClient sobre el mismo directorio
# y permite cerrarlos explicitamente (necesario en Windows).
_clients: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Gestion del cliente
# ---------------------------------------------------------------------------

def get_collection(chroma_path: Path = CHROMA_PATH):
    """
    Devuelve la coleccion ChromaDB para la ruta dada, creando el cliente
    y la coleccion si no existen todavia.

    Usa un cache interno para no abrir multiples clientes sobre el mismo
    directorio. En Windows esto evita que chromadb.sqlite3 quede bloqueado.
    """
    import chromadb

    path_str = str(chroma_path.resolve())

    if path_str not in _clients:
        chroma_path.mkdir(parents=True, exist_ok=True)
        _clients[path_str] = chromadb.PersistentClient(path=path_str)
        log.debug("[ChromaStore] cliente abierto en %s", path_str)

    return _clients[path_str].get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )


def close_client(chroma_path: Path = CHROMA_PATH) -> None:
    """
    Elimina el cliente cacheado para la ruta dada y fuerza la recoleccion
    de basura para que ChromaDB libere los ficheros del SO.

    Necesario en Windows antes de borrar el directorio de Chroma en tests.
    En produccion no hace falta llamarlo.
    """
    path_str = str(chroma_path.resolve())
    if path_str in _clients:
        del _clients[path_str]
        gc.collect()
        log.debug("[ChromaStore] cliente cerrado para %s", path_str)


def close_all_clients() -> None:
    """
    Cierra todos los clientes cacheados.
    Util al final de una suite de tests o al apagar el proceso.
    """
    _clients.clear()
    gc.collect()
    log.debug("[ChromaStore] todos los clientes cerrados")


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def add_chunks(
    chunk_ids:   List[int],
    arxiv_ids:   List[str],
    chunk_idxs:  List[int],
    texts:       List[str],
    char_counts: List[int],
    embeddings:  np.ndarray,
    chroma_path: Path = CHROMA_PATH,
) -> None:
    """
    Inserta o actualiza un lote de chunks en ChromaDB (upsert).

    Los IDs de Chroma tienen la forma "chunk_{chunk_id}" para mantener
    correspondencia directa con la tabla chunks de SQLite.
    """
    collection = get_collection(chroma_path)

    ids       = [f"chunk_{cid}" for cid in chunk_ids]
    metadatas = [
        {"arxiv_id": aid, "chunk_index": cidx, "char_count": cc}
        for aid, cidx, cc in zip(arxiv_ids, chunk_idxs, char_counts)
    ]

    collection.upsert(
        ids        = ids,
        embeddings = embeddings.tolist(),
        documents  = texts,
        metadatas  = metadatas,
    )
    log.debug("[ChromaStore] upsert %d chunks", len(ids))


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def get_existing_chunk_ids(chroma_path: Path = CHROMA_PATH) -> set[int]:
    """
    Devuelve el conjunto de chunk_ids (enteros SQLite) presentes en ChromaDB.
    Nota: con embedded_at como criterio en SQLite esto ya no es necesario
    en el pipeline, pero se conserva para inspeccion y herramientas.
    """
    collection = get_collection(chroma_path)
    if collection.count() == 0:
        return set()
    result = collection.get(include=[])
    return {int(cid.split("_")[1]) for cid in result["ids"]}


def count(chroma_path: Path = CHROMA_PATH) -> int:
    """Devuelve el numero de vectores almacenados en la coleccion."""
    return get_collection(chroma_path).count()


def query(
    query_embedding: np.ndarray,
    n_results:       int  = 40,
    chroma_path:     Path = CHROMA_PATH,
) -> list[dict]:
    """
    Busca los n_results chunks mas cercanos al vector de consulta.

    Devuelve lista de dicts con: chunk_id, arxiv_id, chunk_index,
    char_count, text, distance. Ordenada por distancia ascendente
    (menor distancia = mayor similitud coseno).
    """
    collection = get_collection(chroma_path)

    if collection.count() == 0:
        return []

    n_results = min(n_results, collection.count())
    result    = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results        = n_results,
        include          = ["metadatas", "documents", "distances"],
    )

    hits = []
    for i, chroma_id in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i]
        hits.append({
            "chunk_id":    int(chroma_id.split("_")[1]),
            "arxiv_id":    meta["arxiv_id"],
            "chunk_index": meta["chunk_index"],
            "char_count":  meta.get("char_count", 0),
            "text":        result["documents"][0][i],
            "distance":    result["distances"][0][i],
        })
    return hits