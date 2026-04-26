"""
faiss/index_manager.py
======================
Gestiona el ciclo de vida completo del índice FAISS.

Responsabilidad única
---------------------
Coordinar las operaciones de alto nivel sobre el índice: añadir vectores,
disparar reconstrucciones, buscar y persistir/cargar en disco.

La lógica de *construcción* del índice (entrenamiento de IVFPQ, creación de
FlatL2) está delegada en ``builder.py`` para mantener este fichero enfocado
en la gestión del estado y no en los detalles de FAISS.

Tipo de índice
--------------
``IndexIVFPQ`` (Inverted File + Product Quantization) con fallback a
``IndexFlatL2`` cuando no hay suficientes vectores para entrenar IVFPQ.
El fallback se reemplaza automáticamente en la primera reconstrucción con
datos suficientes.

Mapeo de IDs
------------
FAISS asigna posiciones enteras 0-based internamente. ``_id_map`` es un
array NumPy ``int64`` que traduce ``posición_faiss → chunk_id`` de la BD.
Se persiste en disco junto al índice.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    DEFAULT_NLIST, DEFAULT_M, DEFAULT_NBITS,
    DEFAULT_NPROBE, DEFAULT_REBUILD_EVERY,
)
from . import builder as _builder

log = logging.getLogger(__name__)


class FaissIndexManager:
    """
    Gestiona construcción, actualización, búsqueda y persistencia de un índice FAISS.

    Parámetros
    ----------
    dim           : dimensión del embedding (debe coincidir con el modelo).
    nlist         : número de celdas Voronoi para IndexIVFPQ.
    m             : número de subvectores PQ (debe dividir a ``dim``).
    nbits         : bits por código PQ.
    nprobe        : celdas inspeccionadas durante la búsqueda.
    rebuild_every : chunks añadidos entre reconstrucciones completas.
    index_path    : ruta donde se guarda/carga el índice FAISS (.faiss).
    id_map_path   : ruta donde se guarda/carga el mapa chunk_id (.npy).
    """

    def __init__(
        self,
        dim:           int,
        nlist:         int  = DEFAULT_NLIST,
        m:             int  = DEFAULT_M,
        nbits:         int  = DEFAULT_NBITS,
        nprobe:        int  = DEFAULT_NPROBE,
        rebuild_every: int  = DEFAULT_REBUILD_EVERY,
        index_path:    Optional[Path] = None,
        id_map_path:   Optional[Path] = None,
    ) -> None:
        self.dim           = dim
        self.nlist         = nlist
        self.m             = m
        self.nbits         = nbits
        self.nprobe        = nprobe
        self.rebuild_every = rebuild_every
        self.index_path    = Path(index_path)  if index_path  else None
        self.id_map_path   = Path(id_map_path) if id_map_path else None

        self._validate()

        self._index                     = None
        self._id_map: np.ndarray        = np.empty(0, dtype=np.int64)
        self._added_since_last_rebuild: int  = 0
        self._is_trained: bool          = False

        try:
            import faiss as _faiss
            self._faiss = _faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu (o faiss-gpu) no está instalado. "
                "Ejecuta: pip install faiss-cpu"
            ) from exc

        log.info(
            "[FaissIndex] Configurado — dim=%d nlist=%d m=%d nbits=%d rebuild_every=%d",
            dim, nlist, m, nbits, rebuild_every,
        )

    # ── Validación ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if self.dim % self.m != 0:
            raise ValueError(
                f"dim={self.dim} no es divisible por m={self.m}. "
                "Elige un valor de m que divida exactamente a la dimensión."
            )

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        """Número de vectores actualmente en el índice."""
        return int(self._id_map.shape[0])

    @property
    def index_type(self) -> str:
        """Tipo de índice activo: ``'IndexIVFPQ'``, ``'IndexFlatL2'`` o ``'none'``."""
        if self._index is None:
            return "none"
        return type(self._index).__name__

    def build_stats(self) -> dict:
        """
        Genera un dict con las métricas del índice actual.

        Usado por ``_meta.py`` para registrar cada build en la BD.

        Returns
        -------
        dict con: ``n_vectors``, ``index_type``, ``nlist``, ``m``, ``nbits``,
        ``index_path``, ``id_map_path``.
        """
        nlist_used = (
            self._index.nlist
            if self.index_type == "IndexIVFPQ" and self._index is not None
            else None
        )
        return {
            "n_vectors":   self.total_vectors,
            "index_type":  self.index_type,
            "nlist":       nlist_used,
            "m":           self.m     if self.index_type == "IndexIVFPQ" else None,
            "nbits":       self.nbits if self.index_type == "IndexIVFPQ" else None,
            "index_path":  str(self.index_path)   if self.index_path  else None,
            "id_map_path": str(self.id_map_path)  if self.id_map_path else None,
        }

    # ── Añadir vectores ───────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, chunk_ids: list[int]) -> None:
        """
        Añade vectores al índice con sus ``chunk_ids`` correspondientes.

        Si el índice aún no existe, inicializa un ``IndexFlatL2`` como
        almacenamiento temporal hasta la primera reconstrucción con IVFPQ.

        Parámetros
        ----------
        vectors   : ``np.ndarray`` de shape ``(N, dim)``, dtype float32.
        chunk_ids : lista de N enteros (IDs de la tabla ``chunks``).
        """
        if len(chunk_ids) == 0:
            return

        vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)

        if self._index is None:
            log.info("[FaissIndex] Inicializando índice plano (fallback).")
            self._index = _builder.build_flat(self._faiss, self.dim)

        self._index.add(vectors_c)
        self._id_map = np.concatenate([self._id_map, np.array(chunk_ids, dtype=np.int64)])
        self._added_since_last_rebuild += len(chunk_ids)

        log.debug(
            "[FaissIndex] +%d vectores | total=%d | desde último rebuild: %d/%d",
            len(chunk_ids), self.total_vectors,
            self._added_since_last_rebuild, self.rebuild_every,
        )

    # ── Reconstrucción completa ───────────────────────────────────────────────

    def rebuild(self, db_path: Path) -> dict:
        """
        Reconstruye el índice completo leyendo todos los embeddings de la BD.

        Flujo
        -----
        1. Lee todos los embeddings almacenados en ``chunks`` en lotes.
        2. Si hay suficientes vectores entrena ``IndexIVFPQ``;
           si no, usa ``IndexFlatL2`` como fallback.
        3. Añade todos los vectores al nuevo índice en una sola operación.
        4. Guarda en disco si ``index_path`` está configurado.
        5. Resetea el contador de añadidos.

        Parámetros
        ----------
        db_path : ruta a la BD SQLite (necesaria para leer embeddings).

        Returns
        -------
        dict
            Stats del build. Ver ``build_stats()``.
        """
        from backend.database.chunk_repository import get_all_embeddings_iter

        log.info("[FaissIndex] Iniciando reconstrucción completa del índice…")
        t_start = time.perf_counter()

        all_vectors: list[np.ndarray] = []
        all_ids:     list[int]        = []

        for batch in get_all_embeddings_iter(db_path=db_path):
            for row in batch:
                all_vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
                all_ids.append(row["id"])
            n_read = len(all_vectors)
            if n_read % 10_000 == 0 and n_read > 0:
                log.info("[FaissIndex] Leyendo embeddings de BD… %d leídos", n_read)

        n = len(all_vectors)
        log.info(
            "[FaissIndex] Embeddings leídos: %d (en %.2fs)",
            n, time.perf_counter() - t_start,
        )

        if n == 0:
            log.warning("[FaissIndex] No hay embeddings; índice vacío.")
            self._index  = _builder.build_flat(self._faiss, self.dim)
            self._id_map = np.empty(0, dtype=np.int64)
            self._added_since_last_rebuild = 0
            return self.build_stats()

        matrix    = np.stack(all_vectors).astype(np.float32)
        min_train = _builder.min_train_size(self.nlist, self.nbits)

        if n >= min_train:
            new_index        = _builder.build_ivfpq(
                self._faiss, self.dim, matrix,
                self.nlist, self.m, self.nbits, self.nprobe,
            )
            self._is_trained = True
        else:
            log.info(
                "[FaissIndex] Solo %d vectores (mínimo %d para IVFPQ) → usando FlatL2.",
                n, min_train,
            )
            new_index = _builder.build_flat(self._faiss, self.dim)

        log.info("[FaissIndex] Añadiendo %d vectores al índice…", n)
        t_add = time.perf_counter()
        new_index.add(np.ascontiguousarray(matrix, dtype=np.float32))
        log.info("[FaissIndex] Vectores añadidos en %.2fs.", time.perf_counter() - t_add)

        self._index    = new_index
        self._id_map   = np.array(all_ids, dtype=np.int64)
        self._added_since_last_rebuild = 0

        log.info(
            "[FaissIndex] Reconstrucción completa en %.2fs — tipo: %s | vectores: %d",
            time.perf_counter() - t_start, self.index_type, self.total_vectors,
        )

        if self.index_path:
            self.save()

        return self.build_stats()

    def maybe_rebuild(self, db_path: Path) -> bool:
        """
        Dispara ``rebuild()`` si se han añadido ``>= rebuild_every`` vectores
        desde el último rebuild.

        Parámetros
        ----------
        db_path : necesario para que ``rebuild()`` lea embeddings de la BD.

        Returns
        -------
        bool
            ``True`` si se ejecutó la reconstrucción.
        """
        if self._added_since_last_rebuild >= self.rebuild_every:
            log.info(
                "[FaissIndex] Umbral alcanzado (%d/%d) — reconstruyendo…",
                self._added_since_last_rebuild, self.rebuild_every,
            )
            self.rebuild(db_path)
            return True
        log.debug(
            "[FaissIndex] Umbral no alcanzado (%d/%d).",
            self._added_since_last_rebuild, self.rebuild_every,
        )
        return False

    # ── Búsqueda ──────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Busca los ``top_k`` chunks más cercanos a ``query_vector``.

        Parámetros
        ----------
        query_vector : ``np.ndarray`` 1-D de shape ``(dim,)``, dtype float32.
        top_k        : número de resultados a devolver.

        Returns
        -------
        list[dict]
            Lista de ``{"chunk_id": int, "score": float}`` ordenada por
            distancia ascendente (menor = más cercano). Lista vacía si el
            índice está vacío o no inicializado.
        """
        if self._index is None or self.total_vectors == 0:
            return []

        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.total_vectors)

        distances, indices = self._index.search(q, k)

        return [
            {"chunk_id": int(self._id_map[idx]), "score": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
            if idx >= 0
        ]

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """
        Guarda el índice FAISS y el mapa de IDs en disco.

        ``index_path``  → fichero FAISS serializado.
        ``id_map_path`` → array NumPy con los chunk_ids.

        No hace nada si las rutas no están configuradas.
        """
        if self.index_path is None or self.id_map_path is None:
            log.debug("[FaissIndex] Rutas no configuradas; omitiendo save.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(self.index_path))
        np.save(str(self.id_map_path), self._id_map)

        log.info(
            "[FaissIndex] Guardado — tipo: %s | vectores: %d",
            self.index_type, self.total_vectors,
        )

    def load(self) -> bool:
        """
        Carga el índice y el mapa de IDs desde disco si existen.

        Returns
        -------
        bool
            ``True`` si la carga fue exitosa; ``False`` si los ficheros no existen.
        """
        if self.index_path is None or self.id_map_path is None:
            return False
        if not self.index_path.exists() or not self.id_map_path.exists():
            log.info("[FaissIndex] No se encontraron ficheros de índice en disco.")
            return False

        self._index  = self._faiss.read_index(str(self.index_path))
        self._id_map = np.load(str(self.id_map_path))
        self._added_since_last_rebuild = 0

        log.info(
            "[FaissIndex] Cargado desde disco — tipo=%s n_vectors=%d",
            self.index_type, self.total_vectors,
        )
        return True
