"""
faiss_index.py
==============
Gestiona el ciclo de vida del índice FAISS usado en OmniRetrieve-Engine.

Tipo de índice
--------------
IndexIVFPQ (Inverted File con Product Quantization).
  - Partición del espacio en `nlist` celdas de Voronoi (IVF).
  - Compresión de vectores con Product Quantization (PQ):
      m     = número de subvectores (debe dividir a la dimensión).
      nbits = bits por código de subvector (8 → 256 centroides/subvector).
  - Ventajas: memoria reducida y búsqueda rápida en corpus grandes.
  - Requisito: el índice debe entrenarse sobre un conjunto representativo
    de vectores antes de poder añadir datos.

Fallback
--------
Si no hay suficientes vectores para entrenar un IVFPQ (mínimo nlist * 39
vectores según la heurística de FAISS), se usa IndexFlatL2.
El fallback se reemplaza automáticamente por IVFPQ en la primera reconstrucción
completa con suficientes datos.

Mapeo de IDs
------------
FAISS asigna internamente índices enteros 0-based. Esta clase mantiene un
array NumPy `_id_map` (shape [N], dtype int64) que mapea posición_faiss
→ chunk_id de la base de datos. Se persiste junto al índice en disco.

Reconstrucción automática
--------------------------
El pipeline llama a `maybe_rebuild()` después de cada lote. Este método
dispara `rebuild()` cuando `_added_since_last_rebuild >= rebuild_every`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import time

import numpy as np

log = logging.getLogger(__name__)

# Número mínimo de vectores por celda de Voronoi exigido por FAISS para
# entrenar de forma estable. El umbral real es 39 * nlist.
_MIN_TRAIN_FACTOR = 39

# Parámetros por defecto del índice
DEFAULT_NLIST = 100   # celdas de Voronoi
DEFAULT_M     = 8     # subvectores PQ  (384 / 8 = 48 → válido)
DEFAULT_NBITS = 8     # bits por código (256 centroides/subvector)
DEFAULT_NPROBE = 10   # celdas inspeccionadas en la búsqueda

# Cuántos vecinos se inspeccionan en la búsqueda (afecta recall vs velocidad)
# Se puede ajustar tras la carga con index.nprobe = N


class FaissIndexManager:
    """
    Gestiona construcción, actualización y búsqueda sobre un índice FAISS.

    Parámetros
    ----------
    dim          : dimensión del embedding (debe coincidir con el modelo).
    nlist        : número de celdas Voronoi para IndexIVFPQ.
    m            : número de subvectores PQ (debe dividir a dim).
    nbits        : bits por código PQ.
    nprobe       : celdas inspeccionadas durante la búsqueda.
    rebuild_every: número de chunks añadidos entre reconstrucciones completas.
    index_path   : ruta donde se guarda/carga el índice FAISS (.faiss).
    id_map_path  : ruta donde se guarda/carga el mapa chunk_id (.npy).
    """

    def __init__(
        self,
        dim:          int,
        nlist:        int  = DEFAULT_NLIST,
        m:            int  = DEFAULT_M,
        nbits:        int  = DEFAULT_NBITS,
        nprobe:       int  = DEFAULT_NPROBE,
        rebuild_every: int = 10_000,
        index_path:   Path | None = None,
        id_map_path:  Path | None = None,
    ) -> None:
        self.dim           = dim
        self.nlist         = nlist
        self.m             = m
        self.nbits         = nbits
        self.nprobe        = nprobe
        self.rebuild_every = rebuild_every
        self.index_path    = Path(index_path)  if index_path  else None
        self.id_map_path   = Path(id_map_path) if id_map_path else None

        self._validate_params()

        self._index    = None   # faiss.Index — inicializado al primer uso
        self._id_map: np.ndarray = np.empty(0, dtype=np.int64)  # pos → chunk_id
        self._added_since_last_rebuild: int = 0
        self._is_trained: bool = False

        try:
            import faiss as _faiss  # noqa: F401
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

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self.dim % self.m != 0:
            raise ValueError(
                f"dim={self.dim} no es divisible por m={self.m}. "
                "Elige un valor de m que divida exactamente a la dimensión."
            )

    # ------------------------------------------------------------------
    # Estado del índice
    # ------------------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        """Número de vectores actualmente en el índice."""
        return int(self._id_map.shape[0])

    @property
    def index_type(self) -> str:
        """Tipo de índice activo: 'IndexIVFPQ' o 'IndexFlatL2'."""
        if self._index is None:
            return "none"
        return type(self._index).__name__

    def _min_train_size(self) -> int:
        """
        Mínimo de vectores para entrenar IndexIVFPQ de forma estable.

        FAISS impone dos requisitos simultáneos:
          - IVF: n >= nlist * 39  (heurística de clustering K-means)
          - PQ:  n >= 2^nbits     (un punto por centroide de subvector)
        Se devuelve el máximo de ambos.
        """
        return max(self.nlist * _MIN_TRAIN_FACTOR, 2 ** self.nbits)

    def _effective_nlist(self, n_vectors: int) -> int:
        """
        Calcula el nlist efectivo para n_vectors.

        FAISS recomienda nlist ≈ sqrt(n_vectors) para corpus medianos.
        Limitamos a self.nlist como máximo para no exceder el configurado.
        """
        return max(4, min(self.nlist, int(math.sqrt(max(n_vectors, 1)))))

    # ------------------------------------------------------------------
    # Construcción del índice
    # ------------------------------------------------------------------

    def _build_flat(self) -> "faiss.IndexFlatL2":
        """Construye un índice plano (exacto, sin entrenamiento)."""
        return self._faiss.IndexFlatL2(self.dim)

    def _build_ivfpq(
        self,
        training_vectors: np.ndarray,
    ) -> "faiss.IndexIVFPQ":
        """
        Construye y entrena un IndexIVFPQ sobre los vectores de entrenamiento.

        Parámetros
        ----------
        training_vectors : np.ndarray de shape (N, dim), dtype float32.

        Devuelve
        --------
        Índice FAISS entrenado, listo para añadir vectores.
        """
        n = training_vectors.shape[0]
        effective_nlist = self._effective_nlist(n)

        log.info(
            "[FaissIndex] Entrenando IndexIVFPQ — n_train=%d nlist=%d m=%d nbits=%d",
            n, effective_nlist, self.m, self.nbits,
        )
        log.info("[FaissIndex] Esto puede tardar unos segundos según el tamaño del corpus…")

        quantizer = self._faiss.IndexFlatL2(self.dim)
        index     = self._faiss.IndexIVFPQ(
            quantizer, self.dim, effective_nlist, self.m, self.nbits
        )
        index.nprobe = min(self.nprobe, effective_nlist)

        vectors_c = np.ascontiguousarray(training_vectors, dtype=np.float32)
        t0 = time.perf_counter()
        index.train(vectors_c)
        elapsed = time.perf_counter() - t0
        log.info("[FaissIndex] Entrenamiento completado en %.2fs.", elapsed)
        return index

    # ------------------------------------------------------------------
    # Operación principal: añadir vectores
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, chunk_ids: list[int]) -> None:
        """
        Añade vectores al índice con sus chunk_ids correspondientes.

        Si el índice aún no está entrenado (o no existe), usa IndexFlatL2
        como almacenamiento temporal hasta que se dispare una reconstrucción
        con suficientes datos.

        Parámetros
        ----------
        vectors   : np.ndarray de shape (N, dim), dtype float32.
        chunk_ids : lista de N enteros (IDs de la tabla chunks).
        """
        if len(chunk_ids) == 0:
            return

        vectors_c = np.ascontiguousarray(vectors, dtype=np.float32)

        # Inicializar con índice plano si es la primera llamada
        if self._index is None:
            log.info("[FaissIndex] Inicializando índice plano (fallback).")
            self._index = self._build_flat()

        self._index.add(vectors_c)
        new_ids    = np.array(chunk_ids, dtype=np.int64)
        self._id_map = np.concatenate([self._id_map, new_ids])
        self._added_since_last_rebuild += len(chunk_ids)

        log.debug(
            "[FaissIndex] +%d vectores | total=%d | desde último rebuild: %d/%d",
            len(chunk_ids), self.total_vectors,
            self._added_since_last_rebuild, self.rebuild_every,
        )

    # ------------------------------------------------------------------
    # Reconstrucción completa
    # ------------------------------------------------------------------

    def rebuild(self, db_path: "Path") -> dict:
        """
        Reconstruye el índice completo leyendo todos los embeddings de la BD.

        Flujo
        -----
        1. Lee todos los embeddings almacenados en chunks (iterando en lotes).
        2. Si hay suficientes vectores, entrena y usa IndexIVFPQ.
           De lo contrario, usa IndexFlatL2 como fallback.
        3. Añade todos los vectores al nuevo índice de una sola vez.
        4. Guarda el índice en disco si index_path está configurado.
        5. Resetea el contador de añadidos.

        Parámetros
        ----------
        db_path : ruta a la BD SQLite (necesaria para leer embeddings).

        Devuelve
        --------
        dict con: n_vectors, index_type, nlist, m, nbits, index_path,
                  id_map_path.
        """
        from backend.database.chunk_repository import get_all_embeddings_iter

        log.info("[FaissIndex] Iniciando reconstrucción completa del índice…")
        t_start = time.perf_counter()

        all_vectors: list[np.ndarray] = []
        all_ids:     list[int]        = []

        for batch in get_all_embeddings_iter(db_path=db_path):
            for row in batch:
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                all_vectors.append(vec)
                all_ids.append(row["id"])
            n_read = len(all_vectors)
            if n_read % 10_000 == 0 and n_read > 0:
                log.info("[FaissIndex] Leyendo embeddings de BD… %d leídos", n_read)

        n = len(all_vectors)
        log.info("[FaissIndex] Embeddings leídos: %d (en %.2fs)", n, time.perf_counter() - t_start)

        if n == 0:
            log.warning("[FaissIndex] No hay embeddings; índice vacío.")
            self._index  = self._build_flat()
            self._id_map = np.empty(0, dtype=np.int64)
            self._added_since_last_rebuild = 0
            return self._build_stats(n)

        matrix = np.stack(all_vectors).astype(np.float32)  # (N, dim)

        # Decidir tipo de índice
        min_train = self._min_train_size()
        if n >= min_train:
            new_index       = self._build_ivfpq(matrix)
            self._is_trained = True
        else:
            log.info(
                "[FaissIndex] Solo %d vectores (mínimo %d para IVFPQ) → usando FlatL2.",
                n, min_train,
            )
            new_index = self._build_flat()

        # Añadir todos los vectores al índice nuevo
        matrix_c = np.ascontiguousarray(matrix, dtype=np.float32)
        log.info("[FaissIndex] Añadiendo %d vectores al índice…", n)
        t_add = time.perf_counter()
        new_index.add(matrix_c)
        log.info("[FaissIndex] Vectores añadidos en %.2fs.", time.perf_counter() - t_add)

        self._index    = new_index
        self._id_map   = np.array(all_ids, dtype=np.int64)
        self._added_since_last_rebuild = 0

        t_total = time.perf_counter() - t_start
        log.info(
            "[FaissIndex] Reconstrucción completa en %.2fs — tipo: %s | vectores: %d | nprobe: %d",
            t_total, self.index_type, self.total_vectors,
            self._index.nprobe if hasattr(self._index, 'nprobe') else 1,
        )

        # Persistir en disco
        if self.index_path:
            self.save()

        return self._build_stats(n)

    def _build_stats(self, n_vectors: int) -> dict:
        """Genera el dict de estadísticas para log_faiss_build."""
        nlist_used = None
        if self.index_type == "IndexIVFPQ" and self._index is not None:
            nlist_used = self._index.nlist
        return {
            "n_vectors":   n_vectors,
            "index_type":  self.index_type,
            "nlist":       nlist_used,
            "m":           self.m    if self.index_type == "IndexIVFPQ" else None,
            "nbits":       self.nbits if self.index_type == "IndexIVFPQ" else None,
            "index_path":  str(self.index_path)  if self.index_path  else None,
            "id_map_path": str(self.id_map_path) if self.id_map_path else None,
        }

    # ------------------------------------------------------------------
    # Comprobación de umbral de reconstrucción
    # ------------------------------------------------------------------

    def maybe_rebuild(self, db_path: "Path") -> bool:
        """
        Dispara rebuild() si se han añadido >= rebuild_every vectores desde
        el último rebuild.

        Devuelve True si se ejecutó la reconstrucción.
        """
        if self._added_since_last_rebuild >= self.rebuild_every:
            log.info(
                "[FaissIndex] Umbral de rebuild alcanzado (%d/%d) — iniciando reconstrucción…",
                self._added_since_last_rebuild,
                self.rebuild_every,
            )
            self.rebuild(db_path)
            return True
        log.debug(
            "[FaissIndex] Umbral no alcanzado (%d/%d) — rebuild no necesario.",
            self._added_since_last_rebuild,
            self.rebuild_every,
        )
        return False

    # ------------------------------------------------------------------
    # Búsqueda
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Busca los top_k chunks más cercanos a query_vector.

        Parámetros
        ----------
        query_vector : np.ndarray 1-D de shape (dim,), dtype float32.
        top_k        : número de resultados a devolver.

        Devuelve
        --------
        Lista de dicts con claves: chunk_id, score (distancia L2, menor = mejor).
        Lista vacía si el índice está vacío o no ha sido inicializado.
        """
        if self._index is None or self.total_vectors == 0:
            return []

        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self.total_vectors)

        distances, indices = self._index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS devuelve -1 para posiciones inválidas
                continue
            chunk_id = int(self._id_map[idx])
            results.append({"chunk_id": chunk_id, "score": float(dist)})

        return results

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Guarda el índice FAISS y el mapa de IDs en disco.

        - index_path  → archivo FAISS serializado.
        - id_map_path → array NumPy con los chunk_ids.

        No hace nada si las rutas no están configuradas.
        """
        if self.index_path is None or self.id_map_path is None:
            log.debug("[FaissIndex] Rutas de guardado no configuradas; omitiendo save.")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.id_map_path.parent.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(self.index_path))
        np.save(str(self.id_map_path), self._id_map)

        log.info(
            "[FaissIndex] Índice guardado — tipo: %s | vectores: %d",
            self.index_type, self.total_vectors,
        )
        log.info("[FaissIndex]   → %s", self.index_path)
        log.info("[FaissIndex]   → %s", self.id_map_path)

    def load(self) -> bool:
        """
        Carga el índice y el mapa de IDs desde disco si existen.

        Devuelve True si la carga fue exitosa, False si los archivos no existen.
        """
        if self.index_path is None or self.id_map_path is None:
            return False
        if not self.index_path.exists() or not self.id_map_path.exists():
            log.info("[FaissIndex] No se encontraron archivos de índice en disco.")
            return False

        self._index  = self._faiss.read_index(str(self.index_path))
        self._id_map = np.load(str(self.id_map_path))
        self._added_since_last_rebuild = 0

        log.info(
            "[FaissIndex] Índice cargado desde disco — tipo=%s n_vectors=%d",
            type(self._index).__name__, self.total_vectors,
        )
        return True