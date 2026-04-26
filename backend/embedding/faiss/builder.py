"""
faiss/builder.py
================
Construcción de índices FAISS.

Módulo interno del subpaquete ``faiss``. Encapsula únicamente la lógica de
*crear* un índice, sin saber nada de persistencia, búsqueda ni gestión del
ciclo de vida. Esas responsabilidades pertenecen a ``index_manager.py``.

API pública
-----------
build_flat(faiss_module, dim)
    Crea un IndexFlatL2 exacto, sin entrenamiento.

build_ivfpq(faiss_module, dim, training_vectors, nlist, m, nbits, nprobe)
    Crea y entrena un IndexIVFPQ sobre los vectores proporcionados.

min_train_size(nlist, nbits)
    Mínimo de vectores para entrenar un IVFPQ de forma estable.

effective_nlist(n_vectors, max_nlist)
    Calcula el nlist ajustado al tamaño real del corpus.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from .constants import MIN_TRAIN_FACTOR

log = logging.getLogger(__name__)


def min_train_size(nlist: int, nbits: int) -> int:
    """
    Mínimo de vectores para entrenar IndexIVFPQ de forma estable.

    FAISS impone dos requisitos simultáneos:
      - IVF: ``n >= nlist * MIN_TRAIN_FACTOR``  (K-means clustering)
      - PQ:  ``n >= 2^nbits``                   (un punto por centroide PQ)

    Parámetros
    ----------
    nlist : número de celdas de Voronoi configurado.
    nbits : bits por código PQ.

    Returns
    -------
    int
        El mayor de los dos mínimos.
    """
    return max(nlist * MIN_TRAIN_FACTOR, 2 ** nbits)


def effective_nlist(n_vectors: int, max_nlist: int) -> int:
    """
    Calcula el nlist efectivo para ``n_vectors`` vectores.

    FAISS recomienda ``nlist ≈ sqrt(n_vectors)`` para corpus medianos.
    Se limita a ``max_nlist`` como techo y a ``4`` como suelo.

    Parámetros
    ----------
    n_vectors : número de vectores con los que se entrenará el índice.
    max_nlist : valor máximo permitido (el configurado en ``OrchestratorConfig``).

    Returns
    -------
    int
        Número de celdas de Voronoi a usar en el entrenamiento.
    """
    return max(4, min(max_nlist, int(math.sqrt(max(n_vectors, 1)))))


def build_flat(faiss_module, dim: int):
    """
    Construye un ``IndexFlatL2`` (búsqueda exacta, sin entrenamiento).

    Parámetros
    ----------
    faiss_module : módulo ``faiss`` ya importado.
    dim          : dimensión de los vectores.

    Returns
    -------
    faiss.IndexFlatL2
    """
    log.debug("[builder] Creando IndexFlatL2 (dim=%d).", dim)
    return faiss_module.IndexFlatL2(dim)


def build_ivfpq(
    faiss_module,
    dim:              int,
    training_vectors: np.ndarray,
    nlist:            int,
    m:                int,
    nbits:            int,
    nprobe:           int,
):
    """
    Construye y entrena un ``IndexIVFPQ`` sobre los vectores de entrenamiento.

    Parámetros
    ----------
    faiss_module      : módulo ``faiss`` ya importado.
    dim               : dimensión de los vectores.
    training_vectors  : ``np.ndarray`` de shape ``(N, dim)``, dtype float32.
    nlist             : número de celdas de Voronoi para el entrenamiento.
    m                 : subvectores PQ (debe dividir exactamente a ``dim``).
    nbits             : bits por código PQ.
    nprobe            : celdas inspeccionadas durante la búsqueda.

    Returns
    -------
    faiss.IndexIVFPQ
        Índice entrenado y listo para añadir vectores.
    """
    n               = training_vectors.shape[0]
    eff_nlist       = effective_nlist(n, nlist)
    capped_nprobe   = min(nprobe, eff_nlist)

    log.info(
        "[builder] Entrenando IndexIVFPQ — n_train=%d nlist=%d m=%d nbits=%d",
        n, eff_nlist, m, nbits,
    )
    log.info("[builder] Esto puede tardar unos segundos según el tamaño del corpus…")

    quantizer = faiss_module.IndexFlatL2(dim)
    index     = faiss_module.IndexIVFPQ(quantizer, dim, eff_nlist, m, nbits)
    index.nprobe = capped_nprobe

    vectors_c = np.ascontiguousarray(training_vectors, dtype=np.float32)
    t0 = time.perf_counter()
    index.train(vectors_c)
    log.info("[builder] Entrenamiento completado en %.2fs.", time.perf_counter() - t0)

    return index
