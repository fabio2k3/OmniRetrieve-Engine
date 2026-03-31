"""
embedder.py
===========
Genera embeddings de texto usando sentence-transformers.

Responsabilidad única
---------------------
Convierte listas de texto en vectores float32 normalizados (norma L2 = 1).
No conoce la BD ni gestiona qué chunks procesar — eso es responsabilidad
de EmbeddingPipeline.

Modelo por defecto
------------------
all-MiniLM-L6-v2: 384 dimensiones, ~90 MB, rápido en CPU.
El modelo se descarga automáticamente al primer uso (vía HuggingFace Hub).
Los vectores normalizados permiten usar producto punto como similitud coseno,
lo que reduce la búsqueda a una simple multiplicación de matrices.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """
    Envuelve SentenceTransformer para generar embeddings normalizados.

    La carga del modelo es lazy: ocurre la primera vez que se llama a
    encode() o encode_one(), no en __init__. Esto evita importar
    sentence_transformers en tests que no lo necesitan.

    Uso
    ---
        embedder = Embedder()
        vecs = embedder.encode(["texto uno", "texto dos"])  # (2, 384) float32
        q    = embedder.encode_one("mi consulta")           # (384,) float32
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model     = None   # carga lazy

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """
        Codifica una lista de textos en vectores float32 normalizados.

        Parámetros
        ----------
        texts      : lista de strings a codificar.
        batch_size : lote de inferencia; ajustar según RAM disponible.

        Devuelve
        --------
        np.ndarray de shape (len(texts), dim) y dtype float32.
        Los vectores tienen norma L2 = 1: producto_punto(a, b) == coseno(a, b).
        """
        self._ensure_loaded()
        vecs = self._model.encode(
            texts,
            batch_size          = batch_size,
            normalize_embeddings = True,
            show_progress_bar   = False,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Atajo para codificar un único texto. Devuelve shape (dim,) float32."""
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        """Dimensión del espacio de embeddings (384 para MiniLM-L6-v2)."""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Carga el modelo si todavía no está en memoria."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer   # import lazy
            log.info("[Embedder] Cargando modelo '%s'…", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            log.info(
                "[Embedder] Modelo listo. Dimensión: %d.",
                self._model.get_sentence_embedding_dimension(),
            )