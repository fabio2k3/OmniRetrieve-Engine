"""
embedder.py
===========
Encapsula el modelo sentence-transformers para vectorizar chunks de texto.

Responsabilidad única
---------------------
Recibir listas de strings y devolver arrays NumPy de embeddings normalizados.
No accede a la base de datos ni al índice FAISS; esas responsabilidades
pertenecen a EmbeddingPipeline y FaissIndexManager respectivamente.

Uso típico
----------
    embedder = ChunkEmbedder(model_name="all-MiniLM-L6-v2")
    vectors  = embedder.encode(["texto del chunk A", "texto del chunk B"])
    # vectors.shape == (2, 384)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Nombre de modelo por defecto: equilibrio calidad / velocidad / tamaño.
# Produce embeddings de dimensión 384, lo que facilita particionar IVFPQ
# con m=8 o m=16 (384 es divisible por ambos).
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class ChunkEmbedder:
    """
    Wrapper ligero sobre SentenceTransformer para uso en el pipeline.

    Parámetros
    ----------
    model_name : nombre o ruta del modelo sentence-transformers.
    device     : 'cpu', 'cuda', 'mps' o None (autodetección).
    batch_size : número de frases procesadas en cada llamada interna al modelo.
    normalize  : si True, los vectores se L2-normalizan (recomendado para
                 búsqueda por similitud coseno con FAISS IndexFlatIP).
    """

    def __init__(
        self,
        model_name: str  = DEFAULT_MODEL,
        device:     str | None = None,
        batch_size: int  = 64,
        normalize:  bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize  = normalize

        log.info("[Embedder] Cargando modelo '%s'…", model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Ejecuta: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._dim   = self._model.get_sentence_embedding_dimension()
        log.info(
            "[Embedder] Modelo listo — dim=%d device=%s",
            self._dim,
            self._model.device,
        )

    # ------------------------------------------------------------------
    # Propiedad pública
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensión del vector de salida del modelo."""
        return self._dim

    # ------------------------------------------------------------------
    # API de codificación
    # ------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Codifica una lista de textos y devuelve una matriz de embeddings.

        Parámetros
        ----------
        texts : lista de strings a vectorizar.

        Devuelve
        --------
        np.ndarray de forma (len(texts), dim) y dtype float32.
        Los vectores están L2-normalizados si self.normalize=True.

        Notas
        -----
        - Los textos vacíos o nulos se sustituyen por un string de un espacio
          para evitar errores del modelo; su embedding resultante es un vector
          de valores muy pequeños, que el pipeline puede detectar si lo necesita.
        - show_progress_bar se suprime para mantener los logs limpios cuando
          el pipeline procesa muchos lotes pequeños.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        # Sanitizar entradas nulas o vacías
        clean = [t if t and t.strip() else " " for t in texts]

        vectors: np.ndarray = self._model.encode(
            clean,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Garantizar dtype float32 (FAISS lo requiere)
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Codifica un único texto y devuelve un vector 1-D de shape (dim,).

        Equivalente a encode([text])[0] pero más conveniente para queries.
        """
        return self.encode([text])[0]