"""
web_search/faiss_indexer.py
===========================
Preprocesa, chunkea, embeddea e inserta en FAISS los documentos
recuperados por la búsqueda web.

Responsabilidad única
---------------------
Tomar los resultados crudos de WebSearcher (lista de dicts con 'content',
'title', 'url') y añadirlos al índice FAISS activo SIN tocar:
  - La tabla ``docs`` del crawler.
  - El índice invertido TF (indexing/).
  - El modelo LSI.

Estrategia de IDs
-----------------
Cada chunk recibe un ID string de la forma::

    web:<dominio>:<n>

donde ``<dominio>`` es el hostname del resultado (ej. ``arxiv.org``) y
``<n>`` es el número de chunk dentro de ese documento (0-based).

Ejemplo: ``web:nature.com:0``, ``web:nature.com:1``, ``web:arxiv.org:0``

Estos IDs se usan como ``chunk_id`` en el mapa FAISS (``_id_map``) mediante
una codificación entero negativa única para no colisionar con los IDs
positivos de la BD::

    faiss_int_id = -(hash(web_id) % (2**31))   # siempre negativo

El mapa inverso ``web_id → faiss_int_id`` se mantiene en memoria dentro
de ``WebFaissIndexer`` para poder recuperar metadatos en sesiones largas.

Chunking
--------
Se usa un splitter por párrafos con ventana de solapamiento::

    chunk_size    = 400 caracteres (aprox. 80-100 tokens para MiniLM)
    overlap       = 50  caracteres

Cada chunk conserva el título y la URL de su documento de origen en los
metadatos del RetrievalResult que se devuelve.

Uso
---
    from backend.web_search.faiss_indexer import WebFaissIndexer

    indexer = WebFaissIndexer(faiss_mgr=faiss_mgr, model_name="all-MiniLM-L6-v2")
    results = indexer.index(web_results)   # list[RetrievalResult]
"""

from __future__ import annotations

import hashlib
import logging
import re
import string
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.embedding.faiss.index_manager import FaissIndexManager

from backend.retrieval.protocols import RetrievalResult
from backend.crawler.chunker import make_chunks, clean_text as crawler_clean_text

log = logging.getLogger(__name__)

# ── Pre-limpieza de URLs (antes de pasar al chunker del crawler) ──────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Devuelve el hostname limpio de una URL (sin 'www.')."""
    try:
        host = urlparse(url).hostname or "web"
        return host.removeprefix("www.")
    except Exception:
        return "web"


def _make_web_id(domain: str, chunk_n: int) -> str:
    """Genera el ID legible del chunk: ``web:<domain>:<n>``."""
    return f"web:{domain}:{chunk_n}"


def _web_id_to_faiss_int(web_id: str) -> int:
    """
    Convierte un web_id string a un entero negativo único compatible con FAISS.

    Usamos SHA-256 truncado a 30 bits y lo negamos para que jamás colisione
    con los IDs positivos de la BD SQLite.
    """
    digest = int(hashlib.sha256(web_id.encode()).hexdigest(), 16)
    return -(digest % (2 ** 30))





# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class WebFaissIndexer:
    """
    Preprocesa, chunkea, embeddea e inserta en FAISS los docs web.

    El chunking usa exactamente el mismo algoritmo que el crawler
    (``backend.crawler.chunker.make_chunks``): solapamiento semántico a
    nivel de oración, con los mismos parámetros por defecto.

    Parámetros
    ----------
    faiss_mgr         : instancia activa de ``FaissIndexManager``.
    model_name        : modelo sentence-transformers (debe coincidir con el
                        que usa el resto del pipeline).
    device            : dispositivo de inferencia ('cpu', 'cuda', None).
    chunk_size        : tamaño máximo de cada chunk en caracteres (default: 1000).
    overlap_sentences : oraciones de solapamiento entre chunks (default: 2).
    min_chunk_chars   : longitud mínima para emitir un chunk (default: 100).
    min_sent_chars    : longitud mínima de oración antes de fusionar (default: 20).
    """

    def __init__(
        self,
        faiss_mgr:         "FaissIndexManager",
        model_name:        str        = "all-MiniLM-L6-v2",
        device:            str | None = None,
        chunk_size:        int        = 1000,
        overlap_sentences: int        = 2,
        min_chunk_chars:   int        = 100,
        min_sent_chars:    int        = 20,
    ) -> None:
        self._faiss_mgr         = faiss_mgr
        self._chunk_size        = chunk_size
        self._overlap_sentences = overlap_sentences
        self._min_chunk_chars   = min_chunk_chars
        self._min_sent_chars    = min_sent_chars

        # Carga perezosa del embedder para no penalizar el arranque
        self._model_name = model_name
        self._device     = device
        self._embedder   = None

        # Mapa inverso web_id → faiss_int_id (en memoria, para debug)
        self._web_id_registry: dict[str, int] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def index(self, web_results: list[dict]) -> list[RetrievalResult]:
        """
        Punto de entrada principal.

        Parámetros
        ----------
        web_results : lista de dicts producidos por ``WebSearcher.search()``.
                      Cada dict debe tener al menos: ``content``, ``title``, ``url``.

        Devuelve
        --------
        list[RetrievalResult]
            Un RetrievalResult por chunk generado, listo para pasar al
            CrossEncoderReranker. Los que fallaron no se incluyen.
        """
        if not web_results:
            return []

        all_chunks:      list[str]  = []
        all_web_ids:     list[str]  = []
        all_faiss_ints:  list[int]  = []
        chunk_meta:      list[dict] = []

        for doc in web_results:
            url     = doc.get("url", "")
            title   = doc.get("title", "Sin título")
            content = doc.get("content", "") or doc.get("abstract", "")
            domain  = _extract_domain(url)

            # Eliminar URLs antes de pasar al chunker (el HTML scrapeado
            # suele contener muchas URLs que ensucian los embeddings)
            pre_clean = _URL_RE.sub(" ", content)
            cleaned   = crawler_clean_text(pre_clean)
            if not cleaned:
                log.warning("[WebFaissIndexer] Doc sin contenido: %s", url)
                continue

            chunks = make_chunks(
                cleaned,
                chunk_size        = self._chunk_size,
                overlap_sentences = self._overlap_sentences,
                min_chunk_chars   = self._min_chunk_chars,
                min_sent_chars    = self._min_sent_chars,
            )
            log.debug(
                "[WebFaissIndexer] Doc '%s' → %d chunks (dominio=%s)",
                title[:50], len(chunks), domain,
            )

            for n, chunk_text in enumerate(chunks):
                web_id    = _make_web_id(domain, n)
                faiss_int = _web_id_to_faiss_int(web_id)

                all_chunks.append(chunk_text)
                all_web_ids.append(web_id)
                all_faiss_ints.append(faiss_int)
                chunk_meta.append({
                    "title":  title,
                    "url":    url,
                    "domain": domain,
                    "chunk":  n,
                })
                self._web_id_registry[web_id] = faiss_int

        if not all_chunks:
            log.info("[WebFaissIndexer] Sin chunks para indexar.")
            return []

        # ── Embeddings ────────────────────────────────────────────────────────
        embedder   = self._get_embedder()
        try:
            vectors = embedder.encode(all_chunks)   # (N, dim) float32
        except Exception as exc:
            log.error("[WebFaissIndexer] Error al embeddear: %s", exc, exc_info=True)
            return []

        # ── Inserción en FAISS ────────────────────────────────────────────────
        try:
            self._faiss_mgr.add(vectors, all_faiss_ints)
            log.info(
                "[WebFaissIndexer] %d chunks web añadidos al índice FAISS "
                "(total vectores: %d).",
                len(all_chunks), self._faiss_mgr.total_vectors,
            )
        except Exception as exc:
            log.error("[WebFaissIndexer] Error al insertar en FAISS: %s", exc, exc_info=True)
            return []

        # ── Construir RetrievalResults ─────────────────────────────────────────
        results: list[RetrievalResult] = []
        for chunk_text, web_id, faiss_int, meta in zip(
            all_chunks, all_web_ids, all_faiss_ints, chunk_meta
        ):
            results.append(
                RetrievalResult(
                    chunk_id    = faiss_int,          # entero negativo único
                    arxiv_id    = web_id,             # "web:<domain>:<n>" como identificador
                    chunk_index = meta["chunk"],
                    text        = chunk_text,
                    score       = 0.5,                # score neutro; el cross-encoder lo reemplazará
                    score_type  = "web",
                    metadata    = {
                        "title":  meta["title"],
                        "url":    meta["url"],
                        "source": "web",
                        "domain": meta["domain"],
                    },
                )
            )

        log.info(
            "[WebFaissIndexer] Indexación web completa: %d docs → %d chunks → %d RetrievalResults.",
            len(web_results), len(all_chunks), len(results),
        )
        return results

    # ── Privado ───────────────────────────────────────────────────────────────

    def _get_embedder(self):
        """Carga perezosa del ChunkEmbedder para evitar coste al inicio."""
        if self._embedder is None:
            from backend.embedding.embedder import ChunkEmbedder
            log.info(
                "[WebFaissIndexer] Cargando embedder '%s'…", self._model_name
            )
            self._embedder = ChunkEmbedder(
                model_name=self._model_name,
                device=self._device,
                normalize=True,
            )
        return self._embedder