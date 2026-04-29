"""
pipeline.py
=================
Pipeline completo de refinamiento de consulta.

Integra los cuatro componentes en un flujo cohesivo:

    String (usuario)
         │
         ▼
    1. Expansión Ciega (LCE via LSI)
         │  query_original + términos latentes del corpus
         ▼
    2. Embedding de la query expandida (SentenceTransformer)
         │  vector float32 (dim,)
         ▼
    3. Búsqueda inicial en FAISS IVFPQ
         │  top_k_initial candidatos
         ▼
    4. Pseudo-Retroalimentación de Relevancia (BRF)
         │  vector ajustado hacia el centroide de los top resultados
         ▼
    5. Re-búsqueda en FAISS con el vector refinado
         │  top_k_final candidatos
         ▼
    6. MMR Reranking
         │  selección diversa y relevante
         ▼
    Lista de chunks para RAG

El pipeline expone también retroalimentación explícita (Rocchio) para
el caso en que la UI permita al usuario marcar resultados.

Uso básico
----------
    pipeline = QueryPipeline()
    pipeline.load()
    results  = pipeline.search("attention mechanism in transformers", top_k=10)

Uso con retroalimentación explícita
------------------------------------
    # Primera búsqueda
    results, session_id = pipeline.search_with_session("transformer attention")

    # Usuario marca el chunk 42 como relevante, 99 como irrelevante
    results2 = pipeline.refine(
        session_id     = session_id,
        relevant_ids   = [42],
        irrelevant_ids = [99],
        top_k          = 10,
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from backend.database.schema import DB_PATH, DATA_DIR
from backend.retrieval.lsi_model import LSIModel, MODEL_PATH
from backend.qrf.query_expander import QueryExpander
from backend.qrf.brf     import BlindRelevanceFeedback
from backend.qrf.rocchio import RocchioFeedback
from backend.qrf.mmr     import MMRReranker

log = logging.getLogger(__name__)

_FAISS_DIR   = DATA_DIR / "faiss"
_INDEX_PATH  = _FAISS_DIR / "index.faiss"
_ID_MAP_PATH = _FAISS_DIR / "id_map.npy"


class QueryPipeline:
    """
    Pipeline de recuperación con expansión y retroalimentación de relevancia.

    Parámetros
    ----------
    model_name        : modelo sentence-transformers para embedding.
    lsi_model_path    : ruta al .pkl del modelo LSI.
    index_path        : ruta del archivo .faiss.
    id_map_path       : ruta del .npy con el mapa chunk_id.
    db_path           : base de datos SQLite.
    top_k_initial     : candidatos recuperados en la búsqueda inicial (BRF).
    expand            : activar/desactivar expansión LCE.
    expand_top_dims   : dimensiones latentes a examinar en LCE.
    expand_min_corr   : umbral mínimo de correlación para añadir un término.
    expand_max_terms  : máximo de términos nuevos a añadir.
    brf_alpha         : peso del vector original en BRF.
    brf_top_k         : resultados usados para el centroide BRF.
    mmr_lambda        : balance relevancia/diversidad en MMR.
    rocchio_alpha     : peso del vector original en Rocchio.
    rocchio_beta      : peso de los documentos relevantes en Rocchio.
    rocchio_gamma     : penalización de documentos irrelevantes en Rocchio.
    """

    def __init__(
        self,
        model_name:       str   = "all-MiniLM-L6-v2",
        lsi_model_path:   Path  = MODEL_PATH,
        index_path:       Path  = _INDEX_PATH,
        id_map_path:      Path  = _ID_MAP_PATH,
        db_path:          Path  = DB_PATH,
        top_k_initial:    int   = 20,
        expand:           bool  = True,
        expand_top_dims:  int   = 3,
        expand_min_corr:  float = 0.4,
        expand_max_terms: int   = 8,
        brf_alpha:        float = 0.75,
        brf_top_k:        int   = 5,
        mmr_lambda:       float = 0.6,
        rocchio_alpha:    float = 0.6,
        rocchio_beta:     float = 0.4,
        rocchio_gamma:    float = 0.1,
    ) -> None:
        self.db_path       = Path(db_path)
        self.model_name    = model_name
        self.lsi_model_path = Path(lsi_model_path)
        self.top_k_initial = top_k_initial
        self._expand_enabled = expand

        # Componentes (inicialización perezosa en load())
        self._embedder:  "ChunkEmbedder | None" = None
        self._faiss_mgr: "FaissIndexManager | None" = None
        self._expander:  QueryExpander            = QueryExpander(
            top_dims=expand_top_dims,
            min_correlation=expand_min_corr,
            max_expansion=expand_max_terms,
        )
        self._brf     = BlindRelevanceFeedback(alpha=brf_alpha, top_k_rf=brf_top_k)
        self._rocchio = RocchioFeedback(
            alpha=rocchio_alpha, beta=rocchio_beta, gamma=rocchio_gamma
        )
        self._mmr = MMRReranker(lambda_=mmr_lambda)

        # Caché de vectores de sesión: session_id -> query_vector
        self._session_vectors: dict[str, np.ndarray] = {}

        self._faiss_params = dict(
            nlist=100, m=8, nbits=8, nprobe=10,
            index_path=index_path,
            id_map_path=id_map_path,
        )

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load(self, device: str | None = None) -> None:
        """
        Carga todos los componentes del pipeline.

        Parámetros
        ----------
        device : dispositivo de inferencia para el embedder ('cpu', 'cuda', 'mps').

        Lanza
        -----
        FileNotFoundError si el índice FAISS no existe.
        ImportError si sentence-transformers o faiss no están instalados.
        """
        from backend.embedding.embedder import ChunkEmbedder
        from backend.embedding import FaissIndexManager

        log.info("[QueryPipeline] Cargando modelo de embedding '%s'…", self.model_name)
        self._embedder = ChunkEmbedder(model_name=self.model_name, device=device)

        self._faiss_mgr = FaissIndexManager(
            dim=self._embedder.dim,
            **self._faiss_params,
        )
        loaded = self._faiss_mgr.load()
        if not loaded:
            raise FileNotFoundError(
                f"Índice FAISS no encontrado: {self._faiss_params['index_path']}\n"
                "Ejecuta primero:  python -m backend.tools.embed_chunks"
            )

        if self._expand_enabled:
            log.info("[QueryPipeline] Cargando modelo LSI para expansión…")
            try:
                self._expander.load(self.lsi_model_path, self.db_path)
            except Exception as exc:
                log.warning(
                    "[QueryPipeline] No se pudo cargar el modelo LSI (%s). "
                    "La expansión de consulta estará desactivada.", exc
                )
                self._expand_enabled = False

        log.info(
            "[QueryPipeline] Listo. FAISS: %d vectores (%s) | Expansión: %s",
            self._faiss_mgr.total_vectors,
            self._faiss_mgr.index_type,
            "activa" if self._expand_enabled else "desactivada",
        )

    # ------------------------------------------------------------------
    # Búsqueda principal
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Ejecuta el pipeline completo de recuperación refinada.

        Pasos internos
        --------------
        1. Expande la query con LCE (si está activado).
        2. Embede la query expandida con SentenceTransformer.
        3. Búsqueda inicial en FAISS (top_k_initial candidatos).
        4. Ajusta el vector con BRF (centroide de los top resultados).
        5. Re-búsqueda en FAISS con el vector refinado.
        6. MMR reranking para diversidad.
        7. Enriquece con texto y metadatos desde la BD.

        Parámetros
        ----------
        query  : consulta en lenguaje natural.
        top_k  : número de resultados finales a devolver.

        Devuelve
        --------
        Lista de dicts con claves:
            score, mmr_score, chunk_id, arxiv_id, chunk_index,
            text, char_count, title, authors, abstract, pdf_url,
            expanded_terms (lista de términos añadidos por LCE).
        """
        self._check_loaded()
        t0 = time.monotonic()

        # 1. Expansión LCE
        if self._expand_enabled:
            query_expanded, new_terms = self._expander.expand(query)
        else:
            query_expanded, new_terms = query, []

        # 2. Embedding de la query expandida
        query_vec = self._embedder.encode_single(query_expanded)   # (dim,)

        # 3. Búsqueda inicial
        initial_results = self._faiss_mgr.search(query_vec, top_k=self.top_k_initial)

        # 4. BRF — ajuste del vector
        if initial_results:
            refined_vec = self._brf.adjust(query_vec, initial_results, self.db_path)
        else:
            refined_vec = query_vec

        # 5. Re-búsqueda con vector refinado
        final_candidates = self._faiss_mgr.search(
            refined_vec, top_k=max(top_k * 3, self.top_k_initial)
        )

        # 6. MMR reranking
        reranked = self._mmr.rerank(
            results=final_candidates,
            query_vector=refined_vec,
            top_n=top_k,
            db_path=self.db_path,
        )

        # 7. Enriquecer con texto y metadatos
        results = self._enrich(reranked, new_terms)

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.info(
            "[QueryPipeline] '%s…' → %d resultados en %.1fms "
            "(expansión: +%d términos | BRF: activo | MMR: %.2f)",
            query[:40], len(results), elapsed_ms,
            len(new_terms), self._mmr.lambda_,
        )
        return results

    # ------------------------------------------------------------------
    # Búsqueda con sesión (para retroalimentación explícita)
    # ------------------------------------------------------------------

    def search_with_session(
        self, query: str, top_k: int = 10
    ) -> tuple[list[dict], str]:
        """
        Igual que search(), pero devuelve también un session_id para
        permitir retroalimentación explícita con refine().

        Devuelve
        --------
        (results, session_id)
        """
        results    = self.search(query, top_k=top_k)
        session_id = str(uuid.uuid4())

        # Guardar el vector refinado para Rocchio posterior
        query_expanded, _ = (
            self._expander.expand(query)
            if self._expand_enabled else (query, [])
        )
        query_vec = self._embedder.encode_single(query_expanded)
        self._session_vectors[session_id] = query_vec

        log.info("[QueryPipeline] Sesión creada: %s", session_id)
        return results, session_id

    # ------------------------------------------------------------------
    # Refinamiento con feedback explícito (Rocchio)
    # ------------------------------------------------------------------

    def refine(
        self,
        session_id:     str,
        relevant_ids:   list[int],
        irrelevant_ids: list[int],
        top_k:          int = 10,
    ) -> list[dict]:
        """
        Refina los resultados usando feedback explícito del usuario (Rocchio).

        El vector de la sesión se desplaza hacia los documentos relevantes
        y se aleja de los irrelevantes. La nueva búsqueda usa el vector
        ajustado para recuperar resultados más precisos.

        Parámetros
        ----------
        session_id     : ID devuelto por search_with_session().
        relevant_ids   : chunk_ids que el usuario marcó como relevantes.
        irrelevant_ids : chunk_ids que el usuario marcó como irrelevantes.
        top_k          : número de resultados a devolver.

        Devuelve
        --------
        Lista de dicts en el mismo formato que search(), con el vector
        calibrado por Rocchio.

        Lanza
        -----
        KeyError si session_id no existe.
        """
        self._check_loaded()

        if session_id not in self._session_vectors:
            raise KeyError(
                f"session_id '{session_id}' no encontrado. "
                "Usa search_with_session() para crear una sesión."
            )

        query_vec = self._session_vectors[session_id]

        # Aplicar Rocchio
        refined_vec = self._rocchio.adjust(
            query_id=session_id,
            query_vector=query_vec,
            relevant_ids=relevant_ids,
            irrelevant_ids=irrelevant_ids,
            db_path=self.db_path,
        )

        # Actualizar vector de sesión con el ajustado
        self._session_vectors[session_id] = refined_vec

        # Nueva búsqueda + MMR
        candidates = self._faiss_mgr.search(
            refined_vec, top_k=max(top_k * 3, self.top_k_initial)
        )
        reranked   = self._mmr.rerank(candidates, refined_vec, top_k, self.db_path)
        return self._enrich(reranked, expanded_terms=[])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enrich(
        self,
        raw_results:    list[dict],
        expanded_terms: list[str],
    ) -> list[dict]:
        """Enriquece los resultados FAISS con texto y metadatos de la BD."""
        from backend.database.chunk_repository import get_chunks_by_ids

        if not raw_results:
            return []

        chunk_ids = [r["chunk_id"] for r in raw_results]
        rows      = get_chunks_by_ids(chunk_ids, db_path=self.db_path)
        row_map   = {r["chunk_id"]: r for r in rows}

        results = []
        for raw in raw_results:
            cid = raw["chunk_id"]
            row = row_map.get(cid, {})
            abstract = row.get("abstract") or ""
            results.append({
                "score":          raw.get("score", 0.0),
                "mmr_score":      raw.get("mmr_score"),
                "chunk_id":       cid,
                "arxiv_id":       row.get("arxiv_id", ""),
                "chunk_index":    row.get("chunk_index"),
                "text":           row.get("text", ""),
                "char_count":     row.get("char_count"),
                "title":          row.get("title", ""),
                "authors":        row.get("authors", ""),
                "abstract":       abstract[:300] + ("…" if len(abstract) > 300 else ""),
                "pdf_url":        row.get("pdf_url", ""),
                "expanded_terms": expanded_terms,
            })
        return results

    def _check_loaded(self) -> None:
        if self._embedder is None or self._faiss_mgr is None:
            raise RuntimeError(
                "QueryPipeline no está cargado. Llama a load() primero."
            )

    def clear_session(self, session_id: str | None = None) -> None:
        """Limpia la caché de sesión completa o una sesión concreta."""
        if session_id:
            self._session_vectors.pop(session_id, None)
            self._rocchio.clear_cache(session_id)
        else:
            self._session_vectors.clear()
            self._rocchio.clear_cache()

    @property
    def is_ready(self) -> bool:
        return self._embedder is not None and self._faiss_mgr is not None