"""
orchestrator.py
===============
Coordinador principal de OmniRetrieve-Engine.

Responsabilidad única: crear el estado compartido, arrancar los hilos
daemon y exponer la API pública que usan la CLI y el main.

Toda la lógica de negocio está delegada en módulos internos:
    _faiss.py       → inicialización del índice FAISS.
    _operations.py  → do_index, do_lsi_rebuild, do_embed, do_web_search,
                       build_qrf_pipeline, do_qrf_search,
                       build_rag_pipeline, do_rag_search, do_rag_ask.
    _status.py      → build_status().
    threads/        → un fichero por hilo daemon.
    cli.py          → presentación e interacción con el usuario.

Hilos daemon arrancados por start()
------------------------------------
crawler      — descubrimiento y descarga de artículos arXiv.
indexing     — indexación BM25 incremental cuando hay PDFs nuevos.
lsi_rebuild  — reconstrucción periódica del modelo LSI.
embedding    — embedding incremental de chunks y actualización de FAISS.
qrf_rag      — carga de QueryPipeline y RAGPipeline cuando FAISS está listo.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.database.schema import init_db
from backend.database.embedding_repository import init_embedding_schema
from backend.embedding import FaissIndexManager          # ← import correcto
from backend.retrieval.lsi_retriever import LSIRetriever

from .config      import OrchestratorConfig
from ._faiss      import init_faiss_mgr
from ._operations import (
    do_index, do_lsi_rebuild, do_embed, do_web_search,
    do_qrf_search, do_qrf_search_with_session,
    do_rag_search, do_rag_ask,
)
from ._status     import build_status
from .threads     import (
    run_crawler_thread,
    run_indexing_thread,
    run_lsi_rebuild_thread,
    run_embedding_thread,
    run_qrf_rag_loader_thread,
)
from .cli import run_cli

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordina crawler, indexing BM25, LSI, embedding, QRF, RAG y búsqueda
    web en hilos independientes y expone las queries al exterior.

    Uso
    ---
        cfg = OrchestratorConfig(pdf_threshold=5, embed_threshold=20)
        orc = Orchestrator(cfg)
        orc.start()       # arranca los cinco hilos daemon
        orc.run_cli()     # bucle interactivo en el hilo principal
        orc.stop()        # llamado automáticamente por 'quit' en la CLI
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.cfg = config or OrchestratorConfig()

        # ── Señales de coordinación ──────────────────────────────────────────
        self._shutdown = threading.Event()

        # LSI
        self._lsi_lock             = threading.RLock()
        self._lsi_ready            = threading.Event()
        self._retriever_holder: list[Optional[LSIRetriever]] = [None]

        # FAISS
        self._faiss_lock  = threading.RLock()
        self._faiss_ready = threading.Event()
        self._faiss_mgr: Optional[FaissIndexManager] = None

        # QRF (QueryPipeline — expand + BRF + MMR)
        self._qrf_lock   = threading.RLock()
        self._qrf_ready  = threading.Event()
        self._qrf_holder: list[Optional[object]] = [None]   # type: QueryPipeline | None

        # RAG (RAGPipeline — retrieval + LLM)
        self._rag_lock   = threading.RLock()
        self._rag_ready  = threading.Event()
        self._rag_holder: list[Optional[object]] = [None]   # type: RAGPipeline | None

        # ── Inicialización de BD e índice FAISS ──────────────────────────────
        init_db(self.cfg.db_path)
        init_embedding_schema(self.cfg.db_path)
        self._faiss_mgr, loaded = init_faiss_mgr(self.cfg)
        if loaded:
            self._faiss_ready.set()

        log.info("[Orchestrator] BD inicializada: %s", self.cfg.db_path)

        # ── Hilos daemon ─────────────────────────────────────────────────────
        self._threads: list[threading.Thread] = []

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Arranca los cinco hilos daemon. No bloquea."""
        specs = [
            ("crawler",     self._target_crawler),
            ("indexing",    self._target_indexing),
            ("lsi_rebuild", self._target_lsi_rebuild),
            ("embedding",   self._target_embedding),
            ("qrf_rag",     self._target_qrf_rag_loader),
        ]
        for name, target in specs:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
            log.info("[Orchestrator] Hilo '%s' iniciado.", name)

    def stop(self) -> None:
        """Señala a todos los hilos que deben terminar."""
        log.info("[Orchestrator] Señal de parada enviada.")
        self._shutdown.set()

    def run_cli(self) -> None:
        """Arranca la CLI interactiva en el hilo principal."""
        run_cli(
            shutdown      = self._shutdown,
            lsi_ready     = self._lsi_ready,
            qrf_ready     = self._qrf_ready,
            rag_ready     = self._rag_ready,
            lsi_min_docs  = self.cfg.lsi_min_docs,
            fn_query      = self.query,
            fn_query_web  = self.query_with_web,
            fn_qrf_search = self.qrf_search,
            fn_rag_search = self.rag_search,
            fn_rag_ask    = self.rag_ask,
            fn_status     = self.status,
            fn_index      = lambda: do_index(self.cfg),
            fn_rebuild    = lambda: do_lsi_rebuild(
                self.cfg, self._lsi_lock, self._lsi_ready, self._retriever_holder
            ),
            fn_stop       = self.stop,
        )

    # ── API pública de consulta ───────────────────────────────────────────────

    def query(self, text: str, top_n: int = 10) -> list[dict]:
        """
        Ejecuta una query semántica sobre el modelo LSI actual.

        Returns
        -------
        list[dict]
            Resultados del retriever LSI. Lista vacía si el modelo no está listo.
        """
        if not self._lsi_ready.is_set():
            return []
        with self._lsi_lock:
            retriever = self._retriever_holder[0]
            if retriever is None:
                return []
            return retriever.retrieve(text, top_n=top_n)

    def query_with_web(self, text: str, top_n: int = 10) -> dict:
        """
        Ejecuta una query LSI y activa la búsqueda web si los resultados
        locales no superan el umbral de suficiencia configurado.

        Returns
        -------
        dict con claves: ``results``, ``web_activated``, ``web_results``,
        ``reason``, ``query``, ``indexed``.
        """
        local_results = self.query(text, top_n=top_n)
        return do_web_search(text, local_results, self.cfg)

    def semantic_query(self, text: str, top_k: int = 10) -> list[dict]:
        """
        Búsqueda semántica densa usando el índice FAISS directamente.

        Vectoriza la query, busca en FAISS y enriquece los resultados con
        texto y metadatos de la BD.

        Returns
        -------
        list[dict] con claves: ``chunk_id``, ``score``, ``arxiv_id``,
        ``chunk_index``, ``text``, ``title``. Lista vacía si el índice no
        está listo.
        """
        if not self._faiss_ready.is_set():
            return []

        try:
            from backend.embedding.embedder import ChunkEmbedder
            from backend.database.chunk_repository import get_chunks_by_ids
            embedder  = ChunkEmbedder(model_name=self.cfg.embed_model)
            query_vec = embedder.encode_single(text)
        except Exception as exc:
            log.error("[semantic_query] Error al vectorizar query: %s", exc)
            return []

        with self._faiss_lock:
            if self._faiss_mgr is None:
                return []
            raw_results = self._faiss_mgr.search(query_vec, top_k=top_k)

        if not raw_results:
            return []

        chunk_ids = [r["chunk_id"] for r in raw_results]
        score_map = {r["chunk_id"]: r["score"] for r in raw_results}
        rows      = get_chunks_by_ids(chunk_ids, db_path=self.cfg.db_path)

        results = [
            {
                "chunk_id":    row["chunk_id"],
                "score":       score_map.get(row["chunk_id"], float("inf")),
                "arxiv_id":    row["arxiv_id"],
                "chunk_index": row["chunk_index"],
                "text":        row["text"],
                "title":       row.get("title", ""),
            }
            for row in rows
        ]
        results.sort(key=lambda r: r["score"])
        return results

    def qrf_search(self, text: str, top_k: int = 10) -> list[dict]:
        """
        Búsqueda de alta calidad usando el pipeline QRF completo.

        Pipeline interno: expansión LCE (LSI) → embedding → búsqueda inicial
        FAISS → BRF (Blind Relevance Feedback) → re-búsqueda → MMR reranking.

        Returns
        -------
        list[dict]
            Resultados enriquecidos con: ``score``, ``mmr_score``,
            ``chunk_id``, ``arxiv_id``, ``chunk_index``, ``text``,
            ``title``, ``authors``, ``abstract``, ``pdf_url``,
            ``expanded_terms``.
            Lista vacía si el pipeline no está listo.
        """
        if not self._qrf_ready.is_set():
            return []
        with self._qrf_lock:
            pipeline = self._qrf_holder[0]
            if pipeline is None:
                return []
        return do_qrf_search(text, pipeline, top_k=top_k)

    def qrf_search_with_session(
        self, text: str, top_k: int = 10
    ) -> tuple[list[dict], str]:
        """
        Igual que ``qrf_search`` pero devuelve también un ``session_id``.

        El ``session_id`` permite rondas de refinamiento futuras.
        Nota: el método ``refine()`` (Rocchio) no está integrado de momento.

        Returns
        -------
        tuple[list[dict], str]
            ``(results, session_id)``. En caso de error: ``([], "")``.
        """
        if not self._qrf_ready.is_set():
            return [], ""
        with self._qrf_lock:
            pipeline = self._qrf_holder[0]
            if pipeline is None:
                return [], ""
        return do_qrf_search_with_session(text, pipeline, top_k=top_k)

    def rag_search(self, text: str, top_k: int = 10) -> list[dict]:
        """
        Recuperación densa sin generación LLM (útil para inspección de chunks).

        Usa ``EmbeddingRetriever`` sobre el índice FAISS compartido,
        con ``CrossEncoderReranker`` opcional si ``cfg.rag_use_reranker=True``.

        Returns
        -------
        list[dict] con claves: ``chunk_id``, ``arxiv_id``, ``chunk_index``,
        ``title``, ``text``, ``score``, ``score_type``.
        Lista vacía si el pipeline no está listo.
        """
        if not self._rag_ready.is_set():
            return []
        with self._rag_lock:
            pipeline = self._rag_holder[0]
            if pipeline is None:
                return []
        return do_rag_search(
            text, pipeline,
            top_k=top_k,
            candidate_k=self.cfg.rag_candidate_k,
        )

    def rag_ask(
        self,
        text:       str,
        top_k:      int = 10,
        max_chunks: int | None = None,
        max_chars:  int | None = None,
    ) -> dict:
        """
        Pipeline RAG completo: recuperación densa → contexto → prompt → LLM.

        Parámetros
        ----------
        text       : pregunta del usuario en lenguaje natural.
        top_k      : chunks candidatos a recuperar.
        max_chunks : chunks máximos en el contexto (usa ``cfg.rag_max_chunks`` si None).
        max_chars  : caracteres máximos por chunk (usa ``cfg.rag_max_chars`` si None).

        Returns
        -------
        dict con claves:
            ``query``   — pregunta original.
            ``answer``  — respuesta generada por el LLM.
            ``sources`` — lista de fuentes usadas en el contexto.
            ``error``   — presente solo si ocurrió un error.
        Lista vacía si el pipeline no está listo.
        """
        if not self._rag_ready.is_set():
            return {
                "query": text, "answer": "",
                "sources": [], "error": "RAG pipeline no disponible aún.",
            }
        with self._rag_lock:
            pipeline = self._rag_holder[0]
            if pipeline is None:
                return {
                    "query": text, "answer": "",
                    "sources": [], "error": "RAG pipeline no disponible aún.",
                }
        return do_rag_ask(
            text, pipeline,
            top_k       = top_k,
            candidate_k = self.cfg.rag_candidate_k,
            max_chunks  = max_chunks if max_chunks is not None else self.cfg.rag_max_chunks,
            max_chars   = max_chars  if max_chars  is not None else self.cfg.rag_max_chars,
        )

    def status(self) -> dict:
        """Devuelve un snapshot del estado actual del sistema."""
        return build_status(
            cfg              = self.cfg,
            lsi_lock         = self._lsi_lock,
            retriever_holder = self._retriever_holder,
            faiss_lock       = self._faiss_lock,
            faiss_mgr        = self._faiss_mgr,
            lsi_ready        = self._lsi_ready,
            faiss_ready      = self._faiss_ready,
            qrf_ready        = self._qrf_ready,
            rag_ready        = self._rag_ready,
        )

    # ── Targets de hilos ──────────────────────────────────────────────────────

    def _target_crawler(self) -> None:
        run_crawler_thread(self.cfg, self._shutdown)

    def _target_indexing(self) -> None:
        run_indexing_thread(
            self.cfg, self._shutdown,
            do_index=lambda: do_index(self.cfg),
        )

    def _target_lsi_rebuild(self) -> None:
        run_lsi_rebuild_thread(
            self.cfg, self._shutdown,
            self._lsi_lock, self._lsi_ready, self._retriever_holder,
        )

    def _target_embedding(self) -> None:
        run_embedding_thread(
            self.cfg, self._shutdown,
            do_embed=lambda: do_embed(
                self.cfg, self._faiss_lock, self._faiss_mgr, self._faiss_ready
            ),
        )

    def _target_qrf_rag_loader(self) -> None:
        run_qrf_rag_loader_thread(
            cfg        = self.cfg,
            shutdown   = self._shutdown,
            faiss_ready = self._faiss_ready,
            faiss_mgr  = self._faiss_mgr,
            faiss_lock = self._faiss_lock,
            qrf_holder = self._qrf_holder,
            qrf_lock   = self._qrf_lock,
            qrf_ready  = self._qrf_ready,
            rag_holder = self._rag_holder,
            rag_lock   = self._rag_lock,
            rag_ready  = self._rag_ready,
        )
