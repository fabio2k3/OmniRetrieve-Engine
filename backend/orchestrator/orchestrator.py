"""
orchestrator.py
===============
Coordinador principal de OmniRetrieve-Engine.

Responsabilidad única: crear el estado compartido, arrancar los hilos
daemon y exponer la API pública.

Toda la lógica de negocio está delegada en módulos internos:
    _faiss.py       → inicialización del índice FAISS.
    _operations.py  → todas las operaciones de pipeline.
    _status.py      → build_status().
    threads/        → un fichero por hilo daemon.
    cli.py          → presentación e interacción con el usuario.

Hilos daemon
------------
crawler      — descubrimiento y descarga de artículos arXiv.
indexing     — indexación BM25 incremental (terms + postings) cuando hay PDFs nuevos.
lsi_rebuild  — reconstrucción periódica del modelo LSI.
embedding    — embedding incremental de chunks y actualización de FAISS.
qrf_rag      — carga de QueryPipeline, HybridRetriever, CrossEncoder y RAGPipeline.

API pública principal
---------------------
pipeline_ask()  — flujo unificado: QRF expand → Hybrid → Web → Rerank → RAG.
query()         — búsqueda LSI local standalone.
query_with_web()— búsqueda LSI + fallback web.
qrf_search()    — búsqueda QRF standalone (sin hybrid ni reranking).
rag_search()    — retrieval denso sin LLM.
rag_ask()       — pipeline RAG standalone.
status()        — snapshot del estado del sistema.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.database.schema import init_db
from backend.database.embedding_repository import init_embedding_schema
from backend.embedding import FaissIndexManager
from backend.retrieval.lsi_retriever import LSIRetriever

from .config      import OrchestratorConfig
from ._faiss      import init_faiss_mgr
from ._operations import (
    do_index, do_lsi_rebuild, do_embed, do_web_search,
    do_qrf_search, do_qrf_search_with_session,
    do_rag_search, do_rag_ask,
    do_pipeline_ask,
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
    Coordina crawler, LSI, embedding y el pipeline unificado de consultas.

    Uso
    ---
        orc = Orchestrator()
        orc.start()
        orc.run_cli()
        orc.stop()
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

        # QRF (QueryPipeline — expansión LCE standalone)
        self._qrf_lock   = threading.RLock()
        self._qrf_ready  = threading.Event()
        self._qrf_holder: list[Optional[object]] = [None]

        # HybridRetriever (LSI sparse + FAISS dense, RRF)
        self._hybrid_lock   = threading.RLock()
        self._hybrid_holder: list[Optional[object]] = [None]

        # CrossEncoderReranker
        self._cross_encoder_lock   = threading.RLock()
        self._cross_encoder_holder: list[Optional[object]] = [None]

        # RAGPipeline (generación LLM)
        self._rag_lock   = threading.RLock()
        self._rag_ready  = threading.Event()
        self._rag_holder: list[Optional[object]] = [None]

        # Pipeline unificado listo (los cuatro componentes anteriores cargados)
        self._pipeline_ready = threading.Event()

        # ── Inicialización de BD e índice FAISS ──────────────────────────────
        init_db(self.cfg.db_path)
        init_embedding_schema(self.cfg.db_path)
        self._faiss_mgr, loaded = init_faiss_mgr(self.cfg)
        if loaded:
            self._faiss_ready.set()

        log.info("[Orchestrator] BD inicializada: %s", self.cfg.db_path)
        self._threads: list[threading.Thread] = []

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Arranca los cuatro hilos daemon. No bloquea."""
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
            shutdown         = self._shutdown,
            lsi_ready        = self._lsi_ready,
            qrf_ready        = self._qrf_ready,
            rag_ready        = self._rag_ready,
            pipeline_ready   = self._pipeline_ready,
            lsi_min_docs     = self.cfg.lsi_min_docs,
            fn_pipeline_ask  = self.pipeline_ask,
            fn_query         = self.query,
            fn_query_web     = self.query_with_web,
            fn_qrf_search    = self.qrf_search,
            fn_rag_search    = self.rag_search,
            fn_rag_ask       = self.rag_ask,
            fn_status        = self.status,
            fn_index         = lambda: do_index(self.cfg),
            fn_rebuild       = lambda: do_lsi_rebuild(
                self.cfg, self._lsi_lock, self._lsi_ready, self._retriever_holder
            ),
            fn_stop          = self.stop,
        )

    # ── API pública — pipeline unificado ─────────────────────────────────────

    def pipeline_ask(self, text: str) -> dict:
        """
        Pipeline unificado: QRF expand → HybridRetriever → WebSearch
        → CrossEncoder → RAG → respuesta LLM.

        Es el método principal de consulta del sistema.

        Returns
        -------
        dict con: ``query``, ``expanded_query``, ``expanded_terms``,
        ``answer``, ``sources``, ``web_activated``, ``error`` (si aplica).
        """
        if not self._pipeline_ready.is_set():
            return {
                "query": text, "answer": "",
                "sources": [], "web_activated": False,
                "error": "Pipeline unificado aún no disponible. Espera a que FAISS y LSI estén listos.",
            }

        with self._qrf_lock:
            qrf = self._qrf_holder[0]
        with self._hybrid_lock:
            hybrid = self._hybrid_holder[0]
        with self._cross_encoder_lock:
            cross_enc = self._cross_encoder_holder[0]
        with self._rag_lock:
            rag = self._rag_holder[0]

        if any(x is None for x in (qrf, hybrid, cross_enc, rag)):
            return {
                "query": text, "answer": "",
                "sources": [], "web_activated": False,
                "error": "Algún componente del pipeline no está cargado.",
            }

        return do_pipeline_ask(
            query            = text,
            qrf_pipeline     = qrf,
            hybrid_retriever = hybrid,
            cross_encoder    = cross_enc,
            rag_pipeline     = rag,
            cfg              = self.cfg,
        )

    # ── API pública — modos standalone ───────────────────────────────────────

    def query(self, text: str, top_n: int | None = None) -> list[dict]:
        """Búsqueda LSI local. Devuelve lista vacía si el modelo no está listo."""
        top_n = top_n if top_n is not None else self.cfg.retrieval_top_k
        if not self._lsi_ready.is_set():
            return []
        with self._lsi_lock:
            retriever = self._retriever_holder[0]
            if retriever is None:
                return []
            return retriever.retrieve(text, top_n=top_n)

    def query_with_web(self, text: str, top_n: int | None = None) -> dict:
        """Búsqueda LSI + fallback web si los resultados locales son insuficientes."""
        top_n = top_n if top_n is not None else self.cfg.retrieval_top_k
        local_results = self.query(text, top_n=top_n)
        return do_web_search(text, local_results, self.cfg)

    def semantic_query(self, text: str, top_k: int | None = None) -> list[dict]:
        """Búsqueda semántica densa directa sobre FAISS (sin reranking)."""
        top_k = top_k if top_k is not None else self.cfg.retrieval_top_k
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
            raw = self._faiss_mgr.search(query_vec, top_k=top_k)
        if not raw:
            return []
        chunk_ids = [r["chunk_id"] for r in raw]
        score_map = {r["chunk_id"]: r["score"] for r in raw}
        rows      = get_chunks_by_ids(chunk_ids, db_path=self.cfg.db_path)
        results   = [
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

    def qrf_search(self, text: str, top_k: int | None = None) -> list[dict]:
        """Búsqueda QRF standalone (expand + BRF + MMR sobre FAISS)."""
        top_k = top_k if top_k is not None else self.cfg.retrieval_top_k
        if not self._qrf_ready.is_set():
            return []
        with self._qrf_lock:
            pipeline = self._qrf_holder[0]
            if pipeline is None:
                return []
        return do_qrf_search(text, pipeline, top_k=top_k)

    def qrf_search_with_session(self, text: str, top_k: int | None = None) -> tuple[list[dict], str]:
        """Igual que ``qrf_search`` pero devuelve también un ``session_id``."""
        top_k = top_k if top_k is not None else self.cfg.retrieval_top_k
        if not self._qrf_ready.is_set():
            return [], ""
        with self._qrf_lock:
            pipeline = self._qrf_holder[0]
            if pipeline is None:
                return [], ""
        return do_qrf_search_with_session(text, pipeline, top_k=top_k)

    def rag_search(self, text: str, top_k: int | None = None) -> list[dict]:
        """Retrieval denso sin generación LLM."""
        top_k = top_k if top_k is not None else self.cfg.retrieval_top_k
        if not self._rag_ready.is_set():
            return []
        with self._rag_lock:
            pipeline = self._rag_holder[0]
            if pipeline is None:
                return []
        return do_rag_search(text, pipeline, top_k=top_k, candidate_k=self.cfg.rag_candidate_k)

    def rag_ask(self, text: str, top_k: int | None = None) -> dict:
        """Pipeline RAG standalone: retrieval denso → contexto → respuesta LLM."""
        top_k = top_k if top_k is not None else self.cfg.retrieval_top_k
        if not self._rag_ready.is_set():
            return {"query": text, "answer": "", "sources": [],
                    "error": "RAG pipeline no disponible aún."}
        with self._rag_lock:
            pipeline = self._rag_holder[0]
            if pipeline is None:
                return {"query": text, "answer": "", "sources": [],
                        "error": "RAG pipeline no disponible aún."}
        return do_rag_ask(
            text, pipeline,
            top_k       = top_k,
            candidate_k = self.cfg.rag_candidate_k,
            max_chunks  = self.cfg.rag_max_chunks,
            max_chars   = self.cfg.rag_max_chars,
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
            pipeline_ready   = self._pipeline_ready,
        )

    # ── Targets de hilos ──────────────────────────────────────────────────────

    def _target_crawler(self) -> None:
        run_crawler_thread(self.cfg, self._shutdown)

    def _target_indexing(self) -> None:
        run_indexing_thread(self.cfg, self._shutdown)

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
            cfg                  = self.cfg,
            shutdown             = self._shutdown,
            faiss_ready          = self._faiss_ready,
            lsi_ready            = self._lsi_ready,
            faiss_mgr            = self._faiss_mgr,
            faiss_lock           = self._faiss_lock,
            retriever_holder     = self._retriever_holder,
            lsi_lock             = self._lsi_lock,
            qrf_holder           = self._qrf_holder,
            qrf_lock             = self._qrf_lock,
            qrf_ready            = self._qrf_ready,
            hybrid_holder        = self._hybrid_holder,
            hybrid_lock          = self._hybrid_lock,
            cross_encoder_holder = self._cross_encoder_holder,
            cross_encoder_lock   = self._cross_encoder_lock,
            rag_holder           = self._rag_holder,
            rag_lock             = self._rag_lock,
            rag_ready            = self._rag_ready,
            pipeline_ready       = self._pipeline_ready,
        )
