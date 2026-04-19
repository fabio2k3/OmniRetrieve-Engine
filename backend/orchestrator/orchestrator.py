"""
orchestrator.py
===============
Coordinador principal de OmniRetrieve-Engine.

Responsabilidad única: crear el estado compartido, arrancar los cuatro
hilos daemon y exponer la API pública que usan la CLI y el main.

Toda la lógica de negocio está delegada en módulos internos:
    _faiss.py       → inicialización del índice FAISS.
    _operations.py  → do_index, do_lsi_rebuild, do_embed, do_web_search.
    _status.py      → build_status().
    threads/        → un fichero por hilo daemon.
    cli.py          → presentación e interacción con el usuario.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.database.schema import init_db
from backend.database.embedding_repository import init_embedding_schema
from backend.embedding.faiss_index import FaissIndexManager
from backend.retrieval.lsi_retriever import LSIRetriever

from .config      import OrchestratorConfig
from ._faiss      import init_faiss_mgr
from ._operations import do_index, do_lsi_rebuild, do_embed, do_web_search
from ._status     import build_status
from .threads     import (
    run_crawler_thread,
    run_indexing_thread,
    run_lsi_rebuild_thread,
    run_embedding_thread,
)
from .cli import run_cli

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordina crawler, indexing BM25, LSI, embedding y búsqueda web
    en hilos independientes y expone las queries al exterior.

    Uso
    ---
        cfg = OrchestratorConfig(pdf_threshold=5, embed_threshold=20)
        orc = Orchestrator(cfg)
        orc.start()       # arranca los cuatro hilos daemon
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
        """Arranca los cuatro hilos daemon. No bloquea."""
        specs = [
            ("crawler",     self._target_crawler),
            ("indexing",    self._target_indexing),
            ("lsi_rebuild", self._target_lsi_rebuild),
            ("embedding",   self._target_embedding),
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
            shutdown     = self._shutdown,
            lsi_ready    = self._lsi_ready,
            lsi_min_docs = self.cfg.lsi_min_docs,
            fn_query     = self.query,
            fn_query_web = self.query_with_web,
            fn_status    = self.status,
            fn_index     = lambda: do_index(self.cfg),
            fn_rebuild   = lambda: do_lsi_rebuild(
                self.cfg, self._lsi_lock, self._lsi_ready, self._retriever_holder
            ),
            fn_stop      = self.stop,
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
        Búsqueda semántica densa usando el índice FAISS.

        Vectoriza la query, busca en FAISS y enriquece los resultados con
        texto y metadatos de la BD.

        Returns
        -------
        list[dict] con claves: ``chunk_id``, ``score``, ``arxiv_id``,
        ``chunk_index``, ``text``, ``title``. Lista vacía si el índice no está listo.
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
