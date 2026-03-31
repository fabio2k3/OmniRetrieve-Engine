"""
orchestrator.py
===============
Clase principal del orquestador OmniRetrieve-Engine.

Responsabilidad única: crear y coordinar los tres hilos de fondo
(crawler, indexing, lsi_rebuild) y exponer la API pública que usa
la CLI (query, status, start, stop).

El estado compartido entre hilos (retriever, eventos, lock) vive aquí.
La lógica de cada hilo está en threads.py.
La lógica de presentación está en cli.py.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from backend.database.schema import init_db, get_connection
from backend.database.index_repository import get_index_stats
from backend.indexing.pipeline import IndexingPipeline
from backend.embedding.embedder import Embedder
from backend.embedding.pipeline import EmbeddingPipeline
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever
from backend.retrieval.vector_retriever import VectorRetriever

from .config import OrchestratorConfig
from .threads import (
    run_crawler_thread,
    run_indexing_thread,
    run_lsi_rebuild_thread,
    run_embedding_thread,
)
from .cli import run_cli

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordina crawler, indexing y retrieval en hilos independientes.

    Uso
    ---
        cfg = OrchestratorConfig(pdf_threshold=5, lsi_rebuild_interval=1800)
        orc = Orchestrator(cfg)
        orc.start()       # arranca los tres hilos de fondo
        orc.run_cli()     # bucle interactivo en el hilo principal
        orc.stop()        # llamada automáticamente por 'quit' en la CLI
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.cfg = config or OrchestratorConfig()

        # Señales de coordinación
        self._shutdown  = threading.Event()
        self._lsi_lock  = threading.RLock()
        self._lsi_ready = threading.Event()

        # Retriever LSI compartido en lista mutable para poder hacer swap bajo lock
        self._retriever_holder: list[Optional[LSIRetriever]] = [None]

        # Retriever vectorial — mismo patrón que LSI
        self._vector_lock:    threading.RLock              = threading.RLock()
        self._vector_ready:   threading.Event              = threading.Event()
        self._vector_holder:  list[Optional[VectorRetriever]] = [None]
        self._embedder:       Embedder                     = Embedder(self.cfg.embed_model if config else "all-MiniLM-L6-v2")

        self._threads: list[threading.Thread] = []

        init_db(self.cfg.db_path)
        log.info("[Orchestrator] BD inicializada: %s", self.cfg.db_path)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Arranca los tres hilos de fondo."""
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

    def query(self, text: str, top_n: int = 10) -> list[dict]:
        """
        Ejecuta una query semántica sobre el modelo LSI actual.
        Devuelve lista vacía si el modelo aún no está listo.
        """
        if not self._lsi_ready.is_set():
            return []
        with self._lsi_lock:
            retriever = self._retriever_holder[0]
            if retriever is None:
                return []
            return retriever.retrieve(text, top_n=top_n)

    def vector_query(self, text: str, top_n: int = 10) -> list[dict]:
        """
        Ejecuta una query de similitud vectorial sobre embeddings de chunks.
        Devuelve lista vacía si el índice vectorial aún no está listo.
        """
        if not self._vector_ready.is_set():
            return []
        with self._vector_lock:
            retriever = self._vector_holder[0]
            if retriever is None:
                return []
            return retriever.retrieve(text, top_n=top_n)

    def status(self) -> dict:
        """Devuelve un snapshot del estado actual del sistema."""
        try:
            idx = get_index_stats(db_path=self.cfg.db_path)
        except Exception:
            idx = {}

        try:
            conn      = get_connection(self.cfg.db_path)
            total     = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            indexed   = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1").fetchone()[0]
            pending   = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=0").fetchone()[0]
            unindexed = conn.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
            ).fetchone()[0]
            conn.close()
        except Exception:
            total = indexed = pending = unindexed = -1

        with self._lsi_lock:
            retriever = self._retriever_holder[0]
            lsi_docs  = len(retriever.model.doc_ids) if retriever else 0

        try:
            conn          = get_connection(self.cfg.db_path)
            total_chunks  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded      = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL").fetchone()[0]
            conn.close()
        except Exception:
            total_chunks = embedded = -1

        from backend.embedding.chroma_store import count as _chroma_count
        with self._vector_lock:
            vec_retriever     = self._vector_holder[0]
            vec_chunks_loaded = _chroma_count(self.cfg.chroma_path) if vec_retriever else 0

        return {
            "docs_total":          total,
            "docs_pdf_indexed":    indexed,
            "docs_pdf_pending":    pending,
            "docs_not_in_index":   unindexed,
            "vocab_size":          idx.get("vocab_size", 0),
            "total_postings":      idx.get("total_postings", 0),
            "lsi_docs_in_model":   lsi_docs,
            "lsi_model_ready":     self._lsi_ready.is_set(),
            "total_chunks":        total_chunks,
            "embedded_chunks":     embedded,
            "vector_index_ready":  self._vector_ready.is_set(),
            "vector_chunks_loaded": vec_chunks_loaded,
            "timestamp":           datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    def run_cli(self) -> None:
        """Arranca la CLI interactiva en el hilo principal."""
        run_cli(
            shutdown      = self._shutdown,
            lsi_ready     = self._lsi_ready,
            lsi_min_docs  = self.cfg.lsi_min_docs,
            fn_query      = self.query,
            fn_vquery     = self.vector_query,
            fn_status     = self.status,
            fn_index      = self._do_index,
            fn_rebuild    = self._do_lsi_rebuild,
            fn_embed      = self._do_embed,
            fn_stop       = self.stop,
        )

    # ------------------------------------------------------------------
    # Operaciones internas — usadas por CLI y por los hilos vía callbacks
    # ------------------------------------------------------------------

    def _do_index(self) -> dict:
        """Ejecuta IndexingPipeline incremental y devuelve stats."""
        pipeline = IndexingPipeline(
            db_path=self.cfg.db_path,
            field=self.cfg.index_field,
        )
        stats = pipeline.run(reindex=False)
        log.info("[indexing] Completado — docs=%d términos=%d postings=%d",
                 stats["docs_processed"], stats["terms_added"], stats["postings_added"])
        return stats

    def _do_lsi_rebuild(self) -> dict | None:
        """
        Construye un nuevo LSIModel, lo guarda y actualiza el retriever
        compartido bajo _lsi_lock. Devuelve las stats o None si falla.
        """
        try:
            conn = get_connection(self.cfg.db_path)
            n_indexed = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM postings"
            ).fetchone()[0]
            conn.close()
        except Exception as exc:
            log.warning("[lsi] No se pudo consultar la BD: %s", exc)
            return None

        if n_indexed < self.cfg.lsi_min_docs:
            log.info("[lsi] Solo %d docs indexados (mínimo %d) — omitiendo.",
                     n_indexed, self.cfg.lsi_min_docs)
            return None

        k = min(self.cfg.lsi_k, n_indexed - 1)
        log.info("[lsi] Reconstruyendo modelo (n_indexed=%d, k=%d)…", n_indexed, k)

        try:
            model = LSIModel(k=k)
            stats = model.build(db_path=self.cfg.db_path)
            model.save(path=self.cfg.model_path)

            new_retriever = LSIRetriever(model=model)
            new_retriever.load(model_path=self.cfg.model_path, db_path=self.cfg.db_path)

            with self._lsi_lock:
                self._retriever_holder[0] = new_retriever

            self._lsi_ready.set()
            log.info("[lsi] Modelo actualizado — n_docs=%d varianza=%.1f%%",
                     stats["n_docs"], stats["var_explained"] * 100)
            return stats

        except Exception as exc:
            log.error("[lsi] Error durante rebuild: %s", exc, exc_info=True)
            return None

    def _do_embed(self) -> dict:
        """
        Ejecuta EmbeddingPipeline incremental y recarga el VectorRetriever.

        El embedder se reutiliza entre llamadas (modelo ya cargado en RAM).
        El VectorRetriever se recarga bajo _vector_lock para que vector_query()
        nunca vea un estado intermedio.
        """
        pipeline = EmbeddingPipeline(
            db_path     = self.cfg.db_path,
            chroma_path = self.cfg.chroma_path,
            embedder    = self._embedder,
            batch_size  = self.cfg.embed_batch_size,
        )
        stats = pipeline.run()
        log.info(
            "[embedding] Completado — chunks embebidos=%d",
            stats["chunks_embedded"],
        )

        # Recargar la matriz vectorial solo si se embebió algo nuevo
        if stats["chunks_embedded"] > 0:
            self._reload_vector_index()

        return stats

    def _reload_vector_index(self) -> None:
        """Construye un nuevo VectorRetriever y hace swap bajo lock."""
        try:
            new_retriever = VectorRetriever(embedder=self._embedder)
            new_retriever.load(db_path=self.cfg.db_path, chroma_path=self.cfg.chroma_path)
            with self._vector_lock:
                self._vector_holder[0] = new_retriever
            self._vector_ready.set()
            from backend.embedding.chroma_store import count as _chroma_count
            log.info(
                "[vector] Índice recargado — %d vectores en ChromaDB.",
                _chroma_count(self.cfg.chroma_path),
            )
        except Exception as exc:
            log.error("[vector] Error al recargar índice: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Targets de hilos — delegan en threads.py
    # ------------------------------------------------------------------

    def _target_crawler(self) -> None:
        run_crawler_thread(self.cfg, self._shutdown)

    def _target_indexing(self) -> None:
        run_indexing_thread(self.cfg, self._shutdown, self._do_index)

    def _target_lsi_rebuild(self) -> None:
        run_lsi_rebuild_thread(
            self.cfg,
            self._shutdown,
            self._lsi_lock,
            self._lsi_ready,
            self._retriever_holder,
        )

    def _target_embedding(self) -> None:
        # Cargar el indice vectorial al arrancar, independientemente de si
        # hay chunks pendientes. Esto cubre el caso habitual: sesion nueva
        # con ChromaDB ya poblada de sesiones anteriores. Sin esto,
        # _vector_ready nunca se activa si pending < embed_threshold.
        self._reload_vector_index()
        run_embedding_thread(self.cfg, self._shutdown, self._do_embed)