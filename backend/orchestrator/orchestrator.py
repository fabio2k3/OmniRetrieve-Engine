"""
orchestrator.py
===============
Clase principal del orquestador OmniRetrieve-Engine.

Responsabilidad única: crear y coordinar los cuatro hilos de fondo
(crawler, indexing, lsi_rebuild, embedding) y exponer la API pública
que usa la CLI (query, semantic_query, status, start, stop).

El estado compartido entre hilos (retriever, faiss_mgr, eventos, locks)
vive aquí. La lógica de cada hilo está en threads.py.
La lógica de presentación está en cli.py.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from backend.database.schema import init_db, get_connection
from backend.database.index_repository import get_index_stats
from backend.database.chunk_repository import get_chunks, get_chunk_stats, get_chunks_by_ids
from backend.indexing.pipeline import IndexingPipeline
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever
from backend.embedding.pipeline import EmbeddingPipeline
from backend.retrieval.embedding_retriever import EmbeddingRetriever
from backend.embedding.faiss_index import FaissIndexManager
from backend.database.embedding_repository import init_embedding_schema

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
    Coordina crawler, indexing, retrieval y embedding en hilos independientes.

    Uso
    ---
        cfg = OrchestratorConfig(pdf_threshold=5, embed_threshold=20)
        orc = Orchestrator(cfg)
        orc.start()       # arranca los cuatro hilos de fondo
        orc.run_cli()     # bucle interactivo en el hilo principal
        orc.stop()        # llamada automáticamente por 'quit' en la CLI
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        self.cfg = config or OrchestratorConfig()

        # ── Señales de coordinación ──────────────────────────────────────────
        self._shutdown  = threading.Event()

        # LSI
        self._lsi_lock  = threading.RLock()
        self._lsi_ready = threading.Event()
        self._retriever_holder: list[Optional[LSIRetriever]] = [None]

        # Embedding / FAISS
        self._faiss_lock    = threading.RLock()
        self._faiss_ready   = threading.Event()
        self._faiss_mgr: Optional[FaissIndexManager] = None

        self._threads: list[threading.Thread] = []

        init_db(self.cfg.db_path)
        init_embedding_schema(self.cfg.db_path)
        self._init_faiss_mgr()

        log.info("[Orchestrator] BD inicializada: %s", self.cfg.db_path)

    # ------------------------------------------------------------------
    # Inicialización del índice FAISS
    # ------------------------------------------------------------------

    def _init_faiss_mgr(self) -> None:
        """
        Crea el FaissIndexManager e intenta cargar un índice previo desde disco.

        La dimensión del índice se fija según el modelo configurado.
        Se usa un valor por defecto de 384 (all-MiniLM-L6-v2); si el modelo
        cambia, el primer rebuild usará la dimensión correcta automáticamente.
        """
        # Obtener la dimensión real del modelo para crear el manager correcto.
        # Lo hacemos aquí de forma lazy (sin cargar el modelo completo)
        # usando el valor por defecto de la dimensión conocida del modelo.
        dim = self._resolve_embedding_dim()

        self._faiss_mgr = FaissIndexManager(
            dim           = dim,
            nlist         = self.cfg.embed_nlist,
            m             = self.cfg.embed_m,
            nbits         = self.cfg.embed_nbits,
            nprobe        = self.cfg.embed_nprobe,
            rebuild_every = self.cfg.embed_rebuild_every,
            index_path    = self.cfg.faiss_index_path,
            id_map_path   = self.cfg.faiss_id_map_path,
        )

        loaded = self._faiss_mgr.load()
        if loaded:
            self._faiss_ready.set()
            log.info(
                "[Orchestrator] Índice FAISS cargado desde disco — "
                "tipo=%s vectores=%d",
                self._faiss_mgr.index_type,
                self._faiss_mgr.total_vectors,
            )

    def _resolve_embedding_dim(self) -> int:
        """
        Devuelve la dimensión del modelo de embedding sin cargar los pesos.

        Usa un mapa de dimensiones conocidas para los modelos más comunes.
        Si el modelo no está en el mapa, devuelve 384 como valor por defecto.
        """
        _KNOWN_DIMS = {
            "all-MiniLM-L6-v2":          384,
            "all-MiniLM-L12-v2":         384,
            "all-mpnet-base-v2":         768,
            "multi-qa-MiniLM-L6-cos-v1": 384,
            "allenai-specter":           768,
            "paraphrase-MiniLM-L6-v2":   384,
        }
        return _KNOWN_DIMS.get(self.cfg.embed_model, 384)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Arranca los cuatro hilos de fondo."""
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

    def semantic_query(self, text: str, top_k: int = 10) -> list[dict]:
        """
        Ejecuta una búsqueda semántica densa usando el índice FAISS.

        Para cada chunk recuperado, enriquece el resultado con el texto
        del chunk y los metadatos básicos del documento.

        Parámetros
        ----------
        text  : consulta en lenguaje natural.
        top_k : número de chunks a devolver.

        Devuelve
        --------
        Lista de dicts con claves:
            chunk_id, score, arxiv_id, chunk_index, text, title.
        Lista vacía si el índice no está listo o no hay vectores.
        """
        if not self._faiss_ready.is_set():
            return []

        # 1. Vectorizar la query
        try:
            from backend.embedding.embedder import ChunkEmbedder
            embedder  = ChunkEmbedder(model_name=self.cfg.embed_model)
            query_vec = embedder.encode_single(text)
        except Exception as exc:
            log.error("[semantic_query] Error al vectorizar query: %s", exc)
            return []

        # 2. Buscar en FAISS
        with self._faiss_lock:
            if self._faiss_mgr is None:
                return []
            raw_results = self._faiss_mgr.search(query_vec, top_k=top_k)

        if not raw_results:
            return []

        # 3. Enriquecer con texto y metadatos desde la BD
        chunk_ids = [r["chunk_id"] for r in raw_results]
        score_map = {r["chunk_id"]: r["score"] for r in raw_results}
        rows      = get_chunks_by_ids(chunk_ids, db_path=self.cfg.db_path)

        results = []
        for row in rows:
            results.append({
                "chunk_id":    row["chunk_id"],
                "score":       score_map.get(row["chunk_id"], float("inf")),
                "arxiv_id":    row["arxiv_id"],
                "chunk_index": row["chunk_index"],
                "text":        row["text"],
                "title":       row.get("title", ""),
            })
        results.sort(key=lambda r: r["score"])
        return results

    def status(self) -> dict:
        """Devuelve un snapshot del estado actual del sistema."""
        try:
            idx = get_index_stats(db_path=self.cfg.db_path)
        except Exception:
            idx = {}

        try:
            conn      = get_connection(self.cfg.db_path)
            total     = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            indexed   = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1"
            ).fetchone()[0]
            pending   = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=0"
            ).fetchone()[0]
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
            chunk_stats = get_chunk_stats(self.cfg.db_path)
        except Exception:
            chunk_stats = {"total_chunks": -1, "embedded_chunks": -1, "pending_chunks": -1}

        with self._faiss_lock:
            faiss_vectors = self._faiss_mgr.total_vectors if self._faiss_mgr else 0
            faiss_type    = self._faiss_mgr.index_type    if self._faiss_mgr else "none"

        return {
            "docs_total":            total,
            "docs_pdf_indexed":      indexed,
            "docs_pdf_pending":      pending,
            "docs_not_in_index":     unindexed,
            "vocab_size":            idx.get("vocab_size", 0),
            "total_postings":        idx.get("total_postings", 0),
            "lsi_docs_in_model":     lsi_docs,
            "lsi_model_ready":       self._lsi_ready.is_set(),
            "total_chunks":          chunk_stats["total_chunks"],
            "embedded_chunks":       chunk_stats["embedded_chunks"],
            "pending_chunks":        chunk_stats["pending_chunks"],
            "faiss_vectors":         faiss_vectors,
            "faiss_index_type":      faiss_type,
            "faiss_ready":           self._faiss_ready.is_set(),
            "embed_model":           self.cfg.embed_model,
            "timestamp":             datetime.now(timezone.utc).strftime(
                                         "%Y-%m-%d %H:%M:%S UTC"
                                     ),
        }

    def run_cli(self) -> None:
        """Arranca la CLI interactiva en el hilo principal."""
        run_cli(
            shutdown     = self._shutdown,
            lsi_ready    = self._lsi_ready,
            lsi_min_docs = self.cfg.lsi_min_docs,
            fn_query     = self.query,
            fn_status    = self.status,
            fn_index     = self._do_index,
            fn_rebuild   = self._do_lsi_rebuild,
            fn_stop      = self.stop,
        )

    # ------------------------------------------------------------------
    # Operaciones internas
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

    def _do_embed(self) -> dict:
        """
        Ejecuta EmbeddingPipeline incremental y actualiza el FaissIndexManager
        compartido con el índice reconstruido.

        El FaissIndexManager gestiona internamente la política de rebuild
        (cada embed_rebuild_every chunks). Al terminar, si el índice tiene
        vectores, se activa _faiss_ready para habilitar semantic_query.
        """
        pipeline = EmbeddingPipeline(
            db_path       = self.cfg.db_path,
            model_name    = self.cfg.embed_model,
            batch_size    = self.cfg.embed_batch_size,
            rebuild_every = self.cfg.embed_rebuild_every,
            nlist         = self.cfg.embed_nlist,
            m             = self.cfg.embed_m,
            nbits         = self.cfg.embed_nbits,
            nprobe        = self.cfg.embed_nprobe,
            index_path    = self.cfg.faiss_index_path,
            id_map_path   = self.cfg.faiss_id_map_path,
        )
        stats = pipeline.run(reembed=False)

        # Recargar el índice actualizado en el manager compartido
        if self.cfg.faiss_index_path.exists():
            with self._faiss_lock:
                loaded = self._faiss_mgr.load()
                if loaded:
                    self._faiss_ready.set()
                    log.info(
                        "[embedding] Índice FAISS actualizado — tipo=%s vectores=%d",
                        self._faiss_mgr.index_type,
                        self._faiss_mgr.total_vectors,
                    )

        log.info(
            "[embedding] Completado — chunks=%d lotes=%d rebuilds=%d",
            stats["chunks_processed"],
            stats["batches_processed"],
            stats["rebuilds_triggered"],
        )
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

    # ------------------------------------------------------------------
    # Targets de hilos
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
        run_embedding_thread(self.cfg, self._shutdown, self._do_embed)