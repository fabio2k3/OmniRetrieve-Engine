"""
threads.py
==========
Hilos de fondo del orquestador OmniRetrieve-Engine.

Cada función de este módulo es el target de un hilo daemon independiente.
Reciben el estado compartido por parámetro (config, eventos, lock, retriever)
y no conocen nada de la CLI ni del ciclo principal.

Hilos
-----
run_crawler_thread      — ejecuta el crawler de arXiv de forma continua.
run_indexing_thread     — detecta PDFs nuevos y dispara indexación incremental.
run_lsi_rebuild_thread  — reconstruye el modelo LSI cada N segundos.
run_embedding_thread    — detecta chunks sin embedding y dispara EmbeddingPipeline.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from backend.database.schema import get_connection
from backend.database.chunk_repository import get_chunk_stats
from backend.crawler.crawler import Crawler, CrawlerConfig
from backend.indexing.pipeline import IndexingPipeline
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever
from .config import OrchestratorConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilo 1 — Crawler
# ---------------------------------------------------------------------------

def run_crawler_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
) -> None:
    """
    Ejecuta el crawler de arXiv de forma continua hasta que shutdown se active.

    El crawler tiene su propio evento interno (_stop). Un hilo auxiliar
    (watchdog) espera la señal de shutdown del orquestador y llama a
    crawler.stop() para que run_forever() retorne limpiamente.
    """
    log.info("[crawler] Arrancando…")
    crawler = Crawler(config=CrawlerConfig(
        ids_per_discovery  = cfg.ids_per_discovery,
        batch_size         = cfg.batch_size,
        pdf_batch_size     = cfg.pdf_batch_size,
        discovery_interval = cfg.discovery_interval,
        download_interval  = cfg.download_interval,
        pdf_interval       = cfg.pdf_interval,
    ))

    def _watchdog() -> None:
        shutdown.wait()
        log.info("[crawler] Señal de shutdown — parando crawler…")
        crawler.stop()

    threading.Thread(target=_watchdog, daemon=True, name="crawler-watchdog").start()

    try:
        crawler.run_forever()
    except Exception as exc:
        log.error("[crawler] Error inesperado: %s", exc, exc_info=True)

    log.info("[crawler] Detenido.")


# ---------------------------------------------------------------------------
# Hilo 2 — Indexing watcher
# ---------------------------------------------------------------------------

def run_indexing_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
    do_index: callable,
) -> None:
    """
    Sondea la BD cada index_poll_interval segundos.
    Llama a do_index() cuando hay ≥ pdf_threshold PDFs nuevos sin indexar.

    Parámetros
    ----------
    cfg      : configuración del orquestador.
    shutdown : evento de parada compartido.
    do_index : callable sin argumentos que ejecuta la indexación incremental.
               Separado para facilitar tests y mantener este módulo sin estado.
    """
    log.info("[indexing] Watcher iniciado (umbral=%d PDFs, poll=%ds).",
             cfg.pdf_threshold, int(cfg.index_poll_interval))

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.index_poll_interval)
        if shutdown.is_set():
            break

        try:
            conn = get_connection(cfg.db_path)
            unindexed = conn.execute(
                "SELECT COUNT(*) FROM documents "
                "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
            ).fetchone()[0]
            conn.close()
        except Exception as exc:
            log.warning("[indexing] Error al consultar BD: %s", exc)
            continue

        if unindexed >= cfg.pdf_threshold:
            log.info("[indexing] %d PDFs sin indexar ≥ umbral %d — indexando…",
                     unindexed, cfg.pdf_threshold)
            try:
                do_index()
            except Exception as exc:
                log.error("[indexing] Error durante indexación: %s", exc, exc_info=True)
        else:
            log.debug("[indexing] %d PDFs pendientes (umbral=%d) — esperando.",
                      unindexed, cfg.pdf_threshold)

    log.info("[indexing] Watcher detenido.")


# ---------------------------------------------------------------------------
# Hilo 3 — LSI rebuild
# ---------------------------------------------------------------------------

def run_lsi_rebuild_thread(
    cfg:       OrchestratorConfig,
    shutdown:  threading.Event,
    lsi_lock:  threading.RLock,
    lsi_ready: threading.Event,
    retriever_holder: list,   # lista de un elemento [LSIRetriever | None]
) -> None:
    """
    Reconstruye el modelo LSI cada lsi_rebuild_interval segundos.

    Usa retriever_holder (lista de un elemento) como contenedor mutable
    del retriever compartido, protegido por lsi_lock.

    El primer intento de build ocurre al arrancar el hilo, sin esperar
    el intervalo, para que el sistema esté listo lo antes posible.

    Parámetros
    ----------
    cfg              : configuración del orquestador.
    shutdown         : evento de parada compartido.
    lsi_lock         : RLock que protege retriever_holder durante el swap.
    lsi_ready        : Event que se activa cuando el primer modelo está listo.
    retriever_holder : lista[LSIRetriever | None] de un solo elemento.
    """
    log.info("[lsi] Hilo de rebuild iniciado (intervalo=%ds, k=%d).",
             int(cfg.lsi_rebuild_interval), cfg.lsi_k)

    _try_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.lsi_rebuild_interval)
        if shutdown.is_set():
            break
        _try_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder)

    log.info("[lsi] Hilo de rebuild detenido.")


def _try_rebuild(
    cfg:              OrchestratorConfig,
    lsi_lock:         threading.RLock,
    lsi_ready:        threading.Event,
    retriever_holder: list,
) -> None:
    """
    Intenta construir un nuevo modelo LSI y actualizar el retriever compartido.

    No lanza excepciones: los errores se loguean y se ignoran para que el
    hilo de rebuild no muera ante un fallo puntual.
    """
    try:
        conn = get_connection(cfg.db_path)
        n_indexed = conn.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM postings"
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        log.warning("[lsi] No se pudo consultar la BD: %s", exc)
        return

    if n_indexed < cfg.lsi_min_docs:
        log.info("[lsi] Solo %d docs indexados (mínimo %d) — omitiendo rebuild.",
                 n_indexed, cfg.lsi_min_docs)
        return
    
    # Ajustar k al tamaño real del corpus (SVD requiere k < n_docs)
    k = min(cfg.lsi_k, n_indexed - 1)
    log.info("[lsi] Reconstruyendo modelo (n_indexed=%d, k=%d)…", n_indexed, k)

    try:
        model = LSIModel(k=k)
        stats = model.build(db_path=cfg.db_path)
        model.save(path=cfg.model_path)

        new_retriever = LSIRetriever(model=model)
        new_retriever.load(model_path=cfg.model_path, db_path=cfg.db_path)

        with lsi_lock:
            retriever_holder[0] = new_retriever

        lsi_ready.set()
        log.info("[lsi] Modelo actualizado — n_docs=%d varianza=%.1f%%",
                 stats["n_docs"], stats["var_explained"] * 100)

    except Exception as exc:
        log.error("[lsi] Error durante rebuild: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Hilo 4 — Embedding watcher
# ---------------------------------------------------------------------------

def run_embedding_thread(
    cfg:       OrchestratorConfig,
    shutdown:  threading.Event,
    do_embed:  callable,
) -> None:
    """
    Sondea la BD cada embed_poll_interval segundos.
    Llama a do_embed() cuando hay ≥ embed_threshold chunks sin embedding.

    El primer chequeo ocurre inmediatamente al arrancar el hilo para
    embedir chunks que pudieran haber quedado pendientes en ejecuciones
    anteriores.

    Parámetros
    ----------
    cfg      : configuración del orquestador.
    shutdown : evento de parada compartido.
    do_embed : callable sin argumentos que ejecuta el pipeline de embedding.
               Separado para facilitar tests y mantener este módulo sin estado.
    """
    log.info("[embedding] Watcher iniciado (umbral=%d chunks, poll=%ds).",
             cfg.embed_threshold, int(cfg.embed_poll_interval))

    # Primer intento inmediato al arrancar
    _try_embed(cfg, do_embed)

    while not shutdown.is_set():
        shutdown.wait(timeout=cfg.embed_poll_interval)
        if shutdown.is_set():
            break
        _try_embed(cfg, do_embed)

    log.info("[embedding] Watcher detenido.")


def _try_embed(cfg: OrchestratorConfig, do_embed: callable) -> None:
    """
    Comprueba si hay chunks pendientes de embedding y, si superan el umbral,
    ejecuta do_embed().

    No lanza excepciones: los errores se loguean y se ignoran para que el
    hilo no muera ante un fallo puntual.
    """
    try:
        stats = get_chunk_stats(cfg.db_path)
        pending = stats["pending_chunks"]
    except Exception as exc:
        log.warning("[embedding] Error al consultar BD: %s", exc)
        return

    if pending >= cfg.embed_threshold:
        log.info("[embedding] %d chunks sin embedding ≥ umbral %d — embediendo…",
                 pending, cfg.embed_threshold)
        try:
            do_embed()
        except Exception as exc:
            log.error("[embedding] Error durante embedding: %s", exc, exc_info=True)
    else:
        log.debug("[embedding] %d chunks pendientes (umbral=%d) — esperando.",
                  pending, cfg.embed_threshold)