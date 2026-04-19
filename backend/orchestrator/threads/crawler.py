"""
threads/crawler.py
==================
Hilo de fondo del crawler de arXiv.

Responsabilidad única: arrancar el ``Crawler`` y mantenerlo corriendo hasta
que el orquestador señale el shutdown. Un hilo watchdog auxiliar espera la
señal y llama a ``crawler.stop()`` para que ``run_forever()`` retorne limpiamente.
"""

from __future__ import annotations

import logging
import threading

from backend.crawler.crawler import Crawler, CrawlerConfig
from ..config import OrchestratorConfig

log = logging.getLogger(__name__)


def run_crawler_thread(
    cfg:      OrchestratorConfig,
    shutdown: threading.Event,
) -> None:
    """
    Ejecuta el crawler de forma continua hasta que ``shutdown`` se active.

    Parámetros
    ----------
    cfg      : configuración del orquestador (parámetros del crawler).
    shutdown : evento de parada compartido con el orquestador.
    """
    log.info("[crawler] Arrancando...")
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
        log.info("[crawler] Señal de shutdown — parando crawler...")
        crawler.stop()

    threading.Thread(target=_watchdog, daemon=True, name="crawler-watchdog").start()

    try:
        crawler.run_forever()
    except Exception as exc:
        log.error("[crawler] Error inesperado: %s", exc, exc_info=True)

    log.info("[crawler] Detenido.")
