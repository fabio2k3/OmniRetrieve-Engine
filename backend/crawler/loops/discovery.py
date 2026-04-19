"""
loops/discovery.py
==================
Hilo de descubrimiento de IDs.

Responsabilidad única
---------------------
Interrogar periódicamente a cada cliente registrado para obtener IDs locales
nuevos y persistirlos en el ``IdStore`` con el prefijo de fuente correcto.

Este módulo no sabe nada de metadatos, texto ni chunking.
"""

from __future__ import annotations

import logging
import threading
from typing import List

from ..config   import CrawlerConfig
from ..id_store import IdStore
from ..clients.base_client import BaseClient

logger = logging.getLogger(__name__)


class DiscoveryLoop:
    """
    Recorre cada cliente buscando IDs nuevos cada ``discovery_interval`` segundos.

    Parámetros del constructor
    --------------------------
    config   : configuración compartida del Crawler.
    clients  : lista de clientes registrados.
    id_store : almacén thread-safe de IDs conocidos.
    stop     : evento que señaliza la parada del hilo.
    """

    def __init__(
        self,
        config:   CrawlerConfig,
        clients:  List[BaseClient],
        id_store: IdStore,
        stop:     threading.Event,
    ) -> None:
        self._config   = config
        self._clients  = clients
        self._id_store = id_store
        self._stop     = stop

    def run(self) -> None:
        """
        Bucle principal del hilo de descubrimiento.

        En cada iteración recorre todos los clientes, solicita hasta
        ``ids_per_discovery`` IDs comenzando en ``discovery_start`` y
        añade los nuevos al ``IdStore``. Espera ``discovery_interval``
        segundos entre iteraciones, o termina si ``stop`` se activa.
        """
        cfg = self._config
        while not self._stop.is_set():
            try:
                self._run_cycle(cfg)
            except Exception as exc:
                logger.error("[Discovery] Error inesperado: %s", exc, exc_info=True)
            self._stop.wait(cfg.discovery_interval)

    def _run_cycle(self, cfg: CrawlerConfig) -> None:
        for client in self._clients:
            if self._stop.is_set():
                break
            logger.info(
                "[Discovery] [%s] Buscando %d IDs (offset=%d) ...",
                client.source_name, cfg.ids_per_discovery, cfg.discovery_start,
            )
            local_ids = client.fetch_ids(
                max_results=cfg.ids_per_discovery,
                start=cfg.discovery_start,
            )
            if not local_ids:
                continue

            doc_ids = [client.make_doc_id(lid) for lid in local_ids]
            added   = self._id_store.add_ids(doc_ids)
            logger.info(
                "[Discovery] [%s] %d encontrados → %d nuevos. %s",
                client.source_name, len(doc_ids), added, self._id_store,
            )
            cfg.discovery_start += len(local_ids)
            if added == 0:
                logger.info(
                    "[Discovery] [%s] Página ya conocida (offset=%d) — avanzando.",
                    client.source_name, cfg.discovery_start,
                )
