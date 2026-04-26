"""
crawler.py
==========
Orquestador principal del módulo de adquisición.

Responsabilidad única
---------------------
Crear, arrancar y detener los tres hilos daemon de adquisición.
Toda la lógica de negocio vive en los loops del subpaquete ``loops/``.

Hilos gestionados
-----------------
    DiscoveryLoop  (discovery)  — descubre nuevos IDs en cada fuente.
    DownloaderLoop (downloader) — descarga metadatos de IDs pendientes.
    TextLoop       (text)       — descarga texto completo y genera chunks.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from .clients.base_client  import BaseClient
from .clients.arxiv        import ArxivClient
from .config               import CrawlerConfig
from .id_store             import IdStore
from .loops                import DiscoveryLoop, DownloaderLoop, TextLoop

logger = logging.getLogger(__name__)


class Crawler:
    """
    Gestiona el ciclo de vida de los tres hilos daemon de adquisición.

    Parámetros
    ----------
    config  : configuración del crawler (``CrawlerConfig``).
              Si se omite se usa la configuración por defecto.
    clients : lista de clientes ``BaseClient`` a interrogar.
    client  : alias de compatibilidad hacia atrás (un único cliente).
              Se ignora si ``clients`` está presente.

    Ejemplo de uso
    --------------
    >>> crawler = Crawler()
    >>> crawler.run_forever()          # bloquea hasta Ctrl-C
    """

    def __init__(
        self,
        config:  Optional[CrawlerConfig]    = None,
        clients: Optional[List[BaseClient]] = None,
        client:  Optional[BaseClient]       = None,
    ) -> None:
        self.config = config or CrawlerConfig()

        if clients:
            self._clients: List[BaseClient] = list(clients)
        elif client:
            self._clients = [client]
        else:
            self._clients = [ArxivClient()]

        self._client_map: Dict[str, BaseClient] = {
            c.source_name: c for c in self._clients
        }

        self.id_store = IdStore(self.config.ids_csv)

        # Ajustar offset si ya existen IDs persistidos
        if self.config.discovery_start == 0 and self.id_store.total > 0:
            self.config.discovery_start = self.id_store.total
            logger.info(
                "[Crawler] Offset inicial ajustado a %d.", self.config.discovery_start
            )

        # Importación lazy para no forzar SQLite en tests unitarios
        from ..database.schema import init_db, DB_PATH
        from ..database import crawler_repository as repo
        from ..database.chunk_repository import save_chunks
        init_db(DB_PATH)

        self._stop = threading.Event()

        _loops = [
            DiscoveryLoop(self.config, self._clients, self.id_store, self._stop),
            DownloaderLoop(self.config, self._client_map, self.id_store, repo, DB_PATH, self._stop),
            TextLoop(self.config, self._client_map, repo, save_chunks, DB_PATH, self._stop),
        ]
        _names = ["discovery", "downloader", "text"]
        self._threads = [
            threading.Thread(target=loop.run, name=name, daemon=True)
            for loop, name in zip(_loops, _names)
        ]

    # ── Control del ciclo de vida ─────────────────────────────────────────────

    def start(self) -> None:
        """Arranca los tres hilos daemon. No bloquea."""
        self._stop.clear()
        for t in self._threads:
            t.start()
        sources = [c.source_name for c in self._clients]
        logger.info("[Crawler] Iniciado. Fuentes: %s. Config: %s", sources, self.config)

    def stop(self) -> None:
        """Señaliza la parada y espera a que los tres hilos terminen (timeout=15s)."""
        logger.info("[Crawler] Deteniendo ...")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=15)
        logger.info("[Crawler] Detenido.")

    def run_forever(self) -> None:
        """Arranca el crawler y bloquea hasta ``KeyboardInterrupt``."""
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Crawler] KeyboardInterrupt recibido.")
        finally:
            self.stop()
