"""
loops/downloader.py
===================
Hilo de descarga de metadatos.

Responsabilidad única
---------------------
Consumir IDs pendientes del ``IdStore``, agruparlos por fuente,
llamar a ``fetch_documents()`` de cada cliente y persistir los resultados
en CSV y SQLite.

Este módulo no sabe nada de texto completo ni chunking.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from ..config   import CrawlerConfig
from ..document import Document
from ..id_store import IdStore
from ..clients.base_client import BaseClient
from .._routing import client_for, local_id as extract_local_id

logger = logging.getLogger(__name__)


class DownloaderLoop:
    """
    Descarga metadatos de artículos pendientes en lotes.

    Parámetros del constructor
    --------------------------
    config     : configuración compartida.
    client_map : diccionario ``source_name → BaseClient``.
    id_store   : almacén de IDs.
    repo       : módulo ``crawler_repository`` (inyectado por ``Crawler``).
    db_path    : ruta al fichero SQLite.
    stop       : evento de parada.
    """

    def __init__(
        self,
        config:     CrawlerConfig,
        client_map: Dict[str, BaseClient],
        id_store:   IdStore,
        repo:       Any,
        db_path:    Any,
        stop:       threading.Event,
    ) -> None:
        self._config     = config
        self._client_map = client_map
        self._id_store   = id_store
        self._repo       = repo
        self._db_path    = db_path
        self._stop       = stop

    def run(self) -> None:
        """
        Bucle principal del hilo de metadatos.

        Espera 10 segundos al arranque para dar tiempo al hilo de
        descubrimiento a poblar el ``IdStore``. Luego procesa lotes
        de ``batch_size`` IDs pendientes hasta que no queden más,
        durmiendo ``download_interval`` segundos entre ciclos.
        """
        cfg = self._config
        logger.info("[Downloader] Esperando 10s para IDs iniciales ...")
        self._stop.wait(10)

        while not self._stop.is_set():
            pending = self._id_store.get_pending_batch(cfg.batch_size)
            if not pending:
                logger.info(
                    "[Downloader] Sin IDs pendientes. Durmiendo %ds ...",
                    int(cfg.download_interval),
                )
                self._stop.wait(cfg.download_interval)
                continue

            logger.info("[Downloader] Procesando lote de %d artículos ...", len(pending))
            try:
                self._process_batch(pending)
            except Exception as exc:
                logger.error("[Downloader] Error en lote: %s", exc, exc_info=True)

            self._stop.wait(cfg.download_interval)

    # ── Lógica interna del lote ───────────────────────────────────────────────

    def _process_batch(self, pending: List[str]) -> None:
        cfg           = self._config
        already_saved = Document.load_ids(cfg.documents_csv)

        by_source, failed_sources = self._group_by_source(pending)
        all_docs = self._fetch_all(by_source, failed_sources)
        self._persist(all_docs, already_saved, pending, failed_sources)

    def _group_by_source(
        self, pending: List[str]
    ) -> tuple[Dict[str, List[str]], set]:
        """Agrupa IDs compuestos por fuente y devuelve el mapa de IDs locales."""
        by_source: Dict[str, List[str]] = {}
        for doc_id in pending:
            client = client_for(doc_id, self._client_map)
            if client is None:
                logger.warning("[Downloader] Sin cliente para %r — omitiendo.", doc_id)
                continue
            by_source.setdefault(client.source_name, []).append(extract_local_id(doc_id))
        return by_source, set()

    def _fetch_all(
        self,
        by_source: Dict[str, List[str]],
        failed_sources: set,
    ) -> List[Document]:
        """Llama a fetch_documents por fuente y acumula los resultados."""
        all_docs: List[Document] = []
        for source_name, local_ids in by_source.items():
            client = self._client_map[source_name]
            docs   = client.fetch_documents(local_ids)
            if not docs:
                logger.warning(
                    "[Downloader] [%s] fetch_documents devolvió 0 docs para %d IDs.",
                    source_name, len(local_ids),
                )
                failed_sources.add(source_name)
            else:
                all_docs.extend(docs)
        return all_docs

    def _persist(
        self,
        docs: List[Document],
        already_saved: set,
        pending: List[str],
        failed_sources: set,
    ) -> None:
        """Guarda documentos en CSV + SQLite y marca IDs como descargados."""
        cfg    = self._config
        saved  = skipped = 0

        for doc in docs:
            if doc.doc_id not in already_saved:
                doc.save(cfg.documents_csv)
                already_saved.add(doc.doc_id)
                saved += 1
            else:
                skipped += 1

            self._repo.upsert_document(
                arxiv_id   = doc.doc_id,
                title      = doc.title,
                authors    = doc.authors,
                abstract   = doc.abstract,
                categories = doc.categories,
                published  = doc.published,
                updated    = doc.updated,
                pdf_url    = doc.pdf_url,
                fetched_at = doc.fetched_at,
                db_path    = self._db_path,
            )

        successful_ids = [
            p for p in pending
            if not any(p.startswith(f"{s}:") for s in failed_sources)
        ]
        if successful_ids:
            self._id_store.mark_downloaded(successful_ids)
            logger.info(
                "[Downloader] Guardados %d, omitidos %d duplicados. %s",
                saved, skipped, self._id_store,
            )
