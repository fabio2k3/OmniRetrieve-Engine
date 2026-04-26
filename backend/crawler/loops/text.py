"""
loops/text.py
=============
Hilo de descarga de texto completo y generación de chunks.

Responsabilidad única
---------------------
Para cada documento pendiente ejecuta tres pasos explícitamente separados:

    1. ``client.download_text()``  → texto limpio (responsabilidad del cliente).
    2. ``chunker.make_chunks()``   → fragmentación (algoritmo centralizado).
    3. ``repo / save_chunks()``    → persistencia en SQLite.

Este módulo no sabe nada de descubrimiento de IDs ni de metadatos.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List

from ..config  import CrawlerConfig
from ..chunker import make_chunks
from ..clients.base_client import BaseClient
from .._routing import client_for, local_id as extract_local_id

logger = logging.getLogger(__name__)


class TextLoop:
    """
    Descarga texto completo, lo fragmenta y persiste los chunks en SQLite.

    Parámetros del constructor
    --------------------------
    config      : configuración compartida.
    client_map  : diccionario ``source_name → BaseClient``.
    repo        : módulo ``crawler_repository``.
    save_chunks : callable ``(doc_id, chunks, db_path) → None``.
    db_path     : ruta al fichero SQLite.
    stop        : evento de parada.
    """

    def __init__(
        self,
        config:      CrawlerConfig,
        client_map:  Dict[str, BaseClient],
        repo:        Any,
        save_chunks: Callable,
        db_path:     Any,
        stop:        threading.Event,
    ) -> None:
        self._config      = config
        self._client_map  = client_map
        self._repo        = repo
        self._save_chunks = save_chunks
        self._db_path     = db_path
        self._stop        = stop

    def run(self) -> None:
        """
        Bucle principal del hilo de texto.

        Espera 20 segundos al arranque para que el hilo de metadatos haya
        poblado SQLite. Luego itera en lotes de ``pdf_batch_size`` documentos,
        procesando uno a uno con una pausa de ``pdf_interval`` entre ellos.
        """
        cfg = self._config
        logger.info("[Text] Hilo arrancado. Esperando 20s para que haya metadatos ...")
        self._stop.wait(20)
        logger.info("[Text] Comenzando ciclo de descarga.")

        while not self._stop.is_set():
            try:
                self._log_db_stats()
                pending = self._repo.get_pending_pdf_ids(
                    cfg.pdf_batch_size, db_path=self._db_path
                )
                if not pending:
                    logger.info(
                        "[Text] Sin documentos pendientes. Esperando %ds ...",
                        int(cfg.pdf_interval),
                    )
                    self._stop.wait(cfg.pdf_interval)
                    continue

                logger.info("[Text] %d documentos en cola.", len(pending))
                for i, doc_id in enumerate(pending, 1):
                    if self._stop.is_set():
                        break
                    self._process_document(doc_id, i, len(pending))
                    self._stop.wait(cfg.pdf_interval)

            except Exception as exc:
                logger.error("[Text] Error inesperado: %s", exc, exc_info=True)
                self._stop.wait(cfg.pdf_interval)

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _log_db_stats(self) -> None:
        stats = self._repo.get_stats(db_path=self._db_path)
        logger.info(
            "[Text] DB → total=%d | con_texto=%d | pendientes=%d | errores=%d",
            stats["total_documents"], stats["pdf_indexed"],
            stats["pdf_pending"],     stats["pdf_errors"],
        )

    def _process_document(self, doc_id: str, i: int, total: int) -> None:
        """
        Descarga, fragmenta y persiste el texto de un único documento.

        Si el cliente no está registrado o la descarga falla, registra el
        error en la base de datos y continúa con el siguiente documento.
        """
        cfg    = self._config
        client = client_for(doc_id, self._client_map)

        if client is None:
            logger.error("[Text] Sin cliente para %r — marcando error.", doc_id)
            self._repo.save_pdf_error(
                doc_id,
                "No hay cliente registrado para esta fuente.",
                db_path=self._db_path,
            )
            return

        local   = extract_local_id(doc_id)
        doc_row = self._repo.get_document(doc_id, db_path=self._db_path)
        pdf_url = doc_row["pdf_url"] if doc_row else None

        logger.info(
            "[Text] [%d/%d] [%s] Descargando %s ...",
            i, total, client.source_name, doc_id,
        )

        try:
            # Paso 1 — texto limpio (responsabilidad del cliente)
            full_text = client.download_text(local, pdf_url=pdf_url)
            logger.info("[Text] [%d/%d] %d chars obtenidos.", i, total, len(full_text))

            # Paso 2 — chunking centralizado
            chunks = make_chunks(
                full_text,
                chunk_size=cfg.chunk_size,
                overlap_sentences=cfg.overlap_sentences,
            )
            logger.info("[Text] [%d/%d] %d chunks generados.", i, total, len(chunks))

            # Paso 3 — persistencia
            self._repo.save_pdf_text(doc_id, full_text, db_path=self._db_path)
            self._save_chunks(doc_id, chunks, db_path=self._db_path)

            stats = self._repo.get_stats(db_path=self._db_path)
            logger.info(
                "[Text] OK %s — %d chars | %d chunks | DB: %d/%d con texto",
                doc_id, len(full_text), len(chunks),
                stats["pdf_indexed"], stats["total_documents"],
            )

        except Exception as exc:
            logger.error("[Text] FALLO %s — %s", doc_id, exc)
            self._repo.save_pdf_error(doc_id, str(exc), db_path=self._db_path)
