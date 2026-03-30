"""
crawler.py
Orquesta tres tareas concurrentes:
  1. Discovery  — busca nuevos IDs en arXiv.
  2. Downloader — descarga metadatos de artículos pendientes.
  3. PDF        — descarga y extrae texto de PDFs pendientes en SQLite.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .arxiv_client import ArxivClient
from .document import Document, DOCUMENTS_CSV
from .id_store import IdStore, IDS_CSV
from .pdf_extractor import download_and_extract

logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    ids_per_discovery:  int   = 100
    batch_size:         int   = 10
    pdf_batch_size:     int   = 10   # PDFs por ciclo
    discovery_interval: float = 120.0
    download_interval:  float = 30.0
    pdf_interval:       float = 2.0  # pausa ENTRE PDFs — el rate limit real lo gestiona pdf_extractor
    chunk_size:         int   = 1000
    ids_csv:            Path  = field(default_factory=lambda: IDS_CSV)
    documents_csv:      Path  = field(default_factory=lambda: DOCUMENTS_CSV)
    discovery_start:    int   = 0


class Crawler:
    """
    Tres hilos daemon:
      _discovery_loop  — encuentra IDs nuevos en arXiv.
      _download_loop   — descarga metadatos y los guarda en CSV + SQLite.
      _pdf_loop        — descarga PDFs pendientes y guarda texto en SQLite.
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        client: Optional[ArxivClient]   = None,
    ) -> None:
        self.config   = config or CrawlerConfig()
        self.client   = client or ArxivClient()
        self.id_store = IdStore(self.config.ids_csv)

        # Arranca el offset desde el total de IDs ya conocidos para no
        # re-escanear páginas que ya tenemos. Los papers nuevos siempre
        # aparecen al principio (sortBy=submittedDate desc), así que tras
        # un reset a 0 los encontrará enseguida.
        if self.config.discovery_start == 0 and self.id_store.total > 0:
            self.config.discovery_start = self.id_store.total
            import logging as _l
            _l.getLogger(__name__).info(
                "[Discovery] Offset inicial ajustado a %d (IDs ya conocidos).",
                self.config.discovery_start,
            )

        # Importación lazy para no forzar SQLite en tests que no lo usan
        from ..database.schema import init_db, DB_PATH
        from ..database import crawler_repository as repo
        init_db(DB_PATH)
        self._repo    = repo
        self._db_path = DB_PATH

        self._stop = threading.Event()
        self._t_discovery = threading.Thread(target=self._discovery_loop, name="discovery", daemon=True)
        self._t_download  = threading.Thread(target=self._download_loop,  name="downloader", daemon=True)
        self._t_pdf       = threading.Thread(target=self._pdf_loop,       name="pdf",        daemon=True)

    # ── Control ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._stop.clear()
        self._t_discovery.start()
        self._t_download.start()
        self._t_pdf.start()
        logger.info("[Crawler] Iniciado. Config: %s", self.config)

    def stop(self) -> None:
        logger.info("[Crawler] Deteniendo …")
        self._stop.set()
        for t in (self._t_discovery, self._t_download, self._t_pdf):
            t.join(timeout=15)
        logger.info("[Crawler] Detenido.")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Crawler] KeyboardInterrupt recibido.")
        finally:
            self.stop()

    # ── Hilo 1: descubrimiento de IDs ────────────────────────────────────────
    def _discovery_loop(self) -> None:
        cfg = self.config
        while not self._stop.is_set():
            try:
                logger.info("[Discovery] Buscando %d IDs (offset=%d) …",
                            cfg.ids_per_discovery, cfg.discovery_start)
                ids = self.client.fetch_ids(
                    max_results=cfg.ids_per_discovery,
                    start=cfg.discovery_start,
                )
                if ids:
                    added = self.id_store.add_ids(ids)
                    logger.info("[Discovery] %d IDs encontrados → %d nuevos. %s",
                                len(ids), added, self.id_store)
                    # Siempre avanzar el offset, tanto si encontramos IDs
                    # nuevos como si la página ya era conocida. Resetear
                    # a 0 aquí significaría re-escanear páginas que ya
                    # tenemos en lugar de continuar explorando el índice.
                    cfg.discovery_start += len(ids)
                    if added == 0:
                        logger.info(
                            "[Discovery] Página ya conocida (offset=%d) — "
                            "continuando hacia papers más antiguos.",
                            cfg.discovery_start,
                        )
            except Exception as exc:
                logger.error("[Discovery] Error: %s", exc, exc_info=True)
            self._stop.wait(cfg.discovery_interval)

    # ── Hilo 2: descarga de metadatos ────────────────────────────────────────
    def _download_loop(self) -> None:
        cfg = self.config
        logger.info("[Downloader] Esperando 10s para IDs iniciales …")
        self._stop.wait(10)

        while not self._stop.is_set():
            pending = self.id_store.get_pending_batch(cfg.batch_size)
            if not pending:
                logger.info("[Downloader] Sin IDs pendientes. Durmiendo %ds …",
                            int(cfg.download_interval))
                self._stop.wait(cfg.download_interval)
                continue

            logger.info("[Downloader] Descargando metadatos de %d artículos …", len(pending))
            try:
                already_saved = Document.load_ids(cfg.documents_csv)
                documents     = self.client.fetch_documents(pending)
                saved = skipped = 0

                if not documents:
                    # fetch_documents devolvió [] — error de red silenciado
                    # en _fetch_chunk. No marcamos como descargados para
                    # que se reintenten en el siguiente ciclo.
                    logger.warning(
                        "[Downloader] fetch_documents devolvió 0 documentos "
                        "para %d IDs — posible error de red. Reintentando en el "
                        "próximo ciclo.", len(pending)
                    )
                else:
                    for doc in documents:
                        # 1. CSV
                        if doc.arxiv_id not in already_saved:
                            doc.save(cfg.documents_csv)
                            already_saved.add(doc.arxiv_id)
                            saved += 1
                        else:
                            skipped += 1

                        # 2. SQLite — upsert siempre (idempotente)
                        self._repo.upsert_document(
                            arxiv_id   = doc.arxiv_id,
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

                    # Solo marcar como descargados si realmente se procesaron
                    self.id_store.mark_downloaded(pending)
                    logger.info("[Downloader] Guardados %d, omitidos %d duplicados. %s",
                                saved, skipped, self.id_store)
            except Exception as exc:
                logger.error("[Downloader] Error en batch: %s", exc, exc_info=True)

            self._stop.wait(cfg.download_interval)

    # ── Hilo 3: descarga de PDFs ─────────────────────────────────────────────
    def _pdf_loop(self) -> None:
        cfg = self.config
        logger.info("[PDF] Hilo arrancado. Esperando 20s para que haya metadatos …")
        self._stop.wait(20)
        logger.info("[PDF] Comenzando ciclo de descarga.")

        while not self._stop.is_set():
            try:
                # ── Consulta cuántos PDFs faltan ────────────────────────────
                stats = self._repo.get_stats(db_path=self._db_path)
                logger.info(
                    "[PDF] Estado actual DB → total=%d | con_pdf=%d | "
                    "pendientes=%d | errores=%d",
                    stats["total_documents"], stats["pdf_indexed"],
                    stats["pdf_pending"],     stats["pdf_errors"],
                )

                pending_ids = self._repo.get_pending_pdf_ids(
                    cfg.pdf_batch_size, db_path=self._db_path
                )

                if not pending_ids:
                    logger.info(
                        "[PDF] No hay PDFs pendientes ahora mismo. "
                        "Esperando %ds para el próximo ciclo …", int(cfg.pdf_interval)
                    )
                    self._stop.wait(cfg.pdf_interval)
                    continue

                logger.info(
                    "[PDF] %d PDFs en cola para este ciclo: %s",
                    len(pending_ids), pending_ids,
                )

                for i, arxiv_id in enumerate(pending_ids, 1):
                    if self._stop.is_set():
                        break

                    doc_row = self._repo.get_document(arxiv_id, db_path=self._db_path)
                    pdf_url = doc_row["pdf_url"] if doc_row else None

                    logger.info(
                        "[PDF] [%d/%d] Iniciando descarga de %s — url: %s",
                        i, len(pending_ids), arxiv_id, pdf_url,
                    )

                    try:
                        # Paso 1: descarga y extracción
                        logger.info("[PDF] [%d/%d] Descargando PDF …", i, len(pending_ids))
                        full_text, chunks = download_and_extract(
                            arxiv_id, pdf_url=pdf_url, chunk_size=cfg.chunk_size
                        )
                        logger.info(
                            "[PDF] [%d/%d] PDF descargado y parseado — "
                            "%d chars, %d chunks.",
                            i, len(pending_ids), len(full_text), len(chunks),
                        )

                        # Paso 2: guardar texto en SQLite
                        logger.info("[PDF] [%d/%d] Guardando texto en SQLite …", i, len(pending_ids))
                        self._repo.save_pdf_text(arxiv_id, full_text, db_path=self._db_path)

                        # Paso 3: guardar chunks en SQLite
                        logger.info("[PDF] [%d/%d] Guardando %d chunks en SQLite …", i, len(pending_ids), len(chunks))
                        self._repo.save_chunks(arxiv_id, chunks, db_path=self._db_path)

                        # Confirmación final
                        stats = self._repo.get_stats(db_path=self._db_path)
                        logger.info(
                            "[PDF] ✅ GUARDADO %s — %d chars | %d chunks | "
                            "DB: %d/%d documentos con PDF",
                            arxiv_id, len(full_text), len(chunks),
                            stats["pdf_indexed"], stats["total_documents"],
                        )

                    except Exception as exc:
                        logger.error(
                            "[PDF] ❌ FALLÓ %s — %s — "
                            "marcado como error en DB.",
                            arxiv_id, exc,
                        )
                        self._repo.save_pdf_error(arxiv_id, str(exc), db_path=self._db_path)

                    # Pausa cortés entre PDFs
                    logger.info("[PDF] Pausa de %ds antes del siguiente PDF …", int(cfg.pdf_interval))
                    self._stop.wait(cfg.pdf_interval)

            except Exception as exc:
                logger.error("[PDF] Error inesperado en el ciclo: %s", exc, exc_info=True)
                self._stop.wait(cfg.pdf_interval)