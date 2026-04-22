"""
crawler.py
==========
Orquesta tres tareas concurrentes por cada fuente de datos (cliente):

  1. Discovery  — busca nuevos IDs en cada fuente registrada.
  2. Downloader — descarga metadatos de artículos pendientes.
  3. Text       — descarga el texto completo y genera chunks.

Diseño multi-cliente
--------------------
El Crawler acepta una lista de objetos BaseClient.  Cada cliente decide
cómo obtener sus datos (API, scraping, base de datos externa, etc.).
Una vez que el cliente devuelve el texto limpio, el Crawler aplica el
algoritmo de chunking centralizado (chunker.make_chunks) antes de
persistir en SQLite, manteniendo esa lógica separada del transporte.

Formato de ID compuesto
-----------------------
Todos los IDs almacenados en IdStore, Document y SQLite siguen el formato:

    {source_name}:{local_id}

Ejemplos:
    arxiv:2301.12345
    semantic_scholar:abc123

Esto permite que el resto de módulos (indexing, embedding, retrieval)
trabajen con los mismos identificadores sin necesitar cambios.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .clients.base_client import BaseClient
from .clients.arxiv_client import ArxivClient
from .chunker import make_chunks
from .document import Document, DOCUMENTS_CSV
from .id_store import IdStore, IDS_CSV

logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    ids_per_discovery:  int   = 100
    batch_size:         int   = 10
    pdf_batch_size:     int   = 10    # documentos por ciclo de texto
    discovery_interval: float = 120.0
    download_interval:  float = 30.0
    pdf_interval:       float = 2.0   # pausa entre documentos
    chunk_size:         int   = 1000
    overlap_sentences:  int   = 2     # oraciones de solapamiento entre chunks
    ids_csv:            Path  = field(default_factory=lambda: IDS_CSV)
    documents_csv:      Path  = field(default_factory=lambda: DOCUMENTS_CSV)
    discovery_start:    int   = 0


class Crawler:
    """
    Tres hilos daemon comparten el ciclo de vida del Crawler:

    _discovery_loop — recorre cada cliente buscando IDs nuevos.
    _download_loop  — descarga metadatos de IDs pendientes (todos los clientes).
    _text_loop      — descarga texto completo y genera chunks (todos los clientes).

    Los IDs compuestos permiten al Crawler enrutar cada operación al cliente
    correcto sin que ningún otro módulo necesite conocer el concepto de "fuente".
    """

    def __init__(
        self,
        config:  Optional[CrawlerConfig]    = None,
        clients: Optional[List[BaseClient]] = None,
        # Compatibilidad hacia atrás: acepta un único client= (sin 's')
        client:  Optional[BaseClient]       = None,
    ) -> None:
        self.config = config or CrawlerConfig()

        # Resolver lista de clientes
        if clients:
            self._clients: List[BaseClient] = list(clients)
        elif client:
            self._clients = [client]
        else:
            self._clients = [ArxivClient()]

        # Índice rápido: source_name -> client
        self._client_map: Dict[str, BaseClient] = {
            c.source_name: c for c in self._clients
        }

        self.id_store = IdStore(self.config.ids_csv)

        # Ajustar offset de discovery si ya tenemos IDs
        if self.config.discovery_start == 0 and self.id_store.total > 0:
            self.config.discovery_start = self.id_store.total
            logger.info(
                "[Discovery] Offset inicial ajustado a %d (IDs ya conocidos).",
                self.config.discovery_start,
            )

        # Importación lazy para no forzar SQLite en tests
        from ..database.schema import init_db, DB_PATH
        from ..database import crawler_repository as repo
        from ..database.chunk_repository import save_chunks, get_chunks
        init_db(DB_PATH)
        self._repo        = repo
        self._save_chunks = save_chunks
        self._get_chunks  = get_chunks
        self._db_path     = DB_PATH

        self._stop        = threading.Event()
        self._t_discovery = threading.Thread(
            target=self._discovery_loop, name="discovery", daemon=True
        )
        self._t_download  = threading.Thread(
            target=self._download_loop, name="downloader", daemon=True
        )
        self._t_text      = threading.Thread(
            target=self._text_loop, name="text", daemon=True
        )

    # -- Control --------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._t_discovery.start()
        self._t_download.start()
        self._t_text.start()
        sources = [c.source_name for c in self._clients]
        logger.info("[Crawler] Iniciado. Fuentes: %s. Config: %s", sources, self.config)

    def stop(self) -> None:
        logger.info("[Crawler] Deteniendo ...")
        self._stop.set()
        for t in (self._t_discovery, self._t_download, self._t_text):
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

    # -- Helpers de enrutamiento ----------------------------------------------

    def _client_for(self, doc_id: str) -> Optional[BaseClient]:
        """Devuelve el cliente correspondiente al ID compuesto, o None."""
        try:
            source, _ = BaseClient.parse_doc_id(doc_id)
            return self._client_map.get(source)
        except ValueError:
            logger.warning("[Crawler] ID con formato invalido: %r", doc_id)
            return None

    def _local_id(self, doc_id: str) -> str:
        """Extrae la parte local del ID compuesto."""
        _, local = BaseClient.parse_doc_id(doc_id)
        return local

    # -- Hilo 1: descubrimiento de IDs ----------------------------------------

    def _discovery_loop(self) -> None:
        cfg = self.config
        while not self._stop.is_set():
            try:
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
                    if local_ids:
                        doc_ids = [client.make_doc_id(lid) for lid in local_ids]
                        added   = self.id_store.add_ids(doc_ids)
                        logger.info(
                            "[Discovery] [%s] %d IDs encontrados -> %d nuevos. %s",
                            client.source_name, len(doc_ids), added, self.id_store,
                        )
                        cfg.discovery_start += len(local_ids)
                        if added == 0:
                            logger.info(
                                "[Discovery] [%s] Pagina ya conocida (offset=%d) "
                                "— continuando hacia contenido mas antiguo.",
                                client.source_name, cfg.discovery_start,
                            )
            except Exception as exc:
                logger.error("[Discovery] Error: %s", exc, exc_info=True)
            self._stop.wait(cfg.discovery_interval)

    # -- Hilo 2: descarga de metadatos ----------------------------------------

    def _download_loop(self) -> None:
        cfg = self.config
        logger.info("[Downloader] Esperando 10s para IDs iniciales ...")
        self._stop.wait(10)

        while not self._stop.is_set():
            pending = self.id_store.get_pending_batch(cfg.batch_size)
            if not pending:
                logger.info(
                    "[Downloader] Sin IDs pendientes. Durmiendo %ds ...",
                    int(cfg.download_interval),
                )
                self._stop.wait(cfg.download_interval)
                continue

            logger.info("[Downloader] Descargando metadatos de %d articulos ...", len(pending))
            try:
                already_saved = Document.load_ids(cfg.documents_csv)
                saved = skipped = 0

                # Agrupar IDs pendientes por fuente
                by_source: Dict[str, List[str]] = {}
                failed_sources = set()
                for doc_id in pending:
                    client = self._client_for(doc_id)
                    if client is None:
                        logger.warning("[Downloader] Sin cliente para %r — omitiendo.", doc_id)
                        continue
                    by_source.setdefault(client.source_name, []).append(
                        self._local_id(doc_id)
                    )

                all_docs: List[Document] = []
                for source_name, local_ids in by_source.items():
                    client = self._client_map[source_name]
                    docs   = client.fetch_documents(local_ids)
                    if not docs:
                        logger.warning(
                            "[Downloader] [%s] fetch_documents devolvio 0 documentos "
                            "para %d IDs. Reintentando en el proximo ciclo.",
                            source_name, len(local_ids),
                        )
                        failed_sources.add(source_name)
                    else:
                        all_docs.extend(docs)

                # Solo marcar como descargados los IDs de fuentes que respondieron
                successful_ids = [
                    p for p in pending
                    if not any(p.startswith(f"{s}:") for s in failed_sources)
                ]

                for doc in all_docs:
                    if doc.doc_id not in already_saved:
                        doc.save(cfg.documents_csv)
                        already_saved.add(doc.doc_id)
                        saved += 1
                    else:
                        skipped += 1

                    self._repo.upsert_document(
                        arxiv_id   = doc.doc_id,   # parámetro del repo (no cambia)
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

                if successful_ids:
                    self.id_store.mark_downloaded(successful_ids)
                    logger.info(
                        "[Downloader] Guardados %d, omitidos %d duplicados. %s",
                        saved, skipped, self.id_store,
                    )

            except Exception as exc:
                logger.error("[Downloader] Error en batch: %s", exc, exc_info=True)

            self._stop.wait(cfg.download_interval)

    # -- Hilo 3: descarga de texto y generacion de chunks ---------------------

    def _text_loop(self) -> None:
        """
        Descarga el texto completo de cada documento pendiente y lo fragmenta.

        El flujo tiene tres pasos explicitamente separados:
          1. client.download_text()  -> texto limpio (logica del cliente)
          2. chunker.make_chunks()   -> fragmentos (algoritmo centralizado)
          3. _repo / _save_chunks()  -> persistencia en SQLite
        """
        cfg = self.config
        logger.info("[Text] Hilo arrancado. Esperando 20s para que haya metadatos ...")
        self._stop.wait(20)
        logger.info("[Text] Comenzando ciclo de descarga.")

        while not self._stop.is_set():
            try:
                stats = self._repo.get_stats(db_path=self._db_path)
                logger.info(
                    "[Text] Estado DB -> total=%d | con_texto=%d | "
                    "pendientes=%d | errores=%d",
                    stats["total_documents"], stats["pdf_indexed"],
                    stats["pdf_pending"],     stats["pdf_errors"],
                )

                pending_ids = self._repo.get_pending_pdf_ids(
                    cfg.pdf_batch_size, db_path=self._db_path
                )

                if not pending_ids:
                    logger.info(
                        "[Text] No hay documentos pendientes. Esperando %ds ...",
                        int(cfg.pdf_interval),
                    )
                    self._stop.wait(cfg.pdf_interval)
                    continue

                logger.info("[Text] %d documentos en cola: %s", len(pending_ids), pending_ids)

                for i, doc_id in enumerate(pending_ids, 1):
                    if self._stop.is_set():
                        break

                    client = self._client_for(doc_id)
                    if client is None:
                        logger.error("[Text] Sin cliente para %r — marcando error.", doc_id)
                        self._repo.save_pdf_error(
                            doc_id,
                            "No hay cliente registrado para esta fuente.",
                            db_path=self._db_path,
                        )
                        continue

                    local_id = self._local_id(doc_id)
                    doc_row  = self._repo.get_document(doc_id, db_path=self._db_path)
                    pdf_url  = doc_row["pdf_url"] if doc_row else None

                    logger.info(
                        "[Text] [%d/%d] [%s] Descargando texto de %s ...",
                        i, len(pending_ids), client.source_name, doc_id,
                    )

                    try:
                        # Paso 1: descarga del texto (responsabilidad del cliente)
                        full_text = client.download_text(local_id, pdf_url=pdf_url)
                        logger.info(
                            "[Text] [%d/%d] Texto obtenido — %d chars.",
                            i, len(pending_ids), len(full_text),
                        )

                        # Paso 2: chunking con el algoritmo centralizado
                        chunks = make_chunks(
                            full_text,
                            chunk_size=cfg.chunk_size,
                            overlap_sentences=cfg.overlap_sentences,
                        )
                        logger.info(
                            "[Text] [%d/%d] Chunking completado — %d chunks.",
                            i, len(pending_ids), len(chunks),
                        )

                        # Paso 3: persistencia en SQLite
                        self._repo.save_pdf_text(doc_id, full_text, db_path=self._db_path)
                        self._save_chunks(doc_id, chunks, db_path=self._db_path)

                        stats = self._repo.get_stats(db_path=self._db_path)
                        logger.info(
                            "[Text] OK GUARDADO %s — %d chars | %d chunks | "
                            "DB: %d/%d documentos con texto",
                            doc_id, len(full_text), len(chunks),
                            stats["pdf_indexed"], stats["total_documents"],
                        )

                    except Exception as exc:
                        logger.error(
                            "[Text] FALLO %s — %s — marcado como error.",
                            doc_id, exc,
                        )
                        self._repo.save_pdf_error(doc_id, str(exc), db_path=self._db_path)

                    logger.info(
                        "[Text] Pausa de %ds antes del siguiente documento ...",
                        int(cfg.pdf_interval),
                    )
                    self._stop.wait(cfg.pdf_interval)

            except Exception as exc:
                logger.error("[Text] Error inesperado en el ciclo: %s", exc, exc_info=True)
                self._stop.wait(cfg.pdf_interval)
