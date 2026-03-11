"""
crawler.py
Orchestrates two concurrent tasks:
  1. ID discovery  – periodically queries arXiv for new article IDs.
  2. Downloading   – fetches full metadata for pending IDs in batches.

The ratio between discovery cycles and download cycles is controlled by
``discovery_interval`` and ``download_interval``.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CrawlerConfig:
    """All tunable parameters for the crawler."""

    # How many IDs to request from arXiv per discovery cycle
    ids_per_discovery: int = 100

    # How many articles to download per download cycle
    batch_size: int = 10

    # Seconds between ID-discovery cycles
    discovery_interval: float = 120.0

    # Seconds between download cycles
    download_interval: float = 30.0

    # Paths (defaults come from module-level constants)
    ids_csv: Path = field(default_factory=lambda: IDS_CSV)
    documents_csv: Path = field(default_factory=lambda: DOCUMENTS_CSV)

    # arXiv pagination offset – incremented each discovery cycle
    discovery_start: int = 0


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------
class Crawler:
    """
    Runs two daemon threads:
      * ``_discovery_loop`` – finds new IDs.
      * ``_download_loop``  – downloads pending documents.

    Both threads share an ``IdStore`` instance (thread-safe) and write to the
    same CSV files through their respective helpers.
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        client: Optional[ArxivClient] = None,
    ) -> None:
        self.config = config or CrawlerConfig()
        self.client = client or ArxivClient()
        self.id_store = IdStore(self.config.ids_csv)

        self._stop_event = threading.Event()
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop, name="discovery", daemon=True
        )
        self._download_thread = threading.Thread(
            target=self._download_loop, name="downloader", daemon=True
        )

    # -----------------------------------------------------------------------
    # Public control
    # -----------------------------------------------------------------------
    def start(self) -> None:
        """Start both background threads."""
        logger.info("Crawler starting — config: %s", self.config)
        self._stop_event.clear()
        self._discovery_thread.start()
        self._download_thread.start()
        logger.info("Crawler running. Press Ctrl+C to stop.")

    def stop(self) -> None:
        """Signal both threads to stop and wait for them to finish."""
        logger.info("Crawler stopping …")
        self._stop_event.set()
        self._discovery_thread.join(timeout=15)
        self._download_thread.join(timeout=15)
        logger.info("Crawler stopped.")

    def run_forever(self) -> None:
        """
        Convenience: start the crawler and block until a KeyboardInterrupt
        or until ``stop()`` is called from another thread.
        """
        self.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Crawler] KeyboardInterrupt received.")
        finally:
            self.stop()

    # -----------------------------------------------------------------------
    # Internal loops
    # -----------------------------------------------------------------------
    def _discovery_loop(self) -> None:
        """Periodically fetch new IDs from arXiv and add them to the store."""
        cfg = self.config

        while not self._stop_event.is_set():
            try:
                logger.info(
                    "[Discovery] Fetching up to %d IDs (offset=%d) …",
                    cfg.ids_per_discovery,
                    cfg.discovery_start,
                )
                ids = self.client.fetch_ids(
                    max_results=cfg.ids_per_discovery,
                    start=cfg.discovery_start,
                )
                if ids:
                    added = self.id_store.add_ids(ids)
                    logger.info(
                        "[Discovery] Found %d IDs → %d new. Store: %s",
                        len(ids),
                        added,
                        self.id_store,
                    )
                    # Advance pagination so the next cycle fetches different articles
                    cfg.discovery_start += cfg.ids_per_discovery
                else:
                    # Reset to the beginning when we've reached the end of results
                    logger.info("[Discovery] No IDs returned; resetting offset to 0.")
                    cfg.discovery_start = 0

            except Exception as exc:  # noqa: BLE001
                logger.error("[Discovery] Unexpected error: %s", exc, exc_info=True)

            # Wait for the next discovery cycle (interruptible)
            self._stop_event.wait(cfg.discovery_interval)

    def _download_loop(self) -> None:
        """Continuously download metadata for pending IDs in batches."""
        cfg = self.config

        # Small initial delay so the discovery loop can populate some IDs first
        logger.info("[Downloader] Waiting 10 s for initial IDs …")
        self._stop_event.wait(10)

        while not self._stop_event.is_set():
            pending = self.id_store.get_pending_batch(cfg.batch_size)

            if not pending:
                logger.info(
                    "[Downloader] No pending IDs (total=%d). Sleeping %ds …",
                    self.id_store.total,
                    int(cfg.download_interval),
                )
                self._stop_event.wait(cfg.download_interval)
                continue

            logger.info("[Downloader] Downloading batch of %d IDs …", len(pending))
            try:
                # Load IDs already in documents.csv to guard against duplicates
                # (e.g. crash between save() and mark_downloaded())
                already_saved = Document.load_ids(cfg.documents_csv)

                documents = self.client.fetch_documents(pending)
                saved = skipped = 0
                for doc in documents:
                    if doc.arxiv_id in already_saved:
                        skipped += 1
                        logger.debug("[Downloader] Skipping duplicate %s", doc.arxiv_id)
                    else:
                        doc.save(cfg.documents_csv)
                        already_saved.add(doc.arxiv_id)
                        saved += 1

                self.id_store.mark_downloaded(pending)
                logger.info(
                    "[Downloader] Saved %d, skipped %d duplicates. Store: %s",
                    saved,
                    skipped,
                    self.id_store,
                )

            except Exception as exc:  # noqa: BLE001
                logger.error("[Downloader] Batch error: %s", exc, exc_info=True)

            self._stop_event.wait(cfg.download_interval)