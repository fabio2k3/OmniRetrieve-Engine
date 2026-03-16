"""
id_store.py
Thread-safe management of ids_article.csv.

Responsibilities
----------------
* Persist newly discovered arXiv IDs.
* Track which IDs have already been downloaded.
* Provide the next batch of pending IDs for the downloader.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IDS_CSV = DATA_DIR / "ids_article.csv"

ID_FIELDS = ["arxiv_id", "discovered_at", "downloaded"]


class IdStore:
    """
    Thread-safe, file-backed store for arXiv article IDs.

    Internal state is a dict  arxiv_id -> {"discovered_at": str, "downloaded": bool}
    mirrored to *csv_path* on every mutation.
    """

    def __init__(self, csv_path: Path = IDS_CSV) -> None:
        self._path = csv_path
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # -----------------------------------------------------------------------
    # Internal I/O
    # -----------------------------------------------------------------------
    def _load(self) -> None:
        """Read the CSV file into the in-memory store."""
        if not self._path.exists():
            return
        with self._path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._store[row["arxiv_id"]] = {
                    "discovered_at": row["discovered_at"],
                    "downloaded": row["downloaded"].lower() == "true",
                }

    def _flush(self) -> None:
        """Write the in-memory store back to the CSV file (must hold _lock)."""
        with self._path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ID_FIELDS)
            writer.writeheader()
            for arxiv_id, meta in self._store.items():
                writer.writerow(
                    {
                        "arxiv_id": arxiv_id,
                        "discovered_at": meta["discovered_at"],
                        "downloaded": str(meta["downloaded"]),
                    }
                )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def add_ids(self, ids: List[str]) -> int:
        """
        Add *ids* that have not been seen before.

        Returns
        -------
        int
            Number of **new** IDs actually added.
        """
        now = datetime.utcnow().isoformat()
        new_count = 0
        with self._lock:
            for arxiv_id in ids:
                if arxiv_id not in self._store:
                    self._store[arxiv_id] = {
                        "discovered_at": now,
                        "downloaded": False,
                    }
                    new_count += 1
            if new_count:
                self._flush()
        return new_count

    def get_pending_batch(self, batch_size: int = 10) -> List[str]:
        """Return up to *batch_size* IDs that have not been downloaded yet."""
        with self._lock:
            pending = [
                arxiv_id
                for arxiv_id, meta in self._store.items()
                if not meta["downloaded"]
            ]
        return pending[:batch_size]

    def mark_downloaded(self, ids: List[str]) -> None:
        """Mark the given IDs as downloaded and persist."""
        with self._lock:
            for arxiv_id in ids:
                if arxiv_id in self._store:
                    self._store[arxiv_id]["downloaded"] = True
            self._flush()

    # -----------------------------------------------------------------------
    # Stats helpers
    # -----------------------------------------------------------------------
    @property
    def total(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for m in self._store.values() if not m["downloaded"])

    @property
    def downloaded_count(self) -> int:
        with self._lock:
            return sum(1 for m in self._store.values() if m["downloaded"])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"IdStore(total={self.total}, "
            f"downloaded={self.downloaded_count}, "
            f"pending={self.pending_count})"
        )