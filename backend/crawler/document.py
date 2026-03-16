"""
document.py
Defines the Document dataclass representing an arXiv article,
with serialization helpers and CSV persistence.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import ClassVar, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCUMENTS_CSV = DATA_DIR / "documents.csv"

DOCUMENT_FIELDS: List[str] = [
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published",
    "updated",
    "pdf_url",
    "fetched_at",
]


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """Represents a single arXiv article."""

    # --- core fields --------------------------------------------------------
    arxiv_id: str
    title: str
    authors: str                   # comma-separated list
    abstract: str
    categories: str                # comma-separated list of arXiv category tags
    published: str                 # ISO-8601 date string
    updated: str                   # ISO-8601 date string
    pdf_url: str
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # --- class-level metadata -----------------------------------------------
    FIELDS: ClassVar[List[str]] = DOCUMENT_FIELDS

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain dict (field-order preserved)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Reconstruct a Document from a dict (e.g. a CSV row)."""
        return cls(**{k: data[k] for k in cls.FIELDS})

    # -----------------------------------------------------------------------
    # CSV persistence
    # -----------------------------------------------------------------------
    def save(self, csv_path: Path = DOCUMENTS_CSV) -> None:
        """
        Append this document to *csv_path*.
        Creates the file with a header row if it does not yet exist.
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(self.to_dict())

    # -----------------------------------------------------------------------
    # Class-level CSV helpers
    # -----------------------------------------------------------------------
    @classmethod
    def load_all(cls, csv_path: Path = DOCUMENTS_CSV) -> List["Document"]:
        """Return every Document stored in *csv_path*."""
        if not csv_path.exists():
            return []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            return [cls.from_dict(row) for row in csv.DictReader(fh)]

    @classmethod
    def load_ids(cls, csv_path: Path = DOCUMENTS_CSV) -> set[str]:
        """Return the set of arXiv IDs already stored (fast path – no full parse)."""
        if not csv_path.exists():
            return set()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return {row["arxiv_id"] for row in reader}

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"Document({self.arxiv_id!r}, {self.title[:60]!r})"

    def __repr__(self) -> str:
        return (
            f"Document(arxiv_id={self.arxiv_id!r}, title={self.title[:40]!r}, "
            f"published={self.published!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.arxiv_id == other.arxiv_id

    def __hash__(self) -> int:
        return hash(self.arxiv_id)