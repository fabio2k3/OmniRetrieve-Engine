"""
document.py
===========
Define el dataclass Document que representa un artículo descargado
por cualquier cliente (arXiv, Semantic Scholar, etc.).

Campo principal: doc_id
-----------------------
El identificador único tiene formato compuesto:

    doc_id = "{source}:{local_id}"

Ejemplos:
    arxiv:2301.12345
    semantic_scholar:abc123def456

Compatibilidad retroactiva
--------------------------
El campo se expone también como ``arxiv_id`` (propiedad de solo lectura)
para que el código existente en otros módulos (database, indexing, tools)
no necesite ningún cambio:  ``doc.arxiv_id`` sigue funcionando igual.

El CSV se escribe con cabecera ``doc_id``.  Al leer un CSV antiguo con
cabecera ``arxiv_id`` (generado antes de este cambio) se traduce
automáticamente, por lo que no es necesario migrar los datos.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import ClassVar, List, Optional


# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
DOCUMENTS_CSV = DATA_DIR / "documents.csv"

DOCUMENT_FIELDS: List[str] = [
    "doc_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published",
    "updated",
    "pdf_url",
    "fetched_at",
]

# Nombre antiguo de la columna (para leer CSVs generados antes del cambio)
_LEGACY_ID_FIELD = "arxiv_id"


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------
@dataclass
class Document:
    """
    Representa un documento descargado por cualquier cliente.

    El campo ``doc_id`` contiene el identificador compuesto
    ``"{source}:{local_id}"``, p.ej. ``"arxiv:2301.12345"``.

    El atributo ``arxiv_id`` (propiedad de solo lectura) es un alias de
    ``doc_id`` que mantiene compatibilidad con código existente.
    """

    # --- campos principales -------------------------------------------------
    doc_id:     str
    title:      str
    authors:    str        # lista separada por comas
    abstract:   str
    categories: str        # lista separada por comas de etiquetas de categoría
    published:  str        # fecha ISO-8601
    updated:    str        # fecha ISO-8601
    pdf_url:    str
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # --- metadatos de clase -------------------------------------------------
    FIELDS: ClassVar[List[str]] = DOCUMENT_FIELDS

    # -----------------------------------------------------------------------
    # Compatibilidad retroactiva: arxiv_id -> doc_id
    # -----------------------------------------------------------------------
    @property
    def arxiv_id(self) -> str:
        """
        Alias de doc_id para compatibilidad con código existente.

        El valor ya NO es un ID de arXiv puro — puede ser cualquier ID
        compuesto de la forma ``"{source}:{local_id}"``.
        Este alias evita tener que modificar módulos externos (database,
        indexing, embedding, retrieval) que usan ``doc.arxiv_id``.
        """
        return self.doc_id

    # -----------------------------------------------------------------------
    # Serialización
    # -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Devuelve un diccionario con todos los campos (orden preservado)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """
        Reconstruye un Document desde un diccionario (p.ej. una fila CSV).

        Acepta tanto la clave nueva ``doc_id`` como la antigua ``arxiv_id``
        para poder cargar ficheros CSV generados antes del cambio de nombre.
        """
        # Traducción retroactiva: arxiv_id -> doc_id
        if "doc_id" not in data and _LEGACY_ID_FIELD in data:
            data = dict(data)
            data["doc_id"] = data.pop(_LEGACY_ID_FIELD)
        return cls(**{k: data[k] for k in cls.FIELDS})

    # -----------------------------------------------------------------------
    # Persistencia CSV
    # -----------------------------------------------------------------------
    def save(self, csv_path: Path = DOCUMENTS_CSV) -> None:
        """
        Añade este documento a *csv_path*.
        Crea el fichero con cabecera si no existe.
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(self.to_dict())

    @classmethod
    def load_all(cls, csv_path: Path = DOCUMENTS_CSV) -> List["Document"]:
        """Carga todos los documentos almacenados en *csv_path*."""
        if not csv_path.exists():
            return []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            return [cls.from_dict(row) for row in csv.DictReader(fh)]

    @classmethod
    def load_ids(cls, csv_path: Path = DOCUMENTS_CSV) -> set:
        """
        Devuelve el conjunto de doc_ids ya almacenados (lectura rápida).

        Lee la columna ``doc_id`` si está presente; si el CSV es antiguo
        y solo tiene ``arxiv_id``, usa esa columna en su lugar.
        """
        if not csv_path.exists():
            return set()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            id_col = "doc_id" if "doc_id" in (reader.fieldnames or []) else _LEGACY_ID_FIELD
            return {row[id_col] for row in reader}

    # -----------------------------------------------------------------------
    # Dunders
    # -----------------------------------------------------------------------
    def __str__(self) -> str:
        return f"Document({self.doc_id!r}, {self.title[:60]!r})"

    def __repr__(self) -> str:
        return (
            f"Document(doc_id={self.doc_id!r}, title={self.title[:40]!r}, "
            f"published={self.published!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.doc_id == other.doc_id

    def __hash__(self) -> int:
        return hash(self.doc_id)
