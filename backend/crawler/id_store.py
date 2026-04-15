"""
id_store.py
===========
Gestión thread-safe del fichero ids_article.csv.

Responsabilidades
-----------------
* Persistir IDs compuestos recién descubiertos por cualquier cliente.
* Registrar cuáles han sido ya descargados (metadatos).
* Proporcionar el siguiente lote de IDs pendientes al descargador.

Formato de ID
-------------
Los IDs almacenados son IDs compuestos con formato:

    "{source}:{local_id}"   →   p.ej. "arxiv:2301.12345"

El CSV usa la columna ``doc_id``.  Al leer un fichero antiguo que
tenga la columna ``arxiv_id`` la migra automáticamente en memoria
(sin reescribir el disco) hasta el próximo flush.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IDS_CSV  = DATA_DIR / "ids_article.csv"

# Nombre actual de la columna de ID en el CSV
_ID_COL        = "doc_id"
# Nombre antiguo (para retrocompatibilidad de lectura)
_LEGACY_ID_COL = "arxiv_id"

ID_FIELDS = [_ID_COL, "discovered_at", "downloaded"]


class IdStore:
    """
    Almacén thread-safe de IDs compuestos respaldado en CSV.

    Estado interno:
        dict[doc_id -> {"discovered_at": str, "downloaded": bool}]

    Cada mutación se escribe inmediatamente en el CSV para garantizar
    durabilidad ante reinicios inesperados del proceso.
    """

    def __init__(self, csv_path: Path = IDS_CSV) -> None:
        self._path  = csv_path
        self._lock  = threading.Lock()
        self._store: dict[str, dict] = {}

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # -----------------------------------------------------------------------
    # I/O interno
    # -----------------------------------------------------------------------
    def _load(self) -> None:
        """Lee el CSV en memoria, aceptando cabeceras nuevas y antiguas."""
        if not self._path.exists():
            return
        with self._path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            # Determinar qué columna contiene el ID
            id_col = _ID_COL if _ID_COL in fieldnames else _LEGACY_ID_COL
            for row in reader:
                doc_id = row[id_col]
                self._store[doc_id] = {
                    "discovered_at": row["discovered_at"],
                    "downloaded":    row["downloaded"].lower() == "true",
                }

    def _flush(self) -> None:
        """Escribe el estado en memoria al CSV (debe llamarse bajo _lock)."""
        with self._path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ID_FIELDS)
            writer.writeheader()
            for doc_id, meta in self._store.items():
                writer.writerow({
                    _ID_COL:         doc_id,
                    "discovered_at": meta["discovered_at"],
                    "downloaded":    str(meta["downloaded"]),
                })

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------
    def add_ids(self, ids: List[str]) -> int:
        """
        Añade los IDs compuestos que no hayan sido vistos antes.

        Parámetros
        ----------
        ids : List[str]
            IDs compuestos, p.ej. ``["arxiv:2301.12345", "arxiv:2302.00001"]``.

        Returns
        -------
        int
            Número de IDs **nuevos** realmente añadidos.
        """
        now       = datetime.utcnow().isoformat()
        new_count = 0
        with self._lock:
            for doc_id in ids:
                if doc_id not in self._store:
                    self._store[doc_id] = {
                        "discovered_at": now,
                        "downloaded":    False,
                    }
                    new_count += 1
            if new_count:
                self._flush()
        return new_count

    def get_pending_batch(self, batch_size: int = 10) -> List[str]:
        """Devuelve hasta *batch_size* IDs pendientes de descarga de metadatos."""
        with self._lock:
            pending = [
                doc_id
                for doc_id, meta in self._store.items()
                if not meta["downloaded"]
            ]
        return pending[:batch_size]

    def mark_downloaded(self, ids: List[str]) -> None:
        """Marca los IDs dados como descargados y persiste."""
        with self._lock:
            for doc_id in ids:
                if doc_id in self._store:
                    self._store[doc_id]["downloaded"] = True
            self._flush()

    # -----------------------------------------------------------------------
    # Estadísticas
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

    def __repr__(self) -> str:
        return (
            f"IdStore(total={self.total}, "
            f"downloaded={self.downloaded_count}, "
            f"pending={self.pending_count})"
        )
