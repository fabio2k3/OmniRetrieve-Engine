"""
migrate_db.py
=============
Migra la base de datos SQLite de la version anterior (con columnas
embedding BLOB y embedded_at en chunks) a la version actual donde:
  - embedding BLOB ha sido eliminado (los vectores viven en ChromaDB).
  - embedded_at se conserva como timestamp de control (NULL = pendiente).

El script es IDEMPOTENTE: se puede ejecutar varias veces sin dano.
Detecta automaticamente desde que version parte la BD y solo aplica
los pasos que sean necesarios.

Casos que maneja
----------------
  A) BD muy antigua: chunks tiene embedding BLOB y embedded_at.
     -> Eliminar la columna embedding (SQLite requiere recrear la tabla).
     -> embedded_at se conserva tal cual.

  B) BD intermedia: chunks tiene embedding BLOB pero NO embedded_at.
     -> Recrear tabla sin BLOB, sin embedded_at (se annade en paso C).

  C) BD sin embedded_at (independientemente de si tenia BLOB o no).
     -> ALTER TABLE chunks ADD COLUMN embedded_at TEXT.
     -> Crear indice idx_chunks_embedded.

  D) BD ya migrada: tiene embedded_at, no tiene embedding BLOB.
     -> No hace nada.

Adicionalmente: si la BD vieja tenia chunks con embedding IS NOT NULL
(vectores guardados como BLOB), ofrece la opcion de marcar esos chunks
como ya embebidos (embedded_at = timestamp de migracion) para que el
pipeline no los reprocese. Nota: los vectores del BLOB se PIERDEN —
ChromaDB los generara de nuevo si se resetea embedded_at.

Uso
---
    python -m backend.tools.migrate_db
    python -m backend.tools.migrate_db --db ruta/a/documents.db
    python -m backend.tools.migrate_db --db ruta/a/documents.db --dry-run
    python -m backend.tools.migrate_db --db ruta/a/documents.db --mark-blobs-as-embedded
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.database.schema import DB_PATH

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Inspeccion de la BD
# ---------------------------------------------------------------------------

def _get_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Devuelve {nombre_col: tipo} de una tabla."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1]: r[2] for r in rows}


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in _get_columns(conn, table)


def _has_index(conn: sqlite3.Connection, index_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Pasos de migracion
# ---------------------------------------------------------------------------

def _recreate_chunks_without_blob(conn: sqlite3.Connection) -> None:
    """
    Elimina la columna embedding BLOB recreando la tabla chunks.
    SQLite no soporta DROP COLUMN en versiones antiguas, asi que se usa
    el patron estandar: renombrar -> crear nueva -> copiar -> borrar vieja.

    Conserva: id, arxiv_id, chunk_index, text, char_count, embedded_at
    (si existe), created_at.
    Elimina: embedding BLOB.
    """
    cols = _get_columns(conn, "chunks")
    has_embedded_at = "embedded_at" in cols

    embedded_at_col  = ", embedded_at" if has_embedded_at else ""
    embedded_at_def  = "\n    embedded_at     TEXT," if has_embedded_at else ""

    log.info("  Renombrando chunks -> chunks_old...")
    conn.execute("ALTER TABLE chunks RENAME TO chunks_old")

    log.info("  Creando tabla chunks nueva (sin embedding BLOB)...")
    conn.executescript(f"""
        CREATE TABLE chunks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id        TEXT    NOT NULL
                                REFERENCES documents(arxiv_id) ON DELETE CASCADE,
            chunk_index     INTEGER NOT NULL,
            text            TEXT    NOT NULL,
            char_count      INTEGER,{embedded_at_def}
            created_at      TEXT    NOT NULL,
            UNIQUE(arxiv_id, chunk_index)
        );
    """)

    log.info("  Copiando datos (sin BLOB)...")
    conn.execute(f"""
        INSERT INTO chunks
            (id, arxiv_id, chunk_index, text, char_count{embedded_at_col}, created_at)
        SELECT
            id, arxiv_id, chunk_index, text, char_count{embedded_at_col}, created_at
        FROM chunks_old
    """)

    log.info("  Eliminando tabla chunks_old...")
    conn.execute("DROP TABLE chunks_old")


def _add_embedded_at(conn: sqlite3.Connection) -> None:
    """Anade la columna embedded_at TEXT a la tabla chunks."""
    log.info("  ALTER TABLE chunks ADD COLUMN embedded_at TEXT")
    conn.execute("ALTER TABLE chunks ADD COLUMN embedded_at TEXT")


def _add_embedded_index(conn: sqlite3.Connection) -> None:
    """Crea el indice sobre embedded_at si no existe."""
    log.info("  CREATE INDEX idx_chunks_embedded ON chunks(embedded_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded_at)"
    )


def _recreate_chunks_index(conn: sqlite3.Connection) -> None:
    """Recrea el indice principal de chunks tras la recreacion de tabla."""
    log.info("  Recreando indice idx_chunks_arxiv...")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_arxiv ON chunks(arxiv_id)"
    )


def _mark_existing_blobs_as_embedded(
    conn: sqlite3.Connection,
    timestamp: str,
) -> int:
    """
    Marca como embebidos (embedded_at = timestamp) los chunks que tenian
    un BLOB en la BD vieja. Solo tiene sentido si se quiere asumir que
    esos chunks ya estan en ChromaDB (migracion con datos existentes).
    Devuelve el numero de filas actualizadas.
    """
    cur = conn.execute(
        "UPDATE chunks SET embedded_at = ? WHERE embedded_at IS NULL",
        (timestamp,),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Logica principal
# ---------------------------------------------------------------------------

def migrate(
    db_path:                 Path,
    dry_run:                 bool = False,
    mark_blobs_as_embedded:  bool = False,
) -> None:
    """
    Ejecuta la migracion de la BD.

    Parametros
    ----------
    db_path               : ruta al fichero SQLite.
    dry_run               : si True, solo inspecciona y describe los pasos
                            sin modificar nada.
    mark_blobs_as_embedded: si True, tras migrar marca todos los chunks
                            como ya embebidos (asume que ChromaDB ya tiene
                            sus vectores). Si False, los deja con
                            embedded_at=NULL para que el pipeline los
                            procese de nuevo.
    """
    sep = "─" * 58

    if not db_path.exists():
        log.error("BD no encontrada: %s", db_path)
        return

    log.info(sep)
    log.info("OmniRetrieve — Migracion de BD")
    log.info(sep)
    log.info("BD: %s", db_path)
    if dry_run:
        log.info("MODO DRY-RUN: no se aplicaran cambios")
    log.info(sep)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        cols = _get_columns(conn, "chunks")
        has_blob        = "embedding"    in cols
        has_embedded_at = "embedded_at"  in cols
        has_index       = _has_index(conn, "idx_chunks_embedded")

        # Estadisticas previas
        total    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        with_blob = (
            conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
            if has_blob else 0
        )
        with_ts   = (
            conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL").fetchone()[0]
            if has_embedded_at else 0
        )

        log.info("Estado actual de la tabla chunks:")
        log.info("  Total chunks       : %d", total)
        log.info("  Columna embedding  : %s", "SI (BLOB)" if has_blob else "no")
        log.info("  Con BLOB no nulo   : %d", with_blob)
        log.info("  Columna embedded_at: %s", "si" if has_embedded_at else "NO")
        log.info("  Con timestamp      : %d", with_ts)
        log.info("  Indice embedded_at : %s", "si" if has_index else "NO")
        log.info(sep)

        # Determinar si ya esta migrada
        if not has_blob and has_embedded_at and has_index:
            log.info("La BD ya esta en la version actual. No hay nada que hacer.")
            return

        # Describir pasos necesarios
        steps = []
        if has_blob:
            steps.append("Recrear tabla chunks sin columna embedding BLOB")
        if not has_embedded_at:
            steps.append("ALTER TABLE chunks ADD COLUMN embedded_at TEXT")
        if not has_index:
            steps.append("CREATE INDEX idx_chunks_embedded")
        if mark_blobs_as_embedded and with_blob > 0:
            steps.append(
                f"Marcar {with_blob} chunks (tenian BLOB) como ya embebidos"
            )

        log.info("Pasos a ejecutar:")
        for i, step in enumerate(steps, 1):
            log.info("  %d. %s", i, step)

        if dry_run:
            log.info(sep)
            log.info("DRY-RUN: fin. No se modifico nada.")
            return

        # Hacer backup antes de modificar
        backup = db_path.with_suffix(".db.bak")
        log.info(sep)
        log.info("Creando backup: %s", backup)
        shutil.copy2(db_path, backup)

        # Aplicar pasos
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")

        migration_ts = _now()

        if has_blob:
            log.info("Paso: recrear tabla sin BLOB...")
            _recreate_chunks_without_blob(conn)
            _recreate_chunks_index(conn)
            # Refrescar estado tras recreacion
            cols            = _get_columns(conn, "chunks")
            has_embedded_at = "embedded_at" in cols

        if not has_embedded_at:
            log.info("Paso: añadir columna embedded_at...")
            _add_embedded_at(conn)

        if not _has_index(conn, "idx_chunks_embedded"):
            log.info("Paso: crear indice...")
            _add_embedded_index(conn)

        if mark_blobs_as_embedded and with_blob > 0:
            log.info(
                "Paso: marcar %d chunks como embebidos (embedded_at = %s)...",
                with_blob, migration_ts,
            )
            updated = _mark_existing_blobs_as_embedded(conn, migration_ts)
            log.info("  %d filas actualizadas.", updated)

        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()

        # Verificacion final
        cols_after   = _get_columns(conn, "chunks")
        index_after  = _has_index(conn, "idx_chunks_embedded")
        total_after  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        pending      = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NULL"
        ).fetchone()[0]
        embedded     = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()[0]

        log.info(sep)
        log.info("Migracion completada. Estado final:")
        log.info("  Columnas de chunks : %s", list(cols_after.keys()))
        log.info("  Tiene embedding    : %s", "embedding" in cols_after)
        log.info("  Tiene embedded_at  : %s", "embedded_at" in cols_after)
        log.info("  Indice embedded_at : %s", index_after)
        log.info("  Total chunks       : %d", total_after)
        log.info("  Pendientes (NULL)  : %d", pending)
        log.info("  Embebidos          : %d", embedded)
        log.info("  Backup guardado en : %s", backup)
        log.info(sep)

    except Exception as exc:
        conn.rollback()
        log.error("ERROR durante la migracion: %s", exc, exc_info=True)
        log.error("Se ha hecho rollback. La BD no fue modificada.")
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Migracion de BD SQLite a version con ChromaDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo describe los pasos sin aplicarlos.",
    )
    parser.add_argument(
        "--mark-blobs-as-embedded", action="store_true",
        help=(
            "Marca los chunks que tenian BLOB como ya embebidos. "
            "Usa esto solo si ChromaDB ya tiene sus vectores. "
            "Si no lo usas, el pipeline los vectorizara de nuevo."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    migrate(
        db_path                = args.db,
        dry_run                = args.dry_run,
        mark_blobs_as_embedded = args.mark_blobs_as_embedded,
    )


if __name__ == "__main__":
    main()