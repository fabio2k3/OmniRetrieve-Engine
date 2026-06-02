"""
purge_logs.py
=============
Elimina todas las tablas de log/auditoría de la base de datos,
conservando intactos los documentos y sus datos derivados.

Qué elimina (solo filas de log/auditoría)
------------------------------------------
BD:
  • crawl_log          — historial de ejecuciones del crawler
  • lsi_log            — historial de construcciones del modelo LSI
  • faiss_log          — historial de builds del índice FAISS
  • web_search_log     — auditoría de búsquedas web (nº resultados por query)
  • web_search_results — resultados web cacheados (contenido descargado)
  • index_meta         — metadatos clave/valor de la indexación TF
  • embedding_meta     — metadatos clave/valor del pipeline de embedding

Qué conserva
------------
  • documents          — arxiv_id, title, authors, abstract, full_text, ...
  • chunks             — fragmentos de texto y embeddings
  • terms              — vocabulario TF-IDF
  • postings           — índice invertido de frecuencias
  • Todos los archivos en disco (FAISS, LSI .pkl, PDFs)

Uso
---
    python -m backend.tools.purge_logs
    python -m backend.tools.purge_logs --yes
    python -m backend.tools.purge_logs --db ruta/otra.db
    python -m backend.tools.purge_logs --tables crawl_log lsi_log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection

# ── Colores ANSI ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

# ── Definición de tablas de log ───────────────────────────────────────────────
# Cada entrada: (nombre_tabla, descripción_corta, puede_no_existir)
LOG_TABLES: list[tuple[str, str, bool]] = [
    ("crawl_log",          "Historial de ejecuciones del crawler",        False),
    ("lsi_log",            "Historial de construcciones del modelo LSI",   False),
    ("faiss_log",          "Historial de builds del índice FAISS",         True),   # creada por embedding_repository
    ("web_search_log",     "Auditoría de búsquedas web",                   False),
    ("web_search_results", "Resultados web cacheados",                     False),
    ("index_meta",         "Metadatos clave/valor de la indexación TF",    False),
    ("embedding_meta",     "Metadatos clave/valor del pipeline embedding",  True),   # creada por embedding_repository
]

ALL_TABLE_NAMES = [t[0] for t in LOG_TABLES]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _fmt(n: int) -> str:
    return f"{n:,}"


# ── Lógica principal ──────────────────────────────────────────────────────────

def purge_logs(
    db_path:  Path,
    tables:   list[str],
    dry_run:  bool = False,
) -> dict[str, int]:
    """
    Vacía las tablas de log indicadas.

    Parámetros
    ----------
    db_path : ruta a la BD SQLite.
    tables  : nombres de tablas a vaciar (subconjunto de ALL_TABLE_NAMES).
    dry_run : si True, solo cuenta filas sin eliminar nada.

    Devuelve
    --------
    dict tabla → filas eliminadas (o contadas si dry_run).
    """
    conn = get_connection(db_path)
    stats: dict[str, int] = {}

    try:
        conn.execute("PRAGMA foreign_keys = OFF")

        for table in tables:
            if not _table_exists(conn, table):
                stats[table] = -1   # -1 = tabla no encontrada
                continue

            n = _count(conn, table)
            stats[table] = n

            if not dry_run and n > 0:
                conn.execute(f"DELETE FROM {table}")

        if not dry_run:
            conn.commit()
            # Recuperar espacio en disco
            conn.execute("VACUUM")
            conn.commit()

    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Purgar tablas de log de la base de datos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tablas disponibles:\n"
            + "\n".join(f"  {name:<25} {desc}" for name, desc, _ in LOG_TABLES)
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    parser.add_argument(
        "--tables", nargs="+", metavar="TABLA",
        choices=ALL_TABLE_NAMES,
        default=ALL_TABLE_NAMES,
        help="Tablas a vaciar. Por defecto: todas las de log.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo muestra cuántas filas se eliminarían, sin borrar nada.",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Confirmar automáticamente sin prompt interactivo.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Cabecera ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  OmniRetrieve — Purgar Logs de BD{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"  DB       : {CYAN}{args.db}{RESET}")
    print(f"  Tablas   : {', '.join(args.tables)}")
    if args.dry_run:
        print(f"  {YELLOW}{BOLD}Modo DRY-RUN — no se eliminará ningún dato{RESET}")
    print(f"{'='*60}\n")

    # ── Verificar que la BD existe ────────────────────────────────────────────
    if not args.db.exists():
        print(f"{RED}✗  Base de datos no encontrada: {args.db}{RESET}")
        sys.exit(1)

    # ── Preview de filas a eliminar ───────────────────────────────────────────
    preview = purge_logs(args.db, args.tables, dry_run=True)
    total_rows = sum(v for v in preview.values() if v >= 0)

    print(f"  {'Tabla':<26} {'Filas'}")
    print(f"  {'-'*26} {'-'*10}")
    for name, desc, _ in LOG_TABLES:
        if name not in args.tables:
            continue
        n = preview.get(name, -1)
        if n == -1:
            print(f"  {name:<26} {DIM}(tabla no existe){RESET}")
        elif n == 0:
            print(f"  {name:<26} {DIM}0{RESET}")
        else:
            print(f"  {name:<26} {YELLOW}{_fmt(n)}{RESET}")

    print(f"\n  {BOLD}Total a eliminar: {_fmt(total_rows)} filas{RESET}\n")

    if total_rows == 0:
        print(f"{GREEN}✓  Nada que eliminar — las tablas de log ya están vacías.{RESET}\n")
        sys.exit(0)

    # ── Si es dry-run, terminamos aquí ────────────────────────────────────────
    if args.dry_run:
        print(f"{YELLOW}Dry-run completado. Nada fue eliminado.{RESET}\n")
        sys.exit(0)

    # ── Confirmación ──────────────────────────────────────────────────────────
    if not args.yes:
        resp = input(
            f"  {BOLD}¿Eliminar {_fmt(total_rows)} filas de log? "
            f"Esta acción no se puede deshacer. [s/N]: {RESET}"
        ).strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print(f"\n{YELLOW}Operación cancelada.{RESET}\n")
            sys.exit(0)

    # ── Ejecución ─────────────────────────────────────────────────────────────
    print(f"\n  Eliminando filas…")
    t0 = time.perf_counter()

    stats = purge_logs(args.db, args.tables, dry_run=False)

    elapsed = time.perf_counter() - t0
    deleted  = sum(v for v in stats.values() if v > 0)

    print(f"\n{'='*60}")
    print(f"{GREEN}{BOLD}✓  Purga completada en {elapsed:.2f}s{RESET}")
    print(f"   Filas eliminadas : {BOLD}{_fmt(deleted)}{RESET}")
    print(f"   Espacio liberado : VACUUM ejecutado automáticamente")
    print(f"{'='*60}\n")

    # ── Detalle por tabla ─────────────────────────────────────────────────────
    for name, _, _ in LOG_TABLES:
        if name not in args.tables:
            continue
        n = stats.get(name, -1)
        if n == -1:
            print(f"  {DIM}⊘  {name} (tabla no existe){RESET}")
        elif n == 0:
            print(f"  {DIM}–  {name}: sin filas{RESET}")
        else:
            print(f"  {GREEN}✓{RESET}  {name}: {BOLD}{_fmt(n)}{RESET} filas eliminadas")

    print()


if __name__ == "__main__":
    main()
