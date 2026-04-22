"""
rename_ids.py
=============
Herramienta para renombrar IDs de documentos y chunks en la base de datos.

Caso de uso principal
---------------------
Migrar IDs del formato antiguo (``2301.12345``) al nuevo formato compuesto
(``arxiv:2301.12345``) tras el refactor multi-cliente del crawler.

Tablas afectadas
----------------
Cada renombrado actualiza de forma atómica y consistente:
  - ``documents.arxiv_id``  (clave primaria)
  - ``chunks.arxiv_id``     (clave foránea)
  - ``postings.doc_id``     (clave foránea)

Modos de operación
------------------
  --prefix SOURCE     Añade el prefijo "SOURCE:" a todos los IDs que no
                      lo tengan ya.  Filtrable con --filter.
                      Ej: --prefix arxiv  →  "2301.12345" → "arxiv:2301.12345"

  --rename OLD NEW    Renombra exactamente un ID.

  --mapping FILE      Aplica los renombrados de un CSV con columnas old_id,new_id.

Opciones de seguridad
---------------------
  Por defecto el tool opera en modo DRY-RUN: muestra el plan sin tocar nada.
  Para aplicar cambios hay que pasar --apply explícitamente.

  --backup            Crea una copia .bak de la DB antes de hacer cambios.
  --yes               Omite la confirmación interactiva (útil en scripts).

Uso
---
    # 1. Ver qué cambiaría (sin tocar nada)
    python -m backend.tools.rename_ids --prefix arxiv

    # 2. Aplicar migración con copia de seguridad
    python -m backend.tools.rename_ids --prefix arxiv --backup --apply

    # 3. Solo IDs que empiecen por "23" (glob)
    python -m backend.tools.rename_ids --prefix arxiv --filter "23*" --apply

    # 4. Renombrar un ID concreto
    python -m backend.tools.rename_ids --rename 2301.12345 arxiv:2301.12345 --apply

    # 5. Lote desde CSV (columnas: old_id, new_id)
    python -m backend.tools.rename_ids --mapping migration.csv --apply

    # 6. DB alternativa
    python -m backend.tools.rename_ids --prefix arxiv --db /ruta/otra.db --apply
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection

# ── Colores ANSI ──────────────────────────────────────────────────────────────
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
RESET   = "\033[0m"

W = 62


def _sep(char: str = "─", width: int = W) -> str:
    return DIM + char * width + RESET


def _header(title: str, icon: str = "") -> None:
    print(f"\n{_sep('═')}")
    print(f"{BOLD}  {icon}  {title}{RESET}")
    print(_sep("═"))


def _info(msg: str) -> None:
    print(f"  {CYAN}ℹ{RESET}  {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}", file=sys.stderr)


def _abort(msg: str) -> None:
    _err(msg)
    sys.exit(1)


# =============================================================================
# Construcción del plan de renombrado
# =============================================================================

def _all_ids(conn: sqlite3.Connection) -> List[str]:
    """Devuelve todos los arxiv_id presentes en documents."""
    rows = conn.execute("SELECT arxiv_id FROM documents ORDER BY arxiv_id").fetchall()
    return [r[0] for r in rows]


def build_plan_prefix(
    conn: sqlite3.Connection,
    source: str,
    glob_filter: Optional[str],
) -> Dict[str, str]:
    """
    Construye el mapa old→new para el modo --prefix.

    Solo incluye IDs que:
      - No tengan ya un prefijo (no contengan ':')
      - Coincidan con el glob_filter, si se especificó
    """
    plan: Dict[str, str] = {}
    for old_id in _all_ids(conn):
        if ":" in old_id:
            continue  # ya tiene prefijo — no tocar
        if glob_filter and not fnmatch.fnmatch(old_id, glob_filter):
            continue
        plan[old_id] = f"{source}:{old_id}"
    return plan


def build_plan_rename(old_id: str, new_id: str) -> Dict[str, str]:
    """Plan para el modo --rename (un solo ID)."""
    return {old_id: new_id}


def build_plan_mapping(csv_path: Path) -> Dict[str, str]:
    """
    Lee un CSV con columnas ``old_id`` y ``new_id`` y devuelve el plan.
    Acepta también la primera y segunda columna sin cabecera si los
    nombres no coinciden.
    """
    plan: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        # Determinar nombres de columna
        if "old_id" in fieldnames and "new_id" in fieldnames:
            old_col, new_col = "old_id", "new_id"
        elif len(fieldnames) >= 2:
            old_col, new_col = fieldnames[0], fieldnames[1]
            _warn(f"Columnas detectadas: {old_col!r} → {new_col!r}")
        else:
            _abort("El CSV debe tener al menos 2 columnas (old_id, new_id).")
            return {}
        for row in reader:
            old, new = row[old_col].strip(), row[new_col].strip()
            if old and new:
                plan[old] = new
    return plan


# =============================================================================
# Validación del plan
# =============================================================================

def validate_plan(
    conn: sqlite3.Connection,
    plan: Dict[str, str],
) -> List[str]:
    """
    Verifica el plan y devuelve una lista de errores (vacía = plan válido).

    Comprobaciones:
      - El old_id existe en documents.
      - El new_id no existe ya en documents (salvo que sea el mismo old_id).
      - No hay colisiones internas dentro del plan (dos IDs distintos → mismo destino).
    """
    errors: List[str] = []
    existing_ids = set(_all_ids(conn))
    seen_targets: Dict[str, str] = {}   # new_id → old_id que lo generó

    for old_id, new_id in plan.items():
        if old_id not in existing_ids:
            errors.append(f"ID origen no existe en DB: {old_id!r}")
        if old_id == new_id:
            errors.append(f"Origen y destino son idénticos: {old_id!r}")
        if new_id in existing_ids and new_id not in plan:
            errors.append(f"El destino {new_id!r} ya existe en la DB (para origen {old_id!r})")
        if new_id in seen_targets:
            errors.append(
                f"Colisión de destinos: {old_id!r} y {seen_targets[new_id]!r} "
                f"apuntan al mismo nuevo ID {new_id!r}"
            )
        seen_targets[new_id] = old_id

    return errors


# =============================================================================
# Presentación del plan
# =============================================================================

def _chunk_count(conn: sqlite3.Connection, arxiv_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    return row[0] if row else 0


def _posting_count(conn: sqlite3.Connection, arxiv_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM postings WHERE doc_id = ?", (arxiv_id,)
    ).fetchone()
    return row[0] if row else 0


def show_plan(conn: sqlite3.Connection, plan: Dict[str, str]) -> None:
    """Imprime una tabla con el resumen del plan de renombrado."""
    _header("Plan de renombrado", "📋")

    total_docs     = len(plan)
    total_chunks   = 0
    total_postings = 0

    # Cabecera de tabla
    print(f"\n  {BOLD}{'ID antiguo':<36}  {'ID nuevo':<36}{RESET}")
    print(f"  {DIM}{'─'*36}  {'─'*36}{RESET}")

    for i, (old_id, new_id) in enumerate(sorted(plan.items())):
        n_chunks   = _chunk_count(conn, old_id)
        n_postings = _posting_count(conn, old_id)
        total_chunks   += n_chunks
        total_postings += n_postings

        # Colorear la diferencia entre old y new
        prefix_part = new_id[: len(new_id) - len(old_id)]   # p.ej. "arxiv:"
        suffix_part = old_id

        old_disp = f"{DIM}{old_id[:34]}{RESET}"
        new_disp = f"{CYAN}{prefix_part}{RESET}{suffix_part[:34 - len(prefix_part)]}"

        extra = ""
        if n_chunks:
            extra += f"  {DIM}{n_chunks} chunks{RESET}"
        if n_postings:
            extra += f"  {DIM}{n_postings} postings{RESET}"

        print(f"  {old_disp:<48}→  {new_disp}{extra}")

        # No inundar la terminal si hay muchísimos
        if i == 29 and total_docs > 30:
            remaining = total_docs - 30
            print(f"  {DIM}… y {remaining} más{RESET}")
            break

    print(f"\n  {_sep()}")
    print(f"  {BOLD}Resumen:{RESET}")
    print(f"    Documentos   : {BOLD}{total_docs}{RESET}")
    print(f"    Chunks       : {BOLD}{total_chunks}{RESET}")
    print(f"    Postings     : {BOLD}{total_postings}{RESET}")
    print(f"    Tablas       : documents · chunks · postings")
    print()


# =============================================================================
# Ejecución del renombrado
# =============================================================================

def _rename_in_transaction(
    conn: sqlite3.Connection,
    plan: Dict[str, str],
) -> Tuple[int, int, int]:
    """
    Aplica todos los renombrados en una única transacción atómica.

    Devuelve (docs_updated, chunks_updated, postings_updated).

    Estrategia para claves foráneas
    --------------------------------
    SQLite no soporta ON UPDATE CASCADE por defecto.  En lugar de
    deshabilitar FK globalmente, insertamos temporalmente con el ID
    nuevo, reasignamos los hijos y borramos el registro antiguo.
    Esto respeta la integridad referencial en todo momento.
    """
    docs_updated = chunks_updated = postings_updated = 0

    # Desactivar FK temporalmente para poder hacer UPDATE directo en PKs.
    # Es seguro porque:
    #  1. Todo ocurre en una sola transacción EXCLUSIVE.
    #  2. Actualizamos documentos + hijos de forma coordinada.
    #  3. Al hacer COMMIT la integridad queda restaurada.
    conn.execute("PRAGMA foreign_keys = OFF")

    try:
        with conn:   # transacción — ROLLBACK automático si hay excepción
            for old_id, new_id in plan.items():
                # 1. Documento
                conn.execute(
                    "UPDATE documents SET arxiv_id = ? WHERE arxiv_id = ?",
                    (new_id, old_id),
                )
                docs_updated += conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]

                # 2. Chunks
                cur = conn.execute(
                    "UPDATE chunks SET arxiv_id = ? WHERE arxiv_id = ?",
                    (new_id, old_id),
                )
                chunks_updated += conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]

                # 3. Postings
                conn.execute(
                    "UPDATE postings SET doc_id = ? WHERE doc_id = ?",
                    (new_id, old_id),
                )
                postings_updated += conn.execute(
                    "SELECT changes()"
                ).fetchone()[0]

    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    return docs_updated, chunks_updated, postings_updated


def apply_plan(
    db_path: Path,
    plan: Dict[str, str],
    backup: bool,
    skip_confirm: bool,
) -> None:
    """Aplica el plan de renombrado a la base de datos."""

    # ── Backup opcional ───────────────────────────────────────────────────────
    if backup:
        bak = db_path.with_suffix(".db.bak")
        shutil.copy2(db_path, bak)
        _ok(f"Copia de seguridad creada: {bak}")

    # ── Confirmación interactiva ──────────────────────────────────────────────
    if not skip_confirm:
        print(f"\n  {YELLOW}¿Aplicar {len(plan)} renombrados en {db_path}?{RESET}")
        resp = input("  Escribe 'si' para confirmar: ").strip().lower()
        if resp not in ("si", "sí", "yes", "y"):
            print(f"\n  {DIM}Operación cancelada.{RESET}\n")
            sys.exit(0)

    # ── Ejecución ─────────────────────────────────────────────────────────────
    print()
    _info("Aplicando renombrados …")
    t0 = time.monotonic()

    conn = get_connection(db_path)
    try:
        docs_n, chunks_n, postings_n = _rename_in_transaction(conn, plan)
    except Exception as exc:
        _abort(f"Error durante el renombrado (ROLLBACK aplicado): {exc}")
    finally:
        conn.close()

    elapsed = time.monotonic() - t0

    # ── Resultado ─────────────────────────────────────────────────────────────
    _header("Resultado", "✅")
    _ok(f"documents actualizados : {docs_n}")
    _ok(f"chunks actualizados    : {chunks_n}")
    _ok(f"postings actualizados  : {postings_n}")
    _ok(f"Tiempo                 : {elapsed:.2f}s")
    print()

    if backup:
        _info(f"Para revertir: cp {db_path.with_suffix('.db.bak')} {db_path}")
    print()


# =============================================================================
# Verificación post-renombrado
# =============================================================================

def verify(db_path: Path, plan: Dict[str, str]) -> None:
    """Comprueba que todos los new_ids existen y los old_ids han desaparecido."""
    _header("Verificación", "🔍")
    conn = get_connection(db_path)
    errors = 0

    for old_id, new_id in sorted(plan.items()):
        exists_new = conn.execute(
            "SELECT 1 FROM documents WHERE arxiv_id = ?", (new_id,)
        ).fetchone()
        exists_old = conn.execute(
            "SELECT 1 FROM documents WHERE arxiv_id = ?", (old_id,)
        ).fetchone()

        if exists_new and not exists_old:
            _ok(f"{new_id}")
        elif not exists_new:
            _err(f"Nuevo ID no encontrado: {new_id!r}")
            errors += 1
        elif exists_old:
            _err(f"ID antiguo aún existe: {old_id!r}")
            errors += 1

    # Integridad referencial
    orphan_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks c "
        "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.arxiv_id = c.arxiv_id)"
    ).fetchone()[0]
    orphan_postings = conn.execute(
        "SELECT COUNT(*) FROM postings p "
        "WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.arxiv_id = p.doc_id)"
    ).fetchone()[0]

    conn.close()

    if orphan_chunks == 0 and orphan_postings == 0:
        _ok("Integridad referencial OK (sin chunks ni postings huérfanos)")
    else:
        if orphan_chunks:
            _err(f"{orphan_chunks} chunks huérfanos detectados")
            errors += 1
        if orphan_postings:
            _err(f"{orphan_postings} postings huérfanos detectados")
            errors += 1

    if errors:
        print(f"\n  {RED}{BOLD}{errors} errores de verificación.{RESET}\n")
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}Verificación completada sin errores.{RESET}\n")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.tools.rename_ids",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Modo ─────────────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prefix", metavar="SOURCE",
        help="Añade 'SOURCE:' a todos los IDs sin prefijo. Ej: --prefix arxiv",
    )
    mode.add_argument(
        "--rename", nargs=2, metavar=("OLD", "NEW"),
        help="Renombra un ID concreto. Ej: --rename 2301.12345 arxiv:2301.12345",
    )
    mode.add_argument(
        "--mapping", metavar="FILE", type=Path,
        help="Aplica renombrados desde un CSV con columnas old_id,new_id.",
    )

    # ── Filtrado ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--filter", metavar="GLOB", dest="glob_filter",
        help="(Solo con --prefix) Glob para seleccionar qué IDs procesar. "
             "Ej: --filter '23*' procesa solo IDs que empiecen por 23.",
    )

    # ── Ejecución ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--apply", action="store_true",
        help="Aplica los cambios en la DB.  Sin esta opción: modo DRY-RUN.",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Omite la confirmación interactiva (requiere --apply).",
    )
    p.add_argument(
        "--backup", action="store_true",
        help="Crea una copia .bak de la DB antes de aplicar cambios.",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Verifica la integridad de los IDs renombrados tras aplicar.",
    )
    p.add_argument(
        "--db", metavar="PATH", type=Path, default=DB_PATH,
        help=f"Ruta a la base de datos SQLite (por defecto: {DB_PATH})",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    db_path: Path = args.db

    # ── Validación de argumentos ──────────────────────────────────────────────
    if not db_path.exists():
        _abort(f"Base de datos no encontrada: {db_path}")

    if args.glob_filter and not args.prefix:
        _abort("--filter solo tiene efecto junto con --prefix.")

    if args.yes and not args.apply:
        _abort("--yes requiere --apply.")

    if args.backup and not args.apply:
        _warn("--backup se ignora en modo dry-run (no hay cambios que proteger).")

    # ── Construcción del plan ─────────────────────────────────────────────────
    conn = get_connection(db_path)
    try:
        if args.prefix:
            plan = build_plan_prefix(conn, args.prefix, args.glob_filter)
        elif args.rename:
            plan = build_plan_rename(args.rename[0], args.rename[1])
        else:  # --mapping
            plan = build_plan_mapping(args.mapping)

        if not plan:
            _warn("El plan está vacío: no hay IDs que cumplan los criterios.")
            sys.exit(0)

        # ── Validación del plan ───────────────────────────────────────────────
        errors = validate_plan(conn, plan)
    finally:
        conn.close()

    if errors:
        _header("Errores en el plan", "❌")
        for err in errors:
            _err(err)
        print()
        sys.exit(1)

    # ── Mostrar plan ──────────────────────────────────────────────────────────
    conn = get_connection(db_path)
    show_plan(conn, plan)
    conn.close()

    # ── Dry-run vs apply ──────────────────────────────────────────────────────
    if not args.apply:
        print(f"  {YELLOW}{BOLD}Modo DRY-RUN — no se ha modificado nada.{RESET}")
        print(f"  Usa {BOLD}--apply{RESET} para aplicar los cambios.\n")
        sys.exit(0)

    # ── Aplicar ───────────────────────────────────────────────────────────────
    apply_plan(db_path, plan, backup=args.backup, skip_confirm=args.yes)

    # ── Verificación opcional ─────────────────────────────────────────────────
    if args.verify:
        verify(db_path, plan)


if __name__ == "__main__":
    main()