"""
purge_derived_data.py
=====================
Elimina todos los datos derivados de la base de datos y los archivos
de índice en disco, conservando intactos los documentos (texto, metadatos).

Qué elimina
-----------
BD:
  • chunks            — fragmentos de texto y sus embeddings (BLOB)
  • postings          — índice invertido de frecuencias
  • terms             — vocabulario BM25
  • index_meta        — metadatos de la última indexación
  • faiss_log         — historial de builds del índice FAISS
  • embedding_meta    — metadatos del módulo de embedding
  Además resetea indexed_tfidf_at → NULL en documents.

Disco:
  • index.faiss       — índice vectorial FAISS serializado
  • id_map.npy        — mapa posición → chunk_id del índice FAISS

Qué conserva
------------
  • documents         — arxiv_id, title, authors, abstract, full_text, ...
  • crawl_log         — historial del crawler
  • lsi_log           — historial del modelo LSI
  • web_search_log    — historial de búsquedas web
  • El modelo LSI (.pkl) en disco — no se toca

Uso
---
    python -m backend.tools.purge_derived_data
    python -m backend.tools.purge_derived_data --yes
    python -m backend.tools.purge_derived_data --keep-faiss-files
    python -m backend.tools.purge_derived_data --db ruta/otra.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, DATA_DIR, get_connection

# ── Colores ANSI ──────────────────────────────────────────────────────────────
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
RESET   = "\033[0m"

_FAISS_DIR    = DATA_DIR / "faiss"
_FAISS_INDEX  = _FAISS_DIR / "index.faiss"
_FAISS_ID_MAP = _FAISS_DIR / "id_map.npy"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    return f"{n:,}"


def _preview(db_path: Path) -> dict:
    """Cuenta filas actuales en cada tabla derivada."""
    conn = get_connection(db_path)
    try:
        def _count(table: str) -> int:
            try:
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                return -1

        return {
            "chunks":         _count("chunks"),
            "postings":       _count("postings"),
            "terms":          _count("terms"),
            "index_meta":     _count("index_meta"),
            "faiss_log":      _count("faiss_log"),
            "embedding_meta": _count("embedding_meta"),
            "documents":      _count("documents"),
        }
    finally:
        conn.close()


def _purge(db_path: Path) -> dict:
    """Ejecuta el borrado en una única transacción."""
    conn = get_connection(db_path)
    counts = {}
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA synchronous   = OFF")
        conn.execute("PRAGMA cache_size    = -65536")

        with conn:
            # chunks (incluye embedding BLOB por estar en la misma fila)
            counts["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.execute("DELETE FROM chunks")

            # postings
            counts["postings"] = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
            conn.execute("DELETE FROM postings")

            # terms
            counts["terms"] = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            conn.execute("DELETE FROM terms")

            # index_meta
            counts["index_meta"] = conn.execute("SELECT COUNT(*) FROM index_meta").fetchone()[0]
            conn.execute("DELETE FROM index_meta")

            # faiss_log
            try:
                counts["faiss_log"] = conn.execute("SELECT COUNT(*) FROM faiss_log").fetchone()[0]
                conn.execute("DELETE FROM faiss_log")
            except Exception:
                counts["faiss_log"] = 0

            # embedding_meta
            try:
                counts["embedding_meta"] = conn.execute(
                    "SELECT COUNT(*) FROM embedding_meta"
                ).fetchone()[0]
                conn.execute("DELETE FROM embedding_meta")
            except Exception:
                counts["embedding_meta"] = 0

            # Reset del flag de indexación en documents
            counts["docs_reset"] = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE indexed_tfidf_at IS NOT NULL"
            ).fetchone()[0]
            conn.execute("UPDATE documents SET indexed_tfidf_at = NULL")

    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous  = FULL")
        conn.close()

    return counts


def _delete_faiss_files(index_path: Path, id_map_path: Path) -> list[str]:
    """Elimina los archivos FAISS del disco. Devuelve lista de archivos borrados."""
    deleted = []
    for p in (index_path, id_map_path):
        if p.exists():
            p.unlink()
            deleted.append(str(p))
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Elimina datos derivados conservando los documentos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",
                   type=Path, default=DB_PATH,
                   help="Ruta a la base de datos SQLite.")
    p.add_argument("--faiss-index",
                   type=Path, default=_FAISS_INDEX,
                   help="Ruta del archivo .faiss a eliminar.")
    p.add_argument("--faiss-id-map",
                   type=Path, default=_FAISS_ID_MAP,
                   help="Ruta del archivo id_map.npy a eliminar.")
    p.add_argument("--keep-faiss-files",
                   action="store_true",
                   help="No eliminar los archivos FAISS del disco.")
    p.add_argument("--yes", "-y",
                   action="store_true",
                   help="Salta la confirmación interactiva.")
    args = p.parse_args()

    if not args.db.exists():
        print(f"\n  {RED}Base de datos no encontrada: {args.db}{RESET}\n")
        sys.exit(1)

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Purge de datos derivados — OmniRetrieve-Engine{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")
    print(f"\n  {DIM}DB: {args.db}{RESET}\n")

    prev = _preview(args.db)

    print(f"  {BOLD}Datos actuales{RESET}")
    print(f"  {'─' * 40}")
    print(f"  {'documentos (conservados)':<30} {GREEN}{_fmt(prev['documents'])}{RESET}")
    print(f"  {'─' * 40}")
    print(f"  {'chunks + embeddings':<30} {YELLOW}{_fmt(prev['chunks'])}{RESET}")
    print(f"  {'terms':<30} {YELLOW}{_fmt(prev['terms'])}{RESET}")
    print(f"  {'postings':<30} {YELLOW}{_fmt(prev['postings'])}{RESET}")
    print(f"  {'index_meta':<30} {YELLOW}{_fmt(prev['index_meta'])}{RESET}")
    print(f"  {'faiss_log':<30} {YELLOW}{_fmt(prev['faiss_log'])}{RESET}")
    print(f"  {'embedding_meta':<30} {YELLOW}{_fmt(prev['embedding_meta'])}{RESET}")

    faiss_files = []
    if not args.keep_faiss_files:
        for fp in (args.faiss_index, args.faiss_id_map):
            if fp.exists():
                size_mb = fp.stat().st_size / 1_048_576
                faiss_files.append((fp, size_mb))
                print(f"  {str(fp.name):<30} {YELLOW}{size_mb:.1f} MB  (disco){RESET}")

    print()

    # ── Confirmación ──────────────────────────────────────────────────────────
    if not args.yes:
        print(f"  {RED}{BOLD}⚠  Esta operación es irreversible.{RESET}")
        print(f"  {YELLOW}   Los documentos y su texto completo se conservarán.{RESET}\n")
        try:
            resp = input("  Escribe 'si' para confirmar: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Cancelado.{RESET}\n")
            sys.exit(0)
        if resp not in ("si", "sí", "yes", "y"):
            print(f"\n  {DIM}Operación cancelada.{RESET}\n")
            sys.exit(0)

    # ── Ejecución ─────────────────────────────────────────────────────────────
    print()
    t0 = time.monotonic()

    print(f"  → Purgando BD…", end="", flush=True)
    counts = _purge(args.db)
    elapsed_db = time.monotonic() - t0
    print(f"  {GREEN}✔{RESET}  ({elapsed_db:.2f}s)")

    deleted_files = []
    if not args.keep_faiss_files:
        print(f"  → Eliminando archivos FAISS…", end="", flush=True)
        deleted_files = _delete_faiss_files(args.faiss_index, args.faiss_id_map)
        print(f"  {GREEN}✔{RESET}")

    elapsed = time.monotonic() - t0

    # ── Resultado ─────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Resultado{RESET}  ({elapsed:.2f}s)")
    print(f"{BOLD}{'─' * 56}{RESET}")
    print(f"  {'chunks eliminados':<30} {RED}{_fmt(counts['chunks'])}{RESET}")
    print(f"  {'terms eliminados':<30} {RED}{_fmt(counts['terms'])}{RESET}")
    print(f"  {'postings eliminados':<30} {RED}{_fmt(counts['postings'])}{RESET}")
    print(f"  {'docs con flag reseteado':<30} {YELLOW}{_fmt(counts['docs_reset'])}{RESET}")
    if deleted_files:
        for f in deleted_files:
            print(f"  {'archivo eliminado':<30} {RED}{f}{RESET}")
    else:
        print(f"  {'archivos FAISS':<30} {DIM}conservados{RESET}")
    print(f"\n  {GREEN}Documentos intactos: {_fmt(prev['documents'])}{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}\n")

    print(f"  {CYAN}Próximos pasos sugeridos:{RESET}")
    print(f"  {DIM}1. Reconstruir chunks:  python -m backend.tools.rebuild_chunks{RESET}")
    print(f"  {DIM}2. Crear índice BM25:   python -m backend.tools.build_index{RESET}")
    print(f"  {DIM}3. Crear embeddings:    python -m backend.tools.embed_chunks{RESET}\n")


if __name__ == "__main__":
    main()