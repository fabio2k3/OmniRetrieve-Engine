"""
rebuild_chunks.py
=================
Elimina todos los chunks existentes y los reconstruye documento a documento
usando el nuevo algoritmo de chunking con solapamiento semántico a nivel
de oración.

Cuándo usar este tool
---------------------
- Después de actualizar el algoritmo de chunking (_split_into_chunks).
- Cuando se quiera cambiar chunk_size o overlap_sentences.
- Si los chunks actuales están mal construidos o son de una versión anterior.

Qué hace
--------
1. Muestra estadísticas antes de empezar.
2. Pide confirmación (a menos que se pase --yes).
3. Elimina todos los chunks de la BD (y sus embeddings asociados).
4. Itera sobre cada documento con full_text disponible.
5. Aplica el nuevo algoritmo y guarda los chunks en la BD.
6. Muestra estadísticas finales y avisa de que el índice FAISS es obsoleto.

Uso
---
    python -m backend.tools.rebuild_chunks
    python -m backend.tools.rebuild_chunks --dry-run
    python -m backend.tools.rebuild_chunks --chunk-size 800 --overlap 3
    python -m backend.tools.rebuild_chunks --yes
    python -m backend.tools.rebuild_chunks --db path/to/other.db
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection
from backend.database.chunk_repository import save_chunks, get_chunk_stats
from backend.crawler.chunker import _split_into_chunks

# ── Colores ANSI ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bar(value: int, total: int, width: int = 30) -> str:
    if total == 0:
        return DIM + "─" * width + RESET
    filled = int(width * value / total)
    pct    = value / total
    color  = GREEN if pct >= 0.8 else YELLOW if pct >= 0.4 else CYAN
    return f"{color}{'█' * filled}{RESET}{DIM}{'░' * (width - filled)}{RESET}"


def _print_stats(label: str, stats: dict) -> None:
    total    = stats["total_chunks"]
    embedded = stats["embedded_chunks"]
    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Total chunks     : {BOLD}{total:,}{RESET}")
    print(f"  Embedidos        : {GREEN}{embedded:,}{RESET}  {_bar(embedded, total)}")
    print(f"  Sin embedding    : {YELLOW}{stats['pending_chunks']:,}{RESET}")


def _get_docs_with_text(db_path: Path) -> list[dict]:
    """Devuelve todos los documentos con full_text disponible, ordenados por publicación."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT arxiv_id, title, full_text, text_length
            FROM   documents
            WHERE  pdf_downloaded = 1
              AND  full_text IS NOT NULL
              AND  full_text != ''
            ORDER  BY published DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _delete_all_chunks(db_path: Path) -> int:
    """Elimina todos los chunks (y sus embeddings, por CASCADE o update directo)."""
    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("DELETE FROM chunks")
        conn.commit()
        return count
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Lógica principal
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_chunks(
    db_path:           Path,
    chunk_size:        int,
    overlap_sentences: int,
    dry_run:           bool,
    batch_log_every:   int,
) -> None:

    # ── 1. Estadísticas iniciales ─────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Rebuild de chunks — OmniRetrieve-Engine{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")
    print(f"\n  {DIM}DB         :{RESET} {db_path}")
    print(f"  {DIM}chunk_size :{RESET} {chunk_size} chars")
    print(f"  {DIM}overlap    :{RESET} {overlap_sentences} oraciones")
    print(f"  {DIM}dry_run    :{RESET} {dry_run}")

    docs = _get_docs_with_text(db_path)
    if not docs:
        print(f"\n  {YELLOW}No hay documentos con texto en la BD. Nada que hacer.{RESET}\n")
        return

    stats_before = get_chunk_stats(db_path)
    _print_stats("Estado ANTES", stats_before)

    total_chars = sum(d["text_length"] or 0 for d in docs)
    print(f"\n  Documentos con texto : {BOLD}{len(docs):,}{RESET}")
    print(f"  Texto total          : {BOLD}{total_chars:,.0f} chars{RESET}")

    # Estimación de chunks que se van a generar (referencial)
    est_chunks = sum(
        max(1, (d["text_length"] or 0) // chunk_size)
        for d in docs
    )
    print(f"  Chunks estimados     : {DIM}~{est_chunks:,}{RESET}  "
          f"{DIM}(estimación sin contar overlap){RESET}")

    # ── 2. Advertencias ───────────────────────────────────────────────────────
    print(f"\n  {YELLOW}{BOLD}⚠  ADVERTENCIAS{RESET}")
    print(f"  {YELLOW}• Se eliminarán {stats_before['total_chunks']:,} chunks "
          f"y sus {stats_before['embedded_chunks']:,} embeddings.{RESET}")
    print(f"  {YELLOW}• El índice FAISS quedará obsoleto — ejecuta el pipeline "
          f"de embedding después.{RESET}")
    if stats_before["embedded_chunks"] > 0:
        print(f"  {RED}• Perderás {stats_before['embedded_chunks']:,} vectores "
              f"calculados. Tendrás que re-embedidar todo el corpus.{RESET}")

    if dry_run:
        print(f"\n  {CYAN}{BOLD}Modo DRY-RUN — no se modificará nada.{RESET}\n")

    # ── 3. Ejecución ──────────────────────────────────────────────────────────
    n_docs_ok   = 0
    n_docs_err  = 0
    n_chunks    = 0
    t_start     = time.perf_counter()

    if not dry_run:
        deleted = _delete_all_chunks(db_path)
        print(f"\n  {RED}Eliminados {deleted:,} chunks de la BD.{RESET}")
        print(f"\n  Reconstruyendo chunks...\n")

    else:
        print(f"\n  Simulando rebuild...\n")

    for i, doc in enumerate(docs, 1):
        arxiv_id  = doc["arxiv_id"]
        full_text = doc["full_text"]

        try:
            new_chunks = _split_into_chunks(
                full_text,
                max_chars=chunk_size,
                overlap_sentences=overlap_sentences,
            )

            if not dry_run and new_chunks:
                save_chunks(arxiv_id, new_chunks, db_path=db_path)

            n_chunks  += len(new_chunks)
            n_docs_ok += 1

            # Log de progreso
            if i % batch_log_every == 0 or i == len(docs):
                elapsed = time.perf_counter() - t_start
                pct     = i / len(docs) * 100
                bar     = _bar(i, len(docs), width=20)
                rate    = i / elapsed if elapsed > 0 else 0
                eta     = (len(docs) - i) / rate if rate > 0 else 0
                eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta/3600:.1f}h"

                print(
                    f"  [{i:>6,}/{len(docs):,}]  {bar}  {pct:5.1f}%  "
                    f"{GREEN}{n_chunks:,} chunks{RESET}  "
                    f"{DIM}~{rate:.1f} docs/s  ETA: {eta_str}{RESET}"
                )

        except Exception as exc:
            n_docs_err += 1
            print(f"  {RED}✗ {arxiv_id} — {exc}{RESET}")

    # ── 4. Estadísticas finales ───────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start

    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Resultado{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")
    print(f"  Docs procesados  : {GREEN}{n_docs_ok:,}{RESET}", end="")
    if n_docs_err:
        print(f"  {RED}errores: {n_docs_err}{RESET}", end="")
    print()
    print(f"  Chunks generados : {BOLD}{n_chunks:,}{RESET}")
    if n_docs_ok > 0:
        print(f"  Chunks / doc     : {n_chunks / n_docs_ok:.1f} promedio")
    print(f"  Tiempo total     : {elapsed_total:.1f}s")

    if not dry_run:
        stats_after = get_chunk_stats(db_path)
        _print_stats("Estado DESPUÉS", stats_after)
        print(f"\n  {YELLOW}Recuerda lanzar el pipeline de embedding para re-vectorizar:{RESET}")
        print(f"  {CYAN}python -m backend.embedding.pipeline{RESET}\n")
    else:
        print(f"\n  {CYAN}Simulación completada. Ejecuta sin --dry-run para aplicar.{RESET}\n")

    print(f"{BOLD}{'═' * 56}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _confirm(msg: str) -> bool:
    try:
        answer = input(f"\n  {YELLOW}{msg}{RESET}  [s/N] ").strip().lower()
        return answer in ("s", "si", "sí", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Reconstrucción de chunks con el nuevo algoritmo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",
                   type=Path, default=DB_PATH,
                   help="Ruta a la base de datos SQLite.")
    p.add_argument("--chunk-size",
                   type=int, default=1000, metavar="N",
                   help="Tamaño máximo de cada chunk en caracteres.")
    p.add_argument("--overlap",
                   type=int, default=2, metavar="N",
                   help="Número de oraciones de solapamiento entre chunks.")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Simula el proceso sin modificar la BD.")
    p.add_argument("--yes",
                   action="store_true",
                   help="Salta la confirmación interactiva.")
    p.add_argument("--log-every",
                   type=int, default=100, metavar="N",
                   help="Imprime progreso cada N documentos.")
    args = p.parse_args()

    if not args.db.exists():
        print(f"\n  {RED}Base de datos no encontrada: {args.db}{RESET}\n")
        sys.exit(1)

    if not args.dry_run and not args.yes:
        if not _confirm("¿Confirmas que quieres eliminar y reconstruir todos los chunks?"):
            print(f"\n  {DIM}Operación cancelada.{RESET}\n")
            sys.exit(0)

    rebuild_chunks(
        db_path=args.db,
        chunk_size=args.chunk_size,
        overlap_sentences=args.overlap,
        dry_run=args.dry_run,
        batch_log_every=args.log_every,
    )


if __name__ == "__main__":
    main()