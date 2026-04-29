"""
build_index.py
==============
Construye el índice invertido BM25 (terms + postings) desde los documentos
existentes en la base de datos.

Qué hace
--------
1. Lee todos los documentos con texto disponible (full_text o abstract).
2. Tokeniza y cuenta frecuencias por documento (TextPreprocessor).
3. Persiste el vocabulario en `terms` y las frecuencias en `postings`.
4. Marca cada documento como indexado (indexed_tfidf_at).
5. Guarda metadatos de la ejecución en `index_meta`.

No toca chunks, embeddings ni el índice FAISS.

Uso
---
    python -m backend.tools.build_index
    python -m backend.tools.build_index --field abstract
    python -m backend.tools.build_index --reindex
    python -m backend.tools.build_index --stemming --min-len 4
    python -m backend.tools.build_index --batch-size 500 --db ruta/otra.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection
from backend.indexing.pipeline import IndexingPipeline

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

def _preview(db_path: Path, field: str) -> dict:
    """Cuenta documentos disponibles y estado actual del índice."""
    conn = get_connection(db_path)
    try:
        total_docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1"
        ).fetchone()[0]

        has_text = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND ("
            "  (full_text IS NOT NULL AND full_text != '') OR "
            "  (abstract  IS NOT NULL AND abstract  != '')"
            ")"
        ).fetchone()[0]

        already_indexed = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE indexed_tfidf_at IS NOT NULL"
        ).fetchone()[0]

        try:
            terms    = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
            postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        except Exception:
            terms = postings = 0

        return {
            "total_docs":       total_docs,
            "has_text":         has_text,
            "already_indexed":  already_indexed,
            "pending":          has_text - already_indexed,
            "terms":            terms,
            "postings":         postings,
        }
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Construye el índice BM25 (terms + postings)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",
                   type=Path, default=DB_PATH,
                   help="Ruta a la base de datos SQLite.")
    p.add_argument("--field",
                   choices=["full_text", "abstract", "both"], default="both",
                   help=(
                       "Campo a indexar. "
                       "'both' usa full_text y cae a abstract si no hay texto extraído."
                   ))
    p.add_argument("--batch-size",
                   type=int, default=200, metavar="N",
                   help="Documentos leídos por lote.")
    p.add_argument("--stemming",
                   action="store_true",
                   help="Activar stemming (SnowballStemmer, requiere NLTK).")
    p.add_argument("--min-len",
                   type=int, default=3, metavar="N",
                   help="Longitud mínima de token (caracteres).")
    p.add_argument("--reindex",
                   action="store_true",
                   help=(
                       "Borra terms y postings existentes y reconstruye "
                       "desde cero. Útil tras un cambio de campo o de preprocesado."
                   ))
    args = p.parse_args()

    if not args.db.exists():
        print(f"\n  {RED}Base de datos no encontrada: {args.db}{RESET}\n")
        sys.exit(1)

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Build de índice BM25 — OmniRetrieve-Engine{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}")

    prev = _preview(args.db, args.field)

    print(f"\n  {DIM}DB         : {args.db}{RESET}")
    print(f"  {DIM}Campo      : {args.field}{RESET}")
    print(f"  {DIM}Batch      : {args.batch_size}{RESET}")
    print(f"  {DIM}Stemming   : {args.stemming}{RESET}")
    print(f"  {DIM}Min token  : {args.min_len}{RESET}")
    print(f"  {DIM}Reindex    : {args.reindex}{RESET}\n")

    print(f"  {'─' * 40}")
    print(f"  {'Docs con PDF descargado':<30} {prev['total_docs']:,}")
    print(f"  {'Docs con texto':<30} {prev['has_text']:,}")
    print(f"  {'Ya indexados':<30} {GREEN}{prev['already_indexed']:,}{RESET}")
    print(f"  {'Pendientes':<30} {YELLOW}{prev['pending']:,}{RESET}")
    print(f"  {'Terms actuales':<30} {prev['terms']:,}")
    print(f"  {'Postings actuales':<30} {prev['postings']:,}")
    print(f"  {'─' * 40}\n")

    if prev["pending"] == 0 and not args.reindex:
        print(f"  {GREEN}✔  No hay documentos pendientes de indexar.{RESET}")
        print(f"  {DIM}   Usa --reindex para reconstruir el índice desde cero.{RESET}\n")
        sys.exit(0)

    if args.reindex:
        print(f"  {YELLOW}{BOLD}⚠  --reindex borrará {prev['terms']:,} terms "
              f"y {prev['postings']:,} postings.{RESET}\n")

    # ── Ejecución ─────────────────────────────────────────────────────────────
    t0 = time.monotonic()

    try:
        stats = IndexingPipeline(
            db_path       = args.db,
            field         = args.field,
            batch_size    = args.batch_size,
            use_stemming  = args.stemming,
            min_token_len = args.min_len,
        ).run(reindex=args.reindex)
    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Interrumpido por el usuario.{RESET}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  {RED}Error durante la indexación: {exc}{RESET}\n")
        raise

    elapsed = time.monotonic() - t0

    # ── Resultado ─────────────────────────────────────────────────────────────
    post_prev = _preview(args.db, args.field)

    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"{BOLD}  Resultado{RESET}  ({elapsed:.1f}s)")
    print(f"{BOLD}{'─' * 56}{RESET}")
    print(f"  {'Docs procesados':<30} {GREEN}{stats['docs_processed']:,}{RESET}")
    print(f"  {'Terms añadidos':<30} {GREEN}{stats['terms_added']:,}{RESET}")
    print(f"  {'Postings añadidos':<30} {GREEN}{stats['postings_added']:,}{RESET}")
    print(f"  {'─' * 40}")
    print(f"  {'Terms totales en BD':<30} {post_prev['terms']:,}")
    print(f"  {'Postings totales en BD':<30} {post_prev['postings']:,}")
    if stats['docs_processed'] > 0:
        rate = stats['docs_processed'] / elapsed
        print(f"  {'Velocidad':<30} {rate:.0f} docs/s")
    print(f"{BOLD}{'═' * 56}{RESET}\n")


if __name__ == "__main__":
    main()