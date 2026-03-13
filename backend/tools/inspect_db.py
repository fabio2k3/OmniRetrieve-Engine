"""
inspect_db.py
Muestra el estado actual de la base de datos SQLite en tiempo real.

Uso
---
    python -m backend.tools.inspect_db              # resumen general
    python -m backend.tools.inspect_db --docs 5     # últimos 5 documentos
    python -m backend.tools.inspect_db --doc 2603.11041  # un doc concreto
    python -m backend.tools.inspect_db --watch       # actualiza cada 5s
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection

BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def _bar(value: int, total: int, width: int = 30) -> str:
    if total == 0:
        return "─" * width
    filled = int(width * value / total)
    return f"{GREEN}{'█' * filled}{RESET}{'░' * (width - filled)}"


def show_summary(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        total   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0").fetchone()[0]
        errors  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2").fetchone()[0]
        chunks  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        avg_text = conn.execute(
            "SELECT AVG(text_length) FROM documents WHERE pdf_downloaded = 1"
        ).fetchone()[0] or 0
        last_doc = conn.execute(
            "SELECT arxiv_id, title, indexed_at FROM documents "
            "WHERE pdf_downloaded = 1 ORDER BY indexed_at DESC LIMIT 1"
        ).fetchone()
        last_err = conn.execute(
            "SELECT arxiv_id, index_error FROM documents "
            "WHERE pdf_downloaded = 2 ORDER BY indexed_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  Estado de la DB — {db_path.name}{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"\n  {BOLD}Documentos{RESET}")
    print(f"  Total descubiertos : {BOLD}{total}{RESET}")
    print(f"  Con texto guardado : {GREEN}{indexed}{RESET}  {_bar(indexed, total)}")
    print(f"  Pendientes         : {YELLOW}{pending}{RESET}  {_bar(pending, total)}")
    print(f"  Con error          : {RED}{errors}{RESET}  {_bar(errors, total)}")
    print(f"\n  {BOLD}Chunks{RESET}")
    print(f"  Total chunks       : {CYAN}{chunks}{RESET}")
    print(f"  Chars promedio/doc : {avg_text:,.0f}")
    if last_doc:
        print(f"\n  {BOLD}Último guardado exitosamente{RESET}")
        print(f"  ID     : {GREEN}{last_doc['arxiv_id']}{RESET}")
        print(f"  Título : {last_doc['title'][:60]}")
        print(f"  Cuando : {last_doc['indexed_at']}")
    if last_err:
        print(f"\n  {BOLD}Último error{RESET}")
        print(f"  ID     : {RED}{last_err['arxiv_id']}{RESET}")
        print(f"  Error  : {last_err['index_error'][:80]}")
    print(f"\n{BOLD}{'═'*55}{RESET}\n")


def show_recent_docs(n: int = 5, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.arxiv_id, d.title, d.categories, d.published,
                   d.pdf_downloaded, d.text_length,
                   COUNT(c.id) as n_chunks
            FROM documents d
            LEFT JOIN chunks c ON c.arxiv_id = d.arxiv_id
            GROUP BY d.arxiv_id
            ORDER BY d.indexed_at DESC, d.fetched_at DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    finally:
        conn.close()

    print(f"\n{BOLD}  Últimos {n} documentos{RESET}\n")
    for row in rows:
        status = (
            f"{GREEN}✅ guardado{RESET}"   if row["pdf_downloaded"] == 1 else
            f"{RED}❌ error{RESET}"        if row["pdf_downloaded"] == 2 else
            f"{YELLOW}⏳ pendiente{RESET}"
        )
        print(f"  {BOLD}{row['arxiv_id']}{RESET}  [{status}]")
        print(f"    Título     : {row['title'][:70]}")
        print(f"    Categorías : {row['categories']}")
        print(f"    Publicado  : {str(row['published'])[:10]}")
        if row["pdf_downloaded"] == 1:
            print(f"    Texto      : {row['text_length']:,} chars | {row['n_chunks']} chunks")
        print()


def show_single_doc(arxiv_id: str, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        doc = conn.execute(
            "SELECT * FROM documents WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if not doc:
            print(f"\n  {RED}No encontrado: {arxiv_id}{RESET}\n")
            return
        chunks = conn.execute(
            "SELECT id, chunk_index, char_count, text FROM chunks "
            "WHERE arxiv_id = ? ORDER BY chunk_index",
            (arxiv_id,),
        ).fetchall()
    finally:
        conn.close()

    status = (
        f"{GREEN}✅ guardado{RESET}"   if doc["pdf_downloaded"] == 1 else
        f"{RED}❌ error{RESET}"        if doc["pdf_downloaded"] == 2 else
        f"{YELLOW}⏳ pendiente{RESET}"
    )

    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  {arxiv_id}{RESET}  [{status}]")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  Título     : {doc['title']}")
    print(f"  Autores    : {str(doc['authors'])[:80]}")
    print(f"  Abstract   : {str(doc['abstract'])[:150]}…")
    print(f"  Categorías : {doc['categories']}")
    print(f"  Publicado  : {doc['published']}")
    print(f"  URL        : {doc['pdf_url']}")
    print(f"  Indexado   : {doc['indexed_at']}")
    if doc["index_error"]:
        print(f"  Error      : {RED}{doc['index_error']}{RESET}")
    if doc["text_length"]:
        print(f"  Texto      : {doc['text_length']:,} chars")
    if chunks:
        print(f"\n  {BOLD}{len(chunks)} chunks en DB:{RESET}")
        for c in chunks[:3]:
            preview = c["text"][:120].replace("\n", " ")
            print(f"    [{c['chunk_index']}] ({c['char_count']} chars) {preview}…")
        if len(chunks) > 3:
            print(f"    … y {len(chunks)-3} chunks más")
    print(f"\n{BOLD}{'═'*55}{RESET}\n")


def show_full_text(arxiv_id: str, chars: int = 3000, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        doc = conn.execute(
            "SELECT arxiv_id, title, full_text, text_length, pdf_downloaded "
            "FROM documents WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    finally:
        conn.close()

    if not doc:
        print(f"\n  {RED}No encontrado: {arxiv_id}{RESET}\n")
        return

    if doc["pdf_downloaded"] != 1 or not doc["full_text"]:
        print(f"\n  {YELLOW}El documento {arxiv_id} no tiene texto guardado todavía "
              f"(pdf_downloaded={doc['pdf_downloaded']}){RESET}\n")
        return

    text = doc["full_text"]
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  {arxiv_id} — {doc['title'][:55]}{RESET}")
    print(f"  {CYAN}Total: {doc['text_length']:,} chars{RESET}"
          f"  |  Mostrando primeros {min(chars, len(text)):,}")
    print(f"{BOLD}{'═'*60}{RESET}\n")
    print(text[:chars])
    if len(text) > chars:
        print(f"\n{YELLOW}... [{len(text) - chars:,} chars más no mostrados] ..."
              f"\nUsa --chars N para ver más{RESET}")
    print(f"\n{BOLD}{'═'*60}{RESET}\n")


def show_chunk_text(arxiv_id: str, chunk_idx: int = 0, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()[0]
        chunk = conn.execute(
            "SELECT chunk_index, char_count, text FROM chunks "
            "WHERE arxiv_id = ? AND chunk_index = ?",
            (arxiv_id, chunk_idx),
        ).fetchone()
    finally:
        conn.close()

    if not chunk:
        print(f"\n  {RED}No encontrado: {arxiv_id} chunk {chunk_idx} "
              f"(total chunks: {total}){RESET}\n")
        return

    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  {arxiv_id} — chunk [{chunk_idx}/{total-1}]  "
          f"({chunk['char_count']} chars){RESET}")
    print(f"{BOLD}{'═'*60}{RESET}\n")
    print(chunk["text"])
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"  Navegación:  --chunk {max(0,chunk_idx-1)}  ←  "
          f"actual: {chunk_idx}  →  --chunk {min(total-1, chunk_idx+1)}")
    print(f"{BOLD}{'═'*60}{RESET}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspecciona la base de datos SQLite")
    p.add_argument("--docs",  type=int,  metavar="N",   help="Últimos N documentos")
    p.add_argument("--doc",   type=str,  metavar="ID",  help="Detalle de un documento")
    p.add_argument("--text",  type=str,  metavar="ID",  help="Muestra el full_text guardado")
    p.add_argument("--chunk", type=str,  metavar="ID",  help="Muestra un chunk concreto (usar con --idx)")
    p.add_argument("--idx",   type=int,  metavar="N",   default=0, help="Índice del chunk (default: 0)")
    p.add_argument("--chars", type=int,  metavar="N",   default=3000, help="Chars a mostrar con --text")
    p.add_argument("--watch", action="store_true",      help="Actualiza el resumen cada 5s")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"\n  {RED}La DB no existe todavía: {DB_PATH}{RESET}")
        print("  Arranca el crawler primero:  python -m backend.main\n")
        sys.exit(1)

    if args.text:
        show_full_text(args.text, chars=args.chars)
    elif args.chunk:
        show_chunk_text(args.chunk, chunk_idx=args.idx)
    elif args.doc:
        show_single_doc(args.doc)
    elif args.docs:
        show_recent_docs(args.docs)
    elif args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show_summary()
                print("  Actualizando cada 5s … (Ctrl+C para salir)")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Saliendo.\n")
    else:
        show_summary()


if __name__ == "__main__":
    main()