"""
tools/retry_downloads.py
========================
Herramienta para reintentar descargas con errores en la base de datos.

Uso
---
    # Ver todos los errores
    python -m backend.tools.retry_downloads list

    # Reintentar todos los errores
    python -m backend.tools.retry_downloads retry

    # Reintentar solo IDs concretos
    python -m backend.tools.retry_downloads retry 2301.00001 2301.00002

    # Reintentar errores que contienen cierto texto
    python -m backend.tools.retry_downloads retry --filter "timeout"
    python -m backend.tools.retry_downloads retry --filter "403"

    # Reintentar y ejecutar la descarga inmediatamente (sin esperar al crawler)
    python -m backend.tools.retry_downloads retry --now

    # Ver estadísticas generales
    python -m backend.tools.retry_downloads stats
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# -- Resolver ruta al paquete backend -----------------------------------------
def _find_backend_root() -> Path | None:
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / "backend" / "backend" / "__init__.py").exists():
            return p / "backend"
        p = p.parent
    return None

_backend = _find_backend_root()
if _backend and str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
# -----------------------------------------------------------------------------

from backend.database.schema import DB_PATH, get_connection
from backend.database.crawler_repository import get_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Consultas DB ──────────────────────────────────────────────────────────────

def get_error_docs(db_path: Path, filter_text: str | None = None) -> list[dict]:
    """Devuelve todos los documentos con pdf_downloaded = 2 (error)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT arxiv_id, title, index_error, indexed_at, pdf_url
            FROM documents
            WHERE pdf_downloaded = 2
            ORDER BY indexed_at DESC
        """).fetchall()
        docs = [dict(r) for r in rows]
        if filter_text:
            fl = filter_text.lower()
            docs = [d for d in docs if fl in (d.get("index_error") or "").lower()]
        return docs
    finally:
        conn.close()


def reset_to_pending(arxiv_ids: list[str], db_path: Path) -> int:
    """
    Resetea los documentos indicados a pdf_downloaded=0 (pendiente)
    limpiando el error previo. Devuelve el número de filas afectadas.
    """
    if not arxiv_ids:
        return 0
    conn = get_connection(db_path)
    try:
        placeholders = ",".join("?" * len(arxiv_ids))
        cur = conn.execute(f"""
            UPDATE documents
            SET pdf_downloaded = 0,
                index_error    = NULL,
                indexed_at     = NULL
            WHERE arxiv_id IN ({placeholders})
              AND pdf_downloaded = 2
        """, arxiv_ids)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def download_now(arxiv_ids: list[str], db_path: Path) -> None:
    """
    Descarga inmediatamente los documentos indicados sin esperar al crawler,
    usando la misma lógica que TextLoop._process_document().
    """
    from backend.crawler.config import CrawlerConfig
    from backend.crawler.chunker import make_chunks
    from backend.crawler.clients import ArxivClient
    from backend.crawler._routing import client_for, local_id as extract_local_id
    from backend.database import crawler_repository as repo

    cfg        = CrawlerConfig()
    client_map = {"arxiv": ArxivClient()}

    log.info("[retry] Descargando %d documentos de forma inmediata...", len(arxiv_ids))

    for i, doc_id in enumerate(arxiv_ids, 1):
        client = client_for(doc_id, client_map)
        if client is None:
            log.error("[retry] [%d/%d] Sin cliente para %r — saltando.", i, len(arxiv_ids), doc_id)
            continue

        doc_row = repo.get_document(doc_id, db_path=db_path)
        pdf_url = doc_row["pdf_url"] if doc_row else None
        local   = extract_local_id(doc_id)

        log.info("[retry] [%d/%d] Descargando %s ...", i, len(arxiv_ids), doc_id)
        try:
            full_text = client.download_text(local, pdf_url=pdf_url)
            chunks    = make_chunks(
                full_text,
                chunk_size=cfg.chunk_size,
                overlap_sentences=cfg.overlap_sentences,
            )
            repo.save_pdf_text(doc_id, full_text, db_path=db_path)
            from backend.database.chunk_repository import save_chunks
            save_chunks(doc_id, chunks, db_path=db_path)
            log.info(
                "[retry] [%d/%d] OK %s — %d chars | %d chunks",
                i, len(arxiv_ids), doc_id, len(full_text), len(chunks),
            )
        except Exception as exc:
            log.error("[retry] [%d/%d] FALLO %s — %s", i, len(arxiv_ids), doc_id, exc)
            repo.save_pdf_error(doc_id, str(exc), db_path=db_path)


# ── Comandos CLI ──────────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    """Muestra todos los documentos con error de descarga."""
    docs = get_error_docs(args.db, filter_text=args.filter)
    if not docs:
        print("No hay documentos con errores de descarga.")
        return

    print(f"\n{'─'*80}")
    print(f"  {len(docs)} documento(s) con error de descarga")
    if args.filter:
        print(f"  Filtro aplicado: '{args.filter}'")
    print(f"{'─'*80}\n")

    for d in docs:
        title = (d.get("title") or d["arxiv_id"])[:60]
        error = (d.get("index_error") or "sin mensaje")[:80]
        ts    = (d.get("indexed_at") or "")[:19]
        print(f"  ID    : {d['arxiv_id']}")
        print(f"  Título: {title}")
        print(f"  Error : {error}")
        print(f"  Cuando: {ts}")
        print()


def cmd_retry(args: argparse.Namespace) -> None:
    """Resetea los documentos con error a pendiente (y opcionalmente descarga ya)."""
    if args.ids:
        # IDs explícitos pasados por argumento
        target_ids = list(args.ids)
        # Verificar que realmente tienen error
        docs = get_error_docs(args.db)
        error_ids = {d["arxiv_id"] for d in docs}
        not_found = [i for i in target_ids if i not in error_ids]
        if not_found:
            log.warning("Estos IDs no tienen error en la BD (se ignorarán): %s", not_found)
        target_ids = [i for i in target_ids if i in error_ids]
    else:
        docs = get_error_docs(args.db, filter_text=args.filter)
        target_ids = [d["arxiv_id"] for d in docs]

    if not target_ids:
        print("No hay documentos que coincidan para reintentar.")
        return

    print(f"\nVan a resetearse {len(target_ids)} documento(s):")
    for did in target_ids:
        print(f"  - {did}")

    if not args.yes:
        resp = input("\n¿Continuar? [s/N] ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("Cancelado.")
            return

    reset = reset_to_pending(target_ids, args.db)
    log.info("Reseteados %d documentos a estado 'pendiente'.", reset)

    if args.now:
        download_now(target_ids, args.db)
    else:
        print(
            "\nDocumentos marcados como pendientes. El crawler los procesará "
            "en el próximo ciclo.\n"
            "Usa --now para descargarlos inmediatamente."
        )


def cmd_stats(args: argparse.Namespace) -> None:
    """Muestra estadísticas generales de la base de datos."""
    s = get_stats(db_path=args.db)
    print(f"\n{'─'*40}")
    print(f"  Estadísticas de documentos")
    print(f"{'─'*40}")
    print(f"  Total      : {s.get('total_documents', 0)}")
    print(f"  Con texto  : {s.get('pdf_indexed', 0)}")
    print(f"  Pendientes : {s.get('pdf_pending', 0)}")
    print(f"  Con error  : {s.get('pdf_errors', 0)}")
    print()

    # Agrupar errores por tipo
    conn = get_connection(args.db)
    try:
        rows = conn.execute("""
            SELECT index_error, COUNT(*) as n
            FROM documents
            WHERE pdf_downloaded = 2
            GROUP BY index_error
            ORDER BY n DESC
            LIMIT 10
        """).fetchall()
    finally:
        conn.close()

    if rows:
        print(f"  Tipos de error (top 10):")
        for r in rows:
            msg = (r["index_error"] or "sin mensaje")[:60]
            print(f"    {r['n']:4d}x  {msg}")
    print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Herramienta para reintentar descargas con error en OmniRetrieve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"Ruta a la base de datos SQLite (default: {DB_PATH})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- list -----------------------------------------------------------------
    p_list = sub.add_parser("list", help="Listar documentos con error de descarga.")
    p_list.add_argument(
        "--filter", metavar="TEXTO",
        help="Mostrar solo errores que contengan este texto.",
    )

    # -- retry ----------------------------------------------------------------
    p_retry = sub.add_parser("retry", help="Reintentar descargas con error.")
    p_retry.add_argument(
        "ids", nargs="*", metavar="ARXIV_ID",
        help="IDs concretos a reintentar. Si se omite, reintenta todos los errores.",
    )
    p_retry.add_argument(
        "--filter", metavar="TEXTO",
        help="Reintentar solo los que contengan este texto en el error.",
    )
    p_retry.add_argument(
        "--now", action="store_true",
        help="Descargar inmediatamente sin esperar al crawler.",
    )
    p_retry.add_argument(
        "-y", "--yes", action="store_true",
        help="No pedir confirmación.",
    )

    # -- stats ----------------------------------------------------------------
    sub.add_parser("stats", help="Estadísticas de la base de datos.")

    args = parser.parse_args()

    dispatch = {"list": cmd_list, "retry": cmd_retry, "stats": cmd_stats}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()