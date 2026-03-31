"""
inspect_db.py
=============
Herramienta de inspeccion y diagnostico de OmniRetrieve-Engine.

Muestra el estado completo del sistema: documentos, chunks, indexacion
TF, modelo LSI, embeddings y base de datos vectorial ChromaDB.

Uso
---
    python -m backend.tools.inspect_db                   # resumen completo
    python -m backend.tools.inspect_db --watch           # refresca cada 5s
    python -m backend.tools.inspect_db --docs 10         # ultimos N documentos
    python -m backend.tools.inspect_db --doc 2301.001    # detalle de un documento
    python -m backend.tools.inspect_db --text 2301.001   # full_text del documento
    python -m backend.tools.inspect_db --chunk 2301.001  # chunks del documento
    python -m backend.tools.inspect_db --idx 2           # chunk concreto (con --chunk)
    python -m backend.tools.inspect_db --index           # stats de indexacion TF/LSI
    python -m backend.tools.inspect_db --vectors         # stats de ChromaDB
    python -m backend.tools.inspect_db --errors          # documentos con error
    python -m backend.tools.inspect_db --pending         # documentos pendientes de PDF
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, get_connection
from backend.embedding.chroma_store import CHROMA_PATH, count as chroma_count

# ---------------------------------------------------------------------------
# Colores ANSI
# ---------------------------------------------------------------------------
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"
DIM    = "\033[2m"
RESET  = "\033[0m"

W = 58   # ancho de separadores


def _sep(char: str = "═") -> str:
    return f"{BOLD}{char * W}{RESET}"


def _bar(value: int, total: int, width: int = 28) -> str:
    """Barra de progreso ASCII con color segun porcentaje."""
    if total == 0:
        return DIM + "─" * width + RESET
    pct     = value / total
    filled  = int(width * pct)
    color   = GREEN if pct > 0.7 else YELLOW if pct > 0.3 else RED
    return f"{color}{'█' * filled}{RESET}{DIM}{'░' * (width - filled)}{RESET}"


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "  0%"
    return f"{value/total:5.1%}"


# ---------------------------------------------------------------------------
# Resumen general
# ---------------------------------------------------------------------------

def show_summary(db_path: Path = DB_PATH, chroma_path: Path = CHROMA_PATH) -> None:
    """Muestra el estado completo del sistema en una sola pantalla."""
    conn = get_connection(db_path)
    try:
        # Documentos
        total    = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1").fetchone()[0]
        pending  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0").fetchone()[0]
        errors   = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2").fetchone()[0]

        # Indexacion TF
        not_tfidf = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
        ).fetchone()[0]
        tfidf_done = indexed - not_tfidf
        vocab      = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        postings   = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]

        # Chunks y embeddings
        total_chunks    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()[0]
        pending_chunks  = total_chunks - embedded_chunks

        # LSI — ultimo modelo construido
        lsi_row = conn.execute(
            "SELECT built_at, k, n_docs, n_terms, var_explained "
            "FROM lsi_log ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # Ultimo documento procesado
        last_ok = conn.execute(
            "SELECT arxiv_id, title, indexed_at FROM documents "
            "WHERE pdf_downloaded = 1 ORDER BY indexed_at DESC LIMIT 1"
        ).fetchone()
        last_err = conn.execute(
            "SELECT arxiv_id, index_error FROM documents "
            "WHERE pdf_downloaded = 2 ORDER BY indexed_at DESC LIMIT 1"
        ).fetchone()

    finally:
        conn.close()

    # ChromaDB
    try:
        chroma_vectors = chroma_count(chroma_path)
    except Exception:
        chroma_vectors = -1

    print(f"\n{_sep()}")
    print(f"{BOLD}  OmniRetrieve-Engine — Estado del sistema{RESET}")
    print(f"{DIM}  DB: {db_path}{RESET}")
    print(_sep())

    # ── Documentos ──────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Documentos{RESET}")
    print(f"  {'Descubiertos':<24} {BOLD}{total:>7}{RESET}")
    print(f"  {'Con PDF guardado':<24} {GREEN}{indexed:>7}{RESET}  {_bar(indexed, total)} {_pct(indexed, total)}")
    print(f"  {'Pendientes de PDF':<24} {YELLOW}{pending:>7}{RESET}  {_bar(pending, total)} {_pct(pending, total)}")
    print(f"  {'Con error':<24} {RED}{errors:>7}{RESET}  {_bar(errors, total)} {_pct(errors, total)}")

    # ── Indexacion TF ────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Indexacion TF (indice invertido){RESET}")
    print(f"  {'Docs indexados TF':<24} {GREEN}{tfidf_done:>7}{RESET}  {_bar(tfidf_done, indexed)} {_pct(tfidf_done, indexed)}")
    print(f"  {'Pendientes de indexar':<24} {YELLOW}{not_tfidf:>7}{RESET}  {_bar(not_tfidf, indexed)} {_pct(not_tfidf, indexed)}")
    print(f"  {'Vocabulario (terms)':<24} {CYAN}{vocab:>7,}{RESET}")
    print(f"  {'Postings totales':<24} {CYAN}{postings:>7,}{RESET}")

    # ── LSI ──────────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Modelo LSI{RESET}")
    if lsi_row:
        var_pct = f"{lsi_row['var_explained']:.1%}" if lsi_row["var_explained"] else "—"
        print(f"  {'Docs en modelo':<24} {GREEN}{lsi_row['n_docs']:>7}{RESET}")
        print(f"  {'Terminos en modelo':<24} {CYAN}{lsi_row['n_terms']:>7,}{RESET}")
        print(f"  {'Componentes k':<24} {lsi_row['k']:>7}")
        print(f"  {'Varianza explicada':<24} {CYAN}{var_pct:>7}{RESET}")
        print(f"  {'Construido':<24} {DIM}{str(lsi_row['built_at'])[:19]}{RESET}")
    else:
        print(f"  {YELLOW}Sin modelo LSI todavia. Usa 'rebuild' en la CLI.{RESET}")

    # ── Chunks y Embeddings ──────────────────────────────────────────────────
    print(f"\n  {BOLD}Chunks y Embeddings{RESET}")
    print(f"  {'Chunks totales':<24} {CYAN}{total_chunks:>7,}{RESET}")
    print(f"  {'Embebidos (ChromaDB)':<24} {GREEN}{embedded_chunks:>7,}{RESET}  {_bar(embedded_chunks, total_chunks)} {_pct(embedded_chunks, total_chunks)}")
    print(f"  {'Pendientes de embeber':<24} {YELLOW}{pending_chunks:>7,}{RESET}  {_bar(pending_chunks, total_chunks)} {_pct(pending_chunks, total_chunks)}")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}Base de datos vectorial (ChromaDB){RESET}")
    if chroma_vectors == -1:
        print(f"  {RED}No se pudo conectar a ChromaDB{RESET}")
        print(f"  {DIM}Ruta: {chroma_path}{RESET}")
    else:
        sync_ok = chroma_vectors == embedded_chunks
        sync_icon = f"{GREEN}✔ sincronizado{RESET}" if sync_ok else f"{YELLOW}⚠ desincronizado{RESET}"
        print(f"  {'Vectores almacenados':<24} {GREEN}{chroma_vectors:>7,}{RESET}  {sync_icon}")
        print(f"  {DIM}Ruta: {chroma_path}{RESET}")
        if not sync_ok:
            diff = embedded_chunks - chroma_vectors
            print(f"  {YELLOW}Diferencia SQLite vs Chroma: {diff:+d} "
                  f"(ejecuta 'embed' para sincronizar){RESET}")

    # ── Ultimos eventos ───────────────────────────────────────────────────────
    if last_ok:
        print(f"\n  {BOLD}Ultimo PDF procesado exitosamente{RESET}")
        print(f"  {GREEN}{last_ok['arxiv_id']}{RESET}  {str(last_ok['title'])[:50]}")
        print(f"  {DIM}{str(last_ok['indexed_at'])[:19]}{RESET}")
    if last_err:
        print(f"\n  {BOLD}Ultimo error{RESET}")
        print(f"  {RED}{last_err['arxiv_id']}{RESET}  {str(last_err['index_error'])[:70]}")

    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Stats de indexacion TF/LSI detalladas
# ---------------------------------------------------------------------------

def show_index_stats(db_path: Path = DB_PATH) -> None:
    """Muestra estadisticas detalladas del indice invertido y modelos LSI."""
    conn = get_connection(db_path)
    try:
        # TF
        vocab     = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        postings  = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        docs_in   = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM postings").fetchone()[0]
        top_terms = conn.execute(
            "SELECT word, df FROM terms ORDER BY df DESC LIMIT 10"
        ).fetchall()
        avg_terms = conn.execute(
            "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM postings GROUP BY doc_id)"
        ).fetchone()[0] or 0

        # LSI log — ultimos 5 builds
        lsi_rows = conn.execute(
            "SELECT built_at, k, n_docs, n_terms, var_explained, model_path "
            "FROM lsi_log ORDER BY id DESC LIMIT 5"
        ).fetchall()

        # Docs pendientes TF
        pending_tf = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND indexed_tfidf_at IS NULL"
        ).fetchone()[0]

    finally:
        conn.close()

    print(f"\n{_sep()}")
    print(f"{BOLD}  Indexacion TF — Indice invertido{RESET}")
    print(_sep())
    print(f"  {'Vocabulario (terms)':<30} {CYAN}{vocab:>8,}{RESET}")
    print(f"  {'Postings totales':<30} {CYAN}{postings:>8,}{RESET}")
    print(f"  {'Documentos en indice':<30} {GREEN}{docs_in:>8,}{RESET}")
    print(f"  {'Pendientes de indexar':<30} {YELLOW}{pending_tf:>8,}{RESET}")
    print(f"  {'Terminos promedio/doc':<30} {avg_terms:>8.1f}")

    if top_terms:
        print(f"\n  {BOLD}Top 10 terminos por frecuencia de documento:{RESET}")
        for t in top_terms:
            bar = _bar(t["df"], docs_in, width=20)
            print(f"    {t['word']:<20} df={t['df']:>5}  {bar}")

    print(f"\n{_sep()}")
    print(f"{BOLD}  Historial de modelos LSI{RESET}")
    print(_sep())
    if lsi_rows:
        for i, r in enumerate(lsi_rows):
            marker = f"{GREEN}● actual{RESET}" if i == 0 else f"{DIM}○ anterior{RESET}"
            var    = f"{r['var_explained']:.1%}" if r["var_explained"] else "—"
            print(f"\n  {marker}")
            print(f"  {'Construido':<20} {DIM}{str(r['built_at'])[:19]}{RESET}")
            print(f"  {'Documentos':<20} {r['n_docs']}")
            print(f"  {'Terminos':<20} {r['n_terms']:,}")
            print(f"  {'k (componentes)':<20} {r['k']}")
            print(f"  {'Varianza explicada':<20} {CYAN}{var}{RESET}")
    else:
        print(f"\n  {YELLOW}Sin modelos LSI construidos todavia.{RESET}")

    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Stats de ChromaDB
# ---------------------------------------------------------------------------

def show_vector_stats(db_path: Path = DB_PATH, chroma_path: Path = CHROMA_PATH) -> None:
    """Muestra el estado de los embeddings y la base de datos vectorial."""
    conn = get_connection(db_path)
    try:
        total_chunks    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        embedded        = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()[0]
        pending         = total_chunks - embedded
        last_embedded   = conn.execute(
            "SELECT c.arxiv_id, d.title, c.embedded_at "
            "FROM chunks c JOIN documents d ON d.arxiv_id = c.arxiv_id "
            "WHERE c.embedded_at IS NOT NULL "
            "ORDER BY c.embedded_at DESC LIMIT 1"
        ).fetchone()
        # Distribucion: chunks por documento
        dist = conn.execute(
            "SELECT arxiv_id, COUNT(*) as n, "
            "SUM(CASE WHEN embedded_at IS NOT NULL THEN 1 ELSE 0 END) as emb "
            "FROM chunks GROUP BY arxiv_id ORDER BY n DESC LIMIT 10"
        ).fetchall()
    finally:
        conn.close()

    try:
        chroma_vectors = chroma_count(chroma_path)
        chroma_ok      = True
    except Exception as exc:
        chroma_vectors = 0
        chroma_ok      = False
        chroma_err     = str(exc)

    print(f"\n{_sep()}")
    print(f"{BOLD}  Base de datos vectorial — ChromaDB{RESET}")
    print(f"{DIM}  Ruta: {chroma_path}{RESET}")
    print(_sep())

    print(f"\n  {BOLD}Estado de embeddings (SQLite){RESET}")
    print(f"  {'Chunks totales':<28} {CYAN}{total_chunks:>7,}{RESET}")
    print(f"  {'Embebidos':<28} {GREEN}{embedded:>7,}{RESET}  {_bar(embedded, total_chunks)} {_pct(embedded, total_chunks)}")
    print(f"  {'Pendientes':<28} {YELLOW}{pending:>7,}{RESET}  {_bar(pending, total_chunks)} {_pct(pending, total_chunks)}")

    print(f"\n  {BOLD}Estado de ChromaDB{RESET}")
    if chroma_ok:
        print(f"  {'Vectores en coleccion':<28} {GREEN}{chroma_vectors:>7,}{RESET}")
        if chroma_vectors == embedded:
            print(f"  {GREEN}✔ SQLite y ChromaDB sincronizados{RESET}")
        else:
            diff = embedded - chroma_vectors
            print(f"  {YELLOW}⚠ Desincronizacion detectada: {diff:+d} chunks{RESET}")
            print(f"  {DIM}  Ejecuta 'embed' en la CLI para sincronizar{RESET}")
    else:
        print(f"  {RED}✘ No se pudo conectar a ChromaDB: {chroma_err[:60]}{RESET}")

    if last_embedded:
        print(f"\n  {BOLD}Ultimo chunk embebido{RESET}")
        print(f"  {GREEN}{last_embedded['arxiv_id']}{RESET}  {str(last_embedded['title'])[:50]}")
        print(f"  {DIM}{str(last_embedded['embedded_at'])[:19]}{RESET}")

    if dist:
        print(f"\n  {BOLD}Top 10 documentos por numero de chunks{RESET}")
        print(f"  {'arxiv_id':<14} {'chunks':>6}  {'embeb.':>6}  progreso")
        print(f"  {'─'*14} {'─'*6}  {'─'*6}  {'─'*20}")
        for row in dist:
            bar   = _bar(row["emb"], row["n"], width=16)
            color = GREEN if row["emb"] == row["n"] else YELLOW if row["emb"] > 0 else RED
            print(f"  {row['arxiv_id']:<14} {row['n']:>6,}  {color}{row['emb']:>6,}{RESET}  {bar}")

    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Documentos con error
# ---------------------------------------------------------------------------

def show_errors(n: int = 20, db_path: Path = DB_PATH) -> None:
    """Lista los documentos que fallaron al descargar o extraer el PDF."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, title, index_error, indexed_at "
            "FROM documents WHERE pdf_downloaded = 2 "
            "ORDER BY indexed_at DESC LIMIT ?",
            (n,),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"\n{_sep()}")
    print(f"{BOLD}  Documentos con error  ({total} total){RESET}")
    print(_sep())
    if not rows:
        print(f"\n  {GREEN}✔ Sin errores.{RESET}\n")
        return
    for row in rows:
        print(f"\n  {RED}{row['arxiv_id']}{RESET}  {DIM}{str(row['indexed_at'])[:19]}{RESET}")
        print(f"  Titulo : {str(row['title'])[:65]}")
        print(f"  Error  : {RED}{str(row['index_error'])[:90]}{RESET}")
    if total > n:
        print(f"\n  {DIM}... y {total - n} mas{RESET}")
    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Documentos pendientes de PDF
# ---------------------------------------------------------------------------

def show_pending(n: int = 20, db_path: Path = DB_PATH) -> None:
    """Lista los documentos que aun no tienen PDF descargado."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, title, categories, published "
            "FROM documents WHERE pdf_downloaded = 0 "
            "ORDER BY published DESC LIMIT ?",
            (n,),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"\n{_sep()}")
    print(f"{BOLD}  Pendientes de PDF  ({total} total){RESET}")
    print(_sep())
    if not rows:
        print(f"\n  {GREEN}✔ Sin documentos pendientes.{RESET}\n")
        return
    for row in rows:
        print(f"\n  {YELLOW}{row['arxiv_id']}{RESET}  {DIM}{str(row['published'])[:10]}{RESET}")
        print(f"  {str(row['title'])[:65]}")
        print(f"  {DIM}{row['categories']}{RESET}")
    if total > n:
        print(f"\n  {DIM}... y {total - n} mas{RESET}")
    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Ultimos N documentos
# ---------------------------------------------------------------------------

def show_recent_docs(n: int = 5, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.arxiv_id, d.title, d.categories, d.published,
                   d.pdf_downloaded, d.text_length, d.indexed_tfidf_at,
                   COUNT(c.id)                                           AS n_chunks,
                   SUM(CASE WHEN c.embedded_at IS NOT NULL THEN 1 ELSE 0 END) AS n_emb
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

    print(f"\n{_sep()}")
    print(f"{BOLD}  Ultimos {n} documentos{RESET}")
    print(_sep())
    for row in rows:
        pdf_status = (
            f"{GREEN}✔ PDF{RESET}"     if row["pdf_downloaded"] == 1 else
            f"{RED}✘ error{RESET}"     if row["pdf_downloaded"] == 2 else
            f"{YELLOW}⏳ pendiente{RESET}"
        )
        tf_status = (
            f"{GREEN}✔ TF{RESET}" if row["indexed_tfidf_at"] else f"{YELLOW}— TF{RESET}"
        )
        emb_status = (
            f"{GREEN}✔ emb{RESET}" if row["n_emb"] == row["n_chunks"] and row["n_chunks"] > 0
            else f"{YELLOW}~emb{RESET}" if row["n_emb"] > 0
            else f"{DIM}— emb{RESET}"
        )
        print(f"\n  {BOLD}{row['arxiv_id']}{RESET}  "
              f"[{pdf_status}] [{tf_status}] [{emb_status}]")
        print(f"  {str(row['title'])[:65]}")
        print(f"  {DIM}{row['categories']}  |  {str(row['published'])[:10]}{RESET}")
        if row["pdf_downloaded"] == 1:
            print(f"  {CYAN}{row['text_length']:,} chars  |  "
                  f"{row['n_chunks']} chunks  |  {row['n_emb']} embebidos{RESET}")
    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Detalle de un documento
# ---------------------------------------------------------------------------

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
            "SELECT id, chunk_index, char_count, embedded_at, "
            "SUBSTR(text, 1, 100) AS preview "
            "FROM chunks WHERE arxiv_id = ? ORDER BY chunk_index",
            (arxiv_id,),
        ).fetchall()
    finally:
        conn.close()

    pdf_status = (
        f"{GREEN}✔ PDF guardado{RESET}"  if doc["pdf_downloaded"] == 1 else
        f"{RED}✘ error{RESET}"           if doc["pdf_downloaded"] == 2 else
        f"{YELLOW}⏳ pendiente{RESET}"
    )
    tf_status = (
        f"{GREEN}✔ indexado TF{RESET}" if doc["indexed_tfidf_at"]
        else f"{YELLOW}pendiente TF{RESET}"
    )

    print(f"\n{_sep()}")
    print(f"{BOLD}  {arxiv_id}{RESET}  [{pdf_status}]  [{tf_status}]")
    print(_sep())
    print(f"  {BOLD}Titulo{RESET}     : {doc['title']}")
    print(f"  {BOLD}Autores{RESET}    : {str(doc['authors'])[:80]}")
    print(f"  {BOLD}Abstract{RESET}   : {str(doc['abstract'])[:180]}…")
    print(f"  {BOLD}Categorias{RESET} : {doc['categories']}")
    print(f"  {BOLD}Publicado{RESET}  : {doc['published']}")
    print(f"  {BOLD}URL{RESET}        : {doc['pdf_url']}")
    print(f"  {BOLD}Descargado{RESET} : {doc['indexed_at']}")
    print(f"  {BOLD}Indexado TF{RESET}: {doc['indexed_tfidf_at'] or '—'}")
    if doc["index_error"]:
        print(f"  {BOLD}Error{RESET}      : {RED}{doc['index_error']}{RESET}")
    if doc["text_length"]:
        print(f"  {BOLD}Texto{RESET}      : {doc['text_length']:,} chars")

    if chunks:
        emb_count = sum(1 for c in chunks if c["embedded_at"])
        print(f"\n  {BOLD}{len(chunks)} chunks  ({emb_count} embebidos){RESET}")
        print(f"  {'idx':>4}  {'chars':>6}  {'estado':<14}  preview")
        print(f"  {'─'*4}  {'─'*6}  {'─'*14}  {'─'*30}")
        for c in chunks:
            emb = f"{GREEN}✔ embebido{RESET}" if c["embedded_at"] else f"{YELLOW}pendiente{RESET}"
            preview = str(c["preview"]).replace("\n", " ")
            print(f"  {c['chunk_index']:>4}  {c['char_count']:>6}  {emb:<14}  {DIM}{preview}…{RESET}")
    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Full text de un documento
# ---------------------------------------------------------------------------

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
        print(f"\n  {YELLOW}{arxiv_id} no tiene texto guardado "
              f"(pdf_downloaded={doc['pdf_downloaded']}){RESET}\n")
        return

    text = doc["full_text"]
    print(f"\n{_sep('═')}")
    print(f"{BOLD}  {arxiv_id} — {str(doc['title'])[:50]}{RESET}")
    print(f"  {CYAN}Total: {doc['text_length']:,} chars{RESET}  "
          f"|  Mostrando primeros {min(chars, len(text)):,}")
    print(_sep("─"))
    print(text[:chars])
    if len(text) > chars:
        print(f"\n{YELLOW}... [{len(text)-chars:,} chars mas — usa --chars N]{RESET}")
    print(f"\n{_sep()}\n")


# ---------------------------------------------------------------------------
# Chunks de un documento
# ---------------------------------------------------------------------------

def show_chunk(arxiv_id: str, chunk_idx: int = 0, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()[0]
        chunk = conn.execute(
            "SELECT chunk_index, char_count, embedded_at, text "
            "FROM chunks WHERE arxiv_id = ? AND chunk_index = ?",
            (arxiv_id, chunk_idx),
        ).fetchone()
    finally:
        conn.close()

    if not chunk:
        print(f"\n  {RED}No encontrado: {arxiv_id} chunk {chunk_idx} "
              f"(total: {total}){RESET}\n")
        return

    emb = f"{GREEN}✔ embebido{RESET}" if chunk["embedded_at"] else f"{YELLOW}pendiente{RESET}"
    print(f"\n{_sep()}")
    print(f"{BOLD}  {arxiv_id}  chunk [{chunk_idx}/{total-1}]  "
          f"({chunk['char_count']} chars)  [{emb}]{RESET}")
    print(_sep("─"))
    print(chunk["text"])
    print(_sep())
    print(f"  Navegar:  --chunk {arxiv_id} --idx {max(0,chunk_idx-1)}  ←  "
          f"actual: {chunk_idx}  →  --chunk {arxiv_id} --idx {min(total-1,chunk_idx+1)}")
    print(f"{_sep()}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve-Engine — Inspeccion de la base de datos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",      type=Path, default=DB_PATH,     help="Ruta a SQLite.")
    p.add_argument("--chroma",  type=Path, default=CHROMA_PATH, help="Directorio de ChromaDB.")
    p.add_argument("--watch",   action="store_true",            help="Refresca el resumen cada 5s.")
    p.add_argument("--docs",    type=int,  metavar="N",         help="Ultimos N documentos.")
    p.add_argument("--doc",     type=str,  metavar="ID",        help="Detalle de un documento.")
    p.add_argument("--text",    type=str,  metavar="ID",        help="Full text de un documento.")
    p.add_argument("--chunk",   type=str,  metavar="ID",        help="Chunks de un documento.")
    p.add_argument("--idx",     type=int,  default=0,           help="Indice del chunk (con --chunk).")
    p.add_argument("--chars",   type=int,  default=3000,        help="Chars a mostrar con --text.")
    p.add_argument("--index",   action="store_true",            help="Stats del indice TF e historial LSI.")
    p.add_argument("--vectors", action="store_true",            help="Stats de embeddings y ChromaDB.")
    p.add_argument("--errors",  action="store_true",            help="Documentos con error de descarga.")
    p.add_argument("--pending", action="store_true",            help="Documentos pendientes de PDF.")
    args = p.parse_args()

    db = args.db
    chroma = args.chroma

    if not db.exists():
        print(f"\n  {RED}La BD no existe todavia: {db}{RESET}")
        print("  Arranca el sistema:  python -m backend.orchestrator\n")
        sys.exit(1)

    if args.text:
        show_full_text(args.text, chars=args.chars, db_path=db)
    elif args.chunk:
        show_chunk(args.chunk, chunk_idx=args.idx, db_path=db)
    elif args.doc:
        show_single_doc(args.doc, db_path=db)
    elif args.docs:
        show_recent_docs(args.docs, db_path=db)
    elif args.index:
        show_index_stats(db_path=db)
    elif args.vectors:
        show_vector_stats(db_path=db, chroma_path=chroma)
    elif args.errors:
        show_errors(db_path=db)
    elif args.pending:
        show_pending(db_path=db)
    elif args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show_summary(db_path=db, chroma_path=chroma)
                print(f"  {DIM}Refrescando cada 5s … (Ctrl+C para salir){RESET}")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Saliendo.\n")
    else:
        show_summary(db_path=db, chroma_path=chroma)


if __name__ == "__main__":
    main()