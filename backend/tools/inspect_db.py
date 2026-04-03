"""
inspect_db.py
=============
Herramienta de inspección completa de la base de datos OmniRetrieve-Engine.

Muestra el estado de todos los subsistemas: crawler, indexación TF,
embedding, índice FAISS y modelo LSI.

Uso
---
    python -m backend.tools.inspect_db                  # resumen completo
    python -m backend.tools.inspect_db --watch          # se refresca cada 5s
    python -m backend.tools.inspect_db --docs 10        # últimos N documentos
    python -m backend.tools.inspect_db --doc 2603.11041 # detalle de un doc
    python -m backend.tools.inspect_db --text 2603.11041 --chars 5000
    python -m backend.tools.inspect_db --chunk 2603.11041 --idx 2
    python -m backend.tools.inspect_db --index          # detalle del índice TF
    python -m backend.tools.inspect_db --embedding      # detalle de embedding+FAISS
    python -m backend.tools.inspect_db --crawl-log 10  # historial del crawler
    python -m backend.tools.inspect_db --errors 20     # documentos con error
    python -m backend.tools.inspect_db --categories    # distribución por categoría
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.schema import DB_PATH, DATA_DIR, get_connection

# ── Colores ANSI ─────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA = "\033[95m"
RESET  = "\033[0m"

W = 58   # ancho de pantalla


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: int, total: int, width: int = 22) -> str:
    """Barra de progreso Unicode."""
    if total == 0:
        return DIM + "─" * width + RESET
    filled = int(width * value / total)
    pct    = value / total * 100
    color  = GREEN if pct >= 80 else YELLOW if pct >= 40 else RED
    return f"{color}{'█' * filled}{RESET}{DIM}{'░' * (width - filled)}{RESET}"


def _pct(value: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{value / total * 100:5.1f}%"


def _file_size(path: Path) -> str:
    """Tamaño legible de un archivo: '4.2 MB', '823 KB', etc."""
    if not path or not path.exists():
        return f"{DIM}no existe{RESET}"
    size = path.stat().st_size
    for unit, threshold in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if size >= threshold:
            return f"{size / threshold:.1f} {unit}"
    return f"{size} B"


def _time_ago(ts_iso: str | None) -> str:
    """Convierte un timestamp ISO a 'hace X min / horas / días'."""
    if not ts_iso:
        return f"{DIM}nunca{RESET}"
    try:
        dt    = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        secs  = int(delta.total_seconds())
        if secs < 60:
            return f"{GREEN}hace {secs}s{RESET}"
        if secs < 3600:
            return f"{GREEN}hace {secs // 60}min{RESET}"
        if secs < 86400:
            return f"{YELLOW}hace {secs // 3600}h{RESET}"
        return f"{YELLOW}hace {secs // 86400}d{RESET}"
    except Exception:
        return ts_iso[:19] if ts_iso else f"{DIM}—{RESET}"


def _fmt_ts(ts_iso: str | None) -> str:
    """Formatea un timestamp ISO para mostrar."""
    if not ts_iso:
        return f"{DIM}—{RESET}"
    return ts_iso[:19].replace("T", " ")


def _sep(char: str = "─", width: int = W) -> str:
    return f"  {DIM}{char * width}{RESET}"


def _header(title: str, icon: str = "") -> None:
    label = f"{icon}  {title}" if icon else title
    print(f"\n  {BOLD}{CYAN}{label}{RESET}")
    print(_sep())


def _row(label: str, value: str, width: int = 26) -> None:
    pad = width - len(label)
    print(f"  {DIM}{label}{RESET}{' ' * max(pad, 1)}{value}")


def _row_bar(label: str, value: int, total: int, color: str = GREEN, width: int = 26) -> None:
    pad  = width - len(label)
    vstr = f"{color}{value:,}{RESET}"
    bar  = _bar(value, total)
    pct  = _pct(value, total)
    print(f"  {DIM}{label}{RESET}{' ' * max(pad, 1)}{vstr:30}  {bar}  {pct}")


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de datos por subsistema
# ─────────────────────────────────────────────────────────────────────────────

def _crawler_data(conn) -> dict:
    total   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=0").fetchone()[0]
    errors  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=2").fetchone()[0]
    avg_txt = conn.execute(
        "SELECT AVG(text_length) FROM documents WHERE pdf_downloaded=1"
    ).fetchone()[0] or 0
    last_ok = conn.execute(
        "SELECT arxiv_id, title, indexed_at FROM documents "
        "WHERE pdf_downloaded=1 ORDER BY indexed_at DESC LIMIT 1"
    ).fetchone()
    last_err = conn.execute(
        "SELECT arxiv_id, index_error, indexed_at FROM documents "
        "WHERE pdf_downloaded=2 ORDER BY indexed_at DESC LIMIT 1"
    ).fetchone()
    last_crawl = conn.execute(
        "SELECT started_at, finished_at, ids_discovered, docs_downloaded, "
        "pdfs_indexed, errors FROM crawl_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(
        total=total, indexed=indexed, pending=pending, errors=errors,
        avg_txt=avg_txt, last_ok=last_ok, last_err=last_err,
        last_crawl=last_crawl,
    )


def _index_data(conn) -> dict:
    try:
        vocab    = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        postings = conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0]
        n_docs   = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM postings").fetchone()[0]
        meta     = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM index_meta")}
        pending  = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded=1 AND indexed_tfidf_at IS NULL"
        ).fetchone()[0]
    except Exception:
        return {}
    return dict(vocab=vocab, postings=postings, n_docs=n_docs,
                meta=meta, pending=pending)


def _embedding_data(conn) -> dict:
    total    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    avg_chars = conn.execute(
        "SELECT AVG(char_count) FROM chunks"
    ).fetchone()[0] or 0

    try:
        last_log = conn.execute(
            "SELECT built_at, index_type, n_vectors, nlist, m, nbits "
            "FROM faiss_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except Exception:
        last_log = None

    try:
        meta = {r[0]: r[1] for r in conn.execute(
            "SELECT key, value FROM embedding_meta"
        )}
    except Exception:
        meta = {}

    return dict(
        total=total, embedded=embedded, pending=total - embedded,
        avg_chars=avg_chars, last_log=last_log, meta=meta,
    )


def _lsi_data() -> dict:
    """Lee metadatos del modelo LSI desde disco (no de la BD)."""
    try:
        from backend.retrieval.lsi_model import MODEL_PATH
        lsi_path = MODEL_PATH
    except Exception:
        lsi_path = DATA_DIR / "models" / "lsi_model.pkl"

    result = {"path": lsi_path, "size": _file_size(lsi_path)}
    if lsi_path.exists():
        try:
            import joblib
            model_data = joblib.load(str(lsi_path))
            result["n_docs"]       = len(model_data.get("doc_ids", []))
            result["k"]            = model_data.get("k", "—")
            result["var_explained"]= model_data.get("var_explained", None)
            result["built_at"]     = model_data.get("built_at", None)
        except Exception:
            pass
    return result


def _faiss_paths() -> tuple[Path, Path]:
    idx  = DATA_DIR / "faiss" / "index.faiss"
    imap = DATA_DIR / "faiss" / "id_map.npy"
    return idx, imap


# ─────────────────────────────────────────────────────────────────────────────
# Vistas principales
# ─────────────────────────────────────────────────────────────────────────────

def show_summary(db_path: Path = DB_PATH) -> None:
    """Resumen completo de todos los subsistemas."""
    conn = get_connection(db_path)
    try:
        cr  = _crawler_data(conn)
        idx = _index_data(conn)
        emb = _embedding_data(conn)
    finally:
        conn.close()

    lsi        = _lsi_data()
    faiss_idx, faiss_map = _faiss_paths()
    db_size    = _file_size(db_path)
    now_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  OmniRetrieve-Engine — Inspección de BD{RESET}")
    print(f"  {DIM}{db_path}  ({db_size}){RESET}")
    print(f"  {DIM}{now_str}{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}")

    # ── Crawler ──────────────────────────────────────────────────────────────
    _header("CRAWLER", "🕷️")
    _row_bar("Con PDF guardado", cr["indexed"], cr["total"], GREEN)
    _row_bar("Pendientes",       cr["pending"], cr["total"], YELLOW)
    _row_bar("Con error",        cr["errors"],  cr["total"], RED)
    _row("Total descubiertos",   f"{BOLD}{cr['total']:,}{RESET}")
    _row("Texto promedio / doc", f"{cr['avg_txt']:,.0f} chars")
    if cr["last_ok"]:
        _row("Último guardado",
             f"{GREEN}{cr['last_ok']['arxiv_id']}{RESET}  "
             f"({_time_ago(cr['last_ok']['indexed_at'])})")
    if cr["last_err"]:
        _row("Último error",
             f"{RED}{cr['last_err']['arxiv_id']}{RESET}  "
             f"({_time_ago(cr['last_err']['indexed_at'])})")
    if cr["last_crawl"]:
        lc = cr["last_crawl"]
        _row("Último ciclo crawl",
             f"{_fmt_ts(lc['started_at'])}  "
             f"descubiertos: {lc['ids_discovered']}  "
             f"descargados: {lc['docs_downloaded']}  "
             f"errores: {lc['errors']}")

    # ── Indexación TF ────────────────────────────────────────────────────────
    _header("INDEXACIÓN TF (índice invertido)", "📑")
    if idx:
        total_pdf = cr["indexed"]
        _row_bar("Docs indexados",    idx["n_docs"],  total_pdf, GREEN)
        _row_bar("Pendientes TF",     idx["pending"], total_pdf, YELLOW)
        _row("Vocabulario",           f"{idx['vocab']:,} términos")
        _row("Postings",              f"{idx['postings']:,}")
        _row("Última indexación",     _fmt_ts(idx["meta"].get("last_run_at")))
    else:
        _row("Estado", f"{YELLOW}Sin datos de indexación{RESET}")

    # ── Embedding ────────────────────────────────────────────────────────────
    _header("EMBEDDING", "🔢")
    if emb["total"] > 0:
        _row_bar("Embedidos",    emb["embedded"], emb["total"], GREEN)
        _row_bar("Pendientes",   emb["pending"],  emb["total"], YELLOW)
        _row("Total chunks",     f"{emb['total']:,}")
        _row("Chars prom/chunk", f"{emb['avg_chars']:,.0f}")
        _row("Modelo",           emb["meta"].get("model_name", f"{DIM}desconocido{RESET}"))
        _row("Último embedding", _fmt_ts(emb["meta"].get("last_run_at")))
    else:
        _row("Estado", f"{YELLOW}Sin chunks en la BD{RESET}")

    # ── FAISS ────────────────────────────────────────────────────────────────
    _header("ÍNDICE FAISS", "⚡")
    if emb.get("last_log"):
        ll = emb["last_log"]
        _row("Tipo",             f"{BOLD}{ll['index_type']}{RESET}")
        _row("Vectores",         f"{ll['n_vectors']:,}")
        if ll["nlist"]:
            _row("nlist / m / nbits", f"{ll['nlist']} / {ll['m']} / {ll['nbits']}")
        _row("Última build",     _time_ago(ll["built_at"]))
        _row("Archivo índice",   _file_size(faiss_idx))
        _row("Archivo id_map",   _file_size(faiss_map))
    else:
        _row("Estado", f"{YELLOW}Índice FAISS no construido todavía{RESET}")

    # ── LSI ──────────────────────────────────────────────────────────────────
    _header("MODELO LSI", "🧠")
    if lsi.get("n_docs"):
        var = lsi.get("var_explained")
        var_str = f"{var * 100:.1f}%" if var else f"{DIM}—{RESET}"
        _row("Documentos en modelo", f"{lsi['n_docs']:,}")
        _row("k (componentes)",      str(lsi.get("k", "—")))
        _row("Varianza explicada",   var_str)
        _row("Última build",         _time_ago(lsi.get("built_at")))
        _row("Tamaño modelo",        lsi["size"])
    else:
        _row("Estado", f"{YELLOW}Modelo LSI no encontrado{RESET}")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"  {DIM}Comandos útiles:{RESET}")
    print(f"  {DIM}  --index       detalle del índice TF{RESET}")
    print(f"  {DIM}  --embedding   detalle de embedding y FAISS{RESET}")
    print(f"  {DIM}  --crawl-log N historial del crawler{RESET}")
    print(f"  {DIM}  --errors N    documentos con error{RESET}")
    print(f"  {DIM}  --categories  distribución por categoría{RESET}")
    print(f"  {DIM}  --watch       refresco automático{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_index_stats(db_path: Path = DB_PATH) -> None:
    """Detalle del índice invertido TF."""
    conn = get_connection(db_path)
    try:
        idx = _index_data(conn)
        top_terms = conn.execute(
            "SELECT word, df FROM terms ORDER BY df DESC LIMIT 20"
        ).fetchall()
        docs_by_postings = conn.execute(
            "SELECT doc_id, COUNT(*) as n_terms FROM postings "
            "GROUP BY doc_id ORDER BY n_terms DESC LIMIT 5"
        ).fetchall()
        avg_postings = conn.execute(
            "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM postings GROUP BY doc_id)"
        ).fetchone()[0] or 0
    finally:
        conn.close()

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Índice Invertido TF — Detalle{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    if not idx:
        print(f"  {YELLOW}Sin datos de indexación todavía.{RESET}\n")
        return

    _row("Docs indexados",     f"{idx['n_docs']:,}")
    _row("Pendientes TF",      f"{YELLOW}{idx['pending']:,}{RESET}")
    _row("Vocabulario",        f"{idx['vocab']:,} términos")
    _row("Total postings",     f"{idx['postings']:,}")
    _row("Términos prom/doc",  f"{avg_postings:,.1f}")
    _row("Última indexación",  _fmt_ts(idx["meta"].get("last_run_at")))
    _row("Docs en última run", idx["meta"].get("last_docs_indexed", "—"))
    _row("Términos añadidos",  idx["meta"].get("last_terms_added",  "—"))
    _row("Postings añadidos",  idx["meta"].get("last_postings_added","—"))

    if top_terms:
        print(f"\n  {BOLD}Top 20 términos por document-frequency{RESET}")
        print(_sep())
        max_df = top_terms[0]["df"] if top_terms else 1
        for t in top_terms:
            bar = _bar(t["df"], max_df, width=16)
            print(f"  {CYAN}{t['word']:<22}{RESET}  df={t['df']:>6,}  {bar}")

    if docs_by_postings:
        print(f"\n  {BOLD}Top 5 documentos por riqueza léxica{RESET}")
        print(_sep())
        for r in docs_by_postings:
            print(f"  {r['doc_id']:<20}  {r['n_terms']:,} términos distintos")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_embedding_stats(db_path: Path = DB_PATH) -> None:
    """Detalle del módulo de embedding y del índice FAISS."""
    conn = get_connection(db_path)
    try:
        emb = _embedding_data(conn)
        faiss_history = []
        try:
            faiss_history = conn.execute(
                "SELECT built_at, n_vectors, index_type, nlist, m, nbits "
                "FROM faiss_log ORDER BY id DESC LIMIT 10"
            ).fetchall()
        except Exception:
            pass
        # Distribución de char_count por chunks
        size_dist = conn.execute(
            """
            SELECT
              SUM(CASE WHEN char_count <  500             THEN 1 ELSE 0 END) as lt500,
              SUM(CASE WHEN char_count >= 500  AND char_count < 1000  THEN 1 ELSE 0 END) as s1k,
              SUM(CASE WHEN char_count >= 1000 AND char_count < 2000  THEN 1 ELSE 0 END) as s2k,
              SUM(CASE WHEN char_count >= 2000              THEN 1 ELSE 0 END) as gte2k
            FROM chunks
            """
        ).fetchone()
    finally:
        conn.close()

    faiss_idx, faiss_map = _faiss_paths()

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Embedding y FAISS — Detalle{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    # Chunks
    _header("CHUNKS", "🧩")
    _row("Total chunks",      f"{emb['total']:,}")
    _row_bar("Embedidos",     emb["embedded"], emb["total"], GREEN)
    _row_bar("Pendientes",    emb["pending"],  emb["total"], YELLOW)
    _row("Chars prom/chunk",  f"{emb['avg_chars']:,.0f}")

    if emb["total"] > 0 and size_dist:
        print(f"\n  {BOLD}Distribución de tamaño{RESET}")
        print(_sep())
        for label, val in [
            ("<500 chars",    size_dist["lt500"]),
            ("500–1000",      size_dist["s1k"]),
            ("1000–2000",     size_dist["s2k"]),
            ("≥2000 chars",   size_dist["gte2k"]),
        ]:
            bar = _bar(val or 0, emb["total"], width=20)
            print(f"  {label:<16}  {val or 0:>6,}  {bar}  {_pct(val or 0, emb['total'])}")

    # Modelo
    _header("MODELO DE EMBEDDING", "🤖")
    _row("Modelo",            emb["meta"].get("model_name", f"{DIM}—{RESET}"))
    _row("Último embedding",  _fmt_ts(emb["meta"].get("last_run_at")))
    _row("Chunks procesados", emb["meta"].get("last_chunks_embedded", "—"))

    # FAISS
    _header("ÍNDICE FAISS", "⚡")
    _row("Archivo índice",    f"{faiss_idx}  ({_file_size(faiss_idx)})")
    _row("Archivo id_map",    f"{faiss_map}  ({_file_size(faiss_map)})")

    if emb.get("last_log"):
        ll = emb["last_log"]
        _row("Tipo activo",   f"{BOLD}{ll['index_type']}{RESET}")
        _row("Vectores",      f"{ll['n_vectors']:,}")
        if ll["nlist"]:
            _row("nlist / m / nbits", f"{ll['nlist']} / {ll['m']} / {ll['nbits']}")

    if faiss_history:
        print(f"\n  {BOLD}Historial de builds FAISS (últimos {len(faiss_history)}){RESET}")
        print(_sep())
        print(f"  {DIM}{'Fecha':<22}{'Tipo':<18}{'Vectores':>10}{'nlist':>8}{'m':>5}{'nbits':>7}{RESET}")
        print(_sep("·"))
        for r in faiss_history:
            nlist = str(r["nlist"]) if r["nlist"] else "—"
            m     = str(r["m"])     if r["m"]     else "—"
            nbits = str(r["nbits"]) if r["nbits"] else "—"
            print(f"  {_fmt_ts(r['built_at']):<22}"
                  f"{r['index_type']:<18}"
                  f"{r['n_vectors']:>10,}"
                  f"{nlist:>8}{m:>5}{nbits:>7}")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_crawl_log(n: int = 10, db_path: Path = DB_PATH) -> None:
    """Historial de ejecuciones del crawler."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT started_at, finished_at, ids_discovered, docs_downloaded, "
            "pdfs_indexed, errors, notes FROM crawl_log ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
    finally:
        conn.close()

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Historial del Crawler — últimas {n} ejecuciones{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    if not rows:
        print(f"  {YELLOW}Sin registros de crawl todavía.{RESET}\n")
        return

    print(f"  {DIM}{'Inicio':<22}{'Dur.':<8}{'IDs':>6}{'Docs':>6}{'PDFs':>6}{'Err':>5}{RESET}")
    print(_sep("·"))
    for r in rows:
        # Duración
        dur = "—"
        if r["started_at"] and r["finished_at"]:
            try:
                t0 = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00"))
                s  = int((t1 - t0).total_seconds())
                dur = f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"
            except Exception:
                pass

        err_color = RED if r["errors"] else RESET
        print(f"  {_fmt_ts(r['started_at']):<22}"
              f"{dur:<8}"
              f"{r['ids_discovered']:>6,}"
              f"{r['docs_downloaded']:>6,}"
              f"{r['pdfs_indexed']:>6,}"
              f"{err_color}{r['errors']:>5}{RESET}")
        if r["notes"]:
            print(f"  {DIM}  {r['notes'][:70]}{RESET}")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_errors(n: int = 20, db_path: Path = DB_PATH) -> None:
    """Documentos que fallaron durante la descarga o extracción de PDF."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, title, indexed_at, index_error "
            "FROM documents WHERE pdf_downloaded=2 "
            "ORDER BY indexed_at DESC LIMIT ?",
            (n,)
        ).fetchall()
        total_errors = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE pdf_downloaded=2"
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Documentos con error — {total_errors} total{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    if not rows:
        print(f"  {GREEN}Sin errores registrados. ✅{RESET}\n")
        return

    for r in rows:
        print(f"  {RED}{BOLD}{r['arxiv_id']}{RESET}  {DIM}{_fmt_ts(r['indexed_at'])}{RESET}")
        print(f"  {DIM}{r['title'][:65]}{RESET}")
        print(f"  {RED}{r['index_error'][:90]}{RESET}\n")

    if total_errors > n:
        print(f"  {DIM}… y {total_errors - n} errores más.{RESET}\n")

    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_categories(db_path: Path = DB_PATH) -> None:
    """Distribución de documentos por categoría arXiv."""
    conn = get_connection(db_path)
    try:
        # Expandir la lista de categorías separadas por coma
        rows = conn.execute(
            "SELECT categories FROM documents WHERE categories IS NOT NULL AND categories != ''"
        ).fetchall()
        total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()

    # Contar por categoría raíz (ej: "cs.LG" → "cs")
    from collections import Counter
    cat_full  = Counter()
    cat_root  = Counter()
    for r in rows:
        for cat in r["categories"].split(","):
            cat = cat.strip()
            if cat:
                cat_full[cat] += 1
                cat_root[cat.split(".")[0]] += 1

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Distribución por Categoría arXiv{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    print(f"  {BOLD}Por área (raíz){RESET}")
    print(_sep())
    max_r = cat_root.most_common(1)[0][1] if cat_root else 1
    for cat, count in cat_root.most_common(15):
        bar = _bar(count, max_r, width=18)
        print(f"  {CYAN}{cat:<10}{RESET}  {count:>6,}  {bar}")

    print(f"\n  {BOLD}Top 20 subcategorías{RESET}")
    print(_sep())
    max_f = cat_full.most_common(1)[0][1] if cat_full else 1
    for cat, count in cat_full.most_common(20):
        bar = _bar(count, max_f, width=18)
        print(f"  {CYAN}{cat:<16}{RESET}  {count:>6,}  {bar}")

    print(f"\n  {DIM}Total docs analizados: {total_docs:,}  |  "
          f"Categorías únicas: {len(cat_full)}{RESET}")
    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_recent_docs(n: int = 5, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.arxiv_id, d.title, d.categories, d.published,
                   d.pdf_downloaded, d.text_length,
                   COUNT(c.id) as n_chunks,
                   SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as n_embedded
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

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  Últimos {n} documentos{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")

    for row in rows:
        status = (
            f"{GREEN}✅ guardado{RESET}"   if row["pdf_downloaded"] == 1 else
            f"{RED}❌ error{RESET}"        if row["pdf_downloaded"] == 2 else
            f"{YELLOW}⏳ pendiente{RESET}"
        )
        print(f"  {BOLD}{row['arxiv_id']}{RESET}  [{status}]")
        print(f"  {DIM}Título     :{RESET} {row['title'][:68]}")
        print(f"  {DIM}Categorías :{RESET} {row['categories']}")
        print(f"  {DIM}Publicado  :{RESET} {str(row['published'])[:10]}")
        if row["pdf_downloaded"] == 1:
            emb_info = ""
            if row["n_chunks"]:
                pct = _pct(row["n_embedded"] or 0, row["n_chunks"])
                emb_info = f" | embedidos: {row['n_embedded'] or 0}/{row['n_chunks']} ({pct})"
            print(f"  {DIM}Contenido  :{RESET} {row['text_length']:,} chars | "
                  f"{row['n_chunks']} chunks{emb_info}")
        print()

    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")


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
            "SELECT id, chunk_index, char_count, text, "
            "CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END as has_emb "
            "FROM chunks WHERE arxiv_id = ? ORDER BY chunk_index",
            (arxiv_id,),
        ).fetchall()
    finally:
        conn.close()

    status = (
        f"{GREEN}✅ guardado{RESET}"   if doc["pdf_downloaded"] == 1 else
        f"{RED}❌ error{RESET}"        if doc["pdf_downloaded"] == 2 else
        f"{YELLOW}⏳ pendiente{RESET}"
    )
    n_emb = sum(c["has_emb"] for c in chunks)

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  {arxiv_id}{RESET}  [{status}]")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")
    _row("Título",          doc["title"])
    _row("Autores",         str(doc["authors"] or "—")[:75])
    _row("Categorías",      str(doc["categories"] or "—"))
    _row("Publicado",       str(doc["published"] or "—")[:10])
    _row("URL",             str(doc["pdf_url"] or "—"))
    _row("Descubierto",     _fmt_ts(doc["fetched_at"]))
    _row("PDF indexado",    _fmt_ts(doc["indexed_at"]))
    _row("TF indexado",     _fmt_ts(doc["indexed_tfidf_at"]))
    if doc["index_error"]:
        _row("Error",       f"{RED}{doc['index_error'][:80]}{RESET}")
    if doc["text_length"]:
        _row("Texto",       f"{doc['text_length']:,} chars")

    if chunks:
        emb_bar = _bar(n_emb, len(chunks))
        print(f"\n  {BOLD}Chunks: {len(chunks)}  |  Embedidos: {n_emb}/{len(chunks)}  {emb_bar}{RESET}")
        print(_sep())
        for c in chunks[:5]:
            emb_icon = f"{GREEN}●{RESET}" if c["has_emb"] else f"{YELLOW}○{RESET}"
            preview  = (c["text"] or "")[:100].replace("\n", " ")
            print(f"  {emb_icon} [{c['chunk_index']:>3}]  "
                  f"{DIM}({c['char_count']} chars){RESET}  {preview}…")
        if len(chunks) > 5:
            print(f"  {DIM}  … y {len(chunks) - 5} chunks más{RESET}")

    print(f"\n  {DIM}Abstract:{RESET}")
    print(f"  {str(doc['abstract'] or '—')[:300]}")
    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


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
    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  {arxiv_id} — {doc['title'][:55]}{RESET}")
    print(f"  {CYAN}Total: {doc['text_length']:,} chars{RESET}"
          f"  |  Mostrando primeros {min(chars, len(text)):,}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")
    print(text[:chars])
    if len(text) > chars:
        print(f"\n{YELLOW}  … [{len(text) - chars:,} chars más — usa --chars N para ver más]{RESET}")
    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}\n")


def show_chunk_text(arxiv_id: str, chunk_idx: int = 0, db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()[0]
        chunk = conn.execute(
            "SELECT chunk_index, char_count, text, "
            "CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END as has_emb, "
            "embedded_at "
            "FROM chunks WHERE arxiv_id = ? AND chunk_index = ?",
            (arxiv_id, chunk_idx),
        ).fetchone()
    finally:
        conn.close()

    if not chunk:
        print(f"\n  {RED}No encontrado: {arxiv_id} chunk {chunk_idx} "
              f"(total: {total}){RESET}\n")
        return

    emb_status = (f"{GREEN}● embedido{RESET}  {DIM}({_fmt_ts(chunk['embedded_at'])}){RESET}"
                  if chunk["has_emb"] else f"{YELLOW}○ sin embedding{RESET}")

    print(f"\n{BOLD}{'═' * (W + 4)}{RESET}")
    print(f"{BOLD}  {arxiv_id}  chunk [{chunk_idx}/{total - 1}]  "
          f"({chunk['char_count']} chars)  {emb_status}{RESET}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")
    print(chunk["text"])
    print(f"\n{_sep()}")
    print(f"  Navegar:  --chunk {arxiv_id} --idx {max(0, chunk_idx-1)}  ←  "
          f"{chunk_idx}  →  --idx {min(total - 1, chunk_idx + 1)}")
    print(f"{BOLD}{'═' * (W + 4)}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="OmniRetrieve — Inspección completa de la BD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",         type=Path, default=DB_PATH, help="Ruta a la BD SQLite")
    p.add_argument("--docs",       type=int,  metavar="N",  help="Últimos N documentos")
    p.add_argument("--doc",        type=str,  metavar="ID", help="Detalle de un documento")
    p.add_argument("--text",       type=str,  metavar="ID", help="full_text de un documento")
    p.add_argument("--chars",      type=int,  default=3000, help="Chars a mostrar con --text")
    p.add_argument("--chunk",      type=str,  metavar="ID", help="Texto de un chunk concreto")
    p.add_argument("--idx",        type=int,  default=0,    help="Índice del chunk (con --chunk)")
    p.add_argument("--index",      action="store_true",     help="Detalle del índice invertido TF")
    p.add_argument("--embedding",  action="store_true",     help="Detalle de embedding y FAISS")
    p.add_argument("--crawl-log",  type=int,  metavar="N",  help="Historial del crawler (últimas N runs)")
    p.add_argument("--errors",     type=int,  metavar="N",  help="Documentos con error de descarga")
    p.add_argument("--categories", action="store_true",     help="Distribución por categoría")
    p.add_argument("--watch",      action="store_true",     help="Refresco automático cada 5s")
    args = p.parse_args()

    db = args.db
    if not db.exists():
        print(f"\n  {RED}La BD no existe todavía: {db}{RESET}")
        print("  Arranca el crawler primero:  python -m backend.main\n")
        sys.exit(1)

    if args.text:
        show_full_text(args.text, chars=args.chars, db_path=db)
    elif args.chunk:
        show_chunk_text(args.chunk, chunk_idx=args.idx, db_path=db)
    elif args.doc:
        show_single_doc(args.doc, db_path=db)
    elif args.docs:
        show_recent_docs(args.docs, db_path=db)
    elif args.index:
        show_index_stats(db_path=db)
    elif args.embedding:
        show_embedding_stats(db_path=db)
    elif getattr(args, "crawl_log", None):
        show_crawl_log(args.crawl_log, db_path=db)
    elif args.errors:
        show_errors(args.errors, db_path=db)
    elif args.categories:
        show_categories(db_path=db)
    elif args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                show_summary(db_path=db)
                print("  Actualizando cada 5s … (Ctrl+C para salir)")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Saliendo.\n")
    else:
        show_summary(db_path=db)


if __name__ == "__main__":
    main()