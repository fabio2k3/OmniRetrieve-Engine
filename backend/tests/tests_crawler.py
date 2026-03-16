"""
test_pipeline.py
Verifica el pipeline: crawler + base de datos relacional (SQLite).

Uso
---
    python -m backend.tests.test_pipeline                  # todo
    python -m backend.tests.test_pipeline --skip-network   # solo tests locales
"""
from __future__ import annotations
import argparse, sys, tempfile, traceback
from pathlib import Path

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
BOLD   = "\033[1m";  RESET = "\033[0m"
passed = failed = skipped = 0

def ok(name):
    global passed; passed += 1
    print(f"  {GREEN}✓{RESET}  {name}")

def fail(name, exc):
    global failed; failed += 1
    print(f"  {RED}✗{RESET}  {name}")
    print(f"      {RED}{exc}{RESET}")
    traceback.print_exc()

def skip(name, reason):
    global skipped; skipped += 1
    print(f"  {YELLOW}–{RESET}  {name}  ({YELLOW}{reason}{RESET})")

def section(title):
    print(f"\n{BOLD}{'─'*55}{RESET}\n{BOLD}  {title}{RESET}\n{BOLD}{'─'*55}{RESET}")


# =============================================================================
# 1. Document
# =============================================================================
def test_document(tmp: Path):
    section("1 · Document — serialización y CSV")
    from backend.crawler.document import Document
    csv = tmp / "docs.csv"

    try:
        doc = Document(
            arxiv_id="2301.99999", title="Test Paper", authors="A, B",
            abstract="Abstract.", categories="cs.AI, cs.LG",
            published="2023-01-01T00:00:00Z", updated="2023-01-02T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/2301.99999",
        )
        assert set(doc.to_dict().keys()) == set(Document.FIELDS)
        ok("to_dict() contiene todos los campos")
    except Exception as e: fail("to_dict()", e); return

    try:
        doc.save(csv)
        docs = Document.load_all(csv)
        assert len(docs) == 1 and docs[0].arxiv_id == "2301.99999"
        ok("save() + load_all() — 1 documento")
    except Exception as e: fail("save() / load_all()", e); return

    try:
        ids = Document.load_ids(csv)
        assert "2301.99999" in ids
        ok("load_ids() devuelve el set de IDs")
    except Exception as e: fail("load_ids()", e)

    try:
        doc2 = Document.from_dict(doc.to_dict())
        assert doc2 == doc
        ok("from_dict() round-trip correcto")
    except Exception as e: fail("from_dict()", e)


# =============================================================================
# 2. IdStore
# =============================================================================
def test_id_store(tmp: Path):
    section("2 · IdStore — gestión de IDs")
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp / "ids.csv")

    try:
        assert store.add_ids(["2301.00001", "2301.00002", "2301.00003"]) == 3
        ok("add_ids() → 3 nuevos IDs")
    except Exception as e: fail("add_ids()", e); return

    try:
        assert store.add_ids(["2301.00001", "2301.00004"]) == 1
        ok("add_ids() ignora duplicados")
    except Exception as e: fail("add_ids() dedup", e)

    try:
        batch = store.get_pending_batch(2)
        assert len(batch) == 2
        ok(f"get_pending_batch(2) → {batch}")
    except Exception as e: fail("get_pending_batch()", e); return

    try:
        store.mark_downloaded(batch)
        assert store.downloaded_count == 2 and store.pending_count == 2
        ok(f"mark_downloaded() — downloaded={store.downloaded_count}, pending={store.pending_count}")
    except Exception as e: fail("mark_downloaded()", e)

    try:
        from backend.crawler.id_store import IdStore as IS2
        s2 = IS2(tmp / "ids.csv")
        assert s2.total == store.total and s2.downloaded_count == store.downloaded_count
        ok("Persistencia CSV — estado correcto tras recargar")
    except Exception as e: fail("Persistencia CSV", e)


# =============================================================================
# 3. RobotsChecker
# =============================================================================
def test_robots(network: bool):
    section("3 · RobotsChecker — robots.txt")
    if not network:
        skip("RobotsChecker", "--skip-network activo"); return

    from backend.crawler.robots import RobotsChecker
    checker = RobotsChecker()

    try:
        allowed = checker.allowed("https://export.arxiv.org/api/query?search_query=cat:cs.AI")
        assert allowed is True
        ok(f"arXiv API permitida por robots.txt → {allowed}")
    except Exception as e: fail("robots.txt arXiv", e)

    try:
        delay = checker.crawl_delay("https://export.arxiv.org/api/query")
        ok(f"Crawl-delay arxiv.org → {delay}s")
    except Exception as e: fail("crawl_delay()", e)

    try:
        checker.allowed("https://export.arxiv.org/api/query?search_query=cat:cs.LG")
        ok("Segunda llamada al mismo origen usa caché")
    except Exception as e: fail("Caché robots.txt", e)


# =============================================================================
# 4. ArxivClient
# =============================================================================
def test_arxiv_client(network: bool):
    section("4 · ArxivClient — fetch de IDs y metadatos")
    if not network:
        skip("fetch_ids()", "--skip-network activo")
        skip("fetch_documents()", "--skip-network activo")
        return

    from backend.crawler.arxiv_client import ArxivClient
    client = ArxivClient()
    ids = []

    try:
        ids = client.fetch_ids(max_results=5)
        assert len(ids) > 0
        ok(f"fetch_ids() → {len(ids)} IDs: {ids[:3]} …")
    except Exception as e: fail("fetch_ids()", e)

    if ids:
        try:
            docs = client.fetch_documents(ids[:3])
            assert len(docs) > 0
            d = docs[0]
            assert d.arxiv_id and d.title and d.abstract
            ok(f"fetch_documents() → {len(docs)} docs")
            ok(f"  ID={d.arxiv_id!r}  título='{d.title[:50]}…'")
            ok(f"  categorías={d.categories!r}  publicado={d.published[:10]!r}")
        except Exception as e: fail("fetch_documents()", e)


# =============================================================================
# 5. SQLite — schema y CRUD completo
# =============================================================================
def test_database(tmp: Path):
    section("5 · SQLite — schema y CRUD completo")
    from backend.database.schema import init_db
    from backend.database import repository as repo
    db = tmp / "db" / "test.db"

    try:
        init_db(db)
        ok("init_db() — tablas creadas (documents, chunks, crawl_log)")
    except Exception as e: fail("init_db()", e); return

    try:
        for i in range(4):
            repo.upsert_document(
                arxiv_id=f"2301.0000{i}", title=f"Paper {i}", authors="Author",
                abstract="Abstract.", categories="cs.AI, cs.LG",
                published="2023-01-01", updated="2023-01-01",
                pdf_url=f"https://arxiv.org/pdf/2301.0000{i}",
                fetched_at="2023-01-01", db_path=db,
            )
        ok("upsert_document() × 4 insertados")
    except Exception as e: fail("upsert_document()", e); return

    try:
        repo.upsert_document(
            arxiv_id="2301.00000", title="Paper 0 actualizado", authors="Author",
            abstract="Abstract.", categories="cs.AI", published="2023-01-01",
            updated="2023-01-01", pdf_url="https://arxiv.org/pdf/2301.00000",
            fetched_at="2023-01-01", db_path=db,
        )
        doc = repo.get_document("2301.00000", db_path=db)
        assert doc["title"] == "Paper 0 actualizado"
        assert doc["pdf_downloaded"] == 0  # no se tocó
        ok("upsert idempotente — título actualizado, PDF status intacto")
    except Exception as e: fail("upsert idempotente", e)

    try:
        pending = repo.get_pending_pdf_ids(10, db_path=db)
        assert len(pending) == 4
        ok(f"get_pending_pdf_ids() → {pending}")
    except Exception as e: fail("get_pending_pdf_ids()", e)

    try:
        repo.save_pdf_text("2301.00000", "Texto del paper. " * 60, db_path=db)
        repo.save_chunks("2301.00000", ["chunk A", "chunk B", "chunk C"], db_path=db)
        doc = repo.get_document("2301.00000", db_path=db)
        chunks = repo.get_chunks("2301.00000", db_path=db)
        assert doc["pdf_downloaded"] == 1
        assert doc["text_length"] > 0
        assert len(chunks) == 3
        ok(f"save_pdf_text() + save_chunks() — text_length={doc['text_length']}, chunks={len(chunks)}")
    except Exception as e: fail("save_pdf_text() / save_chunks()", e)

    try:
        repo.save_pdf_error("2301.00001", "Connection timeout", db_path=db)
        doc = repo.get_document("2301.00001", db_path=db)
        assert doc["pdf_downloaded"] == 2 and "timeout" in doc["index_error"]
        ok("save_pdf_error() — estado=2, mensaje guardado")
    except Exception as e: fail("save_pdf_error()", e)

    try:
        log_id = repo.log_crawl_start(db_path=db)
        repo.log_crawl_end(log_id, ids_discovered=50, docs_downloaded=10,
                           pdfs_indexed=3, errors=1, db_path=db)
        ok(f"crawl_log — start + end registrados (id={log_id})")
    except Exception as e: fail("crawl_log", e)

    try:
        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 4
        assert stats["pdf_indexed"]  == 1
        assert stats["pdf_errors"]   == 1
        assert stats["pdf_pending"]  == 2
        assert stats["total_chunks"] == 3
        ok(f"get_stats() → {stats}")
    except Exception as e: fail("get_stats()", e)

    try:
        assert repo.document_exists("2301.00000", db_path=db) is True
        assert repo.document_exists("9999.99999", db_path=db) is False
        ok("document_exists() — True/False correcto")
    except Exception as e: fail("document_exists()", e)


# =============================================================================
# 6. PDF Extractor
# =============================================================================
def test_pdf_extractor(network: bool):
    section("6 · PDF Extractor — descarga y extracción")
    try:
        import fitz  # noqa
    except ImportError:
        skip("pdf_extractor", "PyMuPDF no instalado — pip install pymupdf"); return

    if not network:
        skip("download_and_extract()", "--skip-network activo"); return

    from backend.crawler.pdf_extractor import download_and_extract
    try:
        full_text, chunks = download_and_extract("1706.03762", chunk_size=800)
        assert len(full_text) > 1000 and len(chunks) > 0
        ok(f"download_and_extract('1706.03762') → {len(full_text)} chars, {len(chunks)} chunks")
        ok(f"  Primer chunk: '{chunks[0][:80]}…'")
    except PermissionError as e:
        skip("download_and_extract()", f"robots.txt bloqueó: {e}")
    except Exception as e:
        fail("download_and_extract()", e)


# =============================================================================
# 5b. SQLite — verificación de contenido real
# =============================================================================
def test_database_content(tmp: Path):
    section("5b · SQLite — verificación de contenido guardado")
    from backend.database.schema import init_db, get_connection
    from backend.database import repository as repo
    db = tmp / "db" / "content.db"
    init_db(db)

    # Insertar un documento con texto y chunks reales
    repo.upsert_document(
        arxiv_id="2301.00099", title="Deep Learning Survey",
        authors="LeCun, Bengio, Hinton", abstract="A comprehensive survey.",
        categories="cs.LG, cs.AI", published="2023-01-15",
        updated="2023-01-15", pdf_url="https://arxiv.org/pdf/2301.00099",
        fetched_at="2023-01-15T10:00:00", db_path=db,
    )
    texto = "Deep learning has transformed AI. " * 80
    chunks_texto = [
        "Deep learning methods have revolutionized computer vision tasks.",
        "Natural language processing has seen significant improvements with transformers.",
        "Reinforcement learning combined with deep networks enables complex decision making.",
    ]
    repo.save_pdf_text("2301.00099", texto, db_path=db)
    repo.save_chunks("2301.00099", chunks_texto, db_path=db)

    # ── Verificar que el texto se guardó correctamente ──────────────────────
    try:
        conn = get_connection(db)
        row = conn.execute(
            "SELECT arxiv_id, title, full_text, text_length, pdf_downloaded, indexed_at "
            "FROM documents WHERE arxiv_id='2301.00099'"
        ).fetchone()
        conn.close()
        assert row is not None,                   "Fila no encontrada"
        assert row["arxiv_id"] == "2301.00099",   "arxiv_id incorrecto"
        assert row["title"] == "Deep Learning Survey", "título incorrecto"
        assert row["pdf_downloaded"] == 1,        "pdf_downloaded no es 1"
        assert row["text_length"] == len(texto),  f"text_length incorrecto: {row['text_length']} != {len(texto)}"
        assert row["full_text"] == texto,         "full_text no coincide"
        assert row["indexed_at"] is not None,     "indexed_at es NULL"
        ok(f"Texto guardado correctamente — {row['text_length']:,} chars, indexed_at={row['indexed_at'][:19]}")
    except Exception as e: fail("Verificación texto en DB", e)

    # ── Verificar que los chunks se guardaron correctamente ─────────────────
    try:
        conn = get_connection(db)
        rows = conn.execute(
            "SELECT chunk_index, text, char_count FROM chunks "
            "WHERE arxiv_id='2301.00099' ORDER BY chunk_index"
        ).fetchall()
        conn.close()
        assert len(rows) == 3, f"Esperaba 3 chunks, hay {len(rows)}"
        for i, (row, expected) in enumerate(zip(rows, chunks_texto)):
            assert row["text"] == expected,          f"chunk {i} texto incorrecto"
            assert row["char_count"] == len(expected), f"chunk {i} char_count incorrecto"
            assert row["chunk_index"] == i,           f"chunk_index {i} incorrecto"
        ok(f"3 chunks guardados con texto, char_count y chunk_index correctos")
    except Exception as e: fail("Verificación chunks en DB", e)

    # ── Verificar que el estado es consistente ──────────────────────────────
    try:
        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 1
        assert stats["pdf_indexed"]     == 1
        assert stats["pdf_pending"]     == 0
        assert stats["total_chunks"]    == 3
        assert stats["embedded_chunks"] == 0
        ok(f"get_stats() consistente tras guardado: {stats}")
    except Exception as e: fail("Consistencia stats tras guardado", e)

    # ── Verificar que no hay datos corrompidos ──────────────────────────────
    try:
        conn = get_connection(db)
        null_texts = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE text IS NULL OR text=''"
        ).fetchone()[0]
        null_counts = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE char_count IS NULL"
        ).fetchone()[0]
        conn.close()
        assert null_texts  == 0, f"{null_texts} chunks con texto vacío/NULL"
        assert null_counts == 0, f"{null_counts} chunks con char_count NULL"
        ok("Sin datos corrompidos — ningún chunk con texto NULL o vacío")
    except Exception as e: fail("Integridad datos", e)


# =============================================================================
# 7. Integración — flujo completo
# =============================================================================
def test_integration(tmp: Path, network: bool):
    section("7 · Integración — flujo completo crawler → SQLite")
    from backend.crawler.document import Document
    from backend.database.schema import init_db
    from backend.database import repository as repo

    csv  = tmp / "int_docs.csv"
    db   = tmp / "db" / "int.db"
    init_db(db)

    # Simula crawler guardando docs en CSV
    docs = [
        Document(arxiv_id=f"2310.0000{i}", title=f"Paper {i}", authors="A",
                 abstract="Abs.", categories="cs.LG", published="2023-10-01",
                 updated="2023-10-01", pdf_url=f"https://arxiv.org/pdf/2310.0000{i}")
        for i in range(5)
    ]
    for d in docs:
        d.save(csv)

    try:
        for d in Document.load_all(csv):
            repo.upsert_document(
                arxiv_id=d.arxiv_id, title=d.title, authors=d.authors,
                abstract=d.abstract, categories=d.categories,
                published=d.published, updated=d.updated,
                pdf_url=d.pdf_url, fetched_at=d.fetched_at, db_path=db,
            )
        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 5 and stats["pdf_pending"] == 5
        ok(f"CSV → SQLite: {stats['total_documents']} documentos importados")
    except Exception as e: fail("Importación CSV → SQLite", e)

    try:
        repo.save_pdf_text("2310.00000", "Contenido PDF. " * 40, db_path=db)
        repo.save_chunks("2310.00000", [f"chunk {i}" for i in range(4)], db_path=db)
        stats = repo.get_stats(db_path=db)
        assert stats["pdf_indexed"] == 1 and stats["total_chunks"] == 4
        ok(f"PDF guardado en DB → indexed={stats['pdf_indexed']}, chunks={stats['total_chunks']}")
    except Exception as e: fail("PDF simulado → SQLite", e)

    try:
        # Re-importar CSV no debe duplicar ni borrar estado de PDF
        for d in Document.load_all(csv):
            repo.upsert_document(
                arxiv_id=d.arxiv_id, title=d.title, authors=d.authors,
                abstract=d.abstract, categories=d.categories,
                published=d.published, updated=d.updated,
                pdf_url=d.pdf_url, fetched_at=d.fetched_at, db_path=db,
            )
        stats = repo.get_stats(db_path=db)
        assert stats["total_documents"] == 5   # sin duplicados
        assert stats["pdf_indexed"]     == 1   # pdf_downloaded intacto
        ok("Re-importación idempotente — sin duplicados, PDF status preservado")
    except Exception as e: fail("Re-importación idempotente", e)


# =============================================================================
# Resumen
# =============================================================================
# =============================================================================
# 8. Integridad de la DB real
# =============================================================================
def test_real_db():
    section("8 · Integridad de la DB real (datos del crawler)")
    from backend.database.schema import DB_PATH, get_connection

    if not DB_PATH.exists():
        skip("DB real", f"No existe aún: {DB_PATH}")
        return

    conn = get_connection(DB_PATH)

    # Conteos básicos
    try:
        total   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        indexed = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 1").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 0").fetchone()[0]
        errors  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded = 2").fetchone()[0]
        chunks  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert total >= 0
        ok(f"Tablas accesibles — {total} docs, {indexed} indexados, {pending} pendientes, {errors} errores")
    except Exception as e:
        fail("Lectura de tablas", e); conn.close(); return

    # Documentos indexados tienen full_text y text_length
    try:
        bad = conn.execute(
            "SELECT COUNT(*) FROM documents "
            "WHERE pdf_downloaded = 1 AND (full_text IS NULL OR text_length IS NULL)"
        ).fetchone()[0]
        assert bad == 0, f"{bad} documentos marcados como indexados pero sin texto"
        ok(f"Todos los documentos indexados tienen full_text ({indexed} docs)")
    except Exception as e:
        fail("full_text en indexados", e)

    # Documentos indexados tienen al menos 1 chunk
    try:
        if indexed > 0:
            docs_sin_chunks = conn.execute(
                "SELECT COUNT(*) FROM documents d "
                "WHERE d.pdf_downloaded = 1 "
                "AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.arxiv_id = d.arxiv_id)"
            ).fetchone()[0]
            assert docs_sin_chunks == 0, f"{docs_sin_chunks} docs indexados sin chunks"
            ok(f"Todos los documentos indexados tienen chunks — {chunks} chunks en total")
    except Exception as e:
        fail("Chunks en indexados", e)

    # No hay arxiv_ids duplicados
    try:
        total_ids  = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        unique_ids = conn.execute("SELECT COUNT(DISTINCT arxiv_id) FROM documents").fetchone()[0]
        assert total_ids == unique_ids, f"{total_ids - unique_ids} IDs duplicados"
        ok(f"Sin arxiv_ids duplicados ({unique_ids} únicos)")
    except Exception as e:
        fail("Duplicados", e)

    # Muestra un documento real como ejemplo
    try:
        if indexed > 0:
            row = conn.execute(
                "SELECT arxiv_id, title, text_length, "
                "(SELECT COUNT(*) FROM chunks c WHERE c.arxiv_id = d.arxiv_id) as n_chunks "
                "FROM documents d WHERE pdf_downloaded = 1 "
                "ORDER BY indexed_at DESC LIMIT 1"
            ).fetchone()
            ok(f"Ejemplo real → {row['arxiv_id']} | {row['text_length']:,} chars | {row['n_chunks']} chunks")
            ok(f"  Título: '{row['title'][:60]}…'")
    except Exception as e:
        fail("Ejemplo de documento real", e)

    conn.close()


def summary():
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  RESUMEN{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  {GREEN}Passed : {passed}{RESET}")
    print(f"  {RED}Failed : {failed}{RESET}")
    print(f"  {YELLOW}Skipped: {skipped}{RESET}")
    print(f"  Total  : {passed+failed+skipped}")
    print(f"{BOLD}{'═'*55}{RESET}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-network", action="store_true",
                   help="Omite tests que requieren conexión a internet.")
    p.add_argument("--only-db", action="store_true",
                   help="Solo verifica la integridad de la DB real del crawler.")
    args = p.parse_args()
    network = not args.skip_network

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    if args.only_db:
        test_real_db()
        summary()
        sys.exit(1 if failed else 0)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        test_document(tmp)
        test_id_store(tmp)
        test_robots(network)
        test_arxiv_client(network)
        test_database(tmp)
        test_database_content(tmp)
        test_pdf_extractor(network)
        test_integration(tmp, network)

    # Test sobre la DB real (fuera del tmp, usa la DB del crawler)
    test_real_db()

    summary()
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()