"""
test_full_pipeline.py
=====================
Test de flujo completo (end-to-end) de OmniRetrieve-Engine.

Simula el recorrido real del sistema sobre una base de datos temporal:

    [crawler]  →  documents (full_text, pdf_downloaded=1)
    [indexing] →  terms + postings (freq cruda)
    [retrieval]→  modelo LSI (.pkl) + consultas semánticas

Cada sección valida las invariantes clave del módulo correspondiente,
más la integración entre ellos. Al final se imprime un resumen.

Ejecución
---------
    python -m backend.tests.test_full_pipeline
    python -m pytest backend/tests/test_full_pipeline.py -v
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus de prueba
# ---------------------------------------------------------------------------

CORPUS = [
    {
        "arxiv_id":       "2301.001",
        "title":          "Attention Is All You Need",
        "full_text":      (
            "We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms. Multi-head self-attention "
            "allows the model to jointly attend to information from different "
            "representation subspaces at different positions."
        ),
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2301.002",
        "title":          "BERT: Bidirectional Transformers for Language Understanding",
        "full_text":      (
            "We introduce BERT, a method of pre-training language representations "
            "using bidirectional transformers. Masked language model enables deep "
            "bidirectional pre-training. Fine-tuning yields state-of-the-art results "
            "on natural language processing tasks."
        ),
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2301.003",
        "title":          "Adam: A Method for Stochastic Optimization",
        "full_text":      (
            "We introduce Adam, an algorithm for first-order gradient-based "
            "optimization of stochastic objective functions. The method computes "
            "adaptive learning rates for different parameters from estimates of "
            "first and second moments of the gradients."
        ),
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2301.004",
        "title":          "Deep Residual Learning for Image Recognition",
        "full_text":      (
            "We present a residual learning framework to ease the training of "
            "deep neural networks. Shortcut connections perform identity mapping "
            "added to the outputs of stacked layers. Residual networks are easier "
            "to optimize and gain accuracy with depth."
        ),
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2301.005",
        "title":          "Generative Adversarial Networks",
        "full_text":      (
            "We propose a generative adversarial network framework. A generative "
            "model captures the data distribution while a discriminative model "
            "estimates the probability that a sample came from the training data. "
            "Both models are trained simultaneously in a minimax game."
        ),
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2301.006",
        "title":          "Pending Download Article",
        "full_text":      None,
        "pdf_downloaded": 0,   # ← no debe indexarse con field='full_text'
    },
]


def _create_test_db(path: Path) -> None:
    """
    Crea la base de datos temporal con el schema completo y el corpus de prueba.
    Usa init_db del módulo real para garantizar consistencia con el schema oficial.
    """
    from backend.database.schema import init_db
    path.parent.mkdir(parents=True, exist_ok=True)
    init_db(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    for doc in CORPUS:
        conn.execute(
            """
            INSERT OR IGNORE INTO documents
                (arxiv_id, title, full_text, pdf_downloaded, published, pdf_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc["arxiv_id"], doc["title"], doc["full_text"],
                doc["pdf_downloaded"], "2023-01-01",
                f"https://arxiv.org/pdf/{doc['arxiv_id']}",
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helpers de reporte
# ---------------------------------------------------------------------------

_SEP  = "─" * 60
_SEP2 = "═" * 60
_OK   = "✔ "
_FAIL = "✘ "
_results: list[tuple[str, bool, str]] = []


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = _OK if condition else _FAIL
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))
    _results.append((label, condition, detail))
    if not condition:
        raise AssertionError(f"FALLO: {label}  {detail}")


# ---------------------------------------------------------------------------
# SECCIÓN 1: database — schema
# ---------------------------------------------------------------------------

def _test_schema(db: Path) -> None:
    print(f"\n{_SEP}")
    print("  SECCIÓN 1 — database/schema.py")
    print(_SEP)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    required = {"documents", "terms", "postings", "index_meta", "lsi_log"}
    _check("Tablas requeridas presentes", required <= tables, str(required))

    cols_postings = [r[1] for r in conn.execute("PRAGMA table_info(postings)").fetchall()]
    _check("postings.freq existe (int, sin tfidf_weight)",
           "freq" in cols_postings and "tfidf_weight" not in cols_postings,
           str(cols_postings))

    cols_lsi = [r[1] for r in conn.execute("PRAGMA table_info(lsi_log)").fetchall()]
    _check("lsi_log.n_terms existe", "n_terms" in cols_lsi, str(cols_lsi))

    docs_in_db = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    _check("Corpus cargado", docs_in_db == len(CORPUS), f"{docs_in_db} docs")

    conn.close()


# ---------------------------------------------------------------------------
# SECCIÓN 2: database — crawler_repository
# ---------------------------------------------------------------------------

def _test_crawler_repository(db: Path) -> None:
    print(f"\n{_SEP}")
    print("  SECCIÓN 2 — database/crawler_repository.py")
    print(_SEP)

    from backend.database.crawler_repository import (
        document_exists, get_pending_pdf_ids, get_stats,
        log_crawl_start, log_crawl_end,
    )

    _check("document_exists — doc presente",  document_exists("2301.001", db_path=db))
    _check("document_exists — doc ausente",   not document_exists("9999.999", db_path=db))

    pending = get_pending_pdf_ids(db_path=db)
    _check("get_pending_pdf_ids — devuelve doc sin PDF",
           "2301.006" in pending, f"pending={pending}")
    _check("get_pending_pdf_ids — no incluye docs con PDF",
           "2301.001" not in pending)

    lid = log_crawl_start(db_path=db)
    log_crawl_end(lid, ids_discovered=6, docs_downloaded=5, pdfs_indexed=5, db_path=db)
    stats = get_stats(db_path=db)
    _check("get_stats — total_documents", stats["total_documents"] == 6, str(stats))
    _check("get_stats — pdf_indexed == 5", stats["pdf_indexed"] == 5, str(stats))
    _check("get_stats — pdf_pending == 1", stats["pdf_pending"] == 1, str(stats))


# ---------------------------------------------------------------------------
# SECCIÓN 3: indexing
# ---------------------------------------------------------------------------

def _test_indexing(db: Path) -> None:
    print(f"\n{_SEP}")
    print("  SECCIÓN 3 — indexing/")
    print(_SEP)

    from backend.indexing import TextPreprocessor, IndexingPipeline
    from backend.database.index_repository import (
        get_index_stats, get_top_terms,
        get_postings_for_term, get_postings_for_matrix,
    )

    # preprocessor
    pp = TextPreprocessor()
    tokens = pp.process("Attention mechanisms enable transformer models to attend.")
    _check("preprocessor — tokeniza correctamente",
           "attention" in tokens and "mechanisms" in tokens)
    _check("preprocessor — elimina stopwords",
           "to" not in tokens and "the" not in tokens)
    _check("preprocessor — solo tokens alfabéticos",
           all(t.isalpha() for t in tokens))

    # pipeline primera ejecución — solo 5 docs con pdf_downloaded=1
    pipeline = IndexingPipeline(db_path=db, field="full_text", batch_size=50)
    stats = pipeline.run(reindex=False)
    _check("pipeline — docs_processed == 5",
           stats["docs_processed"] == 5, f"got {stats['docs_processed']}")
    _check("pipeline — terms_added > 0",
           stats["terms_added"] > 0, f"got {stats['terms_added']}")
    _check("pipeline — postings_added > 0",
           stats["postings_added"] > 0, f"got {stats['postings_added']}")

    # postings contienen freq int, sin tfidf_weight
    top = get_top_terms("2301.001", n=10, db_path=db)
    _check("get_top_terms — hay resultados", len(top) > 0)
    _check("postings guardan freq (int)",
           all(isinstance(t["freq"], int) for t in top))
    _check("postings NO guardan tfidf_weight",
           all("tfidf_weight" not in t for t in top))

    # df acumulado
    idx = get_index_stats(db_path=db)
    _check("get_index_stats — total_docs == 5",
           idx["total_docs"] == 5, f"got {idx['total_docs']}")
    _check("get_index_stats — vocab_size > 0",
           idx["vocab_size"] > 0, f"got {idx['vocab_size']}")

    # indexación incremental — no reprocesa
    stats2 = pipeline.run(reindex=False)
    _check("indexación incremental — docs_processed == 0",
           stats2["docs_processed"] == 0, f"got {stats2['docs_processed']}")

    # reindex completo
    stats3 = pipeline.run(reindex=True)
    _check("reindex completo — docs_processed == 5",
           stats3["docs_processed"] == 5, f"got {stats3['docs_processed']}")

    # get_postings_for_matrix — datos crudos para retrieval
    postings_raw, df_map, doc_ids, term_ids, n_docs = get_postings_for_matrix(db_path=db)
    _check("get_postings_for_matrix — 5 documentos",
           len(doc_ids) == 5, f"got {len(doc_ids)}")
    _check("get_postings_for_matrix — hay términos",
           len(term_ids) > 0, f"got {len(term_ids)}")
    _check("get_postings_for_matrix — freq es int",
           all(isinstance(p[2], int) for p in postings_raw))
    _check("get_postings_for_matrix — df es int",
           all(isinstance(v, int) for v in df_map.values()))


# ---------------------------------------------------------------------------
# SECCIÓN 4: retrieval
# ---------------------------------------------------------------------------

def _test_retrieval(db: Path, model_path: Path) -> None:
    print(f"\n{_SEP}")
    print("  SECCIÓN 4 — retrieval/")
    print(_SEP)

    from backend.retrieval import LSIModel, LSIRetriever
    from backend.database.schema import get_connection

    # build
    model = LSIModel(k=3)
    build_stats = model.build(db_path=db)
    _check("LSIModel.build — n_docs == 5",
           build_stats["n_docs"] == 5, f"got {build_stats['n_docs']}")
    _check("LSIModel.build — n_terms > 0",
           build_stats["n_terms"] > 0, f"got {build_stats['n_terms']}")
    _check("LSIModel.build — varianza en (0, 1]",
           0 < build_stats["var_explained"] <= 1.0,
           f"got {build_stats['var_explained']:.4f}")
    _check("LSIModel.build — docs_latent shape == (5, 3)",
           model.docs_latent.shape == (5, 3), str(model.docs_latent.shape))
    _check("LSIModel.build — df_map poblado",
           len(model.df_map) > 0)

    # save / load
    model.save(path=model_path)
    model2 = LSIModel()
    model2.load(path=model_path)
    _check("save/load — doc_ids coinciden",  model2.doc_ids  == model.doc_ids)
    _check("save/load — term_ids coinciden", model2.term_ids == model.term_ids)
    _check("save/load — df_map coincide",    model2.df_map   == model.df_map)

    # retriever
    retriever = LSIRetriever()
    retriever.load(model_path=model_path, db_path=db)
    _check("LSIRetriever.load — word_index poblado",
           len(retriever._word_index) > 0, f"got {len(retriever._word_index)}")
    _check("LSIRetriever.load — meta cargada",
           retriever._meta is not None and len(retriever._meta) > 0)

    # resultados ordenados por score
    results = retriever.retrieve("attention transformer mechanism", top_n=3)
    _check("retrieve — devuelve top_n resultados", len(results) == 3)
    _check("retrieve — scores en orden descendente",
           [r["score"] for r in results] == sorted(
               [r["score"] for r in results], reverse=True))
    _check("retrieve — cada resultado tiene campos requeridos",
           all({"score","arxiv_id","title","abstract","url"} <= set(r) for r in results))

    # relevancia semántica básica
    r_attention = retriever.retrieve("self attention transformer", top_n=5)
    top_attention = r_attention[0]["arxiv_id"]
    _check(f"semántica — 'attention transformer' devuelve 2301.001 o 2301.002",
           top_attention in {"2301.001", "2301.002"},
           f"top={top_attention} '{r_attention[0]['title']}'")

    r_gradient = retriever.retrieve("gradient optimization learning rate", top_n=5)
    top_gradient = r_gradient[0]["arxiv_id"]
    _check(f"semántica — 'gradient optimization' devuelve 2301.003 o 2301.004",
           top_gradient in {"2301.003", "2301.004"},
           f"top={top_gradient} '{r_gradient[0]['title']}'")

    # lsi_log registrado en BD
    conn = get_connection(db)
    log_rows = conn.execute("SELECT * FROM lsi_log").fetchall()
    conn.close()
    _check("lsi_log — 1 registro insertado", len(log_rows) == 1)
    _check("lsi_log — n_terms > 0", log_rows[0]["n_terms"] > 0,
           f"got {log_rows[0]['n_terms']}")
    _check("lsi_log — k == 3", log_rows[0]["k"] == 3)


# ---------------------------------------------------------------------------
# SECCIÓN 5: flujo completo integrado
# ---------------------------------------------------------------------------

def _test_integration(db: Path, model_path: Path) -> None:
    print(f"\n{_SEP}")
    print("  SECCIÓN 5 — integración crawler → indexing → retrieval")
    print(_SEP)

    from backend.database.crawler_repository import upsert_document, save_pdf_text
    from backend.indexing import IndexingPipeline
    from backend.retrieval import LSIModel, LSIRetriever

    # Simular llegada de un documento nuevo vía crawler
    upsert_document(
        arxiv_id="2301.007", title="Diffusion Models for Image Synthesis",
        authors="Ho et al.", abstract="Denoising diffusion probabilistic models.",
        categories="cs.CV", published="2023-01-07", updated="2023-01-07",
        pdf_url="https://arxiv.org/pdf/2301.007", fetched_at="2023-01-07",
        db_path=db,
    )
    save_pdf_text(
        "2301.007",
        "Denoising diffusion probabilistic models achieve high quality image synthesis. "
        "The forward process adds Gaussian noise incrementally. The reverse process "
        "learns to denoise using a neural network. Results surpass GANs on image generation.",
        db_path=db,
    )

    # Indexar incrementalmente (solo el doc nuevo)
    pipeline = IndexingPipeline(db_path=db, field="full_text")
    stats = pipeline.run(reindex=False)
    _check("indexación incremental doc nuevo — docs_processed == 1",
           stats["docs_processed"] == 1, f"got {stats['docs_processed']}")

    # Reconstruir modelo con los 6 docs
    model = LSIModel(k=3)
    build_stats = model.build(db_path=db)
    _check("rebuild modelo — n_docs == 6",
           build_stats["n_docs"] == 6, f"got {build_stats['n_docs']}")
    model.save(path=model_path)

    # El nuevo doc debe ser recuperable
    retriever = LSIRetriever()
    retriever.load(model_path=model_path, db_path=db)
    results = retriever.retrieve("diffusion noise image generation", top_n=6)
    ids = [r["arxiv_id"] for r in results]
    _check("doc nuevo recuperable — 2301.007 en resultados",
           "2301.007" in ids, f"ids={ids}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{_SEP2}")
    print("  OmniRetrieve-Engine — Test de flujo completo")
    print(_SEP2)

    with tempfile.TemporaryDirectory() as tmp:
        db         = Path(tmp) / "db" / "test.db"
        model_path = Path(tmp) / "models" / "lsi_model.pkl"
        db.parent.mkdir(parents=True)
        model_path.parent.mkdir(parents=True)

        print(f"\n  BD temporal: {db}")
        _create_test_db(db)

        _test_schema(db)
        _test_crawler_repository(db)
        _test_indexing(db)
        _test_retrieval(db, model_path)
        _test_integration(db, model_path)

    # ── Resumen ──────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    failed = [(label, detail) for label, ok, detail in _results if not ok]

    print(f"\n{_SEP2}")
    if failed:
        print(f"  ✘  {passed}/{total} checks pasaron.")
        print("\n  Fallos:")
        for label, detail in failed:
            print(f"    · {label}  {detail}")
    else:
        print(f"  ✅  {passed}/{total} checks pasaron — todo correcto.")
    print(_SEP2 + "\n")


if __name__ == "__main__":
    main()
