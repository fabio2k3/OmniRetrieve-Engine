"""
test_pipeline.py
================
Tests de integración del módulo de indexación completo.

Verifica que el pipeline (preprocessor + tfidf + repository) funcione
de extremo a extremo sobre una BD temporal con datos de ejemplo.

"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from backend.indexing.pipeline import IndexingPipeline
from backend.indexing.index_repository import IndexRepository

SAMPLE_DOCS = [
    {
        "arxiv_id": "2301.00001",
        "title": "Fairness in Machine Learning: A Survey",
        "abstract": "We survey fairness definitions and algorithms in machine learning. "
                    "Fairness is a critical concern in AI ethics.",
        "full_text": (
            "Machine learning models can perpetuate and amplify societal biases. "
            "This survey examines fairness definitions including demographic parity, "
            "equalized odds, and individual fairness. We discuss algorithms that "
            "optimize for these criteria and analyze trade-offs between fairness "
            "and accuracy. Ethical AI requires careful consideration of these issues."
        ),
        "categories": "cs.LG, cs.AI",
        "published": "2023-01-05T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length": 400,
    },
    {
        "arxiv_id": "2301.00002",
        "title": "Bias Detection in Natural Language Processing",
        "abstract": "This paper studies bias in NLP models and proposes mitigation techniques.",
        "full_text": (
            "Natural language processing models trained on large corpora inherit "
            "biases from the training data. We propose a framework for detecting "
            "and quantifying gender, racial, and cultural bias in word embeddings "
            "and language models. Our debiasing technique reduces harmful bias "
            "while preserving model performance on downstream tasks. "
            "Responsible AI deployment demands bias-aware model development."
        ),
        "categories": "cs.CL, cs.AI",
        "published": "2023-01-10T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length": 420,
    },
    {
        "arxiv_id": "2301.00003",
        "title": "Explainability and Transparency in Neural Networks",
        "abstract": "We propose methods for explaining neural network decisions.",
        "full_text": (
            "Black-box neural networks pose challenges for transparency and accountability. "
            "We develop explainability methods including saliency maps, SHAP values, "
            "and concept-based explanations. Our experiments on image classification "
            "and sentiment analysis show that explanations can reveal hidden biases "
            "and improve trustworthiness. Transparency is fundamental to ethical AI."
        ),
        "categories": "cs.LG, cs.AI",
        "published": "2023-01-15T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length": 390,
    },
    {
        "arxiv_id": "2301.00004",
        "title": "Privacy-Preserving Machine Learning",
        "abstract": "Differential privacy techniques for machine learning models.",
        "full_text": None,       # Solo abstract disponible
        "categories": "cs.CR, cs.LG",
        "published": "2023-01-20T00:00:00Z",
        "pdf_downloaded": 0,     # Pendiente de descarga
        "text_length": None,
    },
]

def create_test_db(path: Path) -> None:
    """Crea una BD temporal con el esquema mínimo necesario."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS documents (
            arxiv_id        TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            authors         TEXT,
            abstract        TEXT,
            categories      TEXT,
            published       TEXT,
            updated         TEXT,
            pdf_url         TEXT,
            fetched_at      TEXT,
            full_text       TEXT,
            text_length     INTEGER,
            pdf_downloaded  INTEGER NOT NULL DEFAULT 0,
            indexed_at      TEXT,
            index_error     TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id    TEXT    NOT NULL REFERENCES documents(arxiv_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text        TEXT    NOT NULL,
            char_count  INTEGER,
            embedding   BLOB,
            embedded_at TEXT,
            created_at  TEXT NOT NULL,
            UNIQUE(arxiv_id, chunk_index)
        );

        CREATE TABLE IF NOT EXISTS crawl_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at      TEXT NOT NULL,
            finished_at     TEXT,
            ids_discovered  INTEGER DEFAULT 0,
            docs_downloaded INTEGER DEFAULT 0,
            pdfs_indexed    INTEGER DEFAULT 0,
            errors          INTEGER DEFAULT 0,
            notes           TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_doc_pdf_status ON documents(pdf_downloaded);
    """)
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (arxiv_id, title, abstract, full_text, categories,
             published, pdf_downloaded, text_length)
        VALUES
            (:arxiv_id, :title, :abstract, :full_text, :categories,
             :published, :pdf_downloaded, :text_length)
        """,
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()


def run_tests(db_path: Path) -> None:
    sep = "─" * 58

    print(f"\n{sep}")
    print("  TEST 1: Pipeline inicial — campo 'both'")
    print(sep)
    pipeline = IndexingPipeline(db_path=db_path, field="both", batch_size=50)
    stats = pipeline.run(reindex=False, recalculate_idf=True)
    assert stats["docs_processed"] >= 3, "Deberían indexarse al menos 3 docs"
    assert stats["terms_added"] > 0,     "Deberían crearse términos"
    assert stats["postings_added"] > 0,  "Deberían crearse postings"
    print(f"  ✔ docs={stats['docs_processed']}  "
          f"términos={stats['terms_added']}  "
          f"postings={stats['postings_added']}")

    print(f"\n{sep}")
    print("  TEST 2: Estadísticas del índice (IndexRepository)")
    print(sep)
    repo = IndexRepository(db_path)
    idx  = repo.get_index_stats()
    assert idx["vocab_size"]     > 0
    assert idx["total_docs"]     > 0
    assert idx["total_postings"] > 0
    print(f"  ✔ vocabulario={idx['vocab_size']}  "
          f"docs={idx['total_docs']}  "
          f"postings={idx['total_postings']}")

    print(f"\n{sep}")
    print("  TEST 3: Top términos — 2301.00001")
    print(sep)
    top = repo.get_top_terms("2301.00001", n=10)
    assert len(top) > 0, "El documento debería tener términos indexados"
    for t in top:
        print(f"  {t['word']:<20}  tf={t['tf']:.4f}  tfidf={t['tfidf_weight']:.4f}")

    print(f"\n{sep}")
    print("  TEST 4: Posting list de 'fairness'")
    print(sep)
    postings = repo.get_postings_for_term("fairness")
    if postings:
        for p in postings:
            print(f"  doc={p['doc_id']}  tf={p['tf']:.4f}  tfidf={p['tfidf_weight']:.4f}")
    else:
        print("  (no encontrado sin stemming — esperado)")

    print(f"\n{sep}")
    print("  TEST 5: Búsqueda 'bias neural fairness'")
    print(sep)
    results = repo.search(["bias", "neural", "fairness"], top_k=5)
    assert len(results) > 0, "La búsqueda debería devolver resultados"
    for r in results:
        print(f"  [{r['doc_id']}]  score={r['score']:.4f}  {r['title'][:50]}")

    print(f"\n{sep}")
    print("  TEST 6: Indexación incremental (no debe procesar docs ya indexados)")
    print(sep)
    stats2 = pipeline.run(reindex=False)
    assert stats2["docs_processed"] == 0, "No debería haber docs nuevos"
    print(f"  ✔ docs nuevos procesados: {stats2['docs_processed']} (esperado: 0)")

    print(f"\n{sep}")
    print("  TEST 7: Reindex completo (--reindex)")
    print(sep)
    stats3 = pipeline.run(reindex=True)
    assert stats3["docs_processed"] >= 3
    print(f"  ✔ docs re-indexados: {stats3['docs_processed']} (esperado: ≥3)")

    print(f"\n{sep}")
    print("  TEST 8: Exportación de matriz TF-IDF (para módulo LSI)")
    print(sep)
    matrix, doc_ids, term_ids = repo.get_tfidf_matrix()
    assert matrix.shape[0] == len(term_ids)
    assert matrix.shape[1] == len(doc_ids)
    assert (matrix >= 0).all(), "Todos los pesos deben ser ≥ 0"
    print(f"  ✔ matriz shape: {matrix.shape}  "
          f"(terms × docs)  dtype={matrix.dtype}")

    print(f"\n{'═' * 58}")
    print("  ✅ Todos los tests completados correctamente.")
    print(f"{'═' * 58}\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_documents.db"
        print(f"BD temporal: {db_path}")
        create_test_db(db_path)
        run_tests(db_path)


if __name__ == "__main__":
    main()
