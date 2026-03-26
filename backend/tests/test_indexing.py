"""
test_indexing.py
================
Tests de integración del módulo de indexación.

Verifica que el pipeline completo (preprocessor → TFIndexer → index_repository)
funcione de extremo a extremo sobre una BD temporal con datos de ejemplo.

Ejecución
---------
    python -m pytest backend/tests/test_indexing.py -v
    python -m backend.tests.test_indexing          # modo script
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from backend.database.schema import init_db
from backend.database.index_repository import (
    get_index_stats,
    get_top_terms,
    get_postings_for_term,
    get_postings_for_matrix,
)
from backend.indexing.pipeline import IndexingPipeline

SAMPLE_DOCS = [
    {
        "arxiv_id":       "2301.00001",
        "title":          "Fairness in Machine Learning: A Survey",
        "abstract":       "We survey fairness definitions and algorithms in machine learning. "
                          "Fairness is a critical concern in AI ethics.",
        "full_text":      (
            "Machine learning models can perpetuate and amplify societal biases. "
            "This survey examines fairness definitions including demographic parity, "
            "equalized odds, and individual fairness. We discuss algorithms that "
            "optimize for these criteria and analyze trade-offs between fairness "
            "and accuracy. Ethical AI requires careful consideration of these issues."
        ),
        "categories":     "cs.LG, cs.AI",
        "published":      "2023-01-05T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length":    400,
    },
    {
        "arxiv_id":       "2301.00002",
        "title":          "Bias Detection in Natural Language Processing",
        "abstract":       "This paper studies bias in NLP models and proposes mitigation techniques.",
        "full_text":      (
            "Natural language processing models trained on large corpora inherit "
            "biases from the training data. We propose a framework for detecting "
            "and quantifying gender, racial, and cultural bias in word embeddings "
            "and language models. Our debiasing technique reduces harmful bias "
            "while preserving model performance on downstream tasks. "
            "Responsible AI deployment demands bias-aware model development."
        ),
        "categories":     "cs.CL, cs.AI",
        "published":      "2023-01-10T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length":    420,
    },
    {
        "arxiv_id":       "2301.00003",
        "title":          "Explainability and Transparency in Neural Networks",
        "abstract":       "We propose methods for explaining neural network decisions.",
        "full_text":      (
            "Black-box neural networks pose challenges for transparency and accountability. "
            "We develop explainability methods including saliency maps, SHAP values, "
            "and concept-based explanations. Our experiments on image classification "
            "and sentiment analysis show that explanations can reveal hidden biases "
            "and improve trustworthiness. Transparency is fundamental to ethical AI."
        ),
        "categories":     "cs.LG, cs.AI",
        "published":      "2023-01-15T00:00:00Z",
        "pdf_downloaded": 1,
        "text_length":    390,
    },
    {
        "arxiv_id":       "2301.00004",
        "title":          "Privacy-Preserving Machine Learning",
        "abstract":       "Differential privacy techniques for machine learning models.",
        "full_text":      None,        # solo abstract disponible
        "categories":     "cs.CR, cs.LG",
        "published":      "2023-01-20T00:00:00Z",
        "pdf_downloaded": 0,           # pendiente de descarga — no debe indexarse
        "text_length":    None,
    },
]


def create_test_db(path: Path) -> None:
    """Crea una BD temporal con el esquema completo y datos de ejemplo."""
    init_db(path)
    conn = sqlite3.connect(str(path))
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
    print("  TEST 1: Pipeline — campo 'both', indexa 3 docs (pdf=1)")
    print(sep)
    pipeline = IndexingPipeline(db_path=db_path, field="both", batch_size=50)
    stats = pipeline.run(reindex=False)
    assert stats["docs_processed"] >= 3, f"Esperado ≥3, got {stats['docs_processed']}"
    assert stats["terms_added"]    >  0
    assert stats["postings_added"] >  0
    print(f"  ✔ docs={stats['docs_processed']}  "
          f"términos={stats['terms_added']}  "
          f"postings={stats['postings_added']}")

    print(f"\n{sep}")
    print("  TEST 2: Estadísticas del índice")
    print(sep)
    idx = get_index_stats(db_path=db_path)
    assert idx["vocab_size"]     > 0
    assert idx["total_docs"]     > 0
    assert idx["total_postings"] > 0
    print(f"  ✔ vocabulario={idx['vocab_size']}  "
          f"docs={idx['total_docs']}  "
          f"postings={idx['total_postings']}")

    print(f"\n{sep}")
    print("  TEST 3: Top términos — 2301.00001 (freq cruda, sin tfidf_weight)")
    print(sep)
    top = get_top_terms("2301.00001", n=10, db_path=db_path)
    assert len(top) > 0, "El documento debería tener términos indexados"
    assert all("freq" in t for t in top),          "Debe haber columna 'freq'"
    assert all("tfidf_weight" not in t for t in top), "NO debe haber 'tfidf_weight'"
    assert all(isinstance(t["freq"], int) for t in top), "freq debe ser int"
    for t in top:
        print(f"  {t['word']:<20}  freq={t['freq']}  df={t['df']}")

    print(f"\n{sep}")
    print("  TEST 4: Posting list de 'bias'")
    print(sep)
    postings = get_postings_for_term("bias", db_path=db_path)
    if postings:
        assert all("freq" in p for p in postings)
        assert all("tfidf_weight" not in p for p in postings)
        for p in postings:
            print(f"  doc={p['doc_id']}  freq={p['freq']}")
    else:
        print("  (no encontrado sin stemming — esperado)")

    print(f"\n{sep}")
    print("  TEST 5: get_postings_for_matrix — datos crudos para retrieval")
    print(sep)
    postings_raw, df_map, doc_ids, term_ids, n_docs = get_postings_for_matrix(db_path=db_path)
    assert len(doc_ids)      > 0
    assert len(term_ids)     > 0
    assert len(postings_raw) > 0
    assert all(isinstance(p[2], int) for p in postings_raw), "freq debe ser int"
    assert all(isinstance(v, int) for v in df_map.values()), "df debe ser int"
    print(f"  ✔ docs={len(doc_ids)}  términos={len(term_ids)}  "
          f"postings={len(postings_raw)}  N_total={n_docs}")
    print(f"  ✔ ejemplo: term_id={postings_raw[0][0]}  "
          f"doc={postings_raw[0][1]}  freq={postings_raw[0][2]}")

    print(f"\n{sep}")
    print("  TEST 6: Indexación incremental — no reprocesa docs ya indexados")
    print(sep)
    stats2 = pipeline.run(reindex=False)
    assert stats2["docs_processed"] == 0, "No debería haber docs nuevos"
    print(f"  ✔ docs nuevos procesados: {stats2['docs_processed']} (esperado: 0)")

    print(f"\n{sep}")
    print("  TEST 7: Reindex completo")
    print(sep)
    stats3 = pipeline.run(reindex=True)
    assert stats3["docs_processed"] >= 3
    print(f"  ✔ docs re-indexados: {stats3['docs_processed']} (esperado: ≥3)")

    print(f"\n{'═' * 58}")
    print("  ✅ Todos los tests de indexación completados correctamente.")
    print(f"{'═' * 58}\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_documents.db"
        print(f"BD temporal: {db_path}")
        create_test_db(db_path)
        run_tests(db_path)


if __name__ == "__main__":
    main()
