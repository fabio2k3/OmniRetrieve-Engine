"""
test_retrieval.py
=================
Tests de integración del módulo de recuperación LSI.

Verifica que LSIModel y LSIRetriever funcionen correctamente leyendo
desde el índice invertido construido por el módulo indexing.

Ejecución
---------
    python -m pytest backend/tests/test_retrieval.py -v
    python -m backend.tests.test_retrieval          # modo script
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from backend.database.schema import init_db
from backend.indexing.pipeline import IndexingPipeline
from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever

SAMPLE_DOCS = [
    ("2301.001", "Attention Is All You Need",
     "We propose transformer architecture with self-attention mechanisms "
     "for sequence to sequence tasks in natural language processing"),
    ("2301.002", "BERT: Bidirectional Transformers",
     "Pre-training deep bidirectional transformers for language understanding "
     "using masked language model and next sentence prediction objectives"),
    ("2301.003", "Gradient Descent Optimization",
     "Stochastic gradient descent convergence in deep neural networks "
     "with adaptive learning rate methods including Adam and RMSprop"),
    ("2301.004", "Reinforcement Learning Policy Gradient",
     "Policy gradient methods for deep reinforcement learning agents "
     "using actor-critic architectures and proximal policy optimization"),
    ("2301.005", "Convolutional Neural Networks for Vision",
     "Deep convolutional networks for image classification recognition "
     "using residual connections and batch normalization layers"),
]


def create_test_db(path: Path) -> None:
    """Crea BD temporal con esquema completo y documentos de ejemplo."""
    init_db(path)
    conn = sqlite3.connect(str(path))
    for arxiv_id, title, full_text in SAMPLE_DOCS:
        conn.execute(
            """
            INSERT INTO documents
                (arxiv_id, title, abstract, authors, full_text,
                 text_length, pdf_downloaded, published, pdf_url)
            VALUES (?, ?, '', '', ?, ?, 1, '2023-01-01', '')
            """,
            (arxiv_id, title, full_text, len(full_text)),
        )
    conn.commit()
    conn.close()


def run_tests(db_path: Path) -> None:
    sep = "─" * 58

    # ── SETUP: indexar con el pipeline de indexing ────────────────────────
    print(f"\n{sep}")
    print("  SETUP: Indexando corpus con IndexingPipeline…")
    print(sep)
    pipeline  = IndexingPipeline(db_path=db_path, field="full_text", batch_size=50)
    idx_stats = pipeline.run(reindex=True)
    assert idx_stats["docs_processed"] == 5
    assert idx_stats["terms_added"]    >  0
    print(f"  ✔ Índice creado: {idx_stats['terms_added']} términos, "
          f"{idx_stats['postings_added']} postings")

    # ── TEST 1: LSIModel.build() ──────────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 1: LSIModel.build() — construye el espacio latente")
    print(sep)
    model = LSIModel(k=3)
    stats = model.build(db_path=db_path)

    assert stats["n_docs"]           == 5
    assert stats["n_terms"]          >  0
    assert stats["k"]                == 3
    assert 0 < stats["var_explained"] <= 1.0
    assert model.docs_latent.shape   == (5, 3)
    assert len(model.doc_ids)        == 5
    assert len(model.term_ids)       >  0
    assert len(model.df_map)         >  0
    print(f"  ✔ n_docs={stats['n_docs']}  n_terms={stats['n_terms']}  "
          f"k={stats['k']}  varianza={stats['var_explained']:.2%}")

    # ── TEST 2: save / load ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 2: LSIModel.save() / .load()")
    print(sep)
    model_path = db_path.parent / "model.pkl"
    model.save(path=model_path)
    assert model_path.exists(), "El archivo .pkl debe existir"

    model2 = LSIModel()
    model2.load(path=model_path)
    assert model2.k                  == 3
    assert len(model2.doc_ids)       == 5
    assert model2.docs_latent.shape  == (5, 3)
    assert model2.doc_ids            == model.doc_ids
    assert model2.term_ids           == model.term_ids
    assert model2.df_map             == model.df_map
    print("  ✔ Modelo guardado y recargado correctamente")

    # ── TEST 3: LSIRetriever.load() + retrieve() ──────────────────────────
    print(f"\n{sep}")
    print("  TEST 3: LSIRetriever — carga y consulta")
    print(sep)
    retriever = LSIRetriever()
    retriever.load(model_path=model_path, db_path=db_path)

    assert len(retriever._word_index) > 0,    "word_index debe estar poblado"
    assert retriever._meta is not None,        "_meta debe estar cargado"

    results = retriever.retrieve("attention transformer mechanism", top_n=3)
    assert len(results) == 3
    assert all("score"    in r for r in results)
    assert all("arxiv_id" in r for r in results)
    assert all("title"    in r for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Resultados deben estar ordenados"
    print(f"  ✔ top-3 resultados ordenados por score:")
    for r in results:
        print(f"    [{r['arxiv_id']}]  score={r['score']:.4f}  {r['title']}")

    # ── TEST 4: relevancia semántica ──────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 4: Semántica — query sobre gradiente y optimización")
    print(sep)
    results2 = retriever.retrieve("gradient optimization learning rate", top_n=5)
    assert len(results2) > 0
    scores2  = [r["score"] for r in results2]
    assert scores2 == sorted(scores2, reverse=True)
    print(f"  ✔ top resultado: [{results2[0]['arxiv_id']}]  "
          f"score={results2[0]['score']:.4f}  '{results2[0]['title']}'")

    # ── TEST 5: lsi_log registrado en BD ─────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 5: lsi_log registrado en la BD")
    print(sep)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT * FROM lsi_log ORDER BY id DESC LIMIT 1").fetchall()
    conn.close()
    assert len(rows) > 0, "lsi_log debe tener al menos una fila"
    print(f"  ✔ lsi_log contiene {len(rows)} entrada(s)")

    print(f"\n{'═' * 58}")
    print("  ✅ Todos los tests de retrieval completados correctamente.")
    print(f"{'═' * 58}\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_documents.db"
        print(f"BD temporal: {db_path}")
        create_test_db(db_path)
        run_tests(db_path)


if __name__ == "__main__":
    main()
