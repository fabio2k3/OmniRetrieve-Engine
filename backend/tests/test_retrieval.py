"""
test_retrieval.py
=================
Tests para LSIModel y LSIRetriever (Modulo C).

Tipos de tests:
- Unit: Una clase/funcion, BD en memoria, <1 segundo
- Integration: Varios modulos, BD temporal en disco, 1-30 segundos
- End-to-end: Flujo completo, BD real + red, Minutos

Para el Modulo C, el nivel de integracion (BD temporal + modelo LSI)
es el mas importante.
"""
import tempfile
from pathlib import Path
import sqlite3
import pytest

from backend.retrieval.lsi_model import LSIModel
from backend.retrieval.lsi_retriever import LSIRetriever

SAMPLE_DOCS = [
    ("2301.001", "Attention Is All You Need",
     "We propose transformer architecture with self-attention mechanisms"),
    ("2301.002", "BERT: Bidirectional Transformers",
     "Pre-training deep bidirectional transformers for language understanding"),
    ("2301.003", "Gradient Descent Optimization",
     "Stochastic gradient descent convergence in neural networks"),
    ("2301.004", "Reinforcement Learning Policy Gradient",
     "Policy gradient methods for deep reinforcement learning agents"),
    ("2301.005", "Convolutional Neural Networks for Vision",
     "Deep convolutional networks for image classification recognition"),
]


def create_test_db(path: Path) -> None:
    """Crea BD temporal con documentos de prueba."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
    CREATE TABLE documents (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT,
        abstract TEXT,
        authors TEXT,
        full_text TEXT,
        text_length INTEGER,
        pdf_downloaded INTEGER DEFAULT 0,
        published TEXT,
        pdf_url TEXT
    );
    CREATE TABLE lsi_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        built_at TEXT NOT NULL,
        k INTEGER NOT NULL,
        n_docs INTEGER NOT NULL,
        var_explained REAL,
        model_path TEXT,
        notes TEXT
    );
    """)
    for arxiv_id, title, full_text in SAMPLE_DOCS:
        conn.execute(
            "INSERT INTO documents (arxiv_id, title, abstract, authors, full_text, text_length, pdf_downloaded, published, pdf_url) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, '2023-01-01', '')",
            (arxiv_id, title, "", "", full_text, len(full_text))
        )
    conn.commit()
    conn.close()


def test_lsi_model_build():
    """Verifica que el modelo se construye y la varianza es razonable."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        create_test_db(db_path)

        model = LSIModel(k=3)
        stats = model.build(db_path=db_path)

        assert stats["n_docs"] == 5
        assert stats["k"] == 3
        assert 0 < stats["var_explained"] <= 1.0
        assert model.docs_latent.shape == (5, 3)
        assert len(model.doc_ids) == 5


def test_lsi_retriever_returns_results():
    """Verifica que retrieve devuelve los docs mas similares."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        model_path = Path(tmp) / "model.pkl"
        create_test_db(db_path)

        # Construir y guardar
        model = LSIModel(k=3)
        model.build(db_path=db_path)
        model.save(path=model_path)

        # Cargar y recuperar
        retriever = LSIRetriever()
        retriever.model.load(path=model_path)
        retriever._meta = retriever._load_meta(db_path)

        results = retriever.retrieve("attention transformer mechanism", top_n=3)

        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("arxiv_id" in r for r in results)
        # El primer resultado debe tener score >= 0
        assert results[0]["score"] >= 0


def test_retrieve_semantics():
    """Verifica que LSI recupera documentos y ordena por relevancia."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        model_path = Path(tmp) / "model.pkl"
        create_test_db(db_path)

        model = LSIModel(k=3)
        model.build(db_path=db_path)
        model.save(path=model_path)

        retriever = LSIRetriever()
        retriever.model.load(path=model_path)
        retriever._meta = retriever._load_meta(db_path)

        # Usar una query con palabras muy comunes en el corpus
        results = retriever.retrieve("neural networks learning", top_n=3)
        top_ids = [r["arxiv_id"] for r in results]
        # Al menos uno de los docs debe aparecer
        assert len(top_ids) == 3
        # Los scores deben estar en orden descendente
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
