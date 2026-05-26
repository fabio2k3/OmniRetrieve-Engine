"""Fixtures para los tests del módulo indexing."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pytest

SAMPLE_DOCS = [
    {
        "arxiv_id": "2301.00001", "title": "Fairness in Machine Learning: A Survey",
        "abstract": "We survey fairness definitions and algorithms in machine learning.",
        "full_text": (
            "Machine learning models can perpetuate and amplify societal biases. "
            "This survey examines fairness definitions including demographic parity, "
            "equalized odds, and individual fairness. We discuss algorithms that "
            "optimize for these criteria and analyze trade-offs between fairness "
            "and accuracy. Ethical AI requires careful consideration of these issues."
        ),
        "categories": "cs.LG, cs.AI", "published": "2023-01-05T00:00:00Z",
        "pdf_downloaded": 1, "text_length": 400,
    },
    {
        "arxiv_id": "2301.00002", "title": "Bias Detection in Natural Language Processing",
        "abstract": "This paper studies bias in NLP models and proposes mitigation techniques.",
        "full_text": (
            "Natural language processing models trained on large corpora inherit "
            "biases from the training data. We propose a framework for detecting "
            "and quantifying gender, racial, and cultural bias in word embeddings "
            "and language models. Our debiasing technique reduces harmful bias "
            "while preserving model performance on downstream tasks."
        ),
        "categories": "cs.CL, cs.AI", "published": "2023-01-10T00:00:00Z",
        "pdf_downloaded": 1, "text_length": 420,
    },
    {
        "arxiv_id": "2301.00003", "title": "Explainability and Transparency in Neural Networks",
        "abstract": "We propose methods for explaining neural network decisions.",
        "full_text": (
            "Black-box neural networks pose challenges for transparency and accountability. "
            "We develop explainability methods including saliency maps, SHAP values, "
            "and concept-based explanations. Our experiments on image classification "
            "and sentiment analysis show that explanations can reveal hidden biases "
            "and improve trustworthiness. Transparency is fundamental to ethical AI."
        ),
        "categories": "cs.LG, cs.AI", "published": "2023-01-15T00:00:00Z",
        "pdf_downloaded": 1, "text_length": 390,
    },
    {
        "arxiv_id": "2301.00004", "title": "Privacy-Preserving Machine Learning",
        "abstract": "Differential privacy techniques for machine learning models.",
        "full_text": None,
        "categories": "cs.CR, cs.LG", "published": "2023-01-20T00:00:00Z",
        "pdf_downloaded": 0, "text_length": None,
    },
]


def create_test_db(path: Path) -> None:
    from backend.database.schema import init_db
    init_db(path)
    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT OR IGNORE INTO documents "
        "(arxiv_id, title, abstract, full_text, categories, published, pdf_downloaded, text_length) "
        "VALUES (:arxiv_id, :title, :abstract, :full_text, :categories, "
        ":published, :pdf_downloaded, :text_length)",
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test_indexing.db"
    create_test_db(p)
    return p


@pytest.fixture
def pipeline(db_path):
    from backend.indexing.pipeline import IndexingPipeline
    return IndexingPipeline(db_path=db_path, field="both", batch_size=50)
