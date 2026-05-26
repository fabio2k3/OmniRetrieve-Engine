"""
test_lsi_retriever.py
=====================
Tests del LSIRetriever.

Usa DB de archivo real (tmp_path) en lugar de patch() sobre atributos
de módulo, lo que elimina la dependencia frágil del nombre del import.
"""

import sqlite3
import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.retrieval.protocols import RetrievalResult, RetrieverProtocol
from backend.retrieval.lsi_retriever import LSIRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K = 4


def _mock_model(n_docs: int = 3) -> MagicMock:
    model = MagicMock()
    model.doc_ids     = [f"arxiv_{i:04d}" for i in range(n_docs)]
    model.term_ids    = list(range(10))
    model.df_map      = {i: 2 for i in range(10)}
    model.docs_latent = np.eye(n_docs, K, dtype=np.float32)
    q_vec = np.zeros(K, dtype=np.float32)
    q_vec[0] = 1.0
    model.project_query.return_value = q_vec
    return model


def _mock_vectorizer() -> MagicMock:
    v = MagicMock()
    v.vectorize.return_value = np.zeros(10, dtype=np.float32)
    return v


def _make_db_with_chunks(tmp_path, arxiv_ids: list[str]) -> object:
    """
    Crea una BD SQLite real con documentos y chunks para los arxiv_ids dados.
    Devuelve el path al archivo.
    """
    from backend.database.schema import init_db
    from backend.database.chunk_repository import save_chunks

    p = tmp_path / "lsi_test.db"
    init_db(p)

    conn = sqlite3.connect(str(p))
    for aid in arxiv_ids:
        conn.execute(
            "INSERT OR IGNORE INTO documents "
            "(arxiv_id, title, abstract, authors, full_text, pdf_downloaded, published, pdf_url) "
            "VALUES (?, ?, '', '', '', 1, '2023-01-01', '')",
            (aid, f"Title for {aid}"),
        )
    conn.commit()
    conn.close()

    for aid in arxiv_ids:
        save_chunks(aid, [
            f"First chunk text for {aid}.",
            f"Second chunk text for {aid}.",
        ], db_path=p)

    return p


def _retriever_with_db(tmp_path, n_docs=3):
    """Devuelve (retriever, model_mock, db_path) usando DB real."""
    model = _mock_model(n_docs)
    db_path = _make_db_with_chunks(tmp_path, model.doc_ids)

    r = LSIRetriever()
    r._model      = model
    r._vectorizer = _mock_vectorizer()
    r._db_path    = db_path
    return r, model, db_path


# ---------------------------------------------------------------------------
# Tests — RetrieverProtocol
# ---------------------------------------------------------------------------

class TestLSIRetrieverProtocol:
    def test_retrieve_returns_list(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path)
        result = r.retrieve("query", top_n=10)
        assert isinstance(result, list)

    def test_retrieve_returns_retrieval_results(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path)
        results = r.retrieve("query", top_n=10)
        for res in results:
            assert isinstance(res, RetrievalResult)

    def test_retriever_implements_protocol(self):
        assert issubclass(LSIRetriever, RetrieverProtocol)


# ---------------------------------------------------------------------------
# Tests — chunk_ids reales
# ---------------------------------------------------------------------------

class TestRealChunkIds:
    def test_chunk_ids_are_integers(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path)
        results = r.retrieve("query", top_n=10)
        for res in results:
            assert isinstance(res.chunk_id, int)

    def test_no_synthetic_string_chunk_ids(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path)
        results = r.retrieve("query", top_n=10)
        for res in results:
            assert "__lsi__" not in str(res.chunk_id)

    def test_score_type_is_cosine_lsi(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path, n_docs=1)
        results = r.retrieve("query", top_n=5)
        assert all(res.score_type == "cosine_lsi" for res in results)


# ---------------------------------------------------------------------------
# Tests — expansión documento → chunks
# ---------------------------------------------------------------------------

class TestDocumentExpansion:
    def test_returns_chunks_not_documents(self, tmp_path):
        """1 doc × 2 chunks → retrieve devuelve 2 resultados."""
        r, _, _ = _retriever_with_db(tmp_path, n_docs=1)
        results = r.retrieve("query", top_n=10)
        assert len(results) == 2

    def test_top_n_limits_results(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path, n_docs=3)
        results = r.retrieve("query", top_n=3)
        assert len(results) <= 3

    def test_chunks_ordered_by_score_desc(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path, n_docs=3)
        results = r.retrieve("query", top_n=10)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_inherits_doc_score(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path, n_docs=1)
        results = r.retrieve("query", top_n=10)
        # All 2 chunks of the same doc should have the same score
        scores = {res.score for res in results}
        assert len(scores) == 1


# ---------------------------------------------------------------------------
# Tests — casos borde
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query_returns_empty(self, tmp_path):
        r, _, _ = _retriever_with_db(tmp_path)
        assert r.retrieve("", top_n=10) == []
        assert r.retrieve("   ", top_n=10) == []

    def test_raises_if_not_loaded(self):
        r = LSIRetriever()
        with pytest.raises(RuntimeError, match=r"load\(\)"):
            r.retrieve("query")

    def test_all_zero_scores_returns_empty(self, tmp_path):
        r, model, _ = _retriever_with_db(tmp_path, n_docs=3)
        model.project_query.return_value = np.zeros(K, dtype=np.float32)
        results = r.retrieve("query", top_n=10)
        assert results == []