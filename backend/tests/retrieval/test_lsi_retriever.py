"""
test_lsi_retriever.py
=====================
Tests del LSIRetriever refactorizado.

Verifica:
· Implementa RetrieverProtocol (devuelve list[RetrievalResult]).
· chunk_ids son enteros reales (no strings sintéticos).
· Expansión documento → chunks funciona correctamente.
· Manejo de casos borde (query vacía, sin resultados, modelo no cargado).
"""

import sqlite3
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.retrieval.protocols import RetrievalResult, RetrieverProtocol
from backend.retrieval.lsi_retriever import LSIRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K = 4   # dimensión del espacio latente en todos los tests

def _mock_model(n_docs: int = 3) -> MagicMock:
    """
    Mock de LSIModel.
    docs_latent tiene forma (n_docs, K).
    project_query devuelve el primer vector canónico de dim K
    → alta similitud con doc_0, baja con los demás.
    """
    model = MagicMock()
    model.doc_ids     = [f"arxiv_{i:04d}" for i in range(n_docs)]
    model.term_ids    = list(range(10))
    model.df_map      = {i: 2 for i in range(10)}
    # docs_latent: (n_docs, K)
    model.docs_latent = np.eye(n_docs, K, dtype=np.float32)
    # project_query devuelve vector de dim K
    q_vec = np.zeros(K, dtype=np.float32)
    q_vec[0] = 1.0
    model.project_query.return_value = q_vec
    return model


def _mock_vectorizer() -> MagicMock:
    v = MagicMock()
    v.vectorize.return_value = np.zeros(10, dtype=np.float32)
    return v


def _in_memory_db(arxiv_ids: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE documents (
            arxiv_id TEXT PRIMARY KEY, title TEXT, authors TEXT, pdf_url TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT, chunk_index INTEGER, text TEXT
        );
    """)
    for aid in arxiv_ids:
        conn.execute("INSERT INTO documents VALUES (?,?,?,?)",
                     (aid, f"Title {aid}", "Author", "url"))
        conn.execute("INSERT INTO chunks (arxiv_id,chunk_index,text) VALUES (?,0,?)",
                     (aid, f"chunk0 of {aid}"))
        conn.execute("INSERT INTO chunks (arxiv_id,chunk_index,text) VALUES (?,1,?)",
                     (aid, f"chunk1 of {aid}"))
    conn.commit()
    return conn


def _retriever_ready(n_docs: int = 3) -> tuple[LSIRetriever, MagicMock, sqlite3.Connection]:
    """Devuelve un LSIRetriever listo para usar con mocks inyectados."""
    r     = LSIRetriever()
    model = _mock_model(n_docs)
    r._model      = model
    r._vectorizer = _mock_vectorizer()
    conn  = _in_memory_db(model.doc_ids)
    return r, model, conn


# ---------------------------------------------------------------------------
# Tests — protocolo
# ---------------------------------------------------------------------------

class TestLSIRetrieverProtocol:
    def test_implements_retriever_protocol(self):
        assert isinstance(LSIRetriever(), RetrieverProtocol)

    def test_retrieve_returns_list(self):
        r, _, conn = _retriever_ready()
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("test query", top_n=5)
        assert isinstance(results, list)

    def test_retrieve_returns_retrieval_results(self):
        r, _, conn = _retriever_ready()
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("test query", top_n=5)
        assert all(isinstance(x, RetrievalResult) for x in results)


# ---------------------------------------------------------------------------
# Tests — chunk_ids son enteros reales
# ---------------------------------------------------------------------------

class TestRealChunkIds:
    def test_chunk_ids_are_integers(self):
        r, _, conn = _retriever_ready()
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        for res in results:
            assert isinstance(res.chunk_id, int), \
                f"chunk_id debe ser int, no {type(res.chunk_id)}: {res.chunk_id!r}"

    def test_no_synthetic_string_chunk_ids(self):
        r, _, conn = _retriever_ready()
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        for res in results:
            assert "__lsi__" not in str(res.chunk_id)

    def test_score_type_is_cosine_lsi(self):
        r, _, conn = _retriever_ready(n_docs=1)
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=5)
        assert all(res.score_type == "cosine_lsi" for res in results)


# ---------------------------------------------------------------------------
# Tests — expansión documento → chunks
# ---------------------------------------------------------------------------

class TestDocumentExpansion:
    def test_returns_chunks_not_documents(self):
        """1 doc × 2 chunks → retrieve devuelve 2 resultados."""
        r, _, conn = _retriever_ready(n_docs=1)
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        assert len(results) == 2

    def test_top_n_limits_results(self):
        r, _, conn = _retriever_ready(n_docs=3)  # 3 docs × 2 chunks = 6
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=3)
        assert len(results) <= 3

    def test_chunks_ordered_by_score_desc(self):
        r, _, conn = _retriever_ready(n_docs=3)
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_inherits_doc_score(self):
        """Todos los chunks de un documento deben tener el mismo score."""
        r, _, conn = _retriever_ready(n_docs=1)
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        scores = {res.score for res in results}
        assert len(scores) == 1   # todos los chunks del mismo doc tienen el mismo score


# ---------------------------------------------------------------------------
# Tests — casos borde
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query_returns_empty(self):
        r, _, _ = _retriever_ready()
        assert r.retrieve("", top_n=10) == []
        assert r.retrieve("   ", top_n=10) == []

    def test_raises_if_not_loaded(self):
        r = LSIRetriever()
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            r.retrieve("query")

    def test_all_zero_scores_returns_empty(self):
        r, model, conn = _retriever_ready(n_docs=3)
        model.project_query.return_value = np.zeros(K, dtype=np.float32)
        with patch("backend.retrieval.lsi_retriever.get_connection", return_value=conn):
            results = r.retrieve("query", top_n=10)
        assert results == []
