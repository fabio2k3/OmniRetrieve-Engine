"""
test_lsi_query.py
=================
Tests de lsi_query.py — vectorización de queries al espacio LSI.
Sin I/O real: LSIModel y BD completamente mockeados.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.retrieval.lsi_query import QueryVectorizer, build_word_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_model(n_terms=5, n_docs=10, k=3) -> MagicMock:
    model = MagicMock()
    model.term_ids       = list(range(1, n_terms + 1))
    model.df_map         = {i: 2 for i in range(1, n_terms + 1)}
    model.doc_ids        = [f"doc_{i}" for i in range(n_docs)]
    model.docs_latent    = np.random.rand(n_docs, k).astype(np.float32)
    return model


def _word_index(model) -> dict[str, tuple[int, int]]:
    """Construye un word_index sintético sin tocar la BD."""
    words = ["attention", "transformer", "neural", "network", "model"]
    idx = {}
    for i, w in enumerate(words[:len(model.term_ids)]):
        idx[w] = (i, model.df_map.get(i + 1, 1))
    return idx


# ---------------------------------------------------------------------------
# Tests — QueryVectorizer
# ---------------------------------------------------------------------------

class TestQueryVectorizer:
    def test_returns_correct_shape(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        vec   = vz.vectorize("attention transformer")
        assert vec.shape == (5,)

    def test_returns_float32(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        assert vz.vectorize("attention").dtype == np.float32

    def test_known_token_produces_nonzero(self):
        model = _mock_model(n_terms=5)
        wi    = _word_index(model)  # "attention" → (0, 2)
        vz    = QueryVectorizer(model=model, word_index=wi)
        vec   = vz.vectorize("attention")
        assert vec[0] != 0.0

    def test_unknown_token_leaves_zero(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        vec   = vz.vectorize("zzz_unknown_token_zzz")
        assert np.all(vec == 0.0)

    def test_empty_query_returns_zeros(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        vec   = vz.vectorize("")
        assert np.all(vec == 0.0)

    def test_raises_if_model_not_loaded(self):
        model              = _mock_model()
        model.docs_latent  = None   # simula modelo no cargado
        vz = QueryVectorizer(model=model, word_index={})
        with pytest.raises(RuntimeError, match="no cargado"):
            vz.vectorize("query")

    def test_repeated_token_increases_tf(self):
        model = _mock_model(n_terms=5)
        wi    = _word_index(model)
        vz    = QueryVectorizer(model=model, word_index=wi)
        single   = vz.vectorize("attention")
        repeated = vz.vectorize("attention attention attention")
        assert repeated[0] > single[0]


# ---------------------------------------------------------------------------
# Tests — build_word_index
# ---------------------------------------------------------------------------

class TestBuildWordIndex:
    def _mock_db_rows(self):
        """Simula rows devueltas por sqlite: [{term_id, word}, ...]"""
        rows = [
            MagicMock(**{"__getitem__": lambda s, k: {"term_id": i, "word": f"word_{i}"}[k]})
            for i in range(1, 4)
        ]
        # Usar dict-style access
        rows = []
        for i in range(1, 4):
            r = MagicMock()
            r.__getitem__ = lambda s, k, i=i: {"term_id": i, "word": f"word_{i}"}[k]
            rows.append(r)
        return rows

    def test_empty_term_ids_returns_empty(self):
        model = _mock_model()
        model.term_ids = []
        result = build_word_index(model)
        assert result == {}

    def test_index_has_correct_structure(self):
        model = _mock_model(n_terms=3)
        mock_conn = MagicMock()

        rows = []
        for i in range(1, 4):
            r = MagicMock()
            r.__getitem__ = lambda s, k, i=i: {"term_id": i, "word": f"word_{i}"}[k]
            rows.append(r)

        mock_conn.execute.return_value.fetchall.return_value = rows

        with patch("backend.retrieval.lsi_query.get_connection", return_value=mock_conn):
            idx = build_word_index(model)

        assert "word_1" in idx
        row_idx, df = idx["word_1"]
        assert isinstance(row_idx, int)
        assert isinstance(df, int)

    def test_row_idx_is_position_in_term_ids(self):
        model = _mock_model(n_terms=3)
        mock_conn = MagicMock()

        rows = []
        for i in range(1, 4):
            r = MagicMock()
            r.__getitem__ = lambda s, k, i=i: {"term_id": i, "word": f"word_{i}"}[k]
            rows.append(r)

        mock_conn.execute.return_value.fetchall.return_value = rows

        with patch("backend.retrieval.lsi_query.get_connection", return_value=mock_conn):
            idx = build_word_index(model)

        # term_ids = [1, 2, 3] → word_1 debe estar en row_idx=0
        assert idx["word_1"][0] == 0
        assert idx["word_2"][0] == 1
        assert idx["word_3"][0] == 2
