"""
test_lsi_query.py
=================
Tests de lsi_query.py — vectorización de queries al espacio LSI.

Usa DB real (db_with_terms fixture) en lugar de patch() sobre atributos
de módulo para evitar problemas de AttributeError con import indirecto.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.retrieval.lsi_query import QueryVectorizer, build_word_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model(n_terms=5, n_docs=10, k=3) -> MagicMock:
    model = MagicMock()
    model.term_ids    = list(range(1, n_terms + 1))
    model.df_map      = {i: 2 for i in range(1, n_terms + 1)}
    model.doc_ids     = [f"doc_{i}" for i in range(n_docs)]
    model.docs_latent = np.random.rand(n_docs, k).astype(np.float32)

    # project_query: devuelve vector de dim k (suma de filas seleccionadas)
    components = np.eye(k, n_terms, dtype=np.float32)

    class _SVD:
        components_ = components
        def transform(self, x): return x @ self.components_.T

    model.svd = _SVD()
    model.project_query = MagicMock(
        side_effect=lambda q: (model.svd.transform(q.reshape(1, -1)).flatten())
    )
    return model


def _word_index(model) -> dict[str, tuple[int, int]]:
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
        wi    = _word_index(model)   # "attention" → (0, 2)
        vz    = QueryVectorizer(model=model, word_index=wi)
        vec   = vz.vectorize("attention")
        assert vec[0] != 0.0

    def test_unknown_token_leaves_zero(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        vec   = vz.vectorize("zyxwvut")
        assert vec.sum() == pytest.approx(0.0)

    def test_empty_query_returns_zeros(self):
        model = _mock_model(n_terms=5)
        vz    = QueryVectorizer(model=model, word_index=_word_index(model))
        vec   = vz.vectorize("")
        assert vec.sum() == pytest.approx(0.0)

    def test_idf_weight_applied(self):
        model = _mock_model(n_terms=5)
        # df=1 gives higher IDF than df=10
        wi_low_df  = {"rareword":   (0, 1)}
        wi_high_df = {"commonword": (0, 100)}
        vz_low  = QueryVectorizer(model=model, word_index=wi_low_df)
        vz_high = QueryVectorizer(model=model, word_index=wi_high_df)
        assert vz_low.vectorize("rareword")[0] > vz_high.vectorize("commonword")[0]


# ---------------------------------------------------------------------------
# Tests — build_word_index (DB real, sin patch)
# ---------------------------------------------------------------------------

class TestBuildWordIndex:
    def test_empty_term_ids_returns_empty(self):
        model = _mock_model()
        model.term_ids = []
        result = build_word_index(model)
        assert result == {}

    def test_index_has_correct_structure(self, db_with_terms):
        """Usa DB real con terms insertados para evitar patch frágil."""
        model = _mock_model(n_terms=3)
        idx = build_word_index(model, db_path=db_with_terms)
        # db_with_terms tiene term_ids 1,2,3 con palabras word_1, word_2, word_3
        assert len(idx) == 3
        assert "word_1" in idx
        row_idx, df = idx["word_1"]
        assert isinstance(row_idx, int)
        assert isinstance(df, int)

    def test_row_idx_is_position_in_term_ids(self, db_with_terms):
        """term_ids = [1, 2, 3] → word_1 debe estar en row_idx=0."""
        model = _mock_model(n_terms=3)
        idx = build_word_index(model, db_path=db_with_terms)
        assert idx["word_1"][0] == 0
        assert idx["word_2"][0] == 1
        assert idx["word_3"][0] == 2

    def test_missing_term_ids_excluded(self, db_with_terms):
        """term_ids que no están en la BD no aparecen en el índice."""
        model = _mock_model(n_terms=3)
        model.term_ids = [1, 2, 999]   # 999 no existe en db_with_terms
        idx = build_word_index(model, db_path=db_with_terms)
        assert "word_999" not in idx