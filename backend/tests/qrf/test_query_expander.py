"""Tests de QueryExpander — inicialización, expand, vectorize."""
from __future__ import annotations
import numpy as np
import pytest

from .conftest import DIM, MockLSIModel, build_mock_word_index


@pytest.fixture
def expander(mock_model):
    from backend.qrf.query_expander import QueryExpander
    exp = QueryExpander(
        lsi_model=mock_model, top_dims=2, top_terms_per_dim=5,
        min_correlation=0.1, max_expansion=6,
    )
    exp._model = mock_model
    exp._word_index, exp._idx_to_word = build_mock_word_index(mock_model)
    return exp


def test_expander_initialized(expander, mock_model):
    assert expander._model is not None
    assert 0 < len(expander._word_index) <= len(mock_model.term_ids)
    assert len(expander._idx_to_word) == len(mock_model.term_ids)


def test_expand_returns_original_plus_new_terms(expander):
    query = "attention neural transformer"
    expanded, new_terms = expander.expand(query)
    assert isinstance(expanded, str)
    assert expanded.startswith(query)
    assert len(new_terms) <= expander.max_expansion


def test_expand_does_not_repeat_original_tokens(expander):
    query = "attention neural transformer"
    _, new_terms = expander.expand(query)
    originals = {"attention", "neural", "transformer"}
    for t in new_terms:
        assert t not in originals


def test_expand_out_of_vocab_returns_unchanged(expander):
    expanded, terms = expander.expand("zyxwvut qrstuvw")
    assert expanded == "zyxwvut qrstuvw"
    assert terms == []


def test_vectorize_shape_and_dtype(expander, mock_model):
    vec = expander._vectorize("attention neural transformer learning")
    assert vec.shape == (len(mock_model.term_ids),)
    assert vec.dtype == np.float32


def test_vectorize_has_active_terms(expander):
    vec = expander._vectorize("attention neural transformer learning")
    assert vec.sum() > 0
    assert int((vec > 0).sum()) >= 1
