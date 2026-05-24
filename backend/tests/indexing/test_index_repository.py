"""Tests de index_repository — stats, top terms, postings, matriz."""
from __future__ import annotations
import pytest


@pytest.fixture(autouse=True)
def indexed(pipeline):
    """Asegura que el índice esté construido antes de cada test."""
    pipeline.run(reindex=False)


def test_get_index_stats(db_path):
    from backend.database.index_repository import get_index_stats
    idx = get_index_stats(db_path=db_path)
    assert idx["vocab_size"] > 0
    assert idx["total_docs"] > 0
    assert idx["total_postings"] > 0


def test_get_top_terms_has_freq_not_tfidf(db_path):
    from backend.database.index_repository import get_top_terms
    top = get_top_terms("2301.00001", n=10, db_path=db_path)
    assert len(top) > 0
    assert all("freq" in t for t in top)
    assert all("tfidf_weight" not in t for t in top)
    assert all(isinstance(t["freq"], int) for t in top)


def test_get_postings_for_term(db_path):
    from backend.database.index_repository import get_postings_for_term
    postings = get_postings_for_term("bias", db_path=db_path)
    if postings:
        assert all("freq" in p for p in postings)
        assert all("tfidf_weight" not in p for p in postings)


def test_get_postings_for_matrix(db_path):
    from backend.database.index_repository import get_postings_for_matrix
    postings_raw, df_map, doc_ids, term_ids, n_docs = get_postings_for_matrix(db_path=db_path)
    assert len(doc_ids) > 0
    assert len(term_ids) > 0
    assert len(postings_raw) > 0
    assert all(isinstance(p[2], int) for p in postings_raw)
    assert all(isinstance(v, int) for v in df_map.values())
