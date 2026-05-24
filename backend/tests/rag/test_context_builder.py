"""Tests de ContextBuilder — build y build_sources."""
from __future__ import annotations
from backend.rag.context_builder import ContextBuilder
from backend.retrieval.protocols import RetrievalResult


def _sample_results():
    return [
        RetrievalResult(
            chunk_id=1, arxiv_id="2401.00001", chunk_index=0,
            text="Self-attention allows each token to attend to all tokens.",
            score=0.91, score_type="rrf",
            metadata={"title": "Attention Paper", "year": 2017},
        ),
        RetrievalResult(
            chunk_id=2, arxiv_id="2401.00002", chunk_index=1,
            text="BERT uses bidirectional context from masked language modeling.",
            score=0.87, score_type="rrf",
            metadata={"title": "BERT Paper", "year": 2018},
        ),
    ]


def test_build_includes_citation():
    builder = ContextBuilder()
    context = builder.build(_sample_results(), max_chunks=2, max_chars=80)
    assert "[1]" in context
    assert "Attention Paper" in context


def test_build_sources_structure():
    builder = ContextBuilder()
    sources = builder.build_sources(_sample_results(), max_sources=2)
    assert len(sources) == 2
    assert sources[0]["citation"] == 1
    assert sources[0]["arxiv_id"] == "2401.00001"
