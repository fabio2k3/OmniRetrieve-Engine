"""Tests del RAGPipeline — search, ask, con y sin reranker."""
from __future__ import annotations
from backend.rag.pipeline import RAGPipeline
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


class _FakeRetriever:
    def __init__(self, results): self._results = results
    def retrieve(self, query, top_n=20): return self._results[:top_n]


class _FakeReranker:
    def rerank(self, query, candidates, top_k=10):
        return [
            RetrievalResult(
                chunk_id=c.chunk_id, arxiv_id=c.arxiv_id,
                chunk_index=c.chunk_index, text=c.text,
                score=c.score + 1.0, score_type="rerank", metadata=c.metadata,
            )
            for c in reversed(candidates[:top_k])
        ]


class _FakeGenerator:
    def generate(self, prompt):
        # Prompt is in English
        assert "Documents:" in prompt
        assert "Question:" in prompt
        return "Simulated answer [1]."


def test_search_without_reranker():
    rag = RAGPipeline(
        retriever=_FakeRetriever(_sample_results()),
        reranker=None,
        generator=_FakeGenerator(),
    )
    out = rag.search("attention", top_k=2, candidate_k=2)
    assert len(out) == 2
    assert out[0]["score_type"] == "rrf"


def test_ask_with_reranker_returns_answer():
    rag = RAGPipeline(
        retriever=_FakeRetriever(_sample_results()),
        reranker=_FakeReranker(),
        generator=_FakeGenerator(),
    )
    answer = rag.ask(
        query="Explain attention.",
        top_k=2, candidate_k=2,
        max_chunks=2, max_chars=120,
        include_debug=True,
    )
    assert "answer" in answer
    assert answer["answer"].startswith("Simulated answer")
    assert len(answer["sources"]) == 2
    assert "context" in answer
    assert "prompt" in answer


def test_ask_sources_have_rerank_type():
    rag = RAGPipeline(
        retriever=_FakeRetriever(_sample_results()),
        reranker=_FakeReranker(),
        generator=_FakeGenerator(),
    )
    answer = rag.ask("query", top_k=2, candidate_k=2, max_chunks=2, max_chars=120)
    assert answer["sources"][0]["score_type"] == "rerank"