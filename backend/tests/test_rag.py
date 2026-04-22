"""
test_rag.py
===========
Tests unitarios del modulo RAG.

Valida:
- Construccion de contexto y sources.
- Construccion de prompt grounded.
- Flujo search() y ask() del RAGPipeline.

No requiere Ollama real: usa un generador fake inyectado.
"""

from __future__ import annotations

from backend.rag.context_builder import ContextBuilder
from backend.rag.prompt_builder import PromptBuilder
from backend.rag.pipeline import RAGPipeline
from backend.retrieval.protocols import RetrievalResult


class _FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, top_n: int = 20) -> list[RetrievalResult]:
        return self._results[:top_n]


class _FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        out: list[RetrievalResult] = []
        for c in reversed(candidates[:top_k]):
            out.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    arxiv_id=c.arxiv_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    score=c.score + 1.0,
                    score_type="rerank",
                    metadata=c.metadata,
                )
            )
        return out


class _FakeGenerator:
    def generate(self, prompt: str) -> str:
        assert "Documentos:" in prompt
        assert "Pregunta:" in prompt
        return "Respuesta simulada [1]."


def _sample_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=1,
            arxiv_id="2401.00001",
            chunk_index=0,
            text="Self-attention allows each token to attend to all tokens.",
            score=0.91,
            score_type="rrf",
            metadata={"title": "Attention Paper", "year": 2017},
        ),
        RetrievalResult(
            chunk_id=2,
            arxiv_id="2401.00002",
            chunk_index=1,
            text="BERT uses bidirectional context from masked language modeling.",
            score=0.87,
            score_type="rrf",
            metadata={"title": "BERT Paper", "year": 2018},
        ),
    ]


def test_context_builder_build_and_sources() -> None:
    builder = ContextBuilder()
    data = _sample_results()

    context = builder.build(data, max_chunks=2, max_chars=80)
    sources = builder.build_sources(data, max_sources=2)

    assert "[1]" in context
    assert "Attention Paper" in context
    assert len(sources) == 2
    assert sources[0]["citation"] == 1
    assert sources[0]["arxiv_id"] == "2401.00001"


def test_prompt_builder_structure() -> None:
    pb = PromptBuilder()
    prompt = pb.build("Que es self-attention?", "[1] Attention Paper (2017)\n...")

    assert "Documentos:" in prompt
    assert "Pregunta:" in prompt
    assert "Responde con citas." in prompt


def test_rag_pipeline_search_without_reranker() -> None:
    retriever = _FakeRetriever(_sample_results())
    rag = RAGPipeline(retriever=retriever, reranker=None, generator=_FakeGenerator())

    out = rag.search("attention", top_k=2, candidate_k=2)
    assert len(out) == 2
    assert out[0]["score_type"] == "rrf"


def test_rag_pipeline_ask_with_reranker() -> None:
    retriever = _FakeRetriever(_sample_results())
    reranker = _FakeReranker()
    rag = RAGPipeline(retriever=retriever, reranker=reranker, generator=_FakeGenerator())

    answer = rag.ask(
        query="Explica attention.",
        top_k=2,
        candidate_k=2,
        max_chunks=2,
        max_chars=120,
        include_debug=True,
    )

    assert "answer" in answer
    assert answer["answer"].startswith("Respuesta simulada")
    assert len(answer["sources"]) == 2
    assert "context" in answer
    assert "prompt" in answer
    assert answer["sources"][0]["score_type"] == "rerank"
