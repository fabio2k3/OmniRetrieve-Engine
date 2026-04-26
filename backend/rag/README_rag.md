# OmniRetrieve — Modulo RAG

Pipeline de generacion aumentada por recuperacion (RAG) para respuestas grounded sobre chunks recuperados del corpus.

El modulo coordina estas etapas:

1. Recuperacion (`RetrieverProtocol`).
2. Reranking opcional (`RerankerProtocol`).
3. Construccion de contexto con citas (`ContextBuilder`).
4. Construccion de prompt (`PromptBuilder`).
5. Generacion final con Ollama (`Generator`).

---

## Estructura de archivos

```
backend/rag/
├── context_builder.py  ← construye contexto con chunks + citas
├── prompt_builder.py   ← plantilla de prompt grounded
├── generator.py        ← wrapper de Ollama para generacion
├── pipeline.py         ← orquestador RAG (search / ask)
└── __init__.py         ← exports publicos
```

---

## Instalacion

```bash
pip install ollama
```

`ollama` debe estar levantado localmente (por defecto en `http://localhost:11434`).

Tambien necesitas tener descargado el modelo que configures, por ejemplo:

```bash
ollama pull llama3.2:3b
```

---

## Flujo del pipeline

```
query
  ↓
RetrieverProtocol.retrieve(top_n=candidate_k)
  ↓
RerankerProtocol.rerank(top_k)   (opcional)
  ↓
ContextBuilder.build(max_chunks, max_chars)
  ↓
PromptBuilder.build(query, context)
  ↓
Generator.generate(prompt)
  ↓
respuesta + sources
```

---

## Uso programatico

```python
from backend.rag import RAGPipeline, Generator
from backend.retrieval import HybridRetriever, CrossEncoderReranker

# retriever y reranker se inyectan desde los modulos de retrieval
retriever = HybridRetriever(...)
reranker = CrossEncoderReranker()

pipeline = RAGPipeline(
    retriever=retriever,
    reranker=reranker,
    generator=Generator(model="llama3.2:3b"),
)

# Solo retrieval/rerank (sin LLM)
results = pipeline.search("transformer attention", top_k=5)

# Flujo completo RAG
answer = pipeline.ask(
    query="Como funciona self-attention?",
    top_k=8,
    candidate_k=40,
    max_chunks=5,
    max_chars=400,
)

print(answer["answer"])
print(answer["sources"])
```

---

## API publica

### `RAGPipeline.search(query, top_k=10, candidate_k=50)`

Ejecuta retrieval + rerank opcional y devuelve una lista de resultados chunk-level, sin llamar al LLM.

### `RAGPipeline.ask(query, top_k=10, candidate_k=50, max_chunks=5, max_chars=400, include_debug=False)`

Ejecuta el flujo RAG completo y devuelve:

```python
{
    "query": "...",
    "answer": "...",
    "sources": [
        {
            "citation": 1,
            "chunk_id": 123,
            "arxiv_id": "2401.00001",
            "chunk_index": 2,
            "title": "...",
            "year": "2024",
            "score": 0.91,
            "score_type": "rerank",
        }
    ],
    # opcionales si include_debug=True
    "context": "...",
    "prompt": "...",
}
```

---

## Notas de diseno

- El pipeline es desacoplado por inyeccion de dependencias.
- `Generator` esta orientado a Ollama para ejecucion local.
- `ContextBuilder` mantiene trazabilidad con citas numericas.
- `PromptBuilder` fuerza respuesta grounded y salida conservadora si falta evidencia.
