# OmniRetrieve — Módulo `rag`

Pipeline de **generación aumentada por recuperación** (RAG). Coordina cuatro
etapas con responsabilidades separadas: recuperación de chunks, reranking
opcional, construcción del contexto y generación de respuesta con Ollama.

---

## Estructura de archivos

```
backend/rag/
├── pipeline.py         ← RAGPipeline: orquestador de las 4 etapas
├── context_builder.py  ← ContextBuilder: formatea chunks con citas [1], [2]…
├── prompt_builder.py   ← PromptBuilder: mensaje de usuario + constante SYSTEM
├── generator.py        ← Generator: wrapper sobre ollama.chat()
└── __init__.py         ← exports: RAGPipeline, ContextBuilder, PromptBuilder, Generator
```

---

## Instalación

```bash
# Instalar el servidor Ollama: https://ollama.com/download
ollama pull llama3.2:3b
```

---

## Flujo de `RAGPipeline`

```
query (str)
        │
        ▼  retriever.retrieve(query, top_n=candidate_k)
        │  list[RetrievalResult]
        │
        ▼  [si reranker] reranker.rerank(query, retrieved, top_k=top_k)
        │  list[RetrievalResult] reordenados
        │
        ▼  ContextBuilder.build(ranked, max_chunks, max_chars)
        │  "[1] Título (año)\nTexto del chunk…\n\n[2] …"
        │
        ▼  PromptBuilder.build(query, context)
        │  "Documents:\n[contexto]\n\nQuestion: …\n\nAnswer with citations."
        │
        ▼  Generator.generate(prompt)
        │    ollama.chat(system=SYSTEM, user=prompt)
        │
        ▼
        dict {query, answer, sources, [context, prompt si include_debug]}
```

---

## `RAGPipeline`

### Constructor

```python
from backend.rag.pipeline import RAGPipeline

rag = RAGPipeline(
    retriever = my_retriever,        # RetrieverProtocol — obligatorio
    reranker  = my_reranker,         # RerankerProtocol — None = sin reranking
    context_builder = ContextBuilder(),   # por defecto si no se pasa
    prompt_builder  = PromptBuilder(),    # por defecto si no se pasa
    generator       = Generator(),        # por defecto si no se pasa
)
```

Cualquier objeto que implemente `RetrieverProtocol` funciona como retriever:
`LSIRetriever`, `EmbeddingRetriever`, `HybridRetriever` o cualquier stub de tests.

### `search()` — solo recuperación

```python
results = rag.search(
    query       = "How does self-attention work?",
    top_k       = 5,    # resultados finales
    candidate_k = 20,   # candidatos antes del reranker
)
```

Devuelve `list[dict]` con estas claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `chunk_id` | `int` | PK de la tabla `chunks` |
| `arxiv_id` | `str` | ID compuesto del documento |
| `chunk_index` | `int` | Posición del chunk en el documento |
| `title` | `str` | Título del artículo |
| `text` | `str` | Primeros 300 caracteres del chunk |
| `score` | `float` | Puntuación del retriever o reranker |
| `score_type` | `str` | `"cosine_lsi"`, `"cosine"`, `"rrf"`, `"rerank"`, etc. |

### `ask()` — pipeline completo

```python
answer = rag.ask(
    query        = "What are the limitations of attention mechanisms?",
    top_k        = 5,
    candidate_k  = 20,
    max_chunks   = 4,     # chunks incluidos en el contexto LLM
    max_chars    = 400,   # caracteres máximos por chunk en el contexto
    include_debug= True,  # añade "context" y "prompt" a la respuesta
)
```

Devuelve `dict` con:

| Clave | Siempre | Descripción |
|---|---|---|
| `query` | ✓ | Query original |
| `answer` | ✓ | Respuesta del LLM |
| `sources` | ✓ | `list[dict]` — fuentes usadas (ver abajo) |
| `context` | solo si `include_debug=True` | Texto del contexto enviado al LLM |
| `prompt` | solo si `include_debug=True` | Prompt completo (mensaje de usuario) |

### `generate_from_results()` — sin retrieval propio

```python
# Útil cuando el retrieval ya lo hizo el HybridRetriever + CrossEncoder
response = rag.generate_from_results(
    query      = "What is attention?",
    results    = ranked_results,   # list[RetrievalResult]
    max_chunks = 4,
    max_chars  = 400,
)
# → {query, answer, sources}
```

Usado por `do_pipeline_ask()` del orquestador, donde el pipeline unificado
ya produce los chunks finales antes de llamar al RAG.

---

## `ContextBuilder`

Formatea los `RetrievalResult` en un bloque de texto con citas numéricas:

```
[1] Attention Is All You Need (2017)
The transformer model uses self-attention to compute representations of
sequences. Multi-head attention allows the model to attend jointly...

[2] BERT: Pre-training of Deep Bidirectional Transformers (2018)
BERT is designed to pre-train deep bidirectional representations from
unlabeled text by jointly conditioning on both left and right context...
```

### `build(results, max_chunks, max_chars)`

- Selecciona `results[:max_chunks]`
- Cada chunk: `[N] Título (año)\n{text[:max_chars]}…`
- Bloques separados por `\n\n`
- Devuelve `""` si `results` está vacío

### `build_sources(results, max_sources)`

Devuelve `list[dict]` para la UI:

| Clave | Descripción |
|---|---|
| `citation` | Número de cita (1-based, coincide con `[N]` en el contexto) |
| `chunk_id` | PK del chunk |
| `arxiv_id` | ID del documento |
| `chunk_index` | Posición en el documento |
| `title` | Título del artículo |
| `year` | Año de publicación (`"n/a"` si no disponible) |
| `url` | `metadata["pdf_url"]` para papers locales, `metadata["url"]` para resultados web |
| `score` | Puntuación del retriever/reranker |
| `score_type` | Tipo de score |

**Cadenas de fallback:**

```
title → metadata["title"] | metadata["document_title"] | metadata["paper_title"] | arxiv_id
year  → metadata["year"]  | metadata["published"][:4]  | "n/a"
url   → metadata["pdf_url"] | metadata["url"] | ""
```

El campo `url` soporta tanto papers locales (ruta al PDF de arXiv) como
resultados de búsqueda web (URL de Tavily/SiteSearcher), lo que permite
a la UI mostrar el enlace correcto en ambos casos.

---

## `PromptBuilder`

Construye el mensaje de **usuario** que se envía a Ollama:

```
Documents:
[1] Attention Is All You Need (2017)
The transformer model uses self-attention…

[2] BERT (2018)
BERT applies bidirectional training…

Question: How does self-attention work?

Answer with citations.
```

La instrucción del sistema (`SYSTEM`) se mantiene como una constante separada
en `PromptBuilder` pero se inyecta por `Generator` como `{"role": "system"}`,
no concatenada al mensaje de usuario. Esto garantiza que el LLM la trate con
prioridad de sistema.

```python
PromptBuilder.SYSTEM = (
    "You are a scientific assistant specialized in AI and ML research.\n"
    "Answer ONLY using the provided documents.\n"
    "Always respond in ENGLISH, regardless of the language of the query "
    "or the retrieved chunks.\n"
    "If the answer is not in the context, reply exactly: "
    "'Not found in sources.'\n"
    "Cite inline evidence using [1], [2], etc."
)
```

---

## `Generator`

Wrapper delgado sobre `ollama.chat()`.

```python
from backend.rag.generator import Generator

gen = Generator(
    model       = "llama3.2:3b",
    temperature = 0.1,           # baja para respuestas deterministas
)
answer = gen.generate(prompt)
```

### Llamada a Ollama

```python
ollama.chat(
    model    = self.model,
    messages = [
        {"role": "system", "content": PromptBuilder.SYSTEM},  # instrucción de sistema
        {"role": "user",   "content": prompt},                 # contexto + pregunta
    ],
    options = {"temperature": self.temperature},
)
```

### Comportamiento ante errores

| Situación | Devuelve |
|---|---|
| Ollama no instalado | `"[Ollama package not installed. Run: pip install ollama]"` |
| Cualquier otra excepción | `"[Generation error: {mensaje del error}]"` |

Nunca lanza excepciones: cualquier fallo produce una cadena descriptiva.

---

## `RetrieverProtocol` y `RerankerProtocol`

Definidos en `backend/retrieval/protocols.py`:

```python
class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_n: int = 20) -> list[RetrievalResult]: ...

class RerankerProtocol(Protocol):
    def rerank(self, query: str, candidates: list[RetrievalResult],
               top_k: int = 10) -> list[RetrievalResult]: ...
```

Cualquier clase que implemente estas firmas funciona sin herencia explícita
(duck typing via `Protocol`).

---

## Tests

Los tests están en `backend/tests/rag/`.

```bash
pytest backend/tests/rag/ -v

pytest backend/tests/rag/test_context_builder.py -v
pytest backend/tests/rag/test_prompt_builder.py  -v
pytest backend/tests/rag/test_rag_pipeline.py    -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_context_builder.py` | Citas `[N]` en el contexto, `build_sources()` con todos los campos, `max_chunks` respetado, `""` para lista vacía |
| `test_prompt_builder.py` | Secciones `Documents:` / `Question:` / `Answer with citations.` en inglés |
| `test_rag_pipeline.py` | `search()` sin reranker devuelve `list[dict]` con `score_type`; `ask()` con `_FakeReranker` devuelve respuesta y fuentes con `score_type="rerank"`; `_FakeGenerator` verifica que el prompt contiene `Documents:` y `Question:` |

Los tests usan `_FakeRetriever`, `_FakeReranker` y `_FakeGenerator` como
stubs para no depender de Ollama ni de modelos de embedding en disco.