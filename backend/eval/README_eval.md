# OmniRetrieve — Módulo `eval`

Sistema de evaluación offline en tres capas:

1. **Generación de datasets** — crea casos de test a partir de chunks reales.
2. **Evaluación de retrieval** — mide si el retriever devuelve el chunk correcto.
3. **Evaluación RAG** — mide la calidad de la respuesta generada con juez LLM.
4. **Comparación de corridas** — compara dos JSON de resultados y calcula deltas.

---

## Estructura de archivos

```
backend/eval/
├── schema.py              ← EvalCase, EvalDataset (corpus de casos de test)
├── dataset_generator.py   ← DatasetGenerator: genera EvalDataset desde chunks de BD
├── paraphraser.py         ← Paraphraser: casos semánticos via Ollama
├── query_generator.py     ← QueryGenerator: casos "generated" via Ollama
├── generate_dataset.py    ← CLI: genera y guarda el EvalDataset
│
├── retrieval/             ← Evaluación del retriever (ground truth de chunks)
│   ├── _types.py          ← RawHit, MetricSet, AggregatedMetrics
│   ├── metrics.py         ← hit_at_k(), mrr(), ndcg_at_k() — funciones puras
│   ├── scorer.py          ← score_case(): evalúa un EvalCase contra el retriever
│   ├── aggregator.py      ← aggregate(): RawHit[] → AggregatedMetrics
│   ├── runner.py          ← EvalRunner: itera EvalDataset y recoge RawHit
│   ├── report.py          ← format_summary(), save_json(), load_json()
│   ├── generate_dataset.py← CLI: genera EvalDataset para retrieval
│   └── evaluate.py        ← CLI: ejecuta la evaluación de retrieval
│
├── rag/                   ← Evaluación del pipeline RAG (sin ground truth de chunks)
│   ├── schema.py          ← RAGQuery, RAGQuerySet (solo queries, sin expected_chunk_id)
│   ├── _types.py          ← DimensionScore, RAGJudgement, DimensionStats, RAGAggregatedMetrics
│   ├── prompts.py         ← faithfulness_prompt(), answer_relevance_prompt()
│   ├── judge.py           ← OllamaJudge: grammar-constrained JSON + fallback parser
│   ├── scorer.py          ← score_rag_query(): orquesta juez para una consulta
│   ├── aggregator.py      ← aggregate(): RAGJudgement[] → RAGAggregatedMetrics
│   ├── runner.py          ← RAGEvalRunner: itera RAGQuerySet y llama al pipeline
│   ├── report.py          ← format_summary(), save_json(), save_judgements()
│   ├── generate_dataset.py← CLI: convierte EvalDataset en RAGQuerySet
│   ├── generate_queries.py← CLI: genera RAGQuerySet desde texto plano
│   └── evaluate.py        ← CLI: ejecuta la evaluación RAG
│
└── compare/               ← Compara dos JSON de resultados (retrieval o RAG)
    ├── _types.py          ← MetricDelta, ComparisonResult
    ├── differ.py          ← detect_type(), extract_metrics(), compute_deltas()
    ├── report.py          ← format_summary(), save_json(), load_json()
    └── compare.py         ← CLI: python -m backend.eval.compare.compare A.json B.json
```

---

## Capa 1 — Generación de datasets

### `EvalCase` y `EvalDataset` (`schema.py`)

`EvalCase` representa un caso de test con **ground truth de chunk**:

| Campo | Descripción |
|---|---|
| `case_id` | Identificador: `"exact_0042"`, `"semantic_0007"`, `"generated_0015"` |
| `case_type` | `"exact"` \| `"semantic"` \| `"generated"` |
| `query` | Texto que se envía al retriever o pipeline |
| `expected_chunk_id` | `id` (PK) del chunk que debe aparecer en los resultados |
| `expected_arxiv_id` | Documento fuente |
| `expected_chunk_index` | Posición del chunk en el documento |
| `source_text` | Texto íntegro del chunk de referencia |
| `fragment_used` | Semilla usada para generar la query |
| `paraphrase_model` | Modelo Ollama usado (`None` para `exact`) |
| `metadata` | Campo libre (título del doc, char_count, etc.) |

`EvalDataset` es una colección de `EvalCase` con metadatos de generación.
Se serializa en JSON y tiene métodos `exact_cases()`, `semantic_cases()`,
`generated_cases()` y `save(path)` / `EvalDataset.load(path)`.

### Tres tipos de caso

| Tipo | Cómo se genera | Qué estresa | LLM |
|---|---|---|---|
| `exact` | Fragmento literal de 2 oraciones del chunk | Retrieval léxico (LSI) | No |
| `semantic` | Paráfrasis del fragmento (Ollama) | Retrieval semántico (FAISS/dense) | Sí |
| `generated` | Pregunta real de usuario sobre el chunk (Ollama) | Pipeline completo RAG | Sí |

El tipo `generated` es el más representativo: los usuarios hacen preguntas,
no pegan trozos de papers.

### `DatasetGenerator`

```python
from backend.eval.dataset_generator import DatasetGenerator

# Dataset completo (recomendado para eval RAG end-to-end)
gen = DatasetGenerator(
    sample_size       = 100,
    include_exact     = True,
    include_semantic  = False,     # requiere Ollama
    include_generated = True,      # requiere Ollama
    min_chunk_chars   = 200,
    fragment_sentences= 2,
    query_gen_model   = "llama3.2:3b",
    seed              = 42,
)
ds = gen.generate()
ds.save(Path("backend/data/eval/dataset.json"))

print(ds)
# EvalDataset(total=185, exact=95, semantic=0, generated=90, …)
```

El muestreo es **estratificado por documento**: se toma un número proporcional
de chunks de cada `arxiv_id` para garantizar cobertura temática diversa.

### `Paraphraser` (`paraphraser.py`)

Genera paráfrasis con léxico diferente al original para el tipo `semantic`.
Valida con similitud Jaccard: si `Jaccard(original, paráfrasis) > 0.60`
el resultado se rechaza y se reintenta (hasta `max_retries`).

Usa temperatura 0.55 (mayor diversidad léxica que la generación RAG).

### `QueryGenerator` (`query_generator.py`)

Genera preguntas realistas de usuario para el tipo `generated`.
Validaciones más estrictas que `Paraphraser`:

| Validación | Criterio |
|---|---|
| Longitud mínima | ≥ 15 caracteres |
| Longitud máxima | ≤ 250 caracteres |
| Termina en `?` | Obligatorio |
| Similitud Jaccard con el chunk | < 0.25 (umbral más estricto que Paraphraser) |

El LLM NO debe copiar frases del paper; la query debe ser lo que
un usuario escribiría en un buscador.

---

## Capa 2 — Evaluación de retrieval (`eval/retrieval/`)

Mide si el retriever devuelve el chunk correcto para cada `EvalCase`.
Requiere `EvalDataset` (con `expected_chunk_id`).

### Métricas (`metrics.py`)

Funciones puras sin imports del proyecto:

| Función | Descripción |
|---|---|
| `hit_at_k(ranks, k)` | Fracción de casos donde el chunk relevante aparece en top-K |
| `mrr(ranks)` | Mean Reciprocal Rank (promedio de 1/rank) |
| `ndcg_at_k(ranks, k)` | NDCG@K con relevancia binaria |

`ranks` es `list[int | None]`: posición 1-based del chunk esperado, o `None`
si no aparece en los resultados.

### Tipos de datos

**`RawHit`** — resultado de un caso individual:
`case_id`, `case_type`, `expected_chunk_id`, `found`, `rank`, `top_k`, `n_results_returned`

**`MetricSet`** — métricas para un subset (all / exact / semantic):
`hit_at_k`, `mrr`, `ndcg_at_k`, `n_cases`, `n_found`

**`AggregatedMetrics`** — resultados agregados de una corrida completa:
`all`, `exact` y `semantic` (cada uno un `MetricSet` o `None`)

### Flujo de evaluación

```python
from backend.eval.retrieval.runner import EvalRunner
from backend.eval.retrieval.aggregator import aggregate
from backend.eval.retrieval.report import format_summary, save_json
from backend.eval.schema import EvalDataset

ds       = EvalDataset.load(Path("dataset.json"))
runner   = EvalRunner(retriever=my_retriever, top_k=10)
hits     = runner.run(ds)                    # → list[RawHit]
metrics  = aggregate(hits, top_k=10)         # → AggregatedMetrics
print(format_summary(metrics, "LSI v1"))
save_json(metrics, Path("results_lsi.json"))
```

### CLI

```bash
python -m backend.eval.retrieval.evaluate \
  --dataset backend/data/eval/dataset.json \
  --retriever lsi \
  --top-k 10 \
  --output backend/data/eval/results_lsi.json
```

---

## Capa 3 — Evaluación RAG (`eval/rag/`)

Mide la calidad de la respuesta generada por el pipeline RAG.
**No usa ground truth de chunks** — evalúa la respuesta en sí.

### `RAGQuery` y `RAGQuerySet` (`rag/schema.py`)

`RAGQuerySet` es una lista de `RAGQuery` (solo `query_id` + `query`).
Se puede crear desde:
- Un `EvalDataset` (extrae las queries)
- Un fichero de texto plano (una query por línea)
- Programáticamente

```python
from backend.eval.rag.schema import RAGQuerySet

# Desde fichero de texto
qs = RAGQuerySet.from_text_file(Path("mis_queries.txt"))
qs.save(Path("query_set.json"))

# Carga
qs = RAGQuerySet.load(Path("query_set.json"))
```

### Dimensiones evaluadas

| Dimensión | Pregunta al juez |
|---|---|
| `faithfulness` | ¿Está la respuesta fundamentada en los documentos recuperados? Detecta alucinaciones |
| `answer_relevance` | ¿Responde la respuesta la pregunta de forma útil y pertinente? |

La puntuación es **1–5** (escala Likert) normalizada a **[0.0, 1.0]**.

### `OllamaJudge` (`rag/judge.py`)

El juez envía los prompts a Ollama con `format="json"` (grammar-constrained
decoding a nivel de tokens), lo que garantiza JSON válido incluso con modelos
pequeños. Si el cliente Ollama no soporta `format`, reintenta sin él.

Para parsear la respuesta tiene 4 estrategias en cascada:
1. Parseo directo del texto completo
2. Extracción de bloque markdown ` ```json … ``` `
3. Primer objeto `{ … }` encontrado
4. Reparación básica de JSON malformado (comillas faltantes en `reason`)

### Flujo de evaluación RAG

```python
from backend.eval.rag.runner import RAGEvalRunner
from backend.eval.rag.judge import OllamaJudge
from backend.eval.rag.aggregator import aggregate
from backend.eval.rag.report import format_summary, save_json, save_judgements
from backend.eval.rag.schema import RAGQuerySet

qs       = RAGQuerySet.load(Path("query_set.json"))
judge    = OllamaJudge(model="llama3.2:3b", temperature=0.0)
runner   = RAGEvalRunner(pipeline=my_rag, judge=judge)
judgements = runner.run(qs)              # → list[RAGJudgement]
metrics    = aggregate(judgements)       # → RAGAggregatedMetrics
print(format_summary(metrics, "HybridRAG v2"))
save_json(metrics, Path("rag_results.json"))
save_judgements(judgements, Path("rag_judgements.json"))
```

### `score_rag_query()` (`rag/scorer.py`)

```python
score_rag_query(
    query_id        = "q_0001",
    query           = "How does attention work?",
    pipeline_output = rag.ask("How does attention work?", include_debug=True),
    judge           = OllamaJudge(),
) → RAGJudgement
```

Llama al juez para cada dimensión por separado. Si alguna falla, anota el
error en `judge_error` pero devuelve igualmente el `RAGJudgement` sin abortar.

### CLI

```bash
python -m backend.eval.rag.evaluate \
  --queries backend/data/eval/query_set.json \
  --output  backend/data/eval/rag_results.json \
  --judge-model llama3.2:3b
```

---

## Capa 4 — Comparación de corridas (`eval/compare/`)

Compara dos JSON de resultados (retrieval o RAG) y calcula deltas:

```bash
python -m backend.eval.compare.compare \
  backend/data/eval/results_lsi.json \
  backend/data/eval/results_hybrid.json
```

`detect_type()` determina automáticamente si los JSON son de retrieval o RAG
comparando sus claves. `compute_deltas()` genera un `MetricDelta` por métrica
con el valor base, el nuevo valor, la diferencia absoluta y el estado
(`improved`, `degraded`, `neutral`).

---

## Tests

```bash
# Capa de dataset (schema, paraphraser, generators)
pytest backend/tests/eval/ -v

pytest backend/tests/eval/test_eval_schema.py        -v  # EvalCase, EvalDataset
pytest backend/tests/eval/test_paraphraser.py        -v  # _jaccard, Paraphraser
pytest backend/tests/eval/test_dataset_generator.py  -v  # DatasetGenerator
pytest backend/tests/eval/test_query_generator.py    -v  # QueryGenerator, _is_valid

# Evaluación RAG
pytest backend/tests/eval_rag/ -v

pytest backend/tests/eval_rag/test_prompts.py        -v  # faithfulness_prompt, answer_relevance_prompt
pytest backend/tests/eval_rag/test_scorer.py         -v  # score_rag_query con _FixedJudge
pytest backend/tests/eval_rag/test_aggregator.py     -v  # aggregate(), estadísticas por dimensión
pytest backend/tests/eval_rag/test_runner.py         -v  # RAGEvalRunner con RAGQuerySet
pytest backend/tests/eval_rag/test_judge.py          -v  # OllamaJudge, _extract_json()
pytest backend/tests/eval_rag/test_report.py         -v  # format_summary, save_json, save_judgements

# Evaluación de retrieval
pytest backend/tests/eval_retrieval/ -v

# Comparación
pytest backend/tests/eval_compare/ -v
```

### Estrategia de tests

- **Sin Ollama**: `Paraphraser`, `QueryGenerator`, `OllamaJudge` y `RAGEvalRunner`
  se testean con `MagicMock` o clases `_FixedJudge` que devuelven respuestas
  controladas, sin llamadas reales al LLM.
- **Sin BD**: `DatasetGenerator` se testea con una conexión SQLite en memoria
  (`":memory:"`) creada en cada test.
- **Funciones puras**: `metrics.py` se testea directamente con listas de `rank`
  sin ningún mock.