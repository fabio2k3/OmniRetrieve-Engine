# `backend/eval` — Sistema de Evaluación

Pipeline completo para medir y comparar la calidad del sistema RAG.
Cubre desde la generación del dataset hasta la detección de regresiones entre corridas.

---

## Arquitectura completa

```
backend/eval/
│
├── schema.py                  # EvalCase, EvalDataset
├── paraphraser.py             # Paráfrasis semántica con Ollama
├── dataset_generator.py       # Generador de dataset (exact + semantic)
├── generate_dataset.py        # CLI de generación
│
├── retrieval/                 # Bloque 2 — Evaluación de retrieval
│   ├── _types.py              # RawHit, AggregatedMetrics, MetricSet
│   ├── metrics.py             # hit_at_k(), mrr(), ndcg_at_k() — puras
│   ├── scorer.py              # score_case() → RawHit
│   ├── aggregator.py          # aggregate() → AggregatedMetrics
│   ├── runner.py              # EvalRunner
│   ├── report.py              # format_summary(), save_json()
│   └── evaluate.py            # CLI
│
├── rag/                       # Bloque 3 — Evaluación RAG end-to-end
│   ├── _types.py              # DimensionScore, RAGJudgement, RAGAggregatedMetrics
│   ├── prompts.py             # Plantillas de prompt del juez
│   ├── judge.py               # OllamaJudge
│   ├── scorer.py              # score_rag_case() → RAGJudgement
│   ├── aggregator.py          # aggregate() → RAGAggregatedMetrics
│   ├── runner.py              # RAGEvalRunner
│   ├── report.py              # format_summary(), save_json(), save_judgements()
│   └── evaluate.py            # CLI
│
└── compare/                   # Bloque 4 — Comparador de corridas
    ├── _types.py              # MetricDelta, ComparisonResult
    ├── differ.py              # detect_type(), extract_metrics(), compare_reports()
    ├── report.py              # format_summary(), save_json()
    └── compare.py             # CLI
```

---

## Flujo completo

```
┌────────────────────────────────────────────────────────┐
│  1. Generar dataset                                    │
│     python -m backend.eval.generate_dataset            │
│     → backend/data/eval/dataset.json                  │
└───────────────────────┬────────────────────────────────┘
                        │
         ┌──────────────┴─────────────┐
         ▼                            ▼
┌──────────────────────┐  ┌───────────────────────────┐
│  2. Eval Retrieval   │  │  3. Eval RAG end-to-end   │
│  retrieval.evaluate  │  │  rag.evaluate             │
│  → report_ret.json   │  │  → report_rag.json        │
└──────────────────────┘  └───────────────────────────┘
         │                            │
         └──────────────┬─────────────┘
                        ▼
             ┌─────────────────────┐
             │  4. Comparar        │
             │  compare.compare    │
             │  → comparison.json  │
             └─────────────────────┘
```

---

## Uso rápido

### Paso 1 — Generar el dataset

```bash
# Solo exact (sin LLM, rápido)
python -m backend.eval.generate_dataset \
    --sample-size 100 --no-semantic \
    --output backend/data/eval/dataset.json

# Exact + semántico (requiere Ollama)
python -m backend.eval.generate_dataset \
    --sample-size 50 \
    --output backend/data/eval/dataset.json
```

### Paso 2 — Evaluar el retriever

```bash
python -m backend.eval.retrieval.evaluate \
    --dataset  backend/data/eval/dataset.json \
    --retriever hybrid --top-k 10 \
    --output backend/data/eval/report_hybrid.json
```

### Paso 3 — Evaluar el pipeline RAG

```bash
python -m backend.eval.rag.evaluate \
    --dataset    backend/data/eval/dataset.json \
    --output     backend/data/eval/rag_report.json \
    --judgements backend/data/eval/rag_judgements.json
```

### Paso 4 — Comparar dos corridas

```bash
python -m backend.eval.compare.compare \
    --baseline  backend/data/eval/report_v1.json \
    --candidate backend/data/eval/report_v2.json \
    --output    backend/data/eval/comparison.json
```

> **CI/CD:** el comparador devuelve código de salida `1` si hay regresiones, `0` si no.

---

## Tipos de casos del dataset

| Tipo | Descripción | Estresa |
|---|---|---|
| `exact` | Fragmento literal del chunk como query | Retrieval léxico (LSI) |
| `semantic` | Paráfrasis LLM sin vocabulario compartido | Retrieval denso (FAISS) + reranker |

## Dimensiones RAG evaluadas

| Dimensión | Detecta |
|---|---|
| **Faithfulness** | Alucinaciones |
| **Answer Relevance** | Respuestas off-topic |
| **Context Relevance** | Fallos de retrieval |

---

## Tests

```bash
pytest backend/tests/test_eval_dataset.py \
       backend/tests/eval_retrieval/ \
       backend/tests/eval_rag/ \
       backend/tests/eval_compare/ -v
```

Todos los tests usan BD en memoria y Ollama mockeado — sin dependencias externas.

---

## Principio de diseño

Cada archivo tiene exactamente **una razón para cambiar**:

| Archivo | Cambia si… |
|---|---|
| `schema.py` | Cambias la estructura de EvalCase / EvalDataset |
| `paraphraser.py` | Cambias la estrategia de paráfrasis |
| `dataset_generator.py` | Cambias el algoritmo de muestreo |
| `retrieval/metrics.py` | Cambias la fórmula de Hit@K, MRR o NDCG |
| `retrieval/scorer.py` | Cambias cómo se detecta un chunk en los resultados |
| `rag/prompts.py` | Cambias las instrucciones del juez LLM |
| `rag/judge.py` | Cambias el backend LLM (ej. Ollama → OpenAI) |
| `compare/differ.py` | Cambias qué métricas se extraen o cómo se calculan los deltas |
| Cualquier `report.py` | Cambias el formato de salida |
| Cualquier `evaluate.py` | Cambias los argumentos de la CLI |
