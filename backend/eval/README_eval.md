# `backend/eval` — Sistema de Evaluación

Pipeline completo para medir la calidad del sistema RAG, desde la generación
del dataset hasta la detección de regresiones entre corridas.

---

## Índice

1. [Prerequisitos](#prerequisitos)
2. [Arquitectura](#arquitectura)
3. [Tipos de casos de evaluación](#tipos-de-casos-de-evaluación)
4. [Métricas calculadas](#métricas-calculadas)
5. [Guía paso a paso](#guía-paso-a-paso)
6. [Referencia de comandos](#referencia-de-comandos)
7. [Interpretar los resultados](#interpretar-los-resultados)
8. [Tests automáticos](#tests-automáticos)
9. [Principio de diseño](#principio-de-diseño)

---

## Prerequisitos

Antes de correr cualquier evaluación necesitas:

**1. El proyecto indexado** — la BD debe tener chunks y los índices construidos:
```bash
# Verificar que hay datos
python -c "
from backend.database.chunk_repository import get_chunk_stats
from backend.database.schema import DB_PATH
print(get_chunk_stats(DB_PATH))
"
# Esperas: {'total_chunks': N, 'embedded_chunks': N, 'pending_chunks': 0}
```

**2. Ollama corriendo** — necesario para los tipos `semantic` y `generated`, y para la evaluación RAG:
```bash
ollama list    # debe mostrar al menos un modelo (ej. llama3.2:3b)
```

**3. Variable de entorno para Tavily** (solo si quieres eval con búsqueda web):
```powershell
# Windows PowerShell
$env:TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Arquitectura

```
backend/eval/
│
├── schema.py                  # EvalCase, EvalDataset — tipos de datos
├── paraphraser.py             # Paráfrasis semántica con Ollama
├── query_generator.py         # Generación de queries reales con Ollama
├── dataset_generator.py       # Generador principal del dataset
├── generate_dataset.py        # CLI de generación
│
├── retrieval/                 # Bloque 2 — Evaluación del retriever
│   ├── _types.py              # RawHit, AggregatedMetrics, MetricSet
│   ├── metrics.py             # hit_at_k(), mrr(), ndcg_at_k() — funciones puras
│   ├── scorer.py              # score_case() → RawHit
│   ├── aggregator.py          # aggregate() → AggregatedMetrics
│   ├── runner.py              # EvalRunner (dataset + retriever → hits)
│   ├── report.py              # format_summary(), save_json()
│   └── evaluate.py            # CLI
│
├── rag/                       # Bloque 3 — Evaluación RAG end-to-end
│   ├── _types.py              # DimensionScore, RAGJudgement, RAGAggregatedMetrics
│   ├── prompts.py             # Plantillas de prompt del juez LLM
│   ├── judge.py               # OllamaJudge — llama al LLM y parsea la respuesta
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

## Tipos de casos de evaluación

El dataset puede contener tres tipos de casos. Cada uno estresa una parte diferente del sistema:

### `exact`
La query es un **fragmento literal** extraído del interior del chunk (evitando los bordes para no trivializar el test).

```
chunk: "Attention mechanisms allow neural networks to focus on relevant parts..."
query: "allow neural networks to focus on relevant parts of the input"
```

**Para qué sirve:** prueba el retrieval léxico (LSI). Si el sistema no recupera un chunk cuando la query es texto literal de ese chunk, hay un problema de indexación.

**No requiere LLM.** Es el más rápido de generar.

---

### `semantic`
La query es una **paráfrasis** del fragmento generada por un LLM. Las palabras cambian completamente pero el significado se mantiene.

```
fragmento: "allow neural networks to focus on relevant parts of the input"
query:     "enable deep learning models to selectively attend to important features"
```

**Para qué sirve:** prueba el retrieval semántico denso (FAISS + embeddings). Expone debilidades del modelo de embedding cuando no hay solapamiento léxico entre la query y el chunk.

**Requiere Ollama.**

---

### `generated`
La query es una **pregunta real de usuario** generada por un LLM a partir del contenido completo del chunk. Es el tipo más representativo del uso real del sistema.

```
chunk:  "Attention mechanisms allow neural networks to focus on..."
query:  "How do transformers decide which parts of the input to focus on?"
```

**Para qué sirve:** prueba el pipeline completo tal como lo usaría un usuario real. Una buena puntuación en `generated` indica que el sistema responde preguntas reales, no solo recupera fragmentos de texto.

**Requiere Ollama. Recomendado para la evaluación RAG.**

---

### Resumen comparativo

| Tipo | LLM | Velocidad | Qué estresa |
|---|---|---|---|
| `exact` | No | ★★★ Rápido | Indexación léxica (LSI) |
| `semantic` | Sí | ★★ Medio | Retrieval denso (FAISS) |
| `generated` | Sí | ★★ Medio | Pipeline completo (uso real) |

---

## Métricas calculadas

### Métricas de retrieval

Todas se calculan por tipo de caso (`exact`, `semantic`, `generated`) y globalmente.

#### Hit@K
**¿Apareció el chunk correcto en los top-K resultados?**

```
Hit@K = nº de casos donde el chunk correcto está en los top K
        ─────────────────────────────────────────────────────
                      total de casos
```

Con ground truth de un solo chunk por caso, Hit@K = Precision@K = Recall@K.

- **Rango:** 0.0 – 1.0
- **Interpretación:** 0.74 → el chunk correcto aparece en top-K el 74% de las veces
- **Referencia:** >0.70 es aceptable, >0.85 es bueno para un sistema RAG

#### MRR — Mean Reciprocal Rank
**¿En qué posición exacta aparece el chunk correcto?**

```
MRR = media de (1 / posición) para cada caso
    = 1.0 si siempre aparece en posición 1
    = 0.5 si siempre aparece en posición 2
    = 0.0 si no aparece nunca
```

- **Rango:** 0.0 – 1.0
- **Interpretación:** MRR=0.38 → el chunk correcto aparece de media en la posición ~2.6
- **Por qué importa:** Hit@K dice si aparece, MRR dice qué tan arriba aparece. Un MRR alto significa que el chunk relevante sale primero, lo que reduce el trabajo del reranker.

#### NDCG@K — Normalized Discounted Cumulative Gain
**¿Qué tan bien posicionado está el chunk correcto dentro del top-K?**

```
NDCG@K_i = 1 / log₂(posición + 1)   si posición ≤ K
           0                          si no aparece
```

Con relevancia binaria (un solo chunk correcto por query), NDCG@K es similar a MRR pero con una penalización logarítmica en lugar de lineal.

- **Rango:** 0.0 – 1.0
- **Interpretación:** premia más encontrar el chunk en posición 1 que en posición 2, y más en posición 2 que en 3
- **Cuándo usarlo:** cuando quieres penalizar más los fallos a encontrar el chunk en las primeras posiciones

#### Δ Hit@K (delta)
Diferencia de Hit@K entre tipos de caso:

```
Δ (semantic − exact)   → cuánto pierde el sistema cuando la query no comparte léxico
Δ (generated − exact)  → cuánto pierde con queries reales de usuario
```

- **Referencia:** un delta de -0.10 a -0.20 es normal. Más de -0.30 indica que el modelo de embedding es débil en recuperación semántica.

---

### Dimensiones de evaluación RAG

Evaluadas por un LLM-as-judge (Ollama) que puntúa de 1 a 5 y devuelve una justificación. La puntuación se normaliza a [0.0, 1.0].

#### Faithfulness
**¿La respuesta generada está fundamentada en el contexto recuperado?**

El juez verifica que cada afirmación de la respuesta se pueda respaldar directamente con alguna fuente. Penaliza respuestas que introducen hechos externos o contradicen las fuentes.

- **Detecta:** alucinaciones del LLM generador
- **Puntuación baja:** el sistema está inventando información no presente en los documentos
- **Puntuación alta:** la respuesta es fiel a las fuentes, aunque sea incompleta

#### Answer Relevance
**¿La respuesta contesta la pregunta que se hizo?**

El juez evalúa si la respuesta aborda directamente la query, sin desviarse a temas relacionados pero distintos.

- **Detecta:** respuestas off-topic o que responden una pregunta diferente
- **Puntuación baja:** el sistema recuperó chunks correctos pero el LLM generó una respuesta que no responde la pregunta
- **Puntuación alta:** la respuesta es directa y pertinente

#### Context Relevance
**¿El contexto recuperado por el retriever es pertinente para responder la pregunta?**

El juez evalúa los chunks recuperados, no la respuesta generada. Permite separar fallos del retriever de fallos del generador.

- **Detecta:** fallos del pipeline de retrieval (LSI + FAISS + reranker)
- **Puntuación baja:** el retriever está trayendo documentos incorrectos → el LLM no tiene material útil
- **Puntuación alta:** los chunks recuperados son los adecuados para responder

#### Escala de puntuación (todas las dimensiones)

| Puntuación | Significado |
|---|---|
| 1 (0.00) | Muy deficiente |
| 2 (0.25) | Deficiente |
| 3 (0.50) | Aceptable |
| 4 (0.75) | Bueno |
| 5 (1.00) | Excelente |

#### Diagnóstico combinado

| Context Relevance | Faithfulness | Diagnóstico |
|---|---|---|
| Alta | Alta | Sistema funcionando bien |
| Alta | Baja | El LLM alucina aunque tenga buen contexto |
| Baja | Alta | El retriever falla pero el LLM improvisa bien |
| Baja | Baja | Fallo en retrieval y en generación |

---

## Guía paso a paso

### Paso 0 — Verificar prerequisitos

```bash
# Verificar BD con datos
python -c "
from backend.database.chunk_repository import get_chunk_stats
from backend.database.schema import DB_PATH
stats = get_chunk_stats(DB_PATH)
print(stats)
"

# Verificar Ollama (necesario para semantic, generated y eval RAG)
ollama list
```

---

### Paso 1 — Generar el dataset

Empieza con un dataset pequeño para verificar que todo funciona:

```bash
# Dataset mínimo de prueba — solo exact, sin LLM, 10 casos
python -m backend.eval.generate_dataset \
    --exact \
    --sample-size 10 \
    --output backend/data/eval/test_dataset.json \
    --verbose
```

Inspecciona que las queries tienen sentido:

```bash
python -c "
import json
ds = json.load(open('backend/data/eval/test_dataset.json'))
for c in ds['cases'][:3]:
    print('---')
    print('TIPO :', c['case_type'])
    print('QUERY:', c['query'])
    print('CHUNK:', c['source_text'][:100])
"
```

Si las queries parecen frases coherentes de papers científicos, está bien.

**Dataset recomendado para evaluación completa:**

```bash
# Queries reales de usuario + fragmentos exactos (sin paráfrasis)
python -m backend.eval.generate_dataset \
    --exact \
    --generated \
    --sample-size 50 \
    --output backend/data/eval/dataset.json

# Dataset completo con los tres tipos
python -m backend.eval.generate_dataset \
    --exact \
    --semantic \
    --generated \
    --sample-size 50 \
    --output backend/data/eval/dataset_full.json
```

> El dataset se genera **una sola vez** y se reutiliza en todas las evaluaciones, para que los resultados sean comparables entre corridas.

---

### Paso 2 — Evaluar el retriever

Primero aísla FAISS (embedding-only) para tener una línea base:

```bash
python -m backend.eval.retrieval.evaluate \
    --dataset  backend/data/eval/dataset.json \
    --retriever embedding \
    --top-k 20 \
    --output backend/data/eval/report_embedding.json
```

Luego el hybrid completo:

```bash
python -m backend.eval.retrieval.evaluate \
    --dataset  backend/data/eval/dataset.json \
    --retriever hybrid \
    --top-k 20 \
    --output backend/data/eval/report_hybrid.json
```

Con reranker activado (mejora el MRR):

```bash
python -m backend.eval.retrieval.evaluate \
    --dataset  backend/data/eval/dataset.json \
    --retriever hybrid \
    --reranker \
    --top-k 20 \
    --output backend/data/eval/report_hybrid_reranker.json
```

---

### Paso 3 — Evaluar el pipeline RAG completo

Este paso llama al pipeline entero (retrieval + generación LLM) y luego al juez para cada caso. Con 50 casos y llama3.2:3b cuenta con 10-20 minutos:

```bash
python -m backend.eval.rag.evaluate \
    --dataset    backend/data/eval/dataset.json \
    --judge-model llama3.2:3b \
    --top-k 10 \
    --output     backend/data/eval/rag_report.json \
    --judgements backend/data/eval/rag_judgements.json
```

Para inspeccionar qué dijo el juez caso por caso:

```bash
python -c "
import json
js = json.load(open('backend/data/eval/rag_judgements.json'))['judgements']
for j in js[:5]:
    print('---')
    print('QUERY :', j['query'])
    print('ANSWER:', j['answer'][:120])
    if j['faithfulness']:
        print('Faith :', j['faithfulness']['raw_score'], '—', j['faithfulness']['reason'])
    if j['answer_relevance']:
        print('Relev :', j['answer_relevance']['raw_score'], '—', j['answer_relevance']['reason'])
"
```

---

### Paso 4 — Comparar dos corridas

Después de cambiar un parámetro del sistema (modelo de embedding, rrf_k, tamaño de chunk, etc.):

```bash
python -m backend.eval.compare.compare \
    --baseline  backend/data/eval/report_hybrid.json \
    --candidate backend/data/eval/report_hybrid_reranker.json \
    --output    backend/data/eval/comparison_reranker.json
```

Salida esperada:

```
============================================================
  Comparison  [retrieval]
  Baseline  : report_hybrid.json
  Candidate : report_hybrid_reranker.json
  Threshold : ±0.005
============================================================

  ── ✓  Improved ─────────────────────────────────────────
  ✓  overall.mrr          base=0.3821  cand=0.5100  Δ=+0.1279  (+33.5%)
  ✓  overall.ndcg_at_k    base=0.4189  cand=0.5430  Δ=+0.1241  (+29.6%)

  ── ~  Neutral ───────────────────────────────────────────
  ~  overall.hit_at_k     base=0.5426  cand=0.5532  Δ=+0.0106  (+2.0%)

============================================================
  Improved=6  Degraded=0  Neutral=6  Total=12
============================================================
```

> **Integración CI/CD:** el comparador devuelve código de salida `1` si hay regresiones, `0` si no. Útil para bloquear merges automáticamente si una métrica empeora.

---

## Referencia de comandos

### Generar dataset

```bash
python -m backend.eval.generate_dataset \
    --exact                          # incluir casos exact
    --semantic                       # incluir casos semantic (paráfrasis LLM)
    --generated                      # incluir casos generated (queries reales LLM)
    --sample-size 50                 # chunks a muestrear
    --model llama3.2:3b              # modelo Ollama para LLM calls
    --output backend/data/eval/dataset.json
    --min-chars 200                  # tamaño mínimo de chunk
    --seed 42                        # semilla para reproducibilidad
    --verbose
```

### Evaluar retrieval

```bash
python -m backend.eval.retrieval.evaluate \
    --dataset  backend/data/eval/dataset.json \
    --retriever hybrid               # hybrid | embedding | lsi
    --top-k 20                       # ventana de evaluación
    --reranker                       # activar CrossEncoderReranker
    --output backend/data/eval/report.json \
    --verbose
```

### Evaluar RAG

```bash
python -m backend.eval.rag.evaluate \
    --dataset    backend/data/eval/dataset.json \
    --judge-model llama3.2:3b        # modelo Ollama para el juez
    --top-k 10                       # chunks a recuperar por query
    --output     backend/data/eval/rag_report.json \
    --judgements backend/data/eval/rag_judgements.json \
    --verbose
```

### Comparar corridas

```bash
python -m backend.eval.compare.compare \
    --baseline  backend/data/eval/report_v1.json \
    --candidate backend/data/eval/report_v2.json \
    --output    backend/data/eval/comparison.json \
    --threshold 0.005                # mínimo delta para ser "improved/degraded"
```

### Correr tests automáticos

```bash
# Suite completa (sin dependencias externas — todo mockeado)
pytest backend/tests/test_eval_dataset.py \
                    backend/tests/test_query_generator.py \
                    backend/tests/eval_retrieval/ \
                    backend/tests/eval_rag/ \
                    backend/tests/eval_compare/ \
                    backend/tests/retrieval/ \
                    -v
```

---

## Interpretar los resultados

### ¿Mis métricas de retrieval son buenas?

| Hit@K (K=10) | Interpretación |
|---|---|
| < 0.40 | Problema grave — el retriever falla en la mayoría de casos |
| 0.40 – 0.60 | Mejorable — revisar modelo de embedding o parámetros RRF |
| 0.60 – 0.80 | Aceptable para un sistema en desarrollo |
| > 0.80 | Bueno |

### ¿El delta semántico es normal?

Un delta negativo entre `exact` y `semantic`/`generated` es esperado:

| Δ Hit@K | Interpretación |
|---|---|
| > -0.10 | Excelente — el modelo de embedding es muy robusto |
| -0.10 a -0.20 | Normal con modelos generales (MiniLM, MPNet) |
| -0.20 a -0.35 | El modelo de embedding lucha con el dominio científico |
| < -0.35 | Considera cambiar a un modelo domain-specific (specter, scibert) |

### ¿Mis métricas RAG son buenas?

| Dimensión | Puntuación preocupante | Puntuación objetivo |
|---|---|---|
| Faithfulness | < 0.50 | > 0.75 |
| Answer Relevance | < 0.55 | > 0.70 |
| Context Relevance | < 0.45 | > 0.65 |

Si **Context Relevance** es baja → el problema es el retriever, no el LLM.
Si **Faithfulness** es baja con **Context Relevance** alta → el LLM está alucinando.

---

## Tests automáticos

Todos los tests usan BD en memoria y Ollama mockeado — no requieren servicios externos.

```
backend/tests/
├── test_eval_dataset.py          # 26 tests — EvalCase, EvalDataset, DatasetGenerator
├── test_query_generator.py       # 11 tests — QueryGenerator, validaciones
├── eval_retrieval/
│   ├── test_metrics.py           # 32 tests — hit_at_k, mrr, ndcg_at_k (fórmulas puras)
│   ├── test_scorer.py            # 10 tests — score_case()
│   ├── test_aggregator.py        #  9 tests — aggregate()
│   ├── test_runner.py            #  8 tests — EvalRunner
│   └── test_report.py            # 14 tests — format_summary, save_json
├── eval_rag/
│   ├── test_prompts.py           # 14 tests — plantillas de prompt
│   ├── test_judge.py             # 17 tests — OllamaJudge, parseo JSON
│   ├── test_scorer.py            # 10 tests — score_rag_case()
│   ├── test_aggregator.py        # 10 tests — aggregate()
│   ├── test_runner.py            #  8 tests — RAGEvalRunner
│   └── test_report.py            # 18 tests — format_summary, save_json
├── eval_compare/
│   ├── test_differ.py            # 25 tests — detect_type, compute_deltas
│   └── test_report.py            # 12 tests — format_summary, save_json
└── retrieval/
    ├── test_lsi_query.py         # 14 tests — QueryVectorizer
    ├── test_lsi_retriever.py     # 17 tests — LSIRetriever, chunk_ids reales
    └── test_factory.py           # 12 tests — build_hybrid_retriever, etc.
```

---

## Principio de diseño

Cada archivo tiene exactamente **una razón para cambiar**:

| Archivo | Cambia si… |
|---|---|
| `schema.py` | Cambias la estructura de EvalCase / EvalDataset |
| `paraphraser.py` | Cambias la estrategia de paráfrasis |
| `query_generator.py` | Cambias cómo el LLM genera queries de usuario |
| `dataset_generator.py` | Cambias el algoritmo de muestreo |
| `retrieval/metrics.py` | Cambias la fórmula de Hit@K, MRR o NDCG |
| `retrieval/scorer.py` | Cambias cómo se detecta un chunk en los resultados |
| `rag/prompts.py` | Cambias las instrucciones del juez LLM |
| `rag/judge.py` | Cambias el backend LLM (ej. Ollama → OpenAI) |
| `compare/differ.py` | Cambias qué métricas se extraen o cómo se calculan los deltas |
| Cualquier `report.py` | Cambias el formato de salida |
| Cualquier `evaluate.py` | Cambias los argumentos de la CLI |