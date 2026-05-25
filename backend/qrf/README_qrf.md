# OmniRetrieve — Módulo `qrf` (Query Refinement Framework)

Pipeline de búsqueda semántica densa con refinamiento progresivo del vector
de consulta. Combina cuatro técnicas en un flujo de 7 pasos que mejora tanto
la relevancia como la diversidad de los resultados antes de entregarlos al
módulo RAG.

Actúa exclusivamente sobre el **índice FAISS** (búsqueda densa por embeddings).
Para búsqueda híbrida (sparse + dense) ver `HybridRetriever` en `retrieval/`.

---

## Estructura de archivos

```
backend/qrf/
├── pipeline.py          ← QueryPipeline: orquestador de los 7 pasos
├── query_expander.py    ← QueryExpander: expansión LCE vía dimensiones LSI
├── brf.py               ← BlindRelevanceFeedback: ajuste ciego del vector query
├── rocchio.py           ← RocchioFeedback: ajuste con feedback explícito del usuario
├── mmr.py               ← MMRReranker: diversificación con Maximal Marginal Relevance
├── _feedback_utils.py   ← utilidades internas: get_embeddings, cosine_similarity, l2_normalize
└── __init__.py          ← exports públicos
```

---

## Flujo de `QueryPipeline.search()`

```
query (texto libre)
        │
        ▼  1. LCE Expansion
QueryExpander.expand(query)
        │  query_original + términos latentes del SVD
        │  → "attention transformers" + ["self-attention", "encoder", …]
        ▼  2. Embedding
ChunkEmbedder.encode_single(query_expanded)
        │  ndarray float32 (dim,)
        ▼  3. Búsqueda inicial
FaissIndexManager.search(query_vec, top_k=top_k_initial)
        │  top_k_initial candidatos con chunk_id y score L2
        ▼  4. BRF — ajuste ciego
BlindRelevanceFeedback.adjust(query_vec, top_results, db_path)
        │  v_new = α·v_orig + (1-α)·centroide(top_k_rf chunks)
        ▼  5. Re-búsqueda con vector refinado
FaissIndexManager.search(refined_vec, top_k=top_k*3)
        │  candidatos ampliados con el vector corregido
        ▼  6. MMR reranking
MMRReranker.rerank(candidates, refined_vec, top_n, db_path)
        │  selección greedy que maximiza relevancia y diversidad
        ▼  7. Enriquecimiento
_enrich(reranked, expanded_terms)
        │  añade text, title, authors, abstract, pdf_url desde BD
        ▼
list[dict]  — resultados finales para RAG
```

---

## `QueryPipeline` — API

### Inicialización y carga

```python
from backend.qrf.pipeline import QueryPipeline

pipeline = QueryPipeline(
    model_name       = "all-MiniLM-L6-v2",
    top_k_initial    = 20,      # candidatos FAISS en búsqueda inicial (paso 3)
    expand           = True,    # activar LCE
    expand_top_dims  = 3,       # dimensiones latentes LSI a examinar
    expand_min_corr  = 0.4,     # correlación mínima para añadir un término
    expand_max_terms = 8,       # máximo de términos nuevos por consulta
    brf_alpha        = 0.75,    # peso del vector original en BRF (0-1)
    brf_top_k        = 5,       # chunks usados para calcular el centroide BRF
    mmr_lambda       = 0.6,     # balance relevancia/diversidad (1=solo relevancia)
    rocchio_alpha    = 0.6,
    rocchio_beta     = 0.4,
    rocchio_gamma    = 0.1,
)

pipeline.load()   # carga embedder + FAISS + modelo LSI para expansión
                  # si el LSI no está disponible, la expansión se desactiva
                  # silenciosamente con un WARNING
```

`load()` es obligatorio antes de cualquier `search()`. Lanza
`FileNotFoundError` si el índice FAISS no existe en disco.

### Búsqueda simple

```python
results = pipeline.search("attention mechanisms in transformers", top_k=10)
```

Cada resultado es un `dict` con estas claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `chunk_id` | `int` | PK de la tabla `chunks` |
| `arxiv_id` | `str` | ID compuesto del documento |
| `chunk_index` | `int` | Posición del chunk en el documento |
| `text` | `str` | Texto completo del chunk |
| `char_count` | `int` | Longitud del texto |
| `title` | `str` | Título del artículo |
| `authors` | `str` | Autores |
| `abstract` | `str` | Primeros 300 chars del abstract |
| `pdf_url` | `str` | URL del PDF original |
| `score` | `float` | Distancia L2 FAISS (menor = más cercano) |
| `mmr_score` | `float` | Puntuación MMR (mayor = más relevante y diverso) |
| `expanded_terms` | `list[str]` | Términos añadidos por LCE |

### Búsqueda con sesión y refinamiento explícito

```python
# Primera búsqueda — guarda el vector para refinamiento posterior
results, sid = pipeline.search_with_session("transformer attention", top_k=10)

# Usuario marca chunks como relevantes/irrelevantes
results2 = pipeline.refine(
    session_id     = sid,
    relevant_ids   = [results[0]["chunk_id"], results[1]["chunk_id"]],
    irrelevant_ids = [results[-1]["chunk_id"]],
    top_k          = 10,
)

# Limpiar sesión cuando termina la interacción
pipeline.clear_session(sid)          # sesión concreta
pipeline.clear_session()             # todas las sesiones
```

`refine()` lanza `KeyError` si `session_id` no existe. Los refinamientos
son **acumulativos**: el vector se actualiza en `_session_vectors` en cada
llamada, por lo que múltiples rondas de feedback mejoran progresivamente.

### Expansión de query

```python
expanded_query, new_terms = pipeline.expand_query("neural network training")
# expanded_query = "neural network training gradient descent backpropagation"
# new_terms      = ["gradient", "descent", "backpropagation"]
```

---

## Componentes internos

### `QueryExpander` — Expansión LCE

Proyecta la query al espacio LSI e identifica términos correlacionados
en las dimensiones latentes de mayor activación:

1. Tokeniza y vectoriza la query en el espacio TF-IDF del corpus
2. Proyecta al espacio latente: `q_svd = svd.transform(q_tfidf)`
3. Identifica las `top_dims` dimensiones latentes más activadas
4. En cada dimensión, extrae los `top_terms_per_dim` términos con mayor
   peso en `svd.components_`
5. Filtra términos con correlación < `min_correlation` y OOV
6. Añade como máximo `max_expansion` términos al final de la query

El filtro de correlación mínima evita el **query drift**: añadir términos
marginalmente relacionados que alejan el vector del tema original.

```python
from backend.qrf.query_expander import QueryExpander

exp = QueryExpander(
    top_dims          = 3,    # dimensiones latentes a examinar
    top_terms_per_dim = 10,   # candidatos por dimensión
    min_correlation   = 0.4,  # umbral de correlación
    max_expansion     = 8,    # máximo de términos nuevos
)
exp.load(model_path, db_path)   # carga LSI y construye word_index
```

### `BlindRelevanceFeedback` — BRF

Mueve el vector de query hacia el centroide de los `top_k_rf` primeros
resultados FAISS. Los embeddings se leen **desde la BD** (`chunks.embedding`)
y no desde el índice FAISS, evitando el error de aproximación de la
cuantización PQ de `IndexIVFPQ`.

```
v_new = α · v_orig + (1-α) · centroide(top_k_rf)

α=0.75 → 75% vector original, 25% centroide
```

Devuelve el vector original sin cambios si no hay embeddings disponibles
(fallback seguro). Lanza `ValueError` si `alpha` está fuera de `[0, 1]`.

### `RocchioFeedback` — Feedback explícito

Ajusta el vector con señales explícitas del usuario:

```
v_new = α · v_orig
      + β · mean(D_relevantes)
      - γ · mean(D_irrelevantes)
```

Defaults: `α=0.6, β=0.4, γ=0.1`.

Los vectores ajustados se cachean en memoria por `query_id` para permitir
refinamientos acumulativos en la misma sesión:

```python
rocchio = RocchioFeedback(alpha=0.6, beta=0.4, gamma=0.1)
v_adj   = rocchio.adjust("sess1", q_vec, [42, 17], [99], db_path)

rocchio.get_cached("sess1")       # → ndarray del último ajuste
rocchio.cached_queries            # → ["sess1"]
rocchio.clear_cache("sess1")      # elimina sesión concreta
rocchio.clear_cache()             # limpia todo
```

### `MMRReranker` — Diversificación

Selección greedy que maximiza simultáneamente relevancia y diversidad:

```
MMR(d) = λ · sim(d, query) - (1-λ) · max_sim(d, ya_seleccionados)
```

En cada iteración se selecciona el candidato que maximiza `MMR(d)`.

| `lambda_` | Efecto |
|---|---|
| `1.0` | Solo relevancia (sin diversidad) |
| `0.5` | Equilibrio |
| `0.0` | Solo diversidad (sin relevancia) |

Los embeddings también se leen **desde la BD**, no desde FAISS.
Si no hay embeddings disponibles, devuelve los primeros `top_n` candidatos
sin reordenar (fallback). Lanza `ValueError` si `lambda_` ∉ `[0, 1]`.

### `_feedback_utils.py` — Utilidades internas

```python
# Embeddings desde BD (no desde FAISS — evita error de cuantización PQ)
get_embeddings_by_chunk_ids(chunk_ids, db_path)
# → dict {chunk_id: ndarray float32 (dim,)}

cosine_similarity(a, b)   # → float en [-1, 1]
l2_normalize(v)           # → ndarray (dim,); sin cambios si norma ≈ 0
```

**Por qué leer desde la BD y no desde FAISS:** `IndexIVFPQ` almacena
vectores cuantizados con pérdida (error de aproximación PQ). Leer el
embedding serializado en `chunks.embedding` (`BLOB float32`) da el vector
exacto tal como lo produjo el modelo, lo que mejora la precisión de BRF
y MMR.

---

## CLI

```bash
# Búsqueda interactiva (modo de prueba)
python -m backend.qrf.pipeline --query "transformer attention mechanisms" --top-k 10

# Sin expansión LCE
python -m backend.qrf.pipeline --query "..." --no-expand

# Con parámetros personalizados
python -m backend.qrf.pipeline \
  --query "fairness in machine learning" \
  --top-k 10 \
  --brf-alpha 0.8 \
  --mmr-lambda 0.5 \
  --top-k-initial 30
```

---

## Tests

Los tests están en `backend/tests/qrf/`.

```bash
pytest backend/tests/qrf/ -v

pytest backend/tests/qrf/test_query_expander.py -v
pytest backend/tests/qrf/test_feedback.py        -v
pytest backend/tests/qrf/test_mmr.py             -v
pytest backend/tests/qrf/test_pipeline.py        -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_query_expander.py` | `expand()` con términos OOV devuelve query sin cambios; `_vectorize()` shape y dtype; no repite tokens originales; tokens del corpus añadidos |
| `test_feedback.py` | BRF: normalización L2, correlación positiva con el original, fallback sin embeddings, `ValueError` para alpha inválido; Rocchio: normalización, sin feedback devuelve original, caché acumulativo, `clear_cache()` |
| `test_mmr.py` | Sin duplicados en resultados, diversidad vs relevancia, fallback sin embeddings devuelve `results[:top_n]`, `ValueError` para lambda inválido |
| `test_pipeline.py` | `search()` retorna list[dict] con claves correctas, `search_with_session()` devuelve session_id, `refine()` con chunk válido, `KeyError` para session_id inválido, `clear_session()` individual y total |

El `conftest.py` define `MockLSIModel`, `MockEmbedder` y `MockFaissIndex`
inyectados en el pipeline para no depender de modelos en disco. La BD de
prueba (`db_with_chunks`) contiene 3 documentos con 7 chunks y embeddings
reales (generados con `np.random`) persistidos en SQLite.