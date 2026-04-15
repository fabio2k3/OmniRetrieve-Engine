---
noteId: "b3c4d5e0296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Refinamiento de Consulta (`query_refinement`)

Mejora la calidad de las búsquedas semánticas combinando expansión de consulta,
retroalimentación de relevancia y diversificación de resultados. Actúa como capa
intermedia entre la query del usuario y el índice FAISS, aprovechando la arquitectura
híbrida LSI + FAISS IVFPQ + Sentence-Transformers del sistema.

---

## Estructura de archivos

```
backend/query_refinement/
+-- query_expander.py     <- expansión ciega de consulta via LSI (LCE)
+-- brf.py                <- BlindRelevanceFeedback (pseudo-RF de centroide)
+-- rocchio.py            <- RocchioFeedback (retroalimentación explícita)
+-- mmr.py                <- MMRReranker (diversificación de resultados)
+-- _feedback_utils.py    <- utilidades internas compartidas (no API pública)
+-- query_pipeline.py     <- pipeline completo integrado
+-- __init__.py           <- exports públicos
```

---

> **Nota sobre `_feedback_utils.py`**: contiene las funciones internas
> `get_embeddings_by_chunk_ids`, `cosine_similarity` y `l2_normalize`,
> compartidas por `brf.py`, `rocchio.py` y `mmr.py`. El prefijo `_`
> indica que no forma parte de la API pública del módulo.

---

## Instalación

No requiere dependencias adicionales. Usa los módulos ya instalados del sistema:
`sentence-transformers`, `faiss-cpu`, `scikit-learn`, `numpy`.

---

## Cómo ejecutar

```python
from backend.query_refinement import QueryPipeline

pipeline = QueryPipeline()
pipeline.load()

# Búsqueda simple
results = pipeline.search("attention mechanism in transformers", top_k=10)

# Búsqueda con sesión para retroalimentación explícita
results, session_id = pipeline.search_with_session("graph neural networks")

# El usuario marca chunks relevantes e irrelevantes
results2 = pipeline.refine(
    session_id     = session_id,
    relevant_ids   = [42, 7],
    irrelevant_ids = [99],
    top_k          = 10,
)
```

---

## Componentes

### `QueryExpander` (`query_expander.py`)

Expansión ciega de consulta usando el espacio latente LSI antes de tocar FAISS.

**Técnica**: Latent Concept Expansion (LCE).

```
query original
      |
      v
Vectorizacion TF-IDF  ->  q_tfidf (n_terms,)
      |
      v
Proyeccion LSI         ->  q_latent (k,)
      |
      v
Top-N dimensiones activadas
      |
      v
svd.components_[dim]   ->  pesos de terminos en ese concepto
      |
      v
Filtro min_correlation ->  evita Query Drift
      |
      v
query_original + terminos_nuevos
```

**Parámetros**

| Parámetro | Default | Descripción |
|---|---|---|
| `top_dims` | 3 | Dimensiones latentes a examinar |
| `top_terms_per_dim` | 10 | Candidatos por dimensión |
| `min_correlation` | 0.4 | Umbral mínimo de peso en el componente SVD |
| `max_expansion` | 8 | Máximo de términos nuevos a añadir |

**Por qué LCE antes de FAISS**: SentenceTransformer es preciso pero sensible al
vocabulario de la query. Una consulta de 3 palabras puede no activar el embedding
correcto si el corpus usa sinónimos. LCE añade términos estadísticamente relacionados
del corpus, mejorando la cobertura semántica antes de codificar.

```python
from backend.query_refinement.query_expander import QueryExpander

expander = QueryExpander(top_dims=3, min_correlation=0.4, max_expansion=8)
expander.load()

expanded, new_terms = expander.expand("attention transformer")
# expanded  -> "attention transformer neural sequence embedding model"
# new_terms -> ["neural", "sequence", "embedding", "model"]
```

---

### `BlindRelevanceFeedback` (`brf.py`)

Ajusta el vector de query hacia el centroide de los mejores resultados de FAISS.

**Técnica**: Vector Mean Shift con interpolación alpha.

```
v_new = alpha * v_orig + (1 - alpha) * centroide(top_k)
```

Los embeddings se recuperan directamente de la BD (columna `chunks.embedding`)
en lugar de usar `reconstruct()` de FAISS, evitando el error de aproximación
inherente a la cuantización PQ de `IndexIVFPQ`.

**Parámetros**

| Parámetro | Default | Descripción |
|---|---|---|
| `alpha` | 0.75 | Peso del vector original. Mayor = menos ajuste |
| `top_k_rf` | 5 | Resultados usados para calcular el centroide |

```python
from backend.query_refinement.brf import BlindRelevanceFeedback

brf      = BlindRelevanceFeedback(alpha=0.75, top_k_rf=5)
adjusted = brf.adjust(query_vector, top_results, db_path)
```

---

### `RocchioFeedback` (`rocchio.py`)

Retroalimentación explícita: el usuario marca chunks como relevantes o irrelevantes.

**Técnica**: Algoritmo de Rocchio.

```
v_new = alpha * v_orig
      + beta  * (1/|D_r|) * sum(D_r)   <- acerca a los relevantes
      - gamma * (1/|D_n|) * sum(D_n)   <- aleja de los irrelevantes
```

Los vectores ajustados se guardan en caché en memoria por `query_id`, permitiendo
múltiples rondas de refinamiento en la misma sesión sin perder el ajuste acumulado.

**Parámetros**

| Parámetro | Default | Descripción |
|---|---|---|
| `alpha` | 0.6 | Peso del vector original |
| `beta` | 0.4 | Peso de los documentos relevantes |
| `gamma` | 0.1 | Penalización de documentos irrelevantes |

```python
from backend.query_refinement.rocchio import RocchioFeedback

rocchio  = RocchioFeedback(alpha=0.6, beta=0.4, gamma=0.1)
adjusted = rocchio.adjust(
    query_id       = "session_abc",
    query_vector   = query_vec,
    relevant_ids   = [42, 7],
    irrelevant_ids = [99],
    db_path        = db_path,
)

# Reutilizar en la misma sesión
cached = rocchio.get_cached("session_abc")   # vector ya calibrado
```

---

### `MMRReranker` (`mmr.py`)

Diversifica los resultados finales eliminando redundancia.

**Técnica**: Maximal Marginal Relevance (MMR).

```
MMR(d) = lambda * sim(d, query) - (1 - lambda) * max_sim(d, D_seleccionados)
```

Selección iterativa: en cada paso elige el candidato que maximiza
relevancia y minimiza similitud con los ya seleccionados.
Garantiza que el módulo RAG reciba chunks variados y no N párrafos
que dicen lo mismo.

**Parámetros**

| Parámetro | Default | Descripción |
|---|---|---|
| `lambda_` | 0.6 | Balance relevancia/diversidad. 1.0 = solo relevancia, 0.0 = solo diversidad |

```python
from backend.query_refinement.mmr import MMRReranker

mmr      = MMRReranker(lambda_=0.6)
reranked = mmr.rerank(candidates, query_vector, top_n=10, db_path=db_path)
# reranked[i]["mmr_score"] -> score combinado de relevancia y diversidad
```

---

### `QueryPipeline` (`query_pipeline.py`)

Orquesta los cuatro componentes en un flujo cohesivo. Punto de entrada
principal del módulo.

**Parámetros principales**

| Parámetro | Default | Descripción |
|---|---|---|
| `model_name` | `all-MiniLM-L6-v2` | Modelo sentence-transformers |
| `top_k_initial` | 20 | Candidatos en la búsqueda inicial (BRF) |
| `expand` | `True` | Activar/desactivar expansión LCE |
| `expand_top_dims` | 3 | Dimensiones latentes para LCE |
| `expand_min_corr` | 0.4 | Umbral de correlación para LCE |
| `expand_max_terms` | 8 | Máximo de términos nuevos en LCE |
| `brf_alpha` | 0.75 | Peso del vector original en BRF |
| `brf_top_k` | 5 | Resultados para el centroide BRF |
| `mmr_lambda` | 0.6 | Balance relevancia/diversidad en MMR |
| `rocchio_alpha` | 0.6 | Peso original en Rocchio |
| `rocchio_beta` | 0.4 | Peso relevantes en Rocchio |
| `rocchio_gamma` | 0.1 | Penalización irrelevantes en Rocchio |

**Formato de resultado**

Cada elemento de la lista devuelta por `search()` o `refine()` contiene:

| Campo | Tipo | Descripción |
|---|---|---|
| `score` | float | Distancia L2 FAISS (menor = más similar) |
| `mmr_score` | float | Score MMR combinado |
| `chunk_id` | int | ID en la tabla `chunks` |
| `arxiv_id` | str | ID del artículo |
| `chunk_index` | int | Posición del chunk dentro del documento |
| `text` | str | Texto del chunk |
| `char_count` | int | Longitud del chunk en caracteres |
| `title` | str | Título del artículo |
| `authors` | str | Autores |
| `abstract` | str | Abstract (primeros 300 chars) |
| `pdf_url` | str | URL del PDF |
| `expanded_terms` | list[str] | Términos añadidos por LCE |

---

## Uso avanzado

### Sin expansión LCE (solo BRF + MMR)

```python
pipeline = QueryPipeline(expand=False)
pipeline.load()
results  = pipeline.search("diffusion models image synthesis", top_k=10)
```

### Ajuste de parámetros por caso de uso

```python
# Corpus científico denso — más expansión, más diversidad
pipeline = QueryPipeline(
    expand_top_dims  = 5,
    expand_min_corr  = 0.3,
    expand_max_terms = 12,
    brf_alpha        = 0.7,
    mmr_lambda       = 0.5,   # más diversidad para RAG
)

# Query corta de usuario — más conservador
pipeline = QueryPipeline(
    expand_max_terms = 4,
    brf_alpha        = 0.85,  # confiar más en la query original
    mmr_lambda       = 0.7,   # más relevancia
)
```

### Múltiples rondas de refinamiento

```python
results1, sid = pipeline.search_with_session("attention mechanism")

# Ronda 1
results2 = pipeline.refine(sid, relevant_ids=[42], irrelevant_ids=[9])

# Ronda 2 — el vector ya incorpora el ajuste de la ronda 1
results3 = pipeline.refine(sid, relevant_ids=[51], irrelevant_ids=[])

# Limpiar sesión al terminar
pipeline.clear_session(sid)
```

---

## Diseño

- **Sin estado global** — cada instancia de `QueryPipeline` gestiona su propio
  estado. Se pueden instanciar varios pipelines con distintos parámetros en paralelo.
- **Fallback graceful** — si el modelo LSI no está disponible, la expansión LCE
  se desactiva automáticamente y el pipeline continúa con BRF + MMR.
- **Caché en sesión** — `RocchioFeedback` acumula ajustes por `query_id` en
  memoria. No persiste entre reinicios del proceso.
- **Recuperación de vectores desde BD** — BRF y MMR leen los embeddings de la
  columna `chunks.embedding` en lugar de reconstruirlos desde FAISS, evitando
  el error de cuantización PQ.

---

## Tests

```bash
python -m backend.tests.test_query_refinement
python -m pytest backend/tests/test_query_refinement.py -v
```

Los tests usan mocks para LSI, SentenceTransformer y FAISS.
No requieren modelos descargados ni archivos en disco.

---

## Relación con otros módulos

```
query_refinement/
      |
      +-- usa --> retrieval/lsi_model.py        (modelo LSI para LCE)
      +-- usa --> embedding/embedder.py         (vectorizacion de query)
      +-- usa --> embedding/faiss_index.py      (busqueda vectorial)
      +-- usa --> database/chunk_repository.py  (embeddings y metadatos)
      |
      +-- alimenta --> modulo RAG (lista de chunks diversa y relevante)
```