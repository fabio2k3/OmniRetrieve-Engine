# OmniRetrieve — Módulo `retrieval`

Motor de búsqueda semántica con tres estrategias de recuperación y
una segunda etapa de reranking, todos bajo un protocolo común.

| Clase | Estrategia | `score_type` |
|---|---|---|
| `LSIRetriever` | Sparse — TF-IDF + SVD latente | `"cosine_lsi"` |
| `EmbeddingRetriever` | Dense — embeddings + FAISS L2 | `"l2"` |
| `HybridRetriever` | Fusión RRF (LSI + FAISS) | `"rrf"` |
| `CrossEncoderReranker` | Segunda etapa (no retriever) | `"rerank"` |

---

## Estructura de archivos

```
backend/retrieval/
├── protocols.py           ← RetrievalResult, RetrieverProtocol, RerankerProtocol
├── lsi_model.py           ← LSIModel: fase offline (TF-IDF + SVD)
├── lsi_query.py           ← build_word_index(), QueryVectorizer
├── lsi_retriever.py       ← LSIRetriever: búsqueda sparse en tiempo real
├── embedding_retriever.py ← EmbeddingRetriever: búsqueda densa sobre FAISS
├── hybrid_retriever.py    ← HybridRetriever: RRF de sparse + dense
├── reranker.py            ← CrossEncoderReranker: segunda etapa cross-encoder
├── factory.py             ← constructores de alto nivel con dependencias reales
├── build_lsi.py           ← CLI para la fase offline
└── __init__.py            ← exports públicos
```

---

## Protocolo compartido

Todos los retrievers implementan `RetrieverProtocol`:

```python
@runtime_checkable
class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_n: int = 20) -> list[RetrievalResult]: ...

@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(self, query: str, candidates: list[RetrievalResult],
               top_k: int = 10) -> list[RetrievalResult]: ...
```

Resultado de cualquier retriever:

```python
@dataclass
class RetrievalResult:
    chunk_id:    int | str      # PK de la tabla chunks
    arxiv_id:    str            # ID compuesto del documento
    chunk_index: int            # posición del chunk en el documento
    text:        str            # texto del chunk
    score:       float          # score del retriever (interpretación varía por tipo)
    score_type:  str            # "cosine_lsi" | "l2" | "rrf" | "rerank"
    metadata:    dict           # title, authors, pdf_url, … (libre por retriever)
```

---

## LSI — Fase offline (`lsi_model.py`)

### `LSIModel.build()`

```python
from backend.retrieval.lsi_model import LSIModel

model = LSIModel(k=100, n_iter=10)
stats = model.build(
    db_path      = Path("data/db/documents.db"),
    min_df       = 20,    # términos con df < 20 excluidos del SVD
    max_df_ratio = 0.85,  # términos en > 85% de docs excluidos
)
# stats: {n_docs, n_terms, k, var_explained, elapsed_s, min_df, max_df_ratio}
model.save()  # → data/models/lsi_model.pkl
```

**Flujo de `build()`:**

```
get_postings_for_matrix()          ← freq, df, doc_ids, term_ids desde BD
        ↓
Filtrar vocabulario
  [t for t in term_ids if min_df <= df_map[t] <= max_df_abs]
        ↓
Construir matriz TF-IDF sparse  (n_terms_filtrados × n_docs)
  TF(t,d)  = log(1 + freq(t,d))
  IDF(t)   = log((N+1) / (df(t)+1))   ← suavizado Laplace
  W(t,d)   = TF × IDF
        ↓
TruncatedSVD.fit_transform(matrix.T)   ← transpuesta: (n_docs × n_terms)
        ↓  docs_svd: (n_docs × k)
Normalizer L2
        ↓  docs_latent: (n_docs × k), cada fila es un vector unitario
Persistir .pkl + registrar en lsi_log
```

### Filtrado de vocabulario

El filtrado es crítico para la calidad del SVD:

- **`min_df=20`**: elimina hapax legomena y términos muy raros que son
  ruido puro para SVD. Con 6 000+ docs, el 94%+ de los términos sin
  filtrar aparecen en < 20 docs.
- **`max_df_ratio=0.85`**: elimina stop-words de dominio que el IDF no
  captura (p.ej. "paper", "model", "result") porque aparecen en casi
  todos los documentos.

Si el vocabulario queda vacío tras el filtrado, `build()` lanza
`RuntimeError("Vocabulario vacío… Reduce min_df.")`.

### Qué guarda el `.pkl`

| Campo | Tipo | Descripción |
|---|---|---|
| `svd` | `TruncatedSVD` | SVD ajustado; `components_` proyecta queries |
| `normalizer` | `Normalizer` | Normalización L2 consistente con `docs_latent` |
| `docs_latent` | `ndarray (n_docs × k)` | Vectores latentes normalizados de cada doc |
| `doc_ids` | `list[str]` | arxiv_ids en el mismo orden que las columnas |
| `term_ids` | `list[int]` | term_ids en el mismo orden que las filas |
| `df_map` | `dict[int, int]` | `term_id → df` del corpus (para vectorizar queries) |
| `k` | `int` | Componentes latentes del SVD |

`var_explained` y `built_at` se registran en la tabla `lsi_log`, no en el pkl.

---

## LSI — Vectorización de queries (`lsi_query.py`)

### `build_word_index(model, db_path)`

Construye el mapa `word → (row_idx, df)` necesario para vectorizar queries:

- `row_idx`: posición del término en `model.term_ids` (= fila de la matriz TF-IDF)
- `df`: del corpus, leído de `model.df_map`

Términos de `model.term_ids` que no existen en la BD se omiten silenciosamente.

### `QueryVectorizer.vectorize(query)`

```
query (texto libre)
        ↓
TextPreprocessor.process(query)    ← mismos pasos que en indexación
        ↓  list[tokens]
Counter(tokens)                    ← freq por token en la query
        ↓
Para cada token en el vocabulario:
  TF(t,q)  = log(1 + freq)
  IDF(t)   = log((n_docs+1) / (df+1))    ← df real del corpus
  vec[row_idx] = TF × IDF
        ↓
ndarray float32 (n_terms,)         ← tokens OOV ignorados sin error
```

---

## `LSIRetriever`

Retriever sparse que implementa `RetrieverProtocol`. Devuelve `list[RetrievalResult]`
con `chunk_id` enteros reales de la tabla `chunks`.

```python
from backend.retrieval.lsi_retriever import LSIRetriever

r = LSIRetriever(doc_candidates=20)
r.load(model_path=Path("data/models/lsi_model.pkl"),
       db_path=Path("data/db/documents.db"))

results = r.retrieve("attention mechanisms in transformers", top_n=10)
# → list[RetrievalResult] con score_type="cosine_lsi"
```

**Flujo de `retrieve()`:**

```
QueryVectorizer.vectorize(query)       → q_tfidf (n_terms,)
        ↓
LSIModel.project_query(q_tfidf)
  svd.transform(q_tfidf.reshape(1,-1)) → (1, k)
  normalizer.transform(…)              → (k,)
        ↓
cosine_similarity(q_latent, docs_latent) → (n_docs,)
        ↓
top doc_candidates documentos (score > 0)
        ↓
get_chunks_with_metadata_by_arxiv_ids(arxiv_ids)
  cada chunk hereda el score coseno del documento padre
        ↓
sort (score DESC, chunk_index ASC)[:top_n]
        ↓
list[RetrievalResult]  score_type="cosine_lsi"
```

La expansión documento→chunk hace que LSIRetriever opere a nivel de chunk
real, lo que permite fusionarlo con `EmbeddingRetriever` en `HybridRetriever`
usando `chunk_id` como clave de RRF.

`retrieve()` lanza `RuntimeError` si `load()` no fue llamado.
Query vacía o solo espacios devuelve `[]` sin consultar el modelo.

---

## `EmbeddingRetriever`

Retriever denso basado en embeddings + FAISS.

```python
from backend.retrieval.embedding_retriever import EmbeddingRetriever
from backend.embedding.faiss import FaissIndexManager

mgr = FaissIndexManager(dim=384, ...)
mgr.load()

r = EmbeddingRetriever(
    faiss_mgr  = mgr,
    model_name = "all-MiniLM-L6-v2",
)
results = r.retrieve("attention mechanisms", top_n=10)
# → list[RetrievalResult] con score_type="l2", ordenado por distancia ascendente
```

El modelo de embeddings se carga de forma **perezosa** en la primera llamada
a `retrieve()`. El score es distancia L2 (menor = más cercano).

`metadata` incluye: `title`, `authors`, `abstract`, `pdf_url`.

---

## `HybridRetriever`

Fusión de dos retrievers con **Reciprocal Rank Fusion (RRF)**:

```
query
  ├── sparse.retrieve(query, candidate_k)  ← LSIRetriever
  └── dense.retrieve(query, candidate_k)   ← EmbeddingRetriever
        ↓  (opcionalmente en paralelo con ThreadPoolExecutor)
RRF: score(d) = Σᵢ  1 / (rrf_k + rankᵢ(d))
        ↓  sorted por score RRF descending
[CrossEncoderReranker.rerank(…)]   ← segunda etapa opcional
        ↓
list[RetrievalResult]  score_type="rrf"
```

```python
from backend.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    sparse      = lsi_retriever,
    dense       = embedding_retriever,
    candidate_k = 50,      # candidatos por rama antes del RRF
    rrf_k       = 60,      # constante de suavizado (mayor = fusión más suave)
    parallel    = True,    # ThreadPoolExecutor(max_workers=2)
    reranker    = None,    # CrossEncoderReranker opcional
)
results = retriever.retrieve("attention transformers", top_n=10)
```

`metadata` de cada resultado incluye `rrf_k` y `candidate_k` para trazabilidad.
Si un chunk aparece en ambas listas, sus puntuaciones se suman.

---

## `CrossEncoderReranker`

Segunda etapa de precisión. **No es un retriever** — recibe candidatos
recuperados y los reordena.

```python
from backend.retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size = 32,
    max_length = 512,
)
reranked = reranker.rerank(query, candidates, top_k=10)
# → list[RetrievalResult] score_type="rerank"
```

El modelo se carga de forma **perezosa** en la primera llamada. Evalúa
pares `(query, text)` con el cross-encoder y devuelve los `top_k` con
mayor logit score.

`metadata` incluye: `retrieval_score`, `retrieval_score_type` (del retriever
original), `rerank_model` y `rerank_score` para trazabilidad completa.

---

## `factory.py` — Constructores de alto nivel

Único lugar del sistema donde se ensamblan retrievers con sus dependencias
reales. Las importaciones pesadas son a nivel de módulo para que `patch()`
en tests pueda interceptarlas.

```python
from backend.retrieval.factory import (
    build_faiss_manager,
    build_lsi_retriever,
    build_embedding_retriever,
    build_hybrid_retriever,
)

# FaissIndexManager desde disco (rutas por defecto)
mgr = build_faiss_manager(embed_model="all-MiniLM-L6-v2")

# LSIRetriever desde disco
lsi = build_lsi_retriever(doc_candidates=20)

# EmbeddingRetriever (reutiliza mgr si se pasa)
dense = build_embedding_retriever(embed_model="all-MiniLM-L6-v2", faiss_mgr=mgr)

# HybridRetriever completo — faiss_mgr se construye una vez y se comparte
hybrid = build_hybrid_retriever(
    embed_model    = "all-MiniLM-L6-v2",
    with_reranker  = True,
    candidate_k    = 50,
    rrf_k          = 60,
    doc_candidates = 20,
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
```

`build_faiss_manager()` y `build_lsi_retriever()` lanzan `RuntimeError`
si los ficheros no existen en disco.

---

## CLI — Fase offline

```bash
# Construir el modelo LSI con parámetros por defecto (k=100)
python -m backend.retrieval.build_lsi

# Personalizado
python -m backend.retrieval.build_lsi --k 200 --n-iter 15 --min-df 5

# Corpus pequeño de prueba
python -m backend.retrieval.build_lsi --k 50 --max-docs 500 --min-df 1
```

---

## Parámetro `k` — guía de selección

| `k` | Corpus | Nota |
|---|---|---|
| 50–100 | < 5 000 docs | Suficiente para capturar conceptos principales |
| 100–300 | 5 000–50 000 docs | Rango habitual en producción |
| 300–500 | > 50 000 docs | Más costoso; ganancias marginales decrecientes |

`k` se recorta automáticamente si el corpus es menor: `k = min(k, n_docs - 1)`.

---

## Tests

Los tests están en `backend/tests/retrieval/`.

```bash
pytest backend/tests/retrieval/ -v

pytest backend/tests/retrieval/test_lsi_model.py     -v  # build, save/load, var_explained
pytest backend/tests/retrieval/test_lsi_query.py     -v  # build_word_index, QueryVectorizer
pytest backend/tests/retrieval/test_lsi_retriever.py -v  # protocolo, chunk expansion
pytest backend/tests/retrieval/test_retriever.py     -v  # integración: load + retrieve
pytest backend/tests/retrieval/test_factory.py       -v  # constructores con mocks
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_lsi_model.py` | `build()` stats; `save()`/`load()` round-trip; `var_explained` en (0, 1] |
| `test_lsi_query.py` | `build_word_index` con DB real (`db_with_terms`); `QueryVectorizer` shape, dtype, IDF weighting, OOV silencioso |
| `test_lsi_retriever.py` | `RetrieverProtocol` implementado; `chunk_id` enteros; orden score DESC; chunks vacíos → lista vacía |
| `test_retriever.py` | Integración end-to-end con corpus real indexado: `load()` + `retrieve()`, `_vectorizer` inicializado, `lsi_log` registrado |
| `test_factory.py` | Mocks de `FaissIndexManager`, `LSIRetriever`, `HybridRetriever`; `embed_model` obligatorio; `faiss_mgr` compartido |

> **Nota sobre tests:** se usa `min_df=1` en todos los fixtures porque el
> corpus de prueba (5 documentos) es demasiado pequeño para el valor por
> defecto `min_df=20`, lo que vaciaría el vocabulario.

El `conftest.py` define `db_with_terms` (BD SQLite con `terms` reales),
`indexed_db` (corpus indexado con chunks en disco), y `lsi_model`
(fixture que hace `build(min_df=1)` + `save()`).