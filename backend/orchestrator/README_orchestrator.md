# OmniRetrieve — Módulo `orchestrator`

Coordinador central del sistema. Arranca y gestiona **cinco hilos daemon**
que mantienen el corpus actualizado y los modelos de recuperación al día,
y expone una API pública para consultas desde el frontend.

Toda la lógica de negocio está delegada en módulos internos (`_operations.py`,
`_status.py`, `_faiss.py`). Los hilos solo gestionan timing y condiciones
de disparo.

---

## Estructura de archivos

```
backend/orchestrator/
├── orchestrator.py   ← Orchestrator: estado compartido, hilos, API pública
├── config.py         ← OrchestratorConfig: todos los parámetros del sistema
├── _operations.py    ← lógica de indexación, LSI, embedding, QRF, RAG, pipeline
├── _status.py        ← build_status(): snapshot del sistema
├── _faiss.py         ← init_faiss_mgr(), resolve_embedding_dim()
├── main.py           ← CLI: lanzador Streamlit con flags para todos los parámetros
└── threads/
    ├── crawler.py    ← run_crawler_thread()
    ├── indexing.py   ← run_indexing_thread() (watcher TF)
    ├── lsi.py        ← run_lsi_rebuild_thread() (carga rápida + rebuild periódico)
    ├── embedding.py  ← run_embedding_thread() (watcher FAISS)
    └── qrf_rag.py    ← run_qrf_rag_loader_thread() (dos fases en paralelo)
```

---

## Cinco hilos daemon

```
start()
  ↓
  1. "lsi_rebuild"  ──→ run_lsi_rebuild_thread()
  2. "embedding"    ──→ run_embedding_thread()
  3. "qrf_rag"      ──→ run_qrf_rag_loader_thread()
  4. "crawler"      ──→ run_crawler_thread()
  5. "indexing"     ──→ run_indexing_thread()
```

El orden de arranque está optimizado para que **las búsquedas estén disponibles
lo antes posible**: LSI se carga primero (segundos desde el `.pkl`), luego
FAISS, luego el pipeline completo. Crawler e indexing son los últimos porque
no bloquean la disponibilidad de búsqueda.

---

## Coordinación entre hilos

El orquestador usa un conjunto de `threading.Event` para señalizar cuándo
cada componente está listo, y `RLock` + lista de un elemento para hacer
swaps atómicos de los modelos sin bloquear a los lectores:

| Event | Se activa cuando |
|---|---|
| `_shutdown` | `stop()` es llamado |
| `_lsi_ready` | LSIRetriever cargado y listo |
| `_faiss_ready` | FaissIndexManager cargado desde disco |
| `_qrf_ready` | QueryPipeline + HybridRetriever listos |
| `_rag_ready` | CrossEncoder + RAGPipeline listos |
| `_pipeline_ready` | Los cuatro componentes de `pipeline_ask` disponibles |

Patrón de swap seguro para todos los modelos:

```python
# Thread LSI actualiza el retriever
with self._lsi_lock:
    retriever_holder[0] = new_retriever

# API lee sin bloquear otros lectores
with self._lsi_lock:
    retriever = retriever_holder[0]
```

---

## Hilos en detalle

### `crawler` — Descarga continua

Instancia `Crawler` con los parámetros de `cfg` y llama a `run_forever()`.
Un hilo watchdog interno espera `_shutdown` y llama a `crawler.stop()` para
que el bucle termine limpiamente.

### `indexing` — Watcher TF

Sondea la BD cada `index_poll_interval` (30 s). Si hay ≥ `pdf_threshold`
(10) documentos con PDF descargado pero `indexed_tfidf_at IS NULL`, llama
a `do_index()` (`IndexingPipeline.run(reindex=False)`).

```
BD: pdf_downloaded=1 AND indexed_tfidf_at IS NULL ≥ 10
        ↓
do_index(cfg)  →  IndexingPipeline.run()
```

### `lsi_rebuild` — Dos pasos

**Paso 1 — Carga rápida** (segundos): si existe `.pkl` en disco, carga
`LSIRetriever` sin rebuild. Activa `lsi_ready` de inmediato.

**Paso 2 — Rebuild periódico** (cada `lsi_rebuild_interval` = 3600 s):
llama a `do_lsi_rebuild()` → `LSIModel.build()` → swap atómico del retriever.
Si no hay `.pkl` previo (primera ejecución) hace el rebuild directamente.

```
.pkl existe?
  SÍ → carga rápida → lsi_ready ← espera rebuild_interval → rebuild
  NO  → rebuild → lsi_ready ← espera rebuild_interval → rebuild ...
```

### `embedding` — Watcher FAISS

Sondea la BD cada `embed_poll_interval` (60 s). Primer chequeo inmediato
al arrancar para no dejar chunks huérfanos de sesiones anteriores.
Si hay ≥ `embed_threshold` (50) chunks sin embedding, llama a `do_embed()`.

```
chunks pendientes ≥ 50
        ↓
do_embed()  →  EmbeddingPipeline.run()  →  faiss_ready.set()
```

### `qrf_rag` — Dos fases en paralelo

Carga los cuatro componentes del pipeline unificado en dos fases,
aprovechando `ThreadPoolExecutor(max_workers=2)` dentro de cada fase:

```
Esperar faiss_ready
        ↓
Fase 1 (paralelo):
  ├── build_cross_encoder()  →  CrossEncoderReranker
  └── build_rag_pipeline()   →  EmbeddingRetriever + RAGPipeline
        ↓  rag_ready.set()

Esperar lsi_ready
        ↓
Fase 2 (paralelo):
  ├── build_qrf_pipeline()   →  QueryPipeline (QRF)
  └── build_hybrid_retriever()→  HybridRetriever (LSI + FAISS + RRF)
        ↓  qrf_ready.set() + pipeline_ready.set()
```

---

## API pública

### `pipeline_ask(text)` — Pipeline unificado

Flujo completo de extremo a extremo:

```
QRF QueryPipeline.search()     ← expansión LCE + BRF + MMR sobre FAISS
        ↓  chunks candidatos
HybridRetriever.retrieve()     ← LSI sparse + FAISS dense + RRF fusion
        ↓
WebSearchPipeline.run()        ← fallback web si resultados insuficientes
        ↓
CrossEncoderReranker           ← segunda etapa de ranking
        ↓  pipeline_rerank_k chunks finales
RAGPipeline.generate_from_results()  ← contexto + LLM
        ↓
{query, expanded_query, expanded_terms, answer, sources, web_activated}
```

Devuelve `{"error": "..."}` si `_pipeline_ready` no está activado.

### Modos standalone

| Método | Requiere | Devuelve |
|---|---|---|
| `query(text, top_n)` | `lsi_ready` | `list[dict]` — resultados LSI |
| `query_with_web(text, top_n)` | `lsi_ready` | `dict` — LSI + fallback web |
| `semantic_query(text, top_k)` | `faiss_ready` | `list[dict]` — FAISS directo |
| `qrf_search(text, top_k)` | `qrf_ready` | `list[dict]` — QRF standalone |
| `qrf_search_with_session(text, top_k)` | `qrf_ready` | `(list[dict], session_id)` |
| `rag_search(text, top_k)` | `rag_ready` | `list[dict]` — sin LLM |
| `rag_ask(text, top_k)` | `rag_ready` | `dict` — con respuesta LLM |
| `status()` | — | `dict` — snapshot del sistema |

Cada método devuelve lista vacía o dict vacío si el componente necesario
no está listo todavía, sin lanzar excepciones.

### `status()`

```python
orc.status()
# {
#   "uptime_seconds": 3612.4,
#   "threads": {"lsi_rebuild": True, "embedding": True, ...},
#   "lsi_ready": True,
#   "faiss_ready": True,
#   "qrf_ready": True,
#   "rag_ready": True,
#   "pipeline_ready": True,
#   "lsi": {"n_docs": 1234, "k": 100, ...},
#   "faiss": {"total_vectors": 9870, "index_type": "IndexIVFPQ", ...},
#   "db": {"total_documents": 1500, "pdf_indexed": 1234, ...},
#   ...
# }
```

---

## `OrchestratorConfig` — grupos de parámetros

```python
from backend.orchestrator.config import OrchestratorConfig

cfg = OrchestratorConfig(
    # Rutas
    db_path    = Path("data/db/documents.db"),
    model_path = Path("data/models/lsi_model.pkl"),

    # Crawler
    ids_per_discovery  = 100,
    batch_size         = 10,
    pdf_batch_size     = 5,
    discovery_interval = 120.0,   # s
    download_interval  = 30.0,    # s
    pdf_interval       = 60.0,    # s

    # LSI
    lsi_rebuild_interval = 3600.0,  # s entre reconstrucciones
    lsi_k                = 100,     # componentes SVD
    lsi_min_docs         = 10,      # mínimo de docs para construir el modelo
    lsi_doc_candidates   = 20,      # docs LSI a expandir a chunks
    lsi_min_df           = 20,      # términos con df < min_df excluidos del SVD
    lsi_max_df_ratio     = 0.85,    # términos en > 85% de docs excluidos

    # Embedding / FAISS
    embed_model         = "all-MiniLM-L6-v2",
    embed_batch_size    = 256,
    embed_poll_interval = 60.0,    # s entre sondeos
    embed_threshold     = 50,      # chunks pendientes para disparar embedding
    embed_rebuild_every = 10_000,  # chunks añadidos entre rebuilds de FAISS
    embed_nlist         = 100,
    embed_m             = 8,       # debe dividir la dimensión del modelo
    embed_nbits         = 8,
    embed_nprobe        = 10,

    # Indexación TF
    pdf_threshold       = 10,      # PDFs sin indexar para disparar
    index_poll_interval = 30.0,
    index_field         = "both",  # "full_text" | "abstract" | "both"
    index_batch_size    = 100,

    # Web Search
    web_threshold    = 0.35,       # score LSI mínimo para no activar web
    web_min_docs     = 1,
    web_max_results  = 5,
    web_use_fallback = True,       # SiteSearcher si Tavily falla

    # HybridRetriever
    hybrid_candidate_k = 50,       # candidatos por rama antes de RRF
    hybrid_rrf_k       = 60,       # constante RRF
    hybrid_parallel    = True,

    # Pipeline unificado
    pipeline_top_k      = 10,
    pipeline_rerank_k   = 5,
    pipeline_max_chunks = 5,
    pipeline_max_chars  = 400,

    # QRF
    qrf_top_k_initial   = 20,
    qrf_expand          = True,
    qrf_expand_top_dims = 3,
    qrf_expand_min_corr = 0.4,
    qrf_brf_alpha       = 0.75,
    qrf_mmr_lambda      = 0.6,

    # RAG standalone
    rag_use_reranker   = False,
    rag_candidate_k    = 50,
    rag_max_chunks     = 5,
    rag_max_chars      = 400,
)
```

---

## `_operations.py` — lógica desacoplada

Todas las operaciones pesadas viven aquí, separadas de la gestión de hilos:

| Función | Descripción |
|---|---|
| `do_index(cfg)` | Lanza `IndexingPipeline.run(reindex=False)` |
| `do_lsi_rebuild(cfg, lock, ready, holder)` | `LSIModel.build()` + `LSIRetriever.load()` + swap atómico |
| `do_embed(cfg, faiss_lock, faiss_mgr, faiss_ready)` | `EmbeddingPipeline.run()` + `faiss_ready.set()` |
| `do_web_search(query, results, cfg)` | `WebSearchPipeline.run()` con resultados LSI |
| `build_qrf_pipeline(cfg)` | Instancia `QueryPipeline` y llama a `load()` |
| `build_rag_pipeline(cfg, faiss_mgr, faiss_lock)` | `EmbeddingRetriever` + `CrossEncoder` + `RAGPipeline` |
| `build_hybrid_retriever(cfg, ...)` | `LSIRetriever` + `EmbeddingRetriever` + `HybridRetriever` |
| `build_cross_encoder(cfg)` | `CrossEncoderReranker` con modelo `ms-marco-MiniLM-L-6-v2` |
| `do_qrf_search(text, pipeline, top_k)` | `pipeline.search()` → `list[dict]` |
| `do_qrf_search_with_session(text, pipeline, top_k)` | `pipeline.search_with_session()` |
| `do_rag_search(text, pipeline, top_k, candidate_k)` | `pipeline.search()` |
| `do_rag_ask(text, pipeline, ...)` | `pipeline.ask()` |
| `do_pipeline_ask(query, qrf, hybrid, cross_enc, rag, cfg)` | Flujo unificado completo |

---

## Uso programático

```python
from backend.orchestrator import Orchestrator, OrchestratorConfig

orc = Orchestrator(OrchestratorConfig(lsi_k=150, embed_model="all-mpnet-base-v2"))
orc.start()

# Esperar a que el pipeline completo esté listo
orc._pipeline_ready.wait(timeout=120)

# Consulta unificada
result = orc.pipeline_ask("How does self-attention work in transformers?")
print(result["answer"])

# Modo standalone
hits = orc.query("attention mechanisms", top_n=5)

# Estado del sistema
print(orc.status())

orc.stop()
```

---

## CLI

```bash
# Lanzar el servidor Streamlit con parámetros por defecto
python -m backend.orchestrator.main

# Personalizado
python -m backend.orchestrator.main \
  --lsi-k 200 \
  --embed-model all-mpnet-base-v2 \
  --pdf-threshold 5 \
  --web-threshold 0.4 \
  --lsi-min-df 5 \
  --hybrid-candidate-k 100
```

---

## Tests

El orquestador no tiene tests unitarios dedicados — su comportamiento se
verifica a través de los tests de integración y de los módulos que coordina.

Los tests más relevantes que cubren el comportamiento orquestado son:

```bash
# Pipeline completo crawler → indexing → retrieval
pytest backend/tests/integration/test_full_pipeline.py -v

# Módulos individuales coordinados por el orchestrator
pytest backend/tests/retrieval/ -v
pytest backend/tests/embedding/ -v
pytest backend/tests/qrf/       -v
pytest backend/tests/rag/       -v
```