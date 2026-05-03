---
noteId: "4cdafa503c2211f19f111fe4db9ea2b5"
tags: []

---

# Módulo `orchestrator` — Documentación completa

## Índice

1. [Visión general](#1-visión-general)
2. [Estructura de ficheros](#2-estructura-de-ficheros)
3. [Flujo completo de trabajo](#3-flujo-completo-de-trabajo)
4. [Módulos — referencia detallada](#4-módulos--referencia-detallada)
   - [config.py](#41-configpy)
   - [orchestrator.py](#42-orchestratorpy)
   - [_faiss.py](#43-_faisspy)
   - [_operations.py](#44-_operationspy)
   - [_status.py](#45-_statuspy)
   - [cli.py](#46-clipy)
   - [main.py](#47-mainpy)
   - [threads/crawler.py](#48-threadscrawlerpy)
   - [threads/indexing.py](#49-threadsindexingpy)
   - [threads/lsi.py](#410-threadslsipy)
   - [threads/embedding.py](#411-threadsembeddingpy)
5. [Integración de módulos nuevos](#5-integración-de-módulos-nuevos)
   - [new_indexing (BM25)](#51-new_indexing-bm25)
   - [web_search](#52-web_search)
6. [Librerías utilizadas](#6-librerías-utilizadas)
7. [Referencia de comandos CLI](#7-referencia-de-comandos-cli)

---

## 1. Visión general

El módulo `orchestrator` es el **coordinador central** de OmniRetrieve-Engine. Su trabajo es arrancar y supervisar cuatro hilos daemon independientes que mantienen el sistema actualizado de forma continua, y exponer una API pública de consulta que usa la CLI y cualquier módulo externo.

```
                         ┌─────────────────────────────┐
                         │         Orchestrator         │
                         │  (ciclo de vida + API pública)│
                         └──────────────┬──────────────┘
              ┌───────────────┬──────────┴──────────┬───────────────┐
              ▼               ▼                     ▼               ▼
       [crawler]        [indexing]            [lsi_rebuild]   [embedding]
     ArxivClient →    BM25Pipeline →         LSIModel →      EmbeddingPipeline →
      IdStore +         SQLite                LSIRetriever     FaissIndexManager
      Documents
```

**Datos compartidos** entre hilos (protegidos con locks):

| Recurso | Lock | Hilo escritor | Hilo lector |
|---|---|---|---|
| `_retriever_holder` | `_lsi_lock` | `lsi_rebuild` | `query()` |
| `_faiss_mgr` | `_faiss_lock` | `embedding` | `semantic_query()` |

---

## 2. Estructura de ficheros

```
orchestrator/
│
├── __init__.py          Exports públicos del paquete.
├── config.py            OrchestratorConfig — todos los parámetros.
├── orchestrator.py      Orchestrator — coordinación y API pública.
├── _faiss.py            Inicialización del FaissIndexManager (interno).
├── _operations.py       Operaciones de negocio: do_index, do_lsi_rebuild,
│                        do_embed, do_web_search (interno).
├── _status.py           build_status() — snapshot del sistema (interno).
├── cli.py               Bucle interactivo y funciones de presentación.
├── main.py              Entrypoint: parseo de args + arranque del sistema.
│
└── threads/
    ├── __init__.py
    ├── crawler.py       run_crawler_thread — ejecuta el ArxivCrawler.
    ├── indexing.py      run_indexing_thread — watcher de indexación BM25.
    ├── lsi.py           run_lsi_rebuild_thread — rebuild periódico del LSI.
    └── embedding.py     run_embedding_thread — watcher de embedding FAISS.
```

---

## 3. Flujo completo de trabajo

### Arranque

```
main.py → _parse_args() → OrchestratorConfig
       → Orchestrator.__init__()
           → init_db() + init_embedding_schema()
           → init_faiss_mgr() → carga índice FAISS si existe
       → Orchestrator.start()
           → Thread(crawler)     → run_crawler_thread()
           → Thread(indexing)    → run_indexing_thread()
           → Thread(lsi_rebuild) → run_lsi_rebuild_thread()
           → Thread(embedding)   → run_embedding_thread()
       → Orchestrator.run_cli()  → bucle interactivo
```

### Pipeline de datos (en estado estacionario)

```
[crawler]
  ArxivClient.fetch_ids()      → IdStore (CSV)
  ArxivClient.fetch_documents() → Document (CSV + SQLite)
  ArxivClient.download_text()  → texto completo + chunks → SQLite

[indexing] (cada 30s, si ≥ pdf_threshold PDFs nuevos)
  new_indexing.IndexingPipeline.run() → BM25 postings → SQLite

[lsi_rebuild] (cada 3600s)
  LSIModel.build()  → SVD sobre postings
  LSIModel.save()   → disco (.pkl)
  LSIRetriever.load() → retriever_holder[0]

[embedding] (cada 60s, si ≥ embed_threshold chunks nuevos)
  EmbeddingPipeline.run() → vectores → FaissIndexManager
  FaissIndexManager.load() → actualiza _faiss_mgr

[query] (bajo demanda, hilo principal)
  LSIRetriever.retrieve(text) → resultados locales
  └→ [do_web_search] si score < web_threshold
       WebSearchPipeline.run() → Tavily / DuckDuckGo → BD + índice
```

---

## 4. Módulos — referencia detallada

### 4.1 `config.py`

**Propósito:** centralizar todos los parámetros de comportamiento del orquestador.

**Clase:** `OrchestratorConfig` (dataclass)

#### Grupo: Rutas

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `db_path` | `Path` | `data/documents.db` | BD SQLite compartida. |
| `model_path` | `Path` | `data/lsi_model.pkl` | Fichero del modelo LSI. |

#### Grupo: Crawler

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `ids_per_discovery` | `int` | `100` | IDs a descubrir por ciclo. |
| `batch_size` | `int` | `10` | Metadatos por ciclo. |
| `pdf_batch_size` | `int` | `5` | PDFs por ciclo. |
| `discovery_interval` | `float` | `120.0` | Segundos entre ciclos de discovery. |
| `download_interval` | `float` | `30.0` | Segundos entre ciclos de metadatos. |
| `pdf_interval` | `float` | `60.0` | Segundos entre PDFs individuales. |

#### Grupo: Indexing BM25

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `pdf_threshold` | `int` | `10` | PDFs nuevos para disparar indexación. |
| `index_poll_interval` | `float` | `30.0` | Segundos entre sondeos. |
| `index_field` | `str` | `"full_text"` | Campo a indexar: `"full_text"`, `"abstract"` o `"both"`. |
| `index_batch_size` | `int` | `100` | Docs por lote en BM25. |
| `index_use_stemming` | `bool` | `False` | Activar SnowballStemmer. |
| `index_min_token_len` | `int` | `3` | Longitud mínima de token. |

#### Grupo: LSI rebuild

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `lsi_rebuild_interval` | `float` | `3600.0` | Segundos entre rebuilds. |
| `lsi_k` | `int` | `100` | Componentes latentes del SVD. |
| `lsi_min_docs` | `int` | `10` | Mínimo de docs indexados para construir el modelo. |

#### Grupo: Embedding / FAISS

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `embed_model` | `str` | `"all-MiniLM-L6-v2"` | Modelo sentence-transformers. |
| `embed_batch_size` | `int` | `256` | Chunks por lote. |
| `embed_poll_interval` | `float` | `60.0` | Segundos entre sondeos. |
| `embed_threshold` | `int` | `50` | Chunks pendientes para disparar embedding. |
| `embed_rebuild_every` | `int` | `10000` | Chunks entre rebuilds completos de FAISS. |
| `embed_nlist` | `int` | `100` | Celdas Voronoi para IndexIVFPQ. |
| `embed_m` | `int` | `8` | Subvectores PQ. |
| `embed_nbits` | `int` | `8` | Bits por código PQ. |
| `embed_nprobe` | `int` | `10` | Celdas inspeccionadas en búsqueda. |
| `faiss_index_path` | `Path` | `data/faiss/index.faiss` | Fichero .faiss serializado. |
| `faiss_id_map_path` | `Path` | `data/faiss/id_map.npy` | Mapa posición → chunk_id. |

#### Grupo: Web Search

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `web_threshold` | `float` | `0.15` | Score mínimo del LSI para no activar web. |
| `web_min_docs` | `int` | `1` | Docs mínimos que deben superar el umbral. |
| `web_max_results` | `int` | `5` | Máximo resultados a pedir a Tavily/DDG. |
| `web_search_depth` | `str` | `"basic"` | Profundidad Tavily: `"basic"` o `"advanced"`. |
| `web_use_fallback` | `bool` | `True` | Usar DuckDuckGo si Tavily falla. |
| `web_auto_index` | `bool` | `True` | Indexar automáticamente docs web nuevos. |

---

### 4.2 `orchestrator.py`

**Propósito:** crear el estado compartido, arrancar los hilos daemon y exponer la API pública.

**Clase:** `Orchestrator`

#### `__init__(config)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `config` | `OrchestratorConfig \| None` | Configuración. Si es `None` usa los defaults. |

Inicializa BD, esquema de embedding e índice FAISS, y crea los cuatro objetos `Thread` (sin arrancarlos aún).

#### Métodos de ciclo de vida

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `start()` | — | `None` | Arranca los cuatro hilos daemon. No bloquea. |
| `stop()` | — | `None` | Activa `_shutdown` para que todos los hilos terminen. |
| `run_cli()` | — | `None` | Arranca la CLI interactiva. Bloquea hasta `quit`. |

#### Métodos de consulta

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `query(text, top_n=10)` | `str, int` | `list[dict]` | Query LSI local. Lista vacía si el modelo no está listo. |
| `query_with_web(text, top_n=10)` | `str, int` | `dict` | Query LSI + fallback web. Ver `do_web_search`. |
| `semantic_query(text, top_k=10)` | `str, int` | `list[dict]` | Búsqueda densa FAISS. Lista vacía si el índice no está listo. |
| `status()` | — | `dict` | Snapshot del estado. Ver `build_status`. |

---

### 4.3 `_faiss.py`

**Propósito:** inicialización del `FaissIndexManager` separada del constructor del `Orchestrator`.

#### `resolve_embedding_dim(model_name) -> int`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `model_name` | `str` | Nombre del modelo sentence-transformers. |

**Salida:** dimensión del espacio de embedding (sin cargar los pesos). Devuelve `384` si el modelo no está en el mapa de dimensiones conocidas.

#### `init_faiss_mgr(cfg) -> tuple[FaissIndexManager, bool]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `cfg` | `OrchestratorConfig` | Configuración del orquestador. |

**Salida:** `(manager, loaded)` donde `loaded` indica si se cargó un índice previo desde disco. Si `loaded=True`, el índice está listo para búsquedas sin rebuild.

---

### 4.4 `_operations.py`

**Propósito:** encapsular las operaciones de negocio que los hilos watchers o la CLI disparan.

#### `do_index(cfg) -> dict`

Ejecuta `new_indexing.IndexingPipeline` (BM25) en modo incremental.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `cfg` | `OrchestratorConfig` | Configuración. |

**Salida:** `dict` con `docs_processed`, `terms_added`, `postings_added`.

#### `do_lsi_rebuild(cfg, lsi_lock, lsi_ready, retriever_holder) -> dict | None`

Construye un nuevo `LSIModel`, lo persiste en disco y actualiza el retriever compartido bajo `lsi_lock`.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `cfg` | `OrchestratorConfig` | Configuración. |
| `lsi_lock` | `threading.RLock` | Protege `retriever_holder` durante el swap. |
| `lsi_ready` | `threading.Event` | Se activa al primer modelo exitoso. |
| `retriever_holder` | `list` | `[LSIRetriever \| None]` de un elemento. |

**Salida:** `dict` con `n_docs`, `n_terms`, `var_explained`, o `None` si no fue posible.

#### `do_embed(cfg, faiss_lock, faiss_mgr, faiss_ready) -> dict`

Ejecuta `EmbeddingPipeline` incremental y recarga el índice FAISS compartido.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `cfg` | `OrchestratorConfig` | Configuración. |
| `faiss_lock` | `threading.RLock` | Protege `faiss_mgr` durante el reload. |
| `faiss_mgr` | `FaissIndexManager \| None` | Manager compartido. |
| `faiss_ready` | `threading.Event` | Se activa cuando el índice tiene vectores. |

**Salida:** `dict` con `chunks_processed`, `batches_processed`, `rebuilds_triggered`.

#### `do_web_search(query, retriever_results, cfg) -> dict`

Evalúa suficiencia local y, si procede, busca en la web combinando los resultados.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `query` | `str` | Consulta original del usuario. |
| `retriever_results` | `list[dict]` | Resultados del LSIRetriever (con campo `score`). |
| `cfg` | `OrchestratorConfig` | Configuración (umbrales, claves, etc.). |

**Salida:** `dict` con `results`, `web_activated`, `web_results`, `reason`, `query`, `indexed`.

---

### 4.5 `_status.py`

**Propósito:** construir el snapshot de estado del sistema consultando BD, LSI y FAISS.

#### `build_status(cfg, lsi_lock, retriever_holder, faiss_lock, faiss_mgr, lsi_ready, faiss_ready) -> dict`

Consulta todas las fuentes de datos y devuelve un `dict` con las métricas del sistema en tiempo real. Los errores de BD se capturan y devuelven como `-1` para no interrumpir la respuesta.

**Claves del dict devuelto:**

| Clave | Descripción |
|---|---|
| `docs_total` | Total de documentos en BD. |
| `docs_pdf_indexed` | Documentos con texto descargado. |
| `docs_pdf_pending` | Documentos sin texto. |
| `docs_not_in_index` | Documentos con texto pero no indexados. |
| `vocab_size` | Términos únicos en el índice BM25. |
| `total_postings` | Postings totales en el índice. |
| `lsi_docs_in_model` | Documentos incluidos en el modelo LSI actual. |
| `lsi_model_ready` | `True` si el modelo LSI está disponible. |
| `total_chunks` | Chunks en BD. |
| `embedded_chunks` | Chunks con embedding. |
| `pending_chunks` | Chunks sin embedding. |
| `faiss_vectors` | Vectores en el índice FAISS. |
| `faiss_index_type` | Tipo de índice FAISS (`"flat"`, `"ivfpq"`, etc.). |
| `faiss_ready` | `True` si el índice FAISS está listo. |
| `embed_model` | Nombre del modelo de embedding activo. |
| `web_threshold` | Umbral de score para activar búsqueda web. |
| `web_min_docs` | Docs mínimos que deben superar el umbral. |
| `timestamp` | Marca de tiempo UTC de la consulta. |

---

### 4.6 `cli.py`

**Propósito:** interfaz interactiva. Sin estado propio; recibe todos los callables por parámetro.

#### `run_cli(shutdown, lsi_ready, lsi_min_docs, fn_query, fn_query_web, fn_status, fn_index, fn_rebuild, fn_stop)`

Bucle principal. Bloquea el hilo que lo llama hasta que el usuario escribe `quit` o `shutdown` se activa.

**Funciones de presentación públicas:**

| Función | Entrada | Descripción |
|---|---|---|
| `print_banner()` | — | Cabecera del sistema. |
| `print_help()` | — | Lista de comandos disponibles. |
| `print_status(s)` | `dict` | Muestra el snapshot de estado formateado, incluyendo sección web. |
| `print_results(results, query)` | `list[dict], str` | Resultados de una query LSI local. |
| `print_web_results(output)` | `dict` | Resultados combinados local + web, con indicador de fuente. |

---

### 4.7 `main.py`

**Propósito:** entrypoint del módulo (`python -m backend.orchestrator`).

Parsea argumentos de línea de comandos con `argparse`, construye el `OrchestratorConfig` y arranca el `Orchestrator`.

**Grupos de argumentos:** `crawler`, `indexing (BM25)`, `lsi`, `web search`.

---

### 4.8 `threads/crawler.py`

**Propósito:** mantener el `Crawler` corriendo hasta que `shutdown` se active.

#### `run_crawler_thread(cfg, shutdown) -> None`

Crea un `Crawler` con los parámetros del orquestador, lanza un hilo watchdog que espera `shutdown` y llama a `crawler.stop()`, y ejecuta `crawler.run_forever()`.

---

### 4.9 `threads/indexing.py`

**Propósito:** disparar indexación BM25 cuando hay suficientes PDFs nuevos.

#### `run_indexing_thread(cfg, shutdown, do_index) -> None`

Sondea SQLite cada `index_poll_interval` segundos. Llama a `do_index()` cuando hay `≥ pdf_threshold` documentos con texto descargado pero sin indexar.

`do_index` se recibe inyectado (no lo instancia internamente) para facilitar tests y mantener el módulo sin estado.

---

### 4.10 `threads/lsi.py`

**Propósito:** reconstruir el modelo LSI periódicamente.

#### `run_lsi_rebuild_thread(cfg, shutdown, lsi_lock, lsi_ready, retriever_holder) -> None`

Llama a `do_lsi_rebuild()` inmediatamente al arrancar y luego cada `lsi_rebuild_interval` segundos. Si hay menos de `lsi_min_docs` documentos indexados, omite silenciosamente el rebuild.

---

### 4.11 `threads/embedding.py`

**Propósito:** disparar embedding cuando hay suficientes chunks nuevos.

#### `run_embedding_thread(cfg, shutdown, do_embed) -> None`

Primer chequeo inmediato al arrancar (para retomar chunks pendientes de sesiones anteriores). Luego sondea cada `embed_poll_interval` segundos y llama a `do_embed()` cuando hay `≥ embed_threshold` chunks sin embedding.

---

## 5. Integración de módulos nuevos

### 5.1 `new_indexing` (BM25)

El pipeline de indexación anterior (`backend.indexing.pipeline`) ha sido **reemplazado** por `backend.new_indexing.pipeline` en `_operations.do_index`.

**Diferencias clave:**

| Aspecto | Anterior (`indexing`) | Nuevo (`new_indexing`) |
|---|---|---|
| Algoritmo | TF-IDF con postings simples | BM25 (parámetros `k1`, `b`) |
| Preprocesado | Básico | `TextPreprocessor` con stemming opcional |
| Parámetros nuevos en config | — | `index_batch_size`, `index_use_stemming`, `index_min_token_len` |
| Parámetros CLI nuevos | — | `--index-batch`, `--stemming`, `--min-token-len` |

No se requiere ningún cambio en los módulos de retrieval LSI (siguen usando la tabla `postings` con la misma interfaz).

### 5.2 `web_search`

El módulo `backend.web_search` se integra en el orquestador de la siguiente manera:

**Nuevo método en `Orchestrator`:**
- `query_with_web(text, top_n)` → llama a `query()` + `do_web_search()`.

**Nuevo comando en CLI:**
- `wsearch <texto>` → llama a `fn_query_web`.

**Nuevo parámetro en `status()`:**
- `web_threshold`, `web_min_docs` — visibles en el comando `status`.

**Lógica de suficiencia (`web_search.sufficiency.SufficiencyChecker`):**
- Si al menos `web_min_docs` resultados locales tienen `score ≥ web_threshold` → no se activa la web.
- En caso contrario → `WebSearcher.search()` → Tavily (con DuckDuckGo como fallback).
- Los resultados web se guardan en BD y se indexan automáticamente si `web_auto_index=True`.

**Dependencias necesarias para web_search:**
```
pip install tavily-python python-dotenv duckduckgo-search
```
Y crear un fichero `.env` en la raíz del proyecto:
```
TAVILY_API_KEY=tvly-tu-key-aqui
```

---

## 6. Librerías utilizadas

| Librería | Origen | Dónde se usa |
|---|---|---|
| `threading` | stdlib | `orchestrator.py`, todos los threads, `_operations.py` |
| `argparse` | stdlib | `main.py` |
| `logging` | stdlib | Todos los módulos |
| `pathlib` | stdlib | `config.py`, `main.py` |
| `sqlite3` (vía repo) | stdlib | `_status.py`, `threads/indexing.py` |
| `sentence-transformers` | externa | `_operations.do_embed`, `orchestrator.semantic_query` |
| `faiss-cpu` | externa | `_faiss.py`, `_operations.do_embed` |
| `numpy` | externa | FAISS id_map |
| `scipy` / `sklearn` | externa | `LSIModel` (SVD) |
| `tavily-python` | externa | `_operations.do_web_search` → `WebSearcher` |
| `duckduckgo-search` | externa | `_operations.do_web_search` → `DuckDuckGoSearcher` (fallback) |
| `python-dotenv` | externa | `WebSearcher` (lectura de `TAVILY_API_KEY`) |

---

## 7. Referencia de comandos CLI

| Comando | Descripción |
|---|---|
| `query <texto>` | Query LSI local. Devuelve los 10 artículos más relevantes. |
| `wsearch <texto>` | Query LSI + búsqueda web automática si el score es insuficiente. |
| `<texto>` | Atajo de `query`: cualquier texto sin prefijo se trata como query local. |
| `status` | Estado completo del sistema (BD, LSI, FAISS, web). |
| `index` | Fuerza indexación BM25 incremental ahora. |
| `rebuild` | Fuerza reconstrucción del modelo LSI ahora. |
| `help` | Lista de comandos disponibles. |
| `quit` / `exit` | Detiene el sistema y sale. |
