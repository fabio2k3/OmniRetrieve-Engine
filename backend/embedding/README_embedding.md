---
noteId: "5f27afa03c2c11f19f111fe4db9ea2b5"
tags: []

---

# Módulo `embedding` — Documentación completa

## Índice

1. [Visión general](#1-visión-general)
2. [Estructura de ficheros](#2-estructura-de-ficheros)
3. [Flujo completo de trabajo](#3-flujo-completo-de-trabajo)
4. [Módulos — referencia detallada](#4-módulos--referencia-detallada)
   - [embedder.py](#41-embedderpy)
   - [pipeline.py](#42-pipelinepy)
   - [_sync.py](#43-_syncpy)
   - [_batch.py](#44-_batchpy)
   - [_meta.py](#45-_metapy)
   - [main.py](#46-mainpy)
   - [faiss/constants.py](#47-faissconstantspy)
   - [faiss/builder.py](#48-faissbuildepy)
   - [faiss/index_manager.py](#49-faissindex_managerpy)
5. [Librerías utilizadas](#5-librerías-utilizadas)
6. [Referencia de argumentos CLI](#6-referencia-de-argumentos-cli)

---

## 1. Visión general

El módulo `embedding` transforma texto en vectores densos y los indexa en FAISS para permitir búsqueda semántica eficiente sobre el corpus.

```
BD SQLite (chunks sin embedding)
         │
         ▼
  [ChunkEmbedder]          ← sentence-transformers
  encode(texts) → (N, dim) float32
         │
         ▼
  [_batch.process_batch]
  ├── save_chunk_embeddings_batch() → BD (embedding serializado)
  └── FaissIndexManager.add()      → índice en memoria
         │
         ▼ (cada rebuild_every chunks)
  [FaissIndexManager.rebuild]
  ├── builder.build_ivfpq()  → IndexIVFPQ entrenado
  └── save()                 → disco (.faiss + .npy)
```

---

## 2. Estructura de ficheros

```
embedding/
│
├── __init__.py        Exports públicos del paquete.
├── embedder.py        ChunkEmbedder — wrapper sobre sentence-transformers.
├── pipeline.py        EmbeddingPipeline — coordinador del flujo completo.
├── _sync.py           Sincronización FAISS-BD y reset de embeddings (interno).
├── _batch.py          Procesamiento de lotes: vectorizar + persistir (interno).
├── _meta.py           Registro de metadatos y presentación de stats (interno).
├── main.py            Entrypoint CLI (argparse + main).
│
└── faiss/
    ├── __init__.py
    ├── constants.py   Constantes y parámetros por defecto del índice.
    ├── builder.py     Construcción de índices: build_flat, build_ivfpq.
    └── index_manager.py  FaissIndexManager — gestión, búsqueda y persistencia.
```

---

## 3. Flujo completo de trabajo

### `EmbeddingPipeline.run(reembed=False)`

```
1. init_embedding_schema()           ← garantiza que las tablas existen
2. [si reembed] reset_embeddings()   ← borra todos los embeddings de la BD
3. ChunkEmbedder(model_name)         ← carga el modelo (puede tardar)
   FaissIndexManager(dim, ...)       ← crea el manager
   faiss_mgr.load()                  ← intenta cargar índice previo de disco
4. check_and_sync(faiss_mgr, BD)     ← reconstruye si FAISS < BD (desincronía)
5. Para cada lote de chunks pendientes:
   a. process_batch(rows, embedder, faiss_mgr, db_path)
      ├── embedder.encode(texts)      → vectors (N, dim)
      ├── save_chunk_embeddings_batch → BD
      └── faiss_mgr.add(vectors, ids) → índice en memoria
   b. faiss_mgr.maybe_rebuild(db_path)
      └── [si umbral alcanzado] rebuild() → entrena IVFPQ + save()
6. faiss_mgr.save()                  ← guardado final en disco
7. save_run_meta(stats)              ← timestamp, modelo, chunks procesados
8. return stats
```

### Política de rebuild del índice FAISS

El índice se reconstruye completamente cuando `_added_since_last_rebuild >= rebuild_every` (por defecto 10 000). En cada reconstrucción:
- Si hay `>= min_train_size(nlist, nbits)` vectores → `IndexIVFPQ` (comprimido, rápido).
- Si no hay suficientes → `IndexFlatL2` (exacto, sin compresión).

El índice IVFPQ reemplaza automáticamente al FlatL2 en la primera reconstrucción con datos suficientes.

---

## 4. Módulos — referencia detallada

### 4.1 `embedder.py`

**Propósito:** wrapper ligero sobre `SentenceTransformer` para uso en el pipeline.

**Clase:** `ChunkEmbedder`

#### `__init__(model_name, device, batch_size, normalize)`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Nombre o ruta del modelo. |
| `device` | `str \| None` | `None` | `'cpu'`, `'cuda'`, `'mps'` o autodetección. |
| `batch_size` | `int` | `64` | Frases por lote en la inferencia interna. |
| `normalize` | `bool` | `True` | L2-normaliza los vectores (recomendado para similitud coseno). |

**Propiedades:**

| Propiedad | Tipo | Descripción |
|---|---|---|
| `dim` | `int` | Dimensión del espacio de embedding del modelo. |

#### `encode(texts) -> np.ndarray`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `texts` | `Sequence[str]` | Lista de textos a vectorizar. |

**Salida:** `np.ndarray` de shape `(N, dim)`, dtype `float32`. Los textos vacíos se sustituyen por `" "` para evitar errores del modelo.

#### `encode_single(text) -> np.ndarray`

Equivalente a `encode([text])[0]`. Útil para vectorizar queries en `semantic_query`.

**Salida:** `np.ndarray` 1-D de shape `(dim,)`.

---

### 4.2 `pipeline.py`

**Propósito:** coordinar el flujo completo de vectorización delegando en los módulos internos.

**Clase:** `EmbeddingPipeline`

#### `__init__(db_path, model_name, device, batch_size, rebuild_every, nlist, m, nbits, nprobe, index_path, id_map_path)`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `db_path` | `Path` | `data/documents.db` | BD SQLite. |
| `model_name` | `str` | `"all-MiniLM-L6-v2"` | Modelo sentence-transformers. |
| `device` | `str \| None` | `None` | Dispositivo de inferencia. |
| `batch_size` | `int` | `256` | Chunks por lote. |
| `rebuild_every` | `int` | `10000` | Chunks entre rebuilds de FAISS. |
| `nlist` | `int` | `100` | Celdas Voronoi para IVFPQ. |
| `m` | `int` | `8` | Subvectores PQ (debe dividir `dim`). |
| `nbits` | `int` | `8` | Bits por código PQ. |
| `nprobe` | `int` | `10` | Celdas inspeccionadas en búsqueda. |
| `index_path` | `Path` | `data/faiss/index.faiss` | Fichero .faiss. |
| `id_map_path` | `Path` | `data/faiss/id_map.npy` | Fichero .npy de IDs. |

#### `run(reembed=False) -> dict`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `reembed` | `bool` | Si `True`, resetea embeddings y re-vectoriza todo el corpus. |

**Salida:** `dict` con `chunks_processed`, `chunks_skipped`, `batches_processed`, `rebuilds_triggered`, `model_name`, `started_at`, `finished_at`.

---

### 4.3 `_sync.py`

**Propósito:** detectar y corregir desincronizaciones entre el índice FAISS y la BD.

#### `check_and_sync(faiss_mgr, already_embedded, db_path) -> None`

Compara `faiss_mgr.total_vectors` con `already_embedded`. Si hay menos vectores en FAISS que embeddings en BD, llama a `faiss_mgr.rebuild()` para sincronizar.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `faiss_mgr` | `FaissIndexManager` | Gestor del índice. |
| `already_embedded` | `int` | Chunks con embedding en la BD. |
| `db_path` | `Path` | BD SQLite. |

#### `reset_embeddings(db_path) -> int`

Pone a `NULL` todos los embeddings de la tabla `chunks`. Devuelve el número de registros afectados.

---

### 4.4 `_batch.py`

**Propósito:** vectorizar un lote y persistir el resultado en BD e índice FAISS.

#### `process_batch(rows, embedder, faiss_mgr, db_path) -> tuple[int, int]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `rows` | `list[sqlite3.Row]` | Filas con `id`, `arxiv_id`, `chunk_index`, `text`. |
| `embedder` | `ChunkEmbedder` | Instancia ya inicializada. |
| `faiss_mgr` | `FaissIndexManager` | Gestor del índice compartido. |
| `db_path` | `Path` | BD SQLite. |

**Salida:** `(n_processed, n_skipped)` donde `n_skipped` cuenta los chunks con texto vacío que se omitieron.

**Pasos internos:**
1. Filtrar chunks con texto vacío.
2. `embedder.encode(valid_texts)` → `vectors (N, dim)`.
3. `save_chunk_embeddings_batch(db_batch, db_path)` → serializa y persiste en BD.
4. `faiss_mgr.add(vectors, valid_ids)` → añade al índice en memoria.

---

### 4.5 `_meta.py`

**Propósito:** registrar metadatos de cada ejecución y mostrar estadísticas por CLI.

#### `log_faiss_build(faiss_mgr, model_name, db_path) -> None`

Llama a `faiss_mgr.build_stats()` y persiste el resultado en la tabla `faiss_log` de la BD.

#### `save_run_meta(stats, db_path) -> None`

Guarda `last_run_at`, `last_chunks_embedded` y `last_model` en la tabla `embedding_meta`.

#### `print_stats(db_path) -> None`

Lee `embedding_meta` y `faiss_log` y muestra un resumen por stdout. Usada por el flag `--stats` de la CLI.

---

### 4.6 `main.py`

**Propósito:** entrypoint del módulo (`python -m backend.embedding`).

Parsea argumentos con `argparse`, construye un `EmbeddingPipeline` y llama a `.run()`. Si se pasa `--stats`, llama a `print_stats()` y sale.

---

### 4.7 `faiss/constants.py`

**Propósito:** centralizar todos los literales del subpaquete FAISS.

| Constante | Valor | Descripción |
|---|---|---|
| `MIN_TRAIN_FACTOR` | `39` | Vectores mínimos por celda de Voronoi para entrenar K-means. |
| `DEFAULT_NLIST` | `100` | Celdas Voronoi por defecto. |
| `DEFAULT_M` | `8` | Subvectores PQ por defecto. |
| `DEFAULT_NBITS` | `8` | Bits por código PQ por defecto. |
| `DEFAULT_NPROBE` | `10` | Celdas inspeccionadas en búsqueda por defecto. |
| `DEFAULT_REBUILD_EVERY` | `10000` | Chunks entre rebuilds por defecto. |

---

### 4.8 `faiss/builder.py`

**Propósito:** construir índices FAISS. No gestiona estado ni persistencia.

#### `min_train_size(nlist, nbits) -> int`

Calcula el mínimo de vectores para entrenar `IndexIVFPQ` de forma estable. Devuelve `max(nlist * 39, 2^nbits)`.

#### `effective_nlist(n_vectors, max_nlist) -> int`

Calcula el `nlist` ajustado al corpus: `max(4, min(max_nlist, sqrt(n_vectors)))`.

#### `build_flat(faiss_module, dim) -> IndexFlatL2`

Crea un índice de búsqueda exacta sin entrenamiento. Devuelve `faiss.IndexFlatL2(dim)`.

#### `build_ivfpq(faiss_module, dim, training_vectors, nlist, m, nbits, nprobe) -> IndexIVFPQ`

Crea y entrena un `IndexIVFPQ`. Ajusta `nlist` al tamaño real del corpus con `effective_nlist`. Devuelve el índice entrenado, listo para añadir vectores.

---

### 4.9 `faiss/index_manager.py`

**Propósito:** gestionar el ciclo de vida completo del índice FAISS.

**Clase:** `FaissIndexManager`

**Propiedades:**

| Propiedad | Tipo | Descripción |
|---|---|---|
| `total_vectors` | `int` | Número de vectores actualmente en el índice. |
| `index_type` | `str` | `"IndexIVFPQ"`, `"IndexFlatL2"` o `"none"`. |

**Métodos:**

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `add(vectors, chunk_ids)` | `ndarray, list[int]` | `None` | Añade vectores al índice. Inicializa FlatL2 si el índice no existe. |
| `rebuild(db_path)` | `Path` | `dict` | Reconstruye el índice completo desde la BD. Elige IVFPQ o FlatL2 según datos disponibles. Llama a `save()` al terminar. |
| `maybe_rebuild(db_path)` | `Path` | `bool` | Llama a `rebuild()` si `_added_since_last_rebuild >= rebuild_every`. Devuelve `True` si reconstruyó. |
| `search(query_vector, top_k)` | `ndarray, int` | `list[dict]` | Busca los `top_k` chunks más cercanos. Devuelve `[{"chunk_id": int, "score": float}]`. |
| `save()` | — | `None` | Serializa el índice y el mapa de IDs en disco. |
| `load()` | — | `bool` | Carga índice y mapa de IDs desde disco. Devuelve `True` si los ficheros existían. |
| `build_stats()` | — | `dict` | Stats del índice actual: `n_vectors`, `index_type`, `nlist`, `m`, `nbits`, rutas. |

---

## 5. Librerías utilizadas

| Librería | Origen | Dónde se usa | Propósito |
|---|---|---|---|
| `sentence-transformers` | **externa** | `embedder.py` | Modelos de embedding de texto. |
| `numpy` | **externa** | `faiss/`, `_batch.py`, `embedder.py` | Operaciones con arrays de vectores. |
| `faiss-cpu` / `faiss-gpu` | **externa** | `faiss/index_manager.py`, `faiss/builder.py` | Índice vectorial de búsqueda aproximada. |
| `sqlite3` (vía repo) | stdlib | `pipeline.py`, `_batch.py`, `_sync.py`, `_meta.py` | Persistencia de embeddings y metadatos. |
| `pathlib` | stdlib | Todos los módulos | Rutas del sistema de ficheros. |
| `logging` | stdlib | Todos los módulos | Logs estructurados. |
| `math` | stdlib | `faiss/builder.py` | Cálculo de `sqrt` para `effective_nlist`. |
| `time` | stdlib | `faiss/builder.py`, `faiss/index_manager.py` | Medición de tiempos de entrenamiento. |
| `datetime` | stdlib | `_batch.py`, `pipeline.py` | Timestamps de los embeddings. |
| `argparse` | stdlib | `main.py` | Parseo de argumentos CLI. |

---

## 6. Referencia de argumentos CLI

```
python -m backend.embedding [opciones]
```

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--db` | Path | `data/documents.db` | Ruta a la BD SQLite. |
| `--model` | str | `all-MiniLM-L6-v2` | Modelo sentence-transformers. |
| `--device` | str | `None` | `cpu`, `cuda`, `mps` o autodetección. |
| `--batch-size` | int | `256` | Chunks por lote de vectorización. |
| `--rebuild-every` | int | `10000` | Chunks entre rebuilds completos de FAISS. |
| `--nlist` | int | `100` | Celdas Voronoi para IVFPQ. |
| `--m` | int | `8` | Subvectores PQ (debe dividir la dimensión del modelo). |
| `--nbits` | int | `8` | Bits por código PQ. |
| `--nprobe` | int | `10` | Celdas inspeccionadas durante la búsqueda. |
| `--index-path` | Path | `data/faiss/index.faiss` | Ruta del fichero .faiss. |
| `--id-map-path` | Path | `data/faiss/id_map.npy` | Ruta del fichero .npy de IDs. |
| `--reembed` | flag | — | Resetear embeddings y re-vectorizar todo el corpus. |
| `--stats` | flag | — | Mostrar estadísticas del estado actual y salir. |
