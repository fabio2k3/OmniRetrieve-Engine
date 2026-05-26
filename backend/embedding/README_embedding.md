# OmniRetrieve — Módulo `embedding`

Vectoriza los chunks de texto con `sentence-transformers` y gestiona el
índice vectorial FAISS para búsqueda semántica densa. El módulo coordina
cuatro responsabilidades bien separadas: vectorización (`embedder.py`),
procesamiento por lotes (`_batch.py`), sincronización con la BD (`_sync.py`)
y registro de metadatos (`_meta.py`).

---

## Estructura de archivos

```
backend/embedding/
├── pipeline.py        ← EmbeddingPipeline: orquestador principal (8 pasos)
├── embedder.py        ← ChunkEmbedder: wrapper de sentence-transformers
├── _batch.py          ← process_batch(): vectoriza un lote y persiste en BD + FAISS
├── _sync.py           ← check_and_sync(): detecta y corrige desincronizaciones
├── _meta.py           ← log_faiss_build(), save_run_meta(), print_stats()
├── main.py            ← entrypoint CLI (alias de embed_chunks tool)
├── faiss/
│   ├── __init__.py    ← exporta FaissIndexManager
│   ├── index_manager.py ← FaissIndexManager: ciclo de vida completo del índice
│   ├── builder.py     ← build_flat(), build_ivfpq(), min_train_size(), effective_nlist()
│   └── constants.py   ← MIN_TRAIN_FACTOR = 39 (heurística de entrenamiento FAISS)
└── __init__.py        ← exports públicos: EmbeddingPipeline, ChunkEmbedder, FaissIndexManager
```

---

## Instalación

```bash
pip install sentence-transformers faiss-cpu numpy

# GPU (opcional)
pip install faiss-gpu
```

---

## Flujo de `EmbeddingPipeline.run()`

```
1. init_embedding_schema()
       ↓  crea faiss_log y embedding_meta si no existen

2. [Si reembed=True]  reset_embeddings()
       ↓  pone embedding=NULL en todos los chunks

3. ChunkEmbedder(model_name, device)       FaissIndexManager(dim, nlist, m, nbits, …)
       ↓  carga modelo                           ↓  load() desde disco
       └────────────────────────────────────────┘

4. check_and_sync(faiss_mgr, embedded_count, db_path)
       ↓  si FAISS tiene menos vectores que embeddings en BD → rebuild()

5. Para cada lote de chunks con embedding=NULL:
   a. process_batch(rows, embedder, faiss_mgr, db_path)
        ├─ embedder.encode(texts)          → ndarray (N, dim) float32
        ├─ save_chunk_embeddings_batch()   → persiste BLOB en chunks.embedding
        └─ faiss_mgr.add(vectors, ids)     → añade al índice en memoria
   b. faiss_mgr.maybe_rebuild(db_path)
        ↓  rebuild() si added_since_last_rebuild >= rebuild_every

6. faiss_mgr.save()
       ↓  escribe index.faiss + id_map.npy

7. log_faiss_build()    → registra en faiss_log
   save_run_meta()      → persiste model_name, last_run_at, etc. en embedding_meta

8. Devuelve dict con estadísticas del run
```

---

## `ChunkEmbedder`

Wrapper delgado sobre `SentenceTransformer`. Responsabilidad única: recibir
listas de strings y devolver arrays NumPy normalizados. No accede a BD ni FAISS.

```python
from backend.embedding.embedder import ChunkEmbedder

embedder = ChunkEmbedder(
    model_name = "all-MiniLM-L6-v2",  # dim=384, divisible por m=8 y m=16
    device     = None,                 # autodetección (cpu/cuda/mps)
    batch_size = 64,                   # frases por llamada interna al modelo
    normalize  = True,                 # L2-normalización (recomendado para coseno)
)

vecs = embedder.encode(["chunk A", "chunk B"])
# → ndarray float32, shape (2, 384), L2-normalizado

single = embedder.encode_single("una query")
# → ndarray float32, shape (384,)

print(embedder.dim)   # 384
```

Los textos vacíos o `None` se sustituyen por `" "` antes de pasar al modelo
para evitar errores. Los vectores de salida están garantizados en `float32`
(requisito de FAISS).

**Modelo por defecto:** `all-MiniLM-L6-v2` — 384 dimensiones, ~80 MB,
equilibrio calidad/velocidad/tamaño para corpus científico en inglés.

---

## `FaissIndexManager`

Gestiona el ciclo de vida completo del índice FAISS: creación, actualización
incremental, búsqueda y persistencia en disco.

### Tipos de índice

| Tipo | Cuándo | Características |
|---|---|---|
| `IndexFlatL2` | `n < min_train_size(nlist, nbits)` | Búsqueda exacta, sin entrenamiento |
| `IndexIVFPQ` | `n >= min_train_size(nlist, nbits)` | Aproximada, cuantización PQ, rápida |

El umbral mínimo para `IndexIVFPQ` es:

```
min_train_size = max(nlist × 39, 2^nbits)
```

Con los parámetros por defecto (`nlist=100`, `nbits=8`): **mínimo 3 900 vectores**.

La transición de `IndexFlatL2` a `IndexIVFPQ` ocurre automáticamente en el
primer `rebuild()` que supera el umbral.

### `nlist` efectivo

FAISS recomienda `nlist ≈ √n_vectors`. `builder.effective_nlist()` calcula
el valor ajustado automáticamente respetando el techo configurado.

### `m` — subvectores PQ

**Debe dividir exactamente a `dim`.** Con `dim=384`: valores válidos son
`m=8`, `m=16`, `m=24`, `m=32`, etc. El constructor lanza `ValueError` si
no se cumple esta restricción.

### API principal

```python
from backend.embedding.faiss import FaissIndexManager

mgr = FaissIndexManager(
    dim=384, nlist=100, m=8, nbits=8, nprobe=10,
    rebuild_every=10_000,
    index_path=Path("data/faiss/index.faiss"),
    id_map_path=Path("data/faiss/id_map.npy"),
)

# Añadir vectores
mgr.add(vectors, chunk_ids)    # ndarray (N, dim) + list[int]

# Búsqueda
results = mgr.search(query_vec, top_k=10)
# → list[{"chunk_id": int, "score": float}]  ordenada por distancia L2 asc

# Persistencia
mgr.save()                     # escribe .faiss + .npy
mgr.load()                     # → bool (True si encontró ficheros)

# Reconstrucción
mgr.rebuild(db_path)           # lee todos los embeddings de BD y reconstruye
mgr.maybe_rebuild(db_path)     # → bool; rebuild si added >= rebuild_every

# Estado
mgr.total_vectors              # int: vectores en el índice
mgr.index_type                 # str: "IndexFlatL2" | "IndexIVFPQ" | "none"
mgr.build_stats()              # dict: n_vectors, index_type, nlist, m, nbits, paths
```

### Mapa de IDs

El índice FAISS solo conoce posiciones internas (0, 1, 2…). El archivo
`id_map.npy` es un `ndarray int64` que mapea cada posición al `chunk_id`
real de la tabla `chunks`:

```
posición FAISS 0 → id_map[0] = chunk_id 42
posición FAISS 1 → id_map[1] = chunk_id 117
…
```

`search()` traduce automáticamente los índices FAISS a `chunk_id`.

---

## `_batch.py` — Procesamiento por lote

`process_batch(rows, embedder, faiss_mgr, db_path)` ejecuta tres pasos para
cada lote de `sqlite3.Row` con columnas `id`, `text`:

1. Filtra chunks con texto vacío (`n_skipped`)
2. `embedder.encode(valid_texts)` → `ndarray (N, dim)`
3. Serializa: `vec.tobytes()` → `save_chunk_embeddings_batch([(bytes, ts, id), …])`
4. `faiss_mgr.add(vectors, valid_ids)`

Devuelve `(n_processed, n_skipped)`.

---

## `_sync.py` — Sincronización FAISS ↔ BD

`check_and_sync(faiss_mgr, already_embedded, db_path)` se ejecuta al inicio
de cada `run()` para detectar desincronizaciones:

```
Embeddings en BD: 5 000  |  Vectores en FAISS: 3 200
         ↓  desincronización detectada
faiss_mgr.rebuild(db_path)  ← reconstruye desde BD
```

Las desincronizaciones ocurren cuando el proceso se reinicia después de
haber generado embeddings pero antes de haber guardado el índice.

`reset_embeddings(db_path)` pone `embedding=NULL` en todos los chunks.
Se llama cuando `reembed=True` para re-vectorizar con un modelo diferente.

---

## `_meta.py` — Metadatos y estadísticas

| Función | Descripción |
|---|---|
| `log_faiss_build(faiss_mgr, model_name, db_path)` | Registra en `faiss_log`: n_vectors, index_type, nlist, m, nbits, paths |
| `save_run_meta(stats, db_path)` | Persiste en `embedding_meta`: `last_run_at`, `last_chunks_embedded`, `last_model` |
| `print_stats(db_path)` | Imprime resumen por stdout (flag `--stats` de la CLI) |

---

## CLI (via `backend/tools/embed_chunks.py`)

```bash
# Embedizar chunks pendientes (incremental)
python -m backend.tools.embed_chunks

# Modelo diferente
python -m backend.tools.embed_chunks --model all-mpnet-base-v2

# Re-embedizar todo el corpus desde cero
python -m backend.tools.embed_chunks --reembed

# Ajustar parámetros FAISS
python -m backend.tools.embed_chunks --nlist 200 --m 16 --nprobe 20

# Ver estado sin procesar
python -m backend.tools.embed_chunks --stats

# Especificar BD y dispositivo
python -m backend.tools.embed_chunks --db /ruta/a/db.sqlite --device cuda
```

---

## Tests

Los tests están en `backend/tests/embedding/`.

```bash
pytest backend/tests/embedding/ -v

pytest backend/tests/embedding/test_chunk_repository.py     -v
pytest backend/tests/embedding/test_embedding_repository.py -v
pytest backend/tests/embedding/test_faiss_index.py          -v
pytest backend/tests/embedding/test_pipeline.py             -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_chunk_repository.py` | `save_chunks`, `get_chunks`, conteos, `get_unembedded_chunks_iter`, `save_chunk_embeddings_batch`, `get_all_embeddings_iter`, `reset_embeddings` |
| `test_embedding_repository.py` | `log_faiss_build`, `save/get_embedding_meta`, `get_embedding_stats` (claves devueltas) |
| `test_faiss_index.py` | `add`+`search` (chunk_id correcto), `save`/`load` round-trip, `rebuild` → transición a IndexIVFPQ, `maybe_rebuild`, thread-safety con 5 hilos |
| `test_pipeline.py` | Pipeline end-to-end con `MockEmbedder` inyectado: vectores persistidos en BD, añadidos a FAISS, búsqueda devuelve resultados |

El `conftest.py` define `MockEmbedder` (dim=64, normalizado, sin GPU),
`db_path` (BD SQLite en `tmp_path` con 3 docs y 9 chunks) y `faiss_dir`.
Los tests de FAISS usan el `FaissIndexManager` real para verificar la
serialización binaria y el mapa de IDs.

> Los tests usan `dim=64` y corpus pequeños. Para IndexIVFPQ se insertan
> 300 documentos sintéticos extra en `test_rebuild_produces_ivfpq` para
> superar el umbral `min_train_size(nlist=4, nbits=8) = max(156, 256) = 256`.