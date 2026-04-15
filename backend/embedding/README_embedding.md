# OmniRetrieve — Módulo de Embedding

Vectoriza los chunks del corpus arXiv usando sentence-transformers y construye un índice FAISS para búsqueda semántica densa. Complementa el índice invertido TF del módulo `indexing` con recuperación por similitud vectorial.

---

## Estructura de archivos

```
backend/embedding/
├── embedder.py      ← wrapper de SentenceTransformer (ChunkEmbedder)
├── faiss_index.py   ← ciclo de vida del índice FAISS (FaissIndexManager)
├── pipeline.py      ← orquestador + entrypoint CLI (EmbeddingPipeline)
└── __init__.py      ← exports públicos

backend/database/
├── embedding_repository.py  ← tablas faiss_log y embedding_meta
└── chunk_repository.py      ← operaciones sobre la tabla chunks

backend/tools/
├── embed_chunks.py    ← tool para embedizar todos los chunks pendientes
└── rebuild_chunks.py  ← tool para reconstruir chunks con el nuevo algoritmo
```

---

## Instalación

```bash
pip install sentence-transformers faiss-cpu
# Si hay GPU disponible:
pip install sentence-transformers faiss-gpu
```

| Paquete | Para qué |
|---|---|
| `sentence-transformers` | Modelos de embedding de texto |
| `faiss-cpu` / `faiss-gpu` | Índice de similitud vectorial |

---

## Cómo ejecutar

```bash
# Embedizar todos los chunks pendientes
python -m backend.tools.embed_chunks

# Ver estado actual (chunks embedidos vs pendientes)
python -m backend.tools.embed_chunks --stats

# Usar un modelo más potente (768 dims — ajustar --m)
python -m backend.tools.embed_chunks --model all-mpnet-base-v2 --m 16

# Re-vectorizar todo el corpus (tras cambiar de modelo)
python -m backend.tools.embed_chunks --reembed

# Con GPU y lotes grandes
python -m backend.tools.embed_chunks --device cuda --batch-size 512

# Reconstruir chunks con el nuevo algoritmo antes de embedizar
python -m backend.tools.rebuild_chunks
python -m backend.tools.embed_chunks
```

### Parámetros disponibles

```bash
python -m backend.tools.embed_chunks \
  --db             ruta/a/documents.db   \
  --model          all-MiniLM-L6-v2      \
  --device         cpu                   \
  --batch-size     256                   \
  --rebuild-every  10000                 \
  --nlist          100                   \
  --m              8                     \
  --nbits          8                     \
  --nprobe         10                    \
  --reembed                              \
  --stats
```

---

## Arquitectura

```
chunks (BD, embedding IS NULL)
        |
  ChunkEmbedder.encode(texts)
  -----------------------------
  SentenceTransformer -> float32
  L2-normalizacion
        |
  +---------------------+------------------------------+
  |                     |                              |
  v                     v                              |
BD: chunks        FaissIndexManager.add()              |
embedding=BLOB    ----------------------               |
embedded_at=now   acumula vectores                     |
                  contador += N                        |
                          |                            |
                    cada 10.000 chunks                 |
                    maybe_rebuild()                    |
                          |                            |
                  FaissIndexManager.rebuild()           |
                  --------------------------           |
                  lee TODOS los embeddings <-----------+
                  de la BD
                  entrena IndexIVFPQ (si n >= min)
                  o IndexFlatL2 (fallback)
                  guarda index.faiss + id_map.npy
                  registra en faiss_log (BD)
```

### Sincronización al arrancar

Al iniciar el pipeline, compara el número de vectores en el índice FAISS en disco con los embeddings registrados en la BD. Si hay desincronización (p.ej. por una sesión interrumpida), reconstruye el índice completo antes de procesar nuevos chunks.

```
INFO     [Pipeline] Sincronizacion — embeddings en BD: 45000 | vectores en FAISS: 45000
# vs.
WARNING  [Pipeline] Desincronizacion detectada: faltan 34000 vectores en el indice FAISS.
INFO     [Pipeline] Reconstruyendo indice FAISS desde la BD antes de continuar...
```

---

## Componentes

### `ChunkEmbedder` (`embedder.py`)

Wrapper ligero sobre `SentenceTransformer`. No accede a la BD ni al índice.

```python
from backend.embedding.embedder import ChunkEmbedder

embedder = ChunkEmbedder(model_name="all-MiniLM-L6-v2")
vectors  = embedder.encode(["texto A", "texto B"])    # -> (2, 384) float32
query    = embedder.encode_single("attention mechanism")  # -> (384,)
```

### `FaissIndexManager` (`faiss_index.py`)

Gestiona el ciclo de vida completo del índice FAISS: construcción, actualización incremental, reconstrucción periódica, persistencia y búsqueda.

```python
from backend.embedding.faiss_index import FaissIndexManager

mgr = FaissIndexManager(
    dim=384, nlist=100, m=8, nbits=8,
    rebuild_every=10_000,
    index_path=Path("data/faiss/index.faiss"),
    id_map_path=Path("data/faiss/id_map.npy"),
)
mgr.load()                           # carga desde disco si existe
mgr.add(vectors, chunk_ids)          # añade vectores al indice
mgr.maybe_rebuild(db_path)           # reconstruye si se alcanzo el umbral
results = mgr.search(query, top_k=10)
# -> [{"chunk_id": 1234, "score": 0.12}, ...]
```

### `EmbeddingPipeline` (`pipeline.py`)

Orquesta el flujo completo: leer chunks pendientes → vectorizar → guardar en BD → añadir a FAISS → rebuild periódico.

```python
from backend.embedding.pipeline import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    model_name="all-MiniLM-L6-v2",
    batch_size=256,
    rebuild_every=10_000,
)
stats = pipeline.run()
# -> {"chunks_processed": 4821, "rebuilds_triggered": 0, ...}
```

---

## Índice FAISS

### Tipo de índice

| Condición | Índice usado |
|---|---|
| `n_vectors >= max(nlist x 39, 2^nbits)` | `IndexIVFPQ` (comprimido, rápido) |
| Por debajo del umbral | `IndexFlatL2` (exacto, sin compresión) |

El índice pasa automáticamente de `FlatL2` a `IVFPQ` en el primer rebuild donde haya suficientes vectores.

### Parámetros IVFPQ

| Parámetro | Default | Descripción |
|---|---|---|
| `nlist` | 100 | Celdas Voronoi. Recomendado: `sqrt(N)` para el corpus final |
| `m` | 8 | Subvectores PQ. Debe dividir exactamente a la dimensión |
| `nbits` | 8 | Bits por código (256 centroides/subvector) |
| `nprobe` | 10 | Celdas inspeccionadas en búsqueda. Mayor = más recall, más lento |

### Compatibilidad de dimensiones y `m`

| Modelo | Dimensión | `m` válidos |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 8, 16, 32, 48 |
| `all-mpnet-base-v2` | 768 | 16, 32, 48 |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | 8, 16, 32, 48 |
| `allenai-specter` | 768 | 16, 32, 48 |

### Política de rebuild

```
add(vec, id)                  <- por cada chunk en el lote
     |
     v
maybe_rebuild()
     +-- added_since_last < rebuild_every  -> no hace nada
     +-- added_since_last >= rebuild_every
               |
               v
           rebuild()
               +-- lee TODOS los embeddings de la BD
               +-- entrena y construye el indice desde cero
               +-- guarda index.faiss + id_map.npy
               +-- registra en faiss_log
```

### Persistencia en disco

```
backend/data/faiss/
+-- index.faiss   <- indice FAISS serializado
+-- id_map.npy    <- array int64: posicion_faiss -> chunk_id (BD)
```

---

## Tablas de BD

Las tablas propias del módulo las crea `embedding_repository.py` en `backend/database/`.

### `faiss_log`

Historial de cada reconstrucción del índice FAISS.

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK | |
| `built_at` | TEXT | Timestamp ISO de la construcción |
| `n_vectors` | INTEGER | Vectores en el índice |
| `index_type` | TEXT | `IndexIVFPQ` o `IndexFlatL2` |
| `model_name` | TEXT | Modelo usado |
| `nlist` | INTEGER | Celdas Voronoi (solo IVFPQ) |
| `m` | INTEGER | Subvectores PQ (solo IVFPQ) |
| `nbits` | INTEGER | Bits por código (solo IVFPQ) |
| `index_path` | TEXT | Ruta del `.faiss` |
| `id_map_path` | TEXT | Ruta del `.npy` |

### `embedding_meta`

Metadatos clave/valor del módulo.

| Clave | Valor |
|---|---|
| `model_name` | Nombre del modelo usado en la última ejecución |
| `last_run_at` | Timestamp de la última ejecución del pipeline |
| `last_chunks_embedded` | Chunks procesados en la última ejecución |

Los embeddings en sí se almacenan en la tabla `chunks` (columna `embedding BLOB`, `embedded_at TEXT`), gestionada por `chunk_repository.py`.

---

## Tools

### `embed_chunks.py`

Embediza todos los chunks pendientes de la BD. Punto de entrada directo al `EmbeddingPipeline`.

```bash
python -m backend.tools.embed_chunks --stats    # estado actual
python -m backend.tools.embed_chunks            # embedizar pendientes
python -m backend.tools.embed_chunks --reembed  # re-vectorizar todo
```

### `rebuild_chunks.py`

Elimina todos los chunks de la BD y los reconstruye documento a documento con el algoritmo de chunking actualizado. **Invalida todos los embeddings existentes** — hay que re-embedizar después.

```bash
python -m backend.tools.rebuild_chunks --dry-run         # simulacion
python -m backend.tools.rebuild_chunks --chunk-size 800 --overlap 3
python -m backend.tools.rebuild_chunks --yes             # sin confirmacion
```

---

## Tests

```bash
python -m backend.tests.test_embedding
python -m pytest backend/tests/test_embedding.py -v
```

Los tests usan un embedder mock que genera vectores aleatorios normalizados, por lo que **no requieren `sentence-transformers` instalado** para ejecutarse.

---

## Notas de diseño

- **Stateless** — `ChunkEmbedder` y `FaissIndexManager` no comparten estado entre instancias. El `EmbeddingPipeline` los instancia y coordina.
- **Tolerante a interrupciones** — si el proceso se cae a mitad, los embeddings ya guardados en la BD persisten. Al reiniciar, el pipeline detecta la desincronización con FAISS y reconstruye el índice antes de continuar.
- **Separación de responsabilidades** — `ChunkEmbedder` solo vectoriza. `FaissIndexManager` solo gestiona el índice. Las queries SQL están todas en `chunk_repository` y `embedding_repository`.
- **Modelo intercambiable** — cambiar de modelo requiere `--reembed` para re-vectorizar con el nuevo espacio de representación. El índice FAISS se reconstruye automáticamente.