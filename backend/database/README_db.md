# OmniRetrieve — Módulo `database`

Capa de acceso a datos. Una única base de datos **SQLite** con modo WAL
comparte estado entre todos los módulos del sistema. El módulo expone un
repositorio por cada dominio funcional; ningún otro módulo ejecuta SQL
directamente.

---

## Estructura de archivos

```
backend/database/
├── schema.py              ← DDL unificado, init_db(), get_connection()
├── crawler_repository.py  ← documentos: metadatos, texto PDF, estado, crawl_log
├── chunk_repository.py    ← chunks: texto, embeddings (BLOB), iteradores
├── index_repository.py    ← índice TF: terms, postings, index_meta, lsi_log
├── embedding_repository.py← FAISS: faiss_log, embedding_meta (schema adicional)
├── web_repository.py      ← búsqueda web: web_search_results, web_search_log
└── __init__.py            ← exports públicos
```

---

## Configuración SQLite

```python
PRAGMA journal_mode = WAL;   # escrituras concurrentes sin bloquear lecturas
PRAGMA foreign_keys = ON;    # integridad referencial (CASCADE en chunks, postings)
```

Todas las conexiones usan `row_factory = sqlite3.Row`: las filas se acceden
tanto por índice (`row[0]`) como por nombre de columna (`row["title"]`).

Ruta por defecto: `backend/data/db/documents.db`.

---

## Esquema de tablas

### Módulo `crawler`

#### `documents`

| Columna | Tipo | Descripción |
|---|---|---|
| `arxiv_id` | TEXT PK | ID compuesto: `"arxiv:2301.12345"` |
| `title`, `authors`, `abstract`, `categories` | TEXT | Metadatos del artículo |
| `published`, `updated`, `pdf_url`, `fetched_at` | TEXT | Fechas ISO-8601 y URL |
| `full_text` | TEXT | Texto completo extraído (NULL hasta la descarga) |
| `text_length` | INTEGER | Caracteres de `full_text` |
| `pdf_downloaded` | INTEGER | **0** = pendiente · **1** = descargado · **2** = error |
| `indexed_at` | TEXT | Timestamp de extracción del PDF |
| `index_error` | TEXT | Mensaje de error si `pdf_downloaded = 2` |
| `indexed_tfidf_at` | TEXT | Timestamp de indexación TF (NULL = pendiente para indexar) |

Índices: `categories`, `published`, `pdf_downloaded`.

#### `chunks`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | Referenciado por el índice FAISS |
| `arxiv_id` | TEXT FK → documents | Cascade delete |
| `chunk_index` | INTEGER | Posición secuencial en el documento (0-based) |
| `text` | TEXT | Texto del chunk |
| `char_count` | INTEGER | Longitud del texto |
| `embedding` | BLOB | `ndarray.astype(float32).tobytes()` (NULL hasta el embedding) |
| `embedded_at` | TEXT | Timestamp del embedding |
| `created_at` | TEXT | Timestamp de creación |

Constraint `UNIQUE(arxiv_id, chunk_index)`. Índices: `arxiv_id`, `embedded_at`.

#### `crawl_log`

Una fila por ejecución del crawler con `started_at`, `finished_at`,
`ids_discovered`, `docs_downloaded`, `pdfs_indexed`, `errors`, `notes`.

---

### Módulo `indexing` (índice TF)

#### `terms`

| Columna | Tipo | Descripción |
|---|---|---|
| `term_id` | INTEGER PK AUTOINCREMENT | ID interno del término |
| `word` | TEXT UNIQUE | Token normalizado |
| `df` | INTEGER | Nº de documentos que contienen el término |

Índice en `word`.

#### `postings`

| Columna | Tipo | Descripción |
|---|---|---|
| `term_id` | INTEGER FK → terms (CASCADE) | |
| `doc_id` | TEXT FK → documents (CASCADE) | |
| `freq` | INTEGER | Frecuencia cruda del término en el documento |

PK compuesta `(term_id, doc_id)`. Índices: `doc_id`, `term_id`.

#### `index_meta`

Almacén clave/valor (`key` PK, `value`) para auditoría de la indexación:
número de documentos, términos, fecha del último run, etc.

---

### Módulo `retrieval` (LSI)

#### `lsi_log`

| Columna | Tipo | Descripción |
|---|---|---|
| `built_at` | TEXT | Timestamp de construcción del modelo |
| `k` | INTEGER | Componentes latentes |
| `n_docs` | INTEGER | Documentos en el modelo |
| `n_terms` | INTEGER | Términos en el vocabulario |
| `var_explained` | REAL | Varianza explicada por el SVD |
| `model_path` | TEXT | Ruta al `.pkl` serializado |
| `notes` | TEXT | Notas opcionales |

---

### Módulo `embedding` (FAISS)

Estas tablas se crean con `init_embedding_schema()`, **no** con `init_db()`.

#### `faiss_log`

| Columna | Tipo | Descripción |
|---|---|---|
| `built_at` | TEXT | Timestamp de construcción del índice |
| `n_vectors` | INTEGER | Vectores en el índice |
| `index_type` | TEXT | `"IndexFlatL2"` o `"IndexIVFPQ"` |
| `model_name` | TEXT | Nombre del modelo sentence-transformers |
| `nlist`, `m`, `nbits` | INTEGER | Parámetros IVFPQ |
| `index_path`, `id_map_path` | TEXT | Rutas a los ficheros `.faiss` y `.npy` |

#### `embedding_meta`

Almacén clave/valor (`key` PK, `value`) para metadatos del pipeline de
embedding: nombre del modelo, timestamp del último run, etc.

---

### Módulo `web_search`

#### `web_search_results`

| Columna | Tipo | Descripción |
|---|---|---|
| `url` | TEXT UNIQUE | URL de la página (constraint de deduplicación) |
| `searched_at`, `query` | TEXT | Cuándo y con qué query se encontró |
| `title`, `content` | TEXT | Título y texto de la página |
| `score` | REAL | Score asignado por Tavily/SiteSearcher |
| `source` | TEXT | `"web"` (Tavily) o `"web_search"` (SiteSearcher) |

#### `web_search_log`

Una fila por llamada a `WebSearchPipeline.run()` con `query`,
`results_found` y `results_saved`.

---

## `schema.py` — API de inicialización

```python
from backend.database.schema import init_db, get_connection, DB_PATH

init_db()                          # crea tablas e índices si no existen (idempotente)
init_db(Path("/custom/path.db"))   # BD alternativa

conn = get_connection()            # sqlite3.Connection con row_factory=sqlite3.Row
```

`init_db()` ejecuta el DDL completo en una sola llamada a `executescript()`.
Es idempotente: todas las sentencias usan `CREATE TABLE IF NOT EXISTS`.

---

## `crawler_repository.py`

Consumido por: **Crawler** (`DownloaderLoop`, `TextLoop`), **orchestrator**.

| Función | Descripción |
|---|---|
| `upsert_document(arxiv_id, title, …)` | INSERT OR UPDATE de metadatos; no toca `full_text` ni estado PDF |
| `save_pdf_text(arxiv_id, full_text)` | Guarda texto y pone `pdf_downloaded=1`, `indexed_at=now` |
| `save_pdf_error(arxiv_id, error)` | Registra fallo: `pdf_downloaded=2`, `index_error=msg` |
| `get_pending_pdf_ids(limit)` | IDs con `pdf_downloaded=0`, ordenados por fecha desc |
| `get_document(arxiv_id)` | `sqlite3.Row` completa o `None` |
| `document_exists(arxiv_id)` | `bool` rápido sin leer todos los campos |
| `log_crawl_start()` | Inserta en `crawl_log`, devuelve el `id` |
| `log_crawl_end(log_id, …)` | Actualiza la fila con contadores y `finished_at` |
| `get_stats()` | `{total_documents, pdf_indexed, pdf_pending, pdf_errors, total_chunks, …}` |
| `get_document_counts()` | `{total, indexed, pending}` — versión ligera para el orchestrator |
| `get_unindexed_pdf_count()` | Docs con `pdf_downloaded=1` e `indexed_tfidf_at IS NULL` |

---

## `chunk_repository.py`

Consumido por: **Crawler** (save), **Embedding** (lectura/escritura de embeddings),
**QRF** (get_embeddings), **LSIRetriever** (get_chunks_with_metadata).

| Función | Descripción |
|---|---|
| `save_chunks(arxiv_id, texts)` | DELETE previos + INSERT con `chunk_index` secuencial |
| `save_chunk_embedding(chunk_id, embedding_bytes)` | Actualiza un chunk individual |
| `save_chunk_embeddings_batch([(bytes, ts, id), …])` | Batch UPDATE; devuelve nº de filas |
| `reset_embeddings()` | Pone `embedding=NULL` en todos los chunks; devuelve nº de filas |
| `get_chunks(arxiv_id)` | Todos los chunks de un documento, ordenados por `chunk_index` |
| `get_unembedded_chunks(limit)` | Chunks con `embedded_at IS NULL` |
| `get_unembedded_chunks_iter(batch_size)` | Generador por lotes (para pipelines de embedding) |
| `get_all_embeddings_iter(batch_size)` | Generador: solo chunks con `embedding IS NOT NULL` |
| `get_chunks_by_ids(ids)` | Chunks por lista de `id`; chunked queries (límite 900 vars) |
| `get_chunk_count()` | Total de chunks en la tabla |
| `get_embedded_count()` | Chunks con `embedded_at IS NOT NULL` |
| `get_chunk_stats()` | `{total_chunks, embedded_chunks, pending_chunks}` |
| `get_chunks_with_metadata_by_arxiv_ids(ids)` | Chunks + título del documento (para LSIRetriever) |
| `get_chunk_embeddings_by_ids(ids)` | `{chunk_id: ndarray}` (para BRF y MMR del módulo qrf) |

---

## `index_repository.py`

Consumido por: **Indexing** (escritura), **LSI model** (lectura de matriz y términos).

| Función | Descripción |
|---|---|
| `clear_index()` | Elimina todos los registros de `terms` y `postings` |
| `upsert_terms(df_map)` | INSERT de términos nuevos + acumulación de `df` en un batch |
| `flush_postings(batch)` | INSERT OR IGNORE de `(term_id, doc_id, freq)` en batch |
| `mark_documents_indexed(doc_ids)` | Escribe `indexed_tfidf_at=now` en los documentos |
| `get_unindexed_documents(field, batch_size)` | Docs con `pdf_downloaded=1` e `indexed_tfidf_at IS NULL` |
| `get_index_stats()` | `{vocab_size, total_docs, total_postings, last_indexed_at}` |
| `get_top_terms(doc_id, n)` | Top-N términos por frecuencia para un documento |
| `get_postings_for_term(word)` | Documentos que contienen un término |
| `get_postings_for_matrix()` | Devuelve `(postings, df_map, doc_ids, term_ids, n_docs)` — datos crudos para LSI |
| `get_term_words_by_ids(term_ids)` | `[(term_id, word)]`; chunked en lotes de 900 |
| `get_document_metadata(arxiv_ids)` | `{arxiv_id: {title, authors, abstract, pdf_url}}` |
| `get_indexed_doc_count()` | Docs distintos con postings |
| `save_index_meta(stats)` | Persiste metadatos del run en `index_meta` |
| `log_lsi_build(…)` | Inserta en `lsi_log` con `k`, `n_docs`, `var_explained`, etc. |

`get_postings_for_matrix()` devuelve frecuencias crudas (`freq`, `df`).
La fórmula TF-IDF la aplica el módulo `retrieval`, no este repositorio.

---

## `embedding_repository.py`

Consumido por: **EmbeddingPipeline**.

Requiere llamar a `init_embedding_schema(db_path)` antes del primer uso:
crea las tablas `faiss_log` y `embedding_meta` si no existen.

| Función | Descripción |
|---|---|
| `init_embedding_schema()` | Crea `faiss_log` y `embedding_meta` (idempotente) |
| `log_faiss_build(stats)` | Inserta fila en `faiss_log` con parámetros del índice |
| `save_embedding_meta(key, value)` | UPSERT en `embedding_meta` |
| `get_embedding_meta(key)` | Recupera valor por clave, `None` si no existe |
| `get_embedding_stats()` | `{total_chunks, embedded_chunks, pending_chunks, last_build_at, last_index_type, last_n_vectors}` |

---

## `web_repository.py`

Consumido por: **WebSearchPipeline**.

| Función | Descripción |
|---|---|
| `save_web_results(query, results)` | INSERT OR IGNORE en `web_search_results` (dedup por URL); registra en `web_search_log` |
| `get_cached_result(url)` | Devuelve el contenido cacheado de una URL o `None` |
| `get_web_results(limit)` | Últimas N filas de `web_search_results` (para monitorización) |

---

## Convenciones de implementación

**Sin ORM.** Todos los repositorios usan SQL directo vía `sqlite3`.

**Patrón de conexión.** Cada función abre y cierra su propia conexión:

```python
conn = get_connection(db_path)
try:
    result = conn.execute(sql, params).fetchone()
    conn.commit()
    return result
finally:
    conn.close()
```

**Chunked queries.** Las consultas con `IN (?, ?, …)` se dividen en lotes
de 900 para no superar el límite de 999 variables de SQLite.

**`pdf_downloaded` como máquina de estados.**

```
0 (pendiente) ──→ 1 (descargado OK)
0 (pendiente) ──→ 2 (error de descarga o extracción)
2 (error)     ──→ 0 (reset para reintentar, vía retry_downloads.py)
```

---

## Tests

Los tests de la capa de BD están integrados en los tests de los módulos
que consumen cada repositorio.

```bash
# crawler_repository — via test de integración del crawler
pytest backend/tests/crawler/test_backward_compat.py -v
pytest backend/tests/crawler/test_routing.py         -v

# chunk_repository — test dedicado en embedding
pytest backend/tests/embedding/test_chunk_repository.py     -v

# embedding_repository — test dedicado en embedding
pytest backend/tests/embedding/test_embedding_repository.py -v

# index_repository — test dedicado en indexing
pytest backend/tests/indexing/test_index_repository.py      -v

# web_repository — via test_web_search
pytest backend/tests/test_web_search/test_pipeline.py       -v

# pipeline completo de extremo a extremo
pytest backend/tests/integration/test_full_pipeline.py      -v
```

Todos los tests crean su propia BD en `tmp_path` y llaman a `init_db()`
al inicio. Ningún test depende de datos persistentes en disco.