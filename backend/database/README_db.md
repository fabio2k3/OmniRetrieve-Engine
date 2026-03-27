---
noteId: "7c435820296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Base de Datos

Esquema SQLite unificado y repositorios de acceso a datos para todos los módulos del sistema. Es la única capa que toca la base de datos directamente — el resto de módulos accede siempre a través de las funciones de este módulo.

---

## Estructura de archivos

```
backend/database/
├── schema.py              ← DDL completo + get_connection() + init_db()
├── crawler_repository.py  ← operaciones de lectura/escritura del crawler
├── index_repository.py    ← operaciones de lectura/escritura del indexador
└── __init__.py            ← exports públicos de todo el módulo
```

---

## Tablas

### Módulo crawler

| Tabla | Descripción |
|---|---|
| `documents` | Metadatos de cada artículo + texto extraído del PDF |
| `chunks` | Fragmentos de texto a nivel de párrafo (reservado para embeddings) |
| `crawl_log` | Registro de cada ejecución del crawler |

### Módulo indexing

| Tabla | Descripción |
|---|---|
| `terms` | Vocabulario: una fila por token único con su `df` |
| `postings` | Índice invertido: frecuencia cruda (`freq`) por par (término, documento) |
| `index_meta` | Almacén clave/valor para auditoría del indexador |

### Módulo retrieval

| Tabla | Descripción |
|---|---|
| `lsi_log` | Registro de cada construcción del modelo LSI |

---

## Schema

### `documents`

```sql
CREATE TABLE documents (
    arxiv_id         TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    authors          TEXT,
    abstract         TEXT,
    categories       TEXT,
    published        TEXT,
    updated          TEXT,
    pdf_url          TEXT,
    fetched_at       TEXT,          -- timestamp de descarga de metadatos
    full_text        TEXT,          -- texto completo extraído del PDF/HTML
    text_length      INTEGER,
    pdf_downloaded   INTEGER NOT NULL DEFAULT 0,
    --   0 = pendiente de descarga
    --   1 = descargado con éxito
    --   2 = error en descarga/extracción
    indexed_at       TEXT,          -- timestamp de extracción del PDF (crawler)
    index_error      TEXT,          -- mensaje de error si pdf_downloaded = 2
    indexed_tfidf_at TEXT           -- timestamp de indexación TF; NULL = pendiente
);
```

### `postings`

```sql
CREATE TABLE postings (
    term_id  INTEGER NOT NULL REFERENCES terms(term_id) ON DELETE CASCADE,
    doc_id   TEXT    NOT NULL REFERENCES documents(arxiv_id) ON DELETE CASCADE,
    freq     INTEGER NOT NULL DEFAULT 0,   -- frecuencia cruda del término en el doc
    PRIMARY KEY (term_id, doc_id)
);
```

`postings` guarda solo frecuencias crudas (`freq`). Los pesos TF-IDF se calculan en el módulo `retrieval` con la fórmula que considere apropiada.

---

## Repositorios

### `crawler_repository.py`

Operaciones del módulo de adquisición.

| Función | Descripción |
|---|---|
| `upsert_document(...)` | Inserta o actualiza metadatos de un artículo |
| `save_pdf_text(arxiv_id, full_text)` | Persiste el texto extraído y marca `pdf_downloaded=1` |
| `save_pdf_error(arxiv_id, error)` | Registra un fallo y marca `pdf_downloaded=2` |
| `get_pending_pdf_ids(limit)` | IDs de documentos con `pdf_downloaded=0` |
| `document_exists(arxiv_id)` | Comprueba si un documento está en la BD |
| `get_document(arxiv_id)` | Devuelve la fila completa de un documento |
| `save_chunks(arxiv_id, texts)` | Reemplaza los chunks de un documento |
| `log_crawl_start()` | Abre una entrada en `crawl_log`, devuelve su id |
| `log_crawl_end(log_id, ...)` | Cierra la entrada con estadísticas |
| `get_stats()` | Resumen del estado de la BD |

### `index_repository.py`

Operaciones del módulo de indexación y del módulo de recuperación.

| Función | Descripción |
|---|---|
| `clear_index()` | Borra `terms` y `postings` (para reindex completo) |
| `upsert_terms(df_map)` | Inserta términos nuevos y acumula `df` de existentes |
| `flush_postings(batch)` | Inserta o actualiza un lote de `(term_id, doc_id, freq)` |
| `mark_documents_indexed(arxiv_ids)` | Pone `indexed_tfidf_at = now` en los documentos indexados |
| `save_index_meta(stats)` | Persiste metadatos de auditoría en `index_meta` |
| `get_unindexed_documents(field, ...)` | Generador de docs con PDF descargado e `indexed_tfidf_at IS NULL` |
| `get_index_stats()` | vocab_size, total_docs, total_postings, meta |
| `get_top_terms(arxiv_id, n)` | Top N términos por frecuencia de un documento |
| `get_postings_for_term(word)` | Posting list de un término: `[{doc_id, freq}]` |
| `get_postings_for_matrix(max_docs)` | Datos crudos para construir la matriz TF-IDF en retrieval |
| `get_document_metadata(arxiv_ids)` | Metadatos de documentos para mostrar en resultados |

---

## Uso

### Inicializar la BD

```python
from backend.database.schema import init_db

init_db()          # crea todas las tablas si no existen (idempotente)
init_db(db_path)   # ruta personalizada
```

### Desde el crawler

```python
from backend.database.crawler_repository import upsert_document, save_pdf_text

upsert_document(arxiv_id="2301.001", title="...", ...)
save_pdf_text("2301.001", full_text="...")
```

### Desde el indexador

```python
from backend.database.index_repository import (
    get_unindexed_documents, upsert_terms,
    flush_postings, mark_documents_indexed,
)

for arxiv_id, texto in get_unindexed_documents(field="full_text"):
    ...  # tokenizar y contar

all_terms = upsert_terms(df_map)
flush_postings(batch)
mark_documents_indexed(list(doc_ids))
```

### Desde retrieval

```python
from backend.database.index_repository import get_postings_for_matrix

postings, df_map, doc_ids, term_ids, n_docs = get_postings_for_matrix()
# → construir matriz TF-IDF y aplicar SVD
```

---

## Diseño

- **Stateless** — cada función abre y cierra su propia conexión. No hay objetos de conexión persistentes.
- **Idempotente** — todos los inserts usan `INSERT OR IGNORE` o `ON CONFLICT DO UPDATE`. Es seguro llamarlos varias veces con los mismos datos.
- **Sin pesos calculados** — `postings` guarda solo `freq` (entero). Ninguna función de este módulo calcula TF-IDF; esa responsabilidad pertenece a `retrieval/`.
- **WAL mode** — todas las conexiones activan `PRAGMA journal_mode = WAL` para permitir lectores concurrentes mientras el crawler escribe.
