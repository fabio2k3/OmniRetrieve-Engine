---
noteId: "4107f9f0221711f18224b359e9cd45df"
tags: []

---

# OmniRetrieve — Estructura de la Base de Datos

La base de datos SQLite en `backend/data/db/documents.db` es el punto de integración entre todos los módulos del sistema. El módulo de adquisición la escribe; los módulos de indexación y recuperación la leen y extienden.

---

## Ubicación y acceso

```
backend/data/db/documents.db
```

Se crea automáticamente al arrancar el crawler. Usa **WAL mode** (Write-Ahead Logging), lo que permite que múltiples módulos lean simultáneamente mientras el crawler escribe, sin bloqueos.

### Conectarse desde cualquier módulo

```python
from backend.database.schema import get_connection, DB_PATH

conn = get_connection()   # abre conexión con row_factory = sqlite3.Row
row  = conn.execute("SELECT * FROM documents WHERE arxiv_id = ?", ("2301.12345",)).fetchone()
print(row["title"])       # acceso por nombre de columna
conn.close()
```

### Usar el repository (recomendado)

```python
from backend.database import repository as repo

doc    = repo.get_document("2301.12345")
stats  = repo.get_stats()
chunks = repo.get_chunks("2301.12345")
```

---

## Tablas actuales

El módulo de adquisición crea y mantiene **3 tablas**. El módulo de indexación puede añadir las suyas propias extendiendo `schema.py`.

---

### Tabla `documents`

Cada fila representa un artículo de arXiv con sus metadatos y el texto extraído.

```sql
CREATE TABLE documents (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT,
    abstract        TEXT,
    categories      TEXT,
    published       TEXT,
    updated         TEXT,
    pdf_url         TEXT,
    fetched_at      TEXT,
    full_text       TEXT,
    text_length     INTEGER,
    pdf_downloaded  INTEGER NOT NULL DEFAULT 0,
    indexed_at      TEXT,
    index_error     TEXT
);
```

| Columna | Tipo | Descripción |
|---|---|---|
| `arxiv_id` | TEXT PK | ID del paper, ej. `2301.12345` |
| `title` | TEXT | Título completo |
| `authors` | TEXT | Autores separados por coma |
| `abstract` | TEXT | Resumen. **Siempre disponible** desde el primer ciclo |
| `categories` | TEXT | Categorías arXiv, ej. `cs.AI, cs.LG, cs.CV` |
| `published` | TEXT | Fecha de publicación ISO-8601, ej. `2023-01-30T00:00:00Z` |
| `updated` | TEXT | Fecha de última actualización ISO-8601 |
| `pdf_url` | TEXT | URL del PDF, ej. `https://arxiv.org/pdf/2301.12345v2` |
| `fetched_at` | TEXT | Timestamp de cuando se descargaron los metadatos |
| `full_text` | TEXT | Texto completo extraído y limpio (HTML o PDF) |
| `text_length` | INTEGER | Longitud en caracteres de `full_text` |
| `pdf_downloaded` | INTEGER | Estado de extracción de texto (ver abajo) |
| `indexed_at` | TEXT | Timestamp de la última extracción de texto |
| `index_error` | TEXT | Mensaje de error si `pdf_downloaded = 2` |

#### Estado `pdf_downloaded`

| Valor | Significado | `full_text` |
|---|---|---|
| `0` | Pendiente — metadatos descargados, texto no extraído aún | `NULL` |
| `1` | ✅ Texto extraído y guardado correctamente | Disponible |
| `2` | ❌ Error en la extracción (PDF muy grande, 404, timeout, etc.) | `NULL` |

#### Índices disponibles

```sql
idx_doc_categories   ON documents(categories)
idx_doc_published    ON documents(published)
idx_doc_pdf_status   ON documents(pdf_downloaded)
```

---

### Tabla `chunks`

El `full_text` de cada documento dividido en fragmentos de ~1000 caracteres preservando párrafos. Son las unidades de búsqueda para recuperación densa (embeddings).

```sql
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id    TEXT    NOT NULL REFERENCES documents(arxiv_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    char_count  INTEGER,
    embedding   BLOB,
    embedded_at TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(arxiv_id, chunk_index)
);
```

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK | Autoincremental |
| `arxiv_id` | TEXT FK | Referencia al documento padre |
| `chunk_index` | INTEGER | Posición del chunk en el doc (0-based) |
| `text` | TEXT | Texto del fragmento (~1000 chars) |
| `char_count` | INTEGER | Longitud del fragmento |
| `embedding` | BLOB | Vector serializado como `float32` bytes — `NULL` hasta fase 2 |
| `embedded_at` | TEXT | Timestamp de cuando se generó el embedding |
| `created_at` | TEXT | Timestamp de creación del chunk |

#### Trabajar con embeddings (fase 2)

```python
import numpy as np

# Guardar
embedding_bytes = np.array([0.1, 0.2, ...], dtype="float32").tobytes()
repo.save_chunk_embedding(chunk_id=42, embedding=embedding_bytes)

# Leer
row = conn.execute("SELECT embedding FROM chunks WHERE id = 42").fetchone()
vector = np.frombuffer(row["embedding"], dtype="float32")
```

#### Índices disponibles

```sql
idx_chunks_arxiv    ON chunks(arxiv_id)
idx_chunks_embedded ON chunks(embedded_at)
```

---

### Tabla `crawl_log`

Una fila por sesión del crawler, útil para monitorización y auditoría.

```sql
CREATE TABLE crawl_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    ids_discovered  INTEGER DEFAULT 0,
    docs_downloaded INTEGER DEFAULT 0,
    pdfs_indexed    INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    notes           TEXT
);
```

| Columna | Descripción |
|---|---|
| `started_at` | Timestamp de inicio de sesión |
| `finished_at` | `NULL` si el crawler sigue corriendo |
| `ids_discovered` | IDs nuevos encontrados |
| `docs_downloaded` | Metadatos descargados |
| `pdfs_indexed` | Textos extraídos correctamente |
| `errors` | Número de errores en la sesión |

---

## Queries útiles

### Para el módulo de indexación

```sql
-- Documentos con texto listo para indexar
SELECT arxiv_id, title, full_text, text_length, categories, published
FROM documents
WHERE pdf_downloaded = 1
ORDER BY published DESC;

-- Solo abstracts (disponibles para TODOS, incluso sin full_text)
SELECT arxiv_id, title, abstract, categories, published
FROM documents
ORDER BY published DESC;

-- Longitud promedio del corpus (necesario para BM25)
SELECT
    COUNT(*)         AS total_docs,
    AVG(text_length) AS avg_chars,
    MIN(text_length) AS min_chars,
    MAX(text_length) AS max_chars
FROM documents
WHERE pdf_downloaded = 1;

-- Distribución por categoría
SELECT categories, COUNT(*) AS n
FROM documents
WHERE pdf_downloaded = 1
GROUP BY categories
ORDER BY n DESC;
```

### Para el módulo de embeddings

```sql
-- Chunks pendientes de embeddizar
SELECT id, arxiv_id, chunk_index, text
FROM chunks
WHERE embedding IS NULL
ORDER BY arxiv_id, chunk_index;

-- Todos los chunks de un documento
SELECT chunk_index, char_count, text
FROM chunks
WHERE arxiv_id = '2301.12345'
ORDER BY chunk_index;

-- Progreso de embeddings
SELECT
    COUNT(*)                          AS total_chunks,
    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded,
    SUM(CASE WHEN embedding IS NULL     THEN 1 ELSE 0 END) AS pending
FROM chunks;
```

### Monitorización

```sql
-- Estado general
SELECT
    COUNT(*)                                            AS total,
    SUM(CASE WHEN pdf_downloaded = 1 THEN 1 ELSE 0 END) AS con_texto,
    SUM(CASE WHEN pdf_downloaded = 0 THEN 1 ELSE 0 END) AS pendientes,
    SUM(CASE WHEN pdf_downloaded = 2 THEN 1 ELSE 0 END) AS errores
FROM documents;

-- Últimos 10 documentos indexados
SELECT arxiv_id, title, text_length, indexed_at
FROM documents
WHERE pdf_downloaded = 1
ORDER BY indexed_at DESC
LIMIT 10;

-- Errores recientes
SELECT arxiv_id, index_error, indexed_at
FROM documents
WHERE pdf_downloaded = 2
ORDER BY indexed_at DESC;
```

---

## Cómo añadir tablas para el módulo de indexación

Añade las nuevas tablas directamente en `backend/database/schema.py` dentro del bloque `_DDL`:

```python
# En schema.py, dentro de _DDL = """..."""
CREATE TABLE IF NOT EXISTS terms (
    term_id  INTEGER PRIMARY KEY,
    term     TEXT    NOT NULL UNIQUE,
    df       INTEGER DEFAULT 0   -- document frequency
);

CREATE TABLE IF NOT EXISTS postings (
    term_id  INTEGER REFERENCES terms(term_id),
    arxiv_id TEXT    REFERENCES documents(arxiv_id),
    tf       INTEGER DEFAULT 1,
    positions TEXT,              -- JSON: [12, 45, 103, ...]
    PRIMARY KEY (term_id, arxiv_id)
);
```

Luego llama a `init_db()` una vez para crear las tablas nuevas sin tocar las existentes:

```python
from backend.database.schema import init_db, DB_PATH
init_db(DB_PATH)   # CREATE TABLE IF NOT EXISTS — seguro de llamar varias veces
```

---

## Notas importantes

- **WAL mode activo** — el crawler puede escribir mientras otros módulos leen, sin bloqueos.
- **`abstract` siempre disponible** — aunque `full_text` sea `NULL`, el abstract se descarga en el primer ciclo. Puede usarse como fallback para indexar documentos aún sin texto completo.
- **`full_text` ya está limpio** — sin números de página, sin ecuaciones LaTeX, sin referencias bibliográficas, sin afiliaciones de autores. Listo para tokenizar directamente.
- **Chunks opcionales para indexación clásica** — la tabla `chunks` está diseñada para embeddings. Para un índice invertido BM25/TF-IDF es más eficiente trabajar con `full_text` directamente.
- **FK con CASCADE** — si se borra un documento, sus chunks se borran automáticamente.