# OmniRetrieve — Módulo `indexing`

Construye el **índice invertido de frecuencias** sobre el corpus descargado
por el crawler. Lee los textos de la BD, los tokeniza y persiste las
frecuencias crudas en las tablas `terms` y `postings`.

**No calcula pesos TF-IDF.** Solo cuenta y almacena frecuencias. La
fórmula `log(1 + freq) × log((N+1)/(df+1))` la aplica el módulo
`retrieval` al construir la matriz para el SVD de LSI.

---

## Estructura de archivos

```
backend/indexing/
├── preprocessor.py  ← TextPreprocessor: limpieza y tokenización
├── indexer.py       ← TFIndexer: motor de indexación (3 pasos)
├── pipeline.py      ← IndexingPipeline: coordinador + CLI
└── __init__.py      ← exports públicos
```

---

## Instalación

```bash
# Mínimo (usa stopwords básicas de fallback)
# — sin dependencias adicionales —

# Con NLTK (stopwords completas + stemming opcional)
pip install nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

Si NLTK no está instalado el módulo funciona con un conjunto de stopwords
básicas en inglés definido internamente (`_BASIC_STOPWORDS`, ~90 palabras).
Se registra un `WARNING` al iniciarse.

---

## Flujo de `TFIndexer.build()`

```
get_unindexed_documents()         ← docs con indexed_tfidf_at IS NULL
        ↓  (arxiv_id, texto) por lote de batch_size

TextPreprocessor.process(texto)
        ↓  list[str] de tokens normalizados

Counter(tokens)                   ← frecuencia cruda por término por doc
        ↓

Acumular df_map {term: n_docs}    ← cuántos docs contienen cada término

        ↓  (al terminar de leer todos los docs)

upsert_terms(df_map)              → tabla terms (word, df)
        ↓  {word: term_id}

flush_postings(batch)             → tabla postings (term_id, doc_id, freq)
        cada flush_every=5000 postings acumulados

        ↓

mark_documents_indexed(doc_ids)   → indexed_tfidf_at = now()
save_index_meta(stats)            → tabla index_meta
```

---

## `TextPreprocessor`

Transforma texto crudo en una lista de tokens limpios aplicando 9 pasos
en orden estricto:

| Paso | Operación | Ejemplo |
|---|---|---|
| 1 | Minúsculas | `"Attention"` → `"attention"` |
| 2 | Elimina URLs | `"https://arxiv.org/..."` → `" "` |
| 3 | Elimina LaTeX | `"$x^2$"`, `"\sum{...}"` → `" "` |
| 4 | Elimina números aislados | `"\b42\b"` → `" "` |
| 5 | Elimina puntuación | `","`, `"."`, `"–"`, `"\u201c"` → `""` |
| 6 | Tokeniza por espacios | `text.split()` |
| 7 | Filtra por longitud mínima | descarta tokens con `len < min_token_len` (default 3) |
| 8 | Filtra stopwords | elimina `"the"`, `"is"`, `"with"`, etc. |
| 9 | Conserva solo alfabéticos | `token.isalpha()` |
| 10 | Stemming opcional | `"running"` → `"run"` (SnowballStemmer, requiere NLTK) |

```python
from backend.indexing.preprocessor import TextPreprocessor

pp = TextPreprocessor(use_stemming=False, min_token_len=3)
tokens = pp.process("Attention mechanisms in transformer models rely on self-attention.")
# → ['attention', 'mechanisms', 'transformer', 'models', 'rely', 'self']
```

### Stemming

```python
pp_stem = TextPreprocessor(use_stemming=True)
pp_stem.process("training neural networks requires computing gradients")
# → ['train', 'neural', 'network', 'requir', 'comput', 'gradient']
```

`use_stemming=True` sin NLTK instalado se ignora silenciosamente.

---

## `TFIndexer`

Motor de indexación. Responsabilidad única: leer documentos, tokenizar,
contar y persistir.

```python
from backend.indexing.indexer import TFIndexer

indexer = TFIndexer(
    db_path    = Path("data/db/documents.db"),
    field      = "full_text",  # "full_text" | "abstract" | "both"
    batch_size = 100,          # documentos por lote de lectura
    flush_every= 5_000,        # postings acumulados antes de volcar a BD
)
stats = indexer.build(reindex=False)
```

### Parámetro `field`

| Valor | Comportamiento |
|---|---|
| `"full_text"` | Solo indexa `documents.full_text` (requiere `pdf_downloaded=1`) |
| `"abstract"` | Solo indexa `documents.abstract` |
| `"both"` | Usa `full_text` si está disponible, cae a `abstract` si no |

### Modo incremental vs reindex

**Incremental (`reindex=False`):** Solo procesa documentos con
`indexed_tfidf_at IS NULL`. Los documentos ya indexados se omiten.
Es el modo habitual en el orquestador.

**Reindex (`reindex=True`):** Llama a `clear_index()` (borra toda la tabla
`terms` y `postings` en cascade) y reindexea todo el corpus desde cero.
Necesario al cambiar `field` o `min_token_len`.

### Retorno de `build()`

```python
{
    "docs_processed": 1234,    # documentos procesados en esta ejecución
    "terms_added":    8901,    # términos nuevos añadidos al vocabulario
    "postings_added": 145230,  # registros (term_id, doc_id, freq) escritos
    "started_at":     "2024-01-01T10:00:00Z",
    "finished_at":    "2024-01-01T10:02:15Z",
}
```

---

## `IndexingPipeline`

Coordinador delgado que une `TextPreprocessor` + `TFIndexer` y expone
un API limpia y una CLI completa.

```python
from backend.indexing.pipeline import IndexingPipeline

pipeline = IndexingPipeline(
    db_path      = Path("data/db/documents.db"),
    field        = "both",     # "full_text" | "abstract" | "both"
    batch_size   = 100,
    use_stemming = False,
    min_token_len= 3,
)
stats = pipeline.run(reindex=False)
```

---

## CLI

```bash
# Indexación incremental (solo documentos nuevos)
python -m backend.indexing.pipeline

# Reindexar desde cero
python -m backend.indexing.pipeline --reindex

# Solo abstract (útil si aún no hay PDFs descargados)
python -m backend.indexing.pipeline --field abstract

# Con stemming
python -m backend.indexing.pipeline --stemming

# Todos los parámetros
python -m backend.indexing.pipeline \
  --db data/db/documents.db \
  --field full_text \
  --batch-size 200 \
  --min-len 4 \
  --stemming \
  --reindex
```

### Opciones CLI

| Flag | Default | Descripción |
|---|---|---|
| `--db` | `data/db/documents.db` | Ruta a la BD SQLite |
| `--field` | `both` | Campo a indexar (`full_text` / `abstract` / `both`) |
| `--batch-size N` | `100` | Documentos por lote de lectura |
| `--reindex` | off | Limpiar y reconstruir desde cero |
| `--stemming` | off | Activar SnowballStemmer (requiere NLTK) |
| `--min-len N` | `3` | Longitud mínima de token |

---

## Qué se guarda y qué no

| Tabla | Columna | Valor guardado |
|---|---|---|
| `terms` | `word` | Token normalizado (post-preprocesado) |
| `terms` | `df` | Nº de documentos que contienen el término |
| `postings` | `freq` | Nº de ocurrencias del término en el documento |
| `documents` | `indexed_tfidf_at` | Timestamp de indexación (para incremental) |

**No se guardan pesos TF-IDF.** El módulo `retrieval` lee `freq` y `df`
de la BD para construir la matriz TF-IDF al vuelo con:

```
TF(t,d)  = log(1 + freq(t,d))
IDF(t)   = log((N+1) / (df(t)+1))
W(t,d)   = TF(t,d) × IDF(t)
```

El filtro `min_df` (frecuencia mínima de documento) tampoco se aplica
aquí: lo aplica `LSIModel.build()` al leer la matriz desde la BD.

---

## Tests

Los tests están en `backend/tests/indexing/`.

```bash
pytest backend/tests/indexing/ -v

pytest backend/tests/indexing/test_indexing_pipeline.py -v
pytest backend/tests/indexing/test_index_repository.py  -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_indexing_pipeline.py` | Pipeline completa: documentos procesados, docs sin PDF omitidos, modo incremental no reprocesa, `--reindex` reprocesa todo |
| `test_index_repository.py` | `get_index_stats()`, `get_top_terms()`, `get_postings_for_term()`, `get_postings_for_matrix()` |

Los tests usan una BD SQLite en `tmp_path` con 4 documentos: 3 con
`pdf_downloaded=1` y texto, 1 con `pdf_downloaded=0` sin texto. El indexer
solo debe procesar los 3 con PDF.