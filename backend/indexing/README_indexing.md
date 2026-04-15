---
noteId: "7f39b4c0296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Indexación

Construye el índice invertido de frecuencias sobre el corpus de artículos descargado por el crawler. Lee los textos de la base de datos, los tokeniza y persiste las frecuencias crudas en las tablas `terms` y `postings`.

---

## Estructura de archivos

```
backend/indexing/
├── preprocessor.py    ← limpieza y tokenización de texto
├── indexer.py         ← motor de indexación (TFIndexer)
├── pipeline.py        ← orquestador + entrypoint CLI
└── __init__.py        ← exports públicos
```

---

## Instalación

```bash
pip install nltk
```

| Paquete | Para qué |
|---|---|
| `nltk` | Stopwords y stemming. Opcional — si no está instalado se usan stopwords básicas de fallback |

Si está instalado, descargar los recursos necesarios:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

---

## Cómo ejecutar

```bash
# Indexación incremental (solo documentos nuevos con PDF descargado)
python -m backend.indexing.pipeline

# Reindexar desde cero
python -m backend.indexing.pipeline --reindex

# Indexar solo el abstract (sin necesidad de PDF)
python -m backend.indexing.pipeline --field abstract

# Con stemming activado
python -m backend.indexing.pipeline --stemming
```

### Parámetros disponibles

```bash
python -m backend.indexing.pipeline \
  --db       ruta/a/documents.db   # BD a usar (default: data/db/documents.db) \
  --field    full_text             # full_text | abstract | both (default: full_text) \
  --batch-size 100                 # documentos por lote (default: 100) \
  --reindex                        # borrar índice existente y reconstruir \
  --stemming                       # activar SnowballStemmer (requiere NLTK) \
  --min-len  3                     # longitud mínima de token (default: 3)
```

---

## Arquitectura

```
documento (full_text en BD)
        ↓
  TextPreprocessor
  ─────────────────
  minúsculas → elimina URLs/LaTeX/números
  → elimina puntuación → tokeniza
  → filtra stopwords → filtra no-alfa
  → stemming (opcional)
        ↓
  Counter(tokens)     ←── frecuencia cruda por documento
        ↓
  TFIndexer
  ──────────
  upsert_terms(df_map)        →  tabla terms  (word, df)
  flush_postings(batch)       →  tabla postings (term_id, doc_id, freq)
  mark_documents_indexed(ids) →  documents.indexed_tfidf_at = now
```

---

## Preprocesador (`preprocessor.py`)

`TextPreprocessor` aplica estos pasos en orden:

1. Minúsculas
2. Elimina URLs (`https://...`, `www...`)
3. Elimina expresiones LaTeX (`$...$`, `\cmd{...}`)
4. Elimina números aislados (`\b\d+\b`)
5. Elimina puntuación
6. Tokeniza por espacios
7. Filtra tokens por longitud mínima (`min_token_len`, default 3)
8. Filtra stopwords (NLTK inglés, o fallback básico)
9. Conserva solo tokens alfabéticos (`isalpha()`)
10. Stemming opcional (`SnowballStemmer`)

```python
from backend.indexing.preprocessor import TextPreprocessor

pp = TextPreprocessor(use_stemming=False, min_token_len=3)
tokens = pp.process("Attention mechanisms in transformer models.")
# → ['attention', 'mechanisms', 'transformer', 'models']
```

---

## Indexador (`indexer.py`)

`TFIndexer` ejecuta la indexación en tres pasos:

### Paso 1 — Tokenizar

Itera sobre `get_unindexed_documents()` — documentos con `pdf_downloaded=1` e `indexed_tfidf_at IS NULL`. Para cada uno, obtiene un `Counter` de tokens.

### Paso 2 — Persistir vocabulario

Llama a `upsert_terms(df_map)` que inserta términos nuevos y acumula el `df` de los ya existentes en un solo batch.

### Paso 3 — Persistir postings

Vuelca lotes de `(term_id, doc_id, freq)` a la tabla `postings`. Al finalizar, llama a `mark_documents_indexed()` para que estos documentos no se reprocesen en ejecuciones futuras.

### Indexación incremental

Por defecto `reindex=False` — solo se procesan documentos cuyo `indexed_tfidf_at` es `NULL`. Los documentos ya indexados se saltan siempre, incluso si el texto ha cambiado. Para forzar el reprocesado completo usa `--reindex`.

---

## Qué se guarda y qué no

| Tabla | Campo | Valor |
|---|---|---|
| `terms` | `word` | token normalizado |
| `terms` | `df` | nº de documentos que contienen el término |
| `postings` | `freq` | nº de veces que aparece el término en el documento |

**No se guardan pesos TF-IDF.** El módulo `retrieval` lee estas frecuencias crudas y aplica la fórmula que necesite (log-TF × IDF, BM25, etc.) al construir la matriz para el SVD.

---

## Tests

```bash
python -m backend.tests.test_indexing
python -m pytest backend/tests/test_indexing.py -v
```
