# LSI Retrieval Module

**Latent Semantic Indexing (LSI)** for semantic document retrieval on arXiv papers.

## Overview

The retrieval module (`Modulo C`) implements a two-phase semantic search system:

- **Offline phase** (`LSIModel`): Build the latent semantic space from document corpus using TF-IDF + Truncated SVD + L2 normalization
- **Online phase** (`LSIRetriever`): Project queries in the latent space and return top-K similar documents using cosine similarity

Reference: Manning et al. (2008), Ch. 18, Sec. 18.1-18.4

## Architecture

```
TF-IDF Vectorizer
    |
    v
Truncated SVD (k=100)  <- Dimensionality reduction
    |
    v
L2 Normalizer (cosine similarity)
    |
    v
Query Projection & Retrieval
```

## Quick Start

### 1. Build the LSI Model (Offline)

```bash
# Default: k=100 latent concepts
python -m backend.retrieval.build_lsi

# Custom: k=200 concepts
python -m backend.retrieval.build_lsi --k 200

# Custom output path
python -m backend.retrieval.build_lsi --k 150 --out custom_model.pkl
```

Output: `backend/data/models/lsi_model.pkl`

Logs: 
- Console: INFO level
- File: `backend/data/lsi_build.log`

### 2. Retrieve Documents (Online)

```python
from backend.retrieval import LSIRetriever

# Initialize and load model
retriever = LSIRetriever()
retriever.load()  # Loads model + DB metadata

# Query
results = retriever.retrieve("transformer attention mechanisms", top_n=10)

for doc in results:
    print(f"{doc['title']:<50} | score: {doc['score']:.4f}")
```

Output format (each result):
```python
{
    "score": 0.8932,           # Cosine similarity [0.0, 1.0]
    "arxiv_id": "2301.00123",
    "title": "Attention Is All You Need",
    "authors": "Vaswani et al.",
    "abstract": "...(first 300 chars)...",
    "url": "https://arxiv.org/pdf/..."
}
```

## Configuration

### LSIModel Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 100 | Number of latent concepts (dimensions after SVD) |
| `max_features` | 50,000 | Max vocabulary size (TF-IDF) |
| `min_df` | 2 | Ignore terms appearing in fewer than N docs |
| `stop_words` | "english" | Language for stopword removal |
| `n_iter` | 10 | SVD iterations (higher = better quality) |

### LSIRetriever Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_n` | 10 | Number of results to return |
| `db_path` | `DB_PATH` | Path to SQLite database |
| `model_path` | `MODEL_PATH` | Path to saved .pkl model |

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Build (`build()`) | ~5-30s | Corpus size dependent; one-time offline |
| Query (`retrieve()`) | ~1-10ms | Sub-100ms guarantee for <100K docs |
| Memory | ~500MB | For 100K docs, k=100 |

## Database Integration

### LSI Log Table

Tracks model build sessions:

```sql
CREATE TABLE lsi_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    built_at TEXT NOT NULL,           -- ISO timestamp
    k INTEGER NOT NULL,               -- Latent dimensions
    n_docs INTEGER NOT NULL,          -- Documents indexed
    var_explained REAL,               -- Variance explained by SVD
    model_path TEXT,                  -- Model file path
    notes TEXT                        -- Optional notes
);
```

Query build history:
```sql
SELECT built_at, k, n_docs, var_explained FROM lsi_log ORDER BY built_at DESC;
```

## Testing

Run all LSI module tests:

```bash
# From project root
python -m pytest backend/tests/test_retrieval.py -v

# Specific test
python -m pytest backend/tests/test_retrieval.py::test_lsi_model_build -v

# With output
python -m pytest backend/tests/test_retrieval.py -v -s
```

Test types:
- **Unit**: Model construction (< 1s, in-memory)
- **Integration**: Full pipeline with temp DB (1-5s)
- **Semantic**: Query relevance validation

## Logging

### Log Levels

- **INFO**: Build progress, model stats, retrieval ready
- **DEBUG**: Query details, timing breakdowns
- **WARNING**: DB errors, missing tables

### Example Log Output

```
2024-03-25 14:32:15 INFO     retrieval.lsi_model -- [LSIModel] Cargando textos de la BD: backend/data/db/documents.db
2024-03-25 14:32:15 INFO     retrieval.lsi_model -- [LSIModel] 12543 documentos cargados
2024-03-25 14:32:19 INFO     retrieval.lsi_model -- [LSIModel] SVD completado. Varianza=67.23% Tiempo=3.8s
2024-03-25 14:32:20 INFO     retrieval.lsi_model -- [LSIModel] Modelo guardado en backend/data/models/lsi_model.pkl (12543 docs, k=100)
```

## Troubleshooting

### Q: "Modelo no cargado. Llama a .load() primero"

Retrieve called before model loaded:
```python
retriever = LSIRetriever()
# ❌ retriever.retrieve(...)  # Fails
retriever.load()
# ✅ retriever.retrieve(...)  # Works
```

### Q: "ModuleNotFoundError: No module named 'scikit-learn'"

Install dependencies:
```bash
pip install -r backend/requirements.txt
```

### Q: Query returns only low-scoring results

Possible causes:
1. Query vocabulary not in training corpus
2. `k` too small (reduce semantic expressiveness)
3. Corpus too small for SVD stability

Solutions:
- Expand corpus
- Increase `k` parameter
- Verify corpus quality

## Files

| File | Purpose |
|------|---------|
| `lsi_model.py` | Offline model builder (phase 1) |
| `lsi_retriever.py` | Online query engine (phase 2) |
| `build_lsi.py` | CLI entrypoint with logging |
| `test_retrieval.py` | Unit + integration tests |

## References

- Manning et al., "Introduction to Information Retrieval" (2008)
- [Truncated SVD (scikit-learn docs)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [LSI in IR](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

## Next Steps

- [ ] Integrate with `backend/main.py` API endpoint
- [ ] Add query expansion (pseudo-relevance feedback)
- [ ] Hybrid retrieval: LSI + dense embeddings
- [ ] Real-time model updates (incremental SVD)
