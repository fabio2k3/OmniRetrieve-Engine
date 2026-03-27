---
noteId: "658c1130296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Recuperación

Motor de búsqueda semántica basado en Latent Semantic Indexing (LSI). Lee el índice invertido construido por el módulo `indexing`, aplica TF-IDF y SVD truncada para obtener un espacio semántico latente, y permite lanzar consultas en texto libre devolviendo los artículos más relevantes por similitud coseno.

---

## Estructura de archivos

```
backend/retrieval/
├── lsi_model.py      ← fase offline: construye el modelo LSI
├── lsi_retriever.py  ← fase online: responde consultas semánticas
├── build_lsi.py      ← entrypoint CLI para la fase offline
└── __init__.py       ← exports públicos
```

---

## Instalación

```bash
pip install numpy scipy scikit-learn joblib
```

| Paquete | Para qué |
|---|---|
| `numpy` | Álgebra lineal y matrices densas |
| `scipy` | Matrices sparse (`lil_matrix`, `csr_matrix`) |
| `scikit-learn` | `TruncatedSVD`, `Normalizer`, `cosine_similarity` |
| `joblib` | Serialización del modelo `.pkl` |

---

## Flujo en dos fases

```
FASE OFFLINE (una vez, o cada N horas)
──────────────────────────────────────
index_repository.get_postings_for_matrix()
        ↓
  freq + df por (term, doc)
        ↓
  Calcular TF-IDF:
    TF(t,d)  = log(1 + freq)
    IDF(t)   = log((N+1) / (df+1))
    W(t,d)   = TF × IDF
        ↓
  Matriz sparse  (n_terms × n_docs)
        ↓
  TruncatedSVD (k componentes)  +  Normalizer L2
        ↓
  docs_latent  (n_docs × k)
        ↓
  Guardar .pkl  →  registrar en lsi_log


FASE ONLINE (por cada consulta)
────────────────────────────────
query en texto libre
        ↓
  TextPreprocessor (mismo que indexing)
        ↓
  Vectorizar con TF × IDF del corpus
        ↓
  SVD.transform()  →  q_latent  (k,)
        ↓
  cosine_similarity(q_latent, docs_latent)
        ↓
  top-N por score  →  resultados con metadatos
```

---

## Cómo ejecutar

### Fase offline — construir el modelo

```bash
# Con parámetros por defecto (k=100)
python -m backend.retrieval.build_lsi

# Personalizado
python -m backend.retrieval.build_lsi --k 200 --n-iter 15

# Limitar documentos (útil para pruebas)
python -m backend.retrieval.build_lsi --k 50 --max-docs 1000
```

### Parámetros disponibles

```bash
python -m backend.retrieval.build_lsi \
  --db        ruta/a/documents.db   # BD a usar (default: data/db/documents.db) \
  --k         100                   # componentes latentes del SVD (default: 100) \
  --n-iter    10                    # iteraciones del algoritmo SVD (default: 10) \
  --out       ruta/modelo.pkl       # ruta de salida del .pkl \
  --max-docs  5000                  # limitar número de documentos
```

### Fase online — consultas programáticas

```python
from backend.retrieval import LSIRetriever

retriever = LSIRetriever()
retriever.load()   # carga el .pkl y los metadatos de la BD

results = retriever.retrieve("transformer attention mechanisms", top_n=10)

for r in results:
    print(f"{r['score']:.4f}  [{r['arxiv_id']}]  {r['title']}")
```

Cada resultado contiene:

```python
{
    "score":    0.8932,          # similitud coseno [0.0, 1.0]
    "arxiv_id": "2301.00123",
    "title":    "Attention Is All You Need",
    "authors":  "Vaswani et al.",
    "abstract": "...(primeros 300 caracteres)...",
    "url":      "https://arxiv.org/pdf/2301.00123",
}
```

---

## Fórmulas

```
TF(t, d)  = log(1 + freq(t, d))          suavizado logarítmico
IDF(t)    = log((N + 1) / (df(t) + 1))   suavizado Laplace (evita div/0)
W(t, d)   = TF(t, d) × IDF(t)
```

La vectorización de la query usa la misma fórmula TF × IDF, con el `df` real del corpus (guardado en el modelo `.pkl`) para que la proyección al espacio latente sea coherente con la fase offline.

Referencia: Manning et al., *Introduction to Information Retrieval* (2008), Cap. 18.

---

## El modelo `.pkl`

`LSIModel.save()` serializa:

| Campo | Tipo | Descripción |
|---|---|---|
| `svd` | `TruncatedSVD` | SVD ajustado, contiene `components_` para proyectar queries |
| `normalizer` | `Normalizer` | Normalización L2 ajustada |
| `docs_latent` | `np.ndarray (n_docs × k)` | Vectores latentes de cada documento |
| `doc_ids` | `list[str]` | arxiv_ids en el mismo orden que las columnas de la matriz |
| `term_ids` | `list[int]` | term_ids en el mismo orden que las filas de la matriz |
| `df_map` | `dict[int, int]` | `term_id → df` del corpus (para vectorizar queries) |
| `k` | `int` | Número de componentes latentes |

El modelo se guarda en `backend/data/models/lsi_model.pkl` por defecto.

---

## Parámetro k

`k` controla el número de componentes latentes del SVD. Es el parámetro más importante:

| k | Uso recomendado |
|---|---|
| 50–100 | Corpus pequeño (< 5.000 docs) |
| 100–300 | Corpus mediano (5.000–50.000 docs) |
| 300–500 | Corpus grande (> 50.000 docs) |

Valores más altos capturan más matices semánticos pero aumentan el tiempo de build y el uso de memoria. `k` se ajusta automáticamente si el corpus es menor que el valor configurado (`k = min(k, n_docs - 1)`).

---

## Rendimiento

| Operación | Tiempo estimado | Notas |
|---|---|---|
| `build()` | 5–60s | Depende del tamaño del corpus; operación offline |
| `retrieve()` | < 10ms | Para corpus < 100K docs |
| Memoria modelo | ~50–500 MB | Para k=100, 10K–100K docs |

---

## Tests

```bash
python -m backend.tests.test_retrieval
python -m pytest backend/tests/test_retrieval.py -v
```
