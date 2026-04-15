---
noteId: "2d0468b0296711f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo Orquestador

Coordina los cuatro módulos del sistema (crawler, indexing, embedding, retrieval) en tiempo real. Ejecuta cada módulo en su propio hilo de fondo y expone una CLI interactiva para lanzar consultas mientras el sistema está corriendo.

---

## Estructura de archivos

```
backend/orchestrator/
+-- config.py        <- dataclass OrchestratorConfig con todos los parametros
+-- threads.py       <- funciones target de los cuatro hilos de fondo
+-- orchestrator.py  <- clase Orchestrator (estado compartido + API publica)
+-- cli.py           <- bucle interactivo y funciones de presentacion
+-- main.py          <- entrypoint CLI + argparse
+-- __init__.py      <- exports publicos
```

---

## Cómo ejecutar

```bash
# Arranque con parametros por defecto
python -m backend.orchestrator

# Para pruebas rapidas
python -m backend.orchestrator \
  --pdf-threshold 3 \
  --lsi-interval 300 \
  --lsi-k 50 \
  --lsi-min-docs 5 \
  --embed-threshold 10
```

### Parámetros disponibles

```bash
python -m backend.orchestrator \
  --db                ruta/a/documents.db  \
  --model             ruta/a/modelo.pkl    \

  # Crawler
  --ids-per-discovery 100    \
  --batch-size        10     \
  --pdf-batch         5      \
  --discovery-interval 120   \
  --download-interval  30    \
  --pdf-interval       60    \

  # Indexing
  --pdf-threshold     10     \
  --index-poll        30     \
  --index-field       full_text \

  # LSI
  --lsi-interval      3600   \
  --lsi-k             100    \
  --lsi-min-docs      10     \

  # Embedding
  --embed-model       all-MiniLM-L6-v2 \
  --embed-batch       256              \
  --embed-poll        60               \
  --embed-threshold   50               \
  --embed-rebuild-every 10000          \
  --embed-nlist       100              \
  --embed-m           8                \
  --embed-nbits       8                \
  --embed-nprobe      10
```

---

## Arquitectura: 4 hilos + CLI

```
+----------------------------------------------------------------+
|                         Orchestrator                           |
|                                                                |
|  Hilo 1: crawler         Hilo 2: indexing watcher             |
|  ----------------        ----------------------               |
|  run_forever()      ->   sondea BD cada 30s                   |
|  descarga IDs,           si delta >= pdf_threshold            |
|  metadatos y PDFs        -> IndexingPipeline.run()            |
|                                                                |
|  Hilo 3: lsi_rebuild     Hilo 4: embedding watcher            |
|  -------------------     -----------------------              |
|  duerme N segundos  ->   sondea BD cada 60s                   |
|  reconstruye modelo      si chunks_pendientes >= umbral       |
|  actualiza retriever     -> EmbeddingPipeline.run()           |
|  LSI bajo RLock          actualiza indice FAISS               |
|                                                                |
|  Hilo main: CLI                                               |
|  ---------------                                              |
|  input() interactivo                                          |
|  query | semantic | status | index | rebuild | quit           |
+----------------------------------------------------------------+
```

### Hilo 1 — Crawler

Llama directamente a `Crawler.run_forever()`. Un hilo auxiliar (watchdog) vigila la señal de shutdown y llama a `crawler.stop()` para que el hilo termine limpiamente.

### Hilo 2 — Indexing watcher

Cada `index_poll_interval` segundos consulta cuántos documentos tienen `pdf_downloaded=1` e `indexed_tfidf_at IS NULL`. Si el número es mayor o igual a `pdf_threshold`, lanza `IndexingPipeline.run()` de forma incremental.

### Hilo 3 — LSI rebuild

Intenta construir el modelo al arrancar (si hay suficientes docs). Luego duerme `lsi_rebuild_interval` segundos y repite. Cuando el modelo está listo, actualiza el `LSIRetriever` compartido bajo un `RLock` y activa el evento `_lsi_ready`.

### Hilo 4 — Embedding watcher

Cada `embed_poll_interval` segundos consulta cuántos chunks tienen `embedding IS NULL`. Si supera `embed_threshold`, lanza `EmbeddingPipeline.run()`. Al terminar, recarga el índice FAISS en el `FaissIndexManager` compartido y activa `_faiss_ready` para habilitar `semantic_query()`.

El primer chequeo ocurre inmediatamente al arrancar el hilo para procesar chunks pendientes de sesiones anteriores sin esperar el intervalo completo.

---

## Sincronización entre hilos

| Mecanismo | Tipo | Propósito |
|---|---|---|
| `_shutdown` | `threading.Event` | Señal de parada limpia para todos los hilos |
| `_lsi_lock` | `threading.RLock` | Protege el `LSIRetriever` durante el swap al rebuildar |
| `_lsi_ready` | `threading.Event` | Indica que ya existe al menos un modelo LSI cargado |
| `_retriever_holder` | `list[LSIRetriever]` | Contenedor mutable para el swap bajo lock |
| `_faiss_lock` | `threading.RLock` | Protege el `FaissIndexManager` durante la recarga |
| `_faiss_ready` | `threading.Event` | Indica que el índice FAISS está listo para búsquedas |

---

## API pública

### `query(text, top_n)` — búsqueda LSI

Recuperación semántica basada en el modelo LSI (índice invertido + SVD). Devuelve lista vacía si el modelo no está listo todavía.

```python
results = orc.query("graph neural networks", top_n=10)
# -> [{"arxiv_id": "...", "score": 0.94, "title": "...", ...}, ...]
```

### `semantic_query(text, top_k)` — búsqueda densa FAISS

Recuperación por similitud vectorial densa. Vectoriza la query con el mismo modelo de embedding, busca en el índice FAISS y enriquece los resultados con texto y metadatos de la BD.

```python
results = orc.semantic_query("attention mechanism transformer", top_k=10)
# -> [{"chunk_id": 1234, "score": 0.08, "arxiv_id": "...",
#      "chunk_index": 3, "text": "...", "title": "..."}, ...]
```

Devuelve lista vacía si el índice FAISS no está listo todavía.

### `status()` — snapshot del sistema

```python
snapshot = orc.status()
# Claves devueltas:
# docs_total, docs_pdf_indexed, docs_pdf_pending, docs_not_in_index
# vocab_size, total_postings
# lsi_docs_in_model, lsi_model_ready
# total_chunks, embedded_chunks, pending_chunks
# faiss_vectors, faiss_index_type, faiss_ready, embed_model
# timestamp
```

---

## CLI interactiva

Una vez arrancado el sistema, el hilo principal queda en modo interactivo:

```
==============================================================
  OmniRetrieve-Engine — Orquestador
==============================================================
  Escribe 'help' para ver los comandos disponibles.

query>
```

### Comandos disponibles

| Comando | Descripción |
|---|---|
| `query <texto>` | Busca con el modelo LSI (índice invertido + SVD) |
| `semantic <texto>` | Busca con FAISS (embedding denso) |
| `<texto>` | Atajo: cualquier texto sin prefijo se trata como query LSI |
| `status` | Muestra el estado actual del sistema |
| `index` | Fuerza una indexación TF incremental ahora |
| `rebuild` | Fuerza una reconstrucción del modelo LSI ahora |
| `help` | Muestra los comandos disponibles |
| `quit` / `exit` | Detiene el sistema y sale |

### Ejemplo de sesión

```
query> transformer attention mechanism
  Resultados LSI para: 'transformer attention mechanism'
  ----------------------------------------------------------
   1. [2301.001]  score=0.9412
      Attention Is All You Need
      We propose a new network architecture based solely on attention...

query> semantic efficient attention

  Resultados semanticos para: 'efficient attention'
  ----------------------------------------------------------
   1. chunk_id=4821  score=0.04  [2301.001, chunk 3]
      "Multi-head attention allows the model to attend to different
       positions simultaneously. Unlike recurrent layers, attention
       is computed in parallel across the entire sequence..."

query> status

  Estado del sistema  (2024-03-26 17:00:00 UTC)
  ----------------------------------------------------------
  Documentos en BD        : 1243
  PDFs descargados        : 876
  Pendientes de indexar   : 0
  Vocabulario (terms)     : 42891
  Docs en modelo LSI      : 876      listo: si
  Chunks totales          : 18432
  Chunks embedidos        : 18432
  Vectores en FAISS       : 18432    tipo: IndexIVFPQ
  Modelo de embedding     : all-MiniLM-L6-v2

query> quit
  Sistema detenido. Hasta luego.
```

---

## Uso programático

```python
from backend.orchestrator import Orchestrator, OrchestratorConfig

cfg = OrchestratorConfig(
    pdf_threshold        = 5,
    lsi_rebuild_interval = 1800,
    lsi_k                = 100,
    embed_threshold      = 50,
    embed_model          = "all-MiniLM-L6-v2",
)

orc = Orchestrator(cfg)
orc.start()       # arranca los 4 hilos de fondo

# Consulta LSI
lsi_results = orc.query("graph neural networks", top_n=5)

# Consulta semantica densa
dense_results = orc.semantic_query("attention mechanism", top_k=10)

snapshot = orc.status()
orc.stop()        # parada limpia
```

---

## Logs

Cada hilo identifica sus mensajes con un prefijo:

```
[crawler]    -- hilo de adquisicion
[indexing]   -- watcher de indexacion TF
[lsi]        -- hilo de rebuild del modelo LSI
[embedding]  -- watcher de embedding y FAISS
```

---

## Notas sobre rate limiting

Si el sistema recibe errores `HTTP 429` de arXiv, aumentar los intervalos del crawler:

```bash
python -m backend.orchestrator \
  --download-interval 60 \
  --batch-size 5 \
  --pdf-interval 60 \
  --pdf-batch 3
```