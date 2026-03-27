---
noteId: "2d0468b0296711f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo Orquestador

Coordina los tres módulos del sistema (crawler, indexing, retrieval) en tiempo real sobre datos reales. Ejecuta cada módulo en su propio hilo de fondo y expone una CLI interactiva para lanzar consultas semánticas mientras el sistema está corriendo.

---

## Estructura de archivos

```
backend/orchestrator/
├── config.py        ← dataclass OrchestratorConfig con todos los parámetros
├── threads.py       ← funciones target de los tres hilos de fondo
├── orchestrator.py  ← clase Orchestrator (estado compartido + API pública)
├── cli.py           ← bucle interactivo y funciones de presentación
├── main.py          ← entrypoint CLI + argparse
└── __init__.py      ← exports públicos
```

---

## Cómo ejecutar

```bash
# Arranque con parámetros por defecto
python -m backend.orchestrator

# Para pruebas rápidas (umbral bajo, rebuild cada 5 min)
python -m backend.orchestrator \
  --pdf-threshold 3 \
  --lsi-interval 300 \
  --lsi-k 50 \
  --lsi-min-docs 5
```

### Parámetros disponibles

```bash
python -m backend.orchestrator \
  --db                ruta/a/documents.db  # BD SQLite (default: data/db/documents.db) \
  --model             ruta/a/modelo.pkl    # modelo LSI (default: data/models/lsi_model.pkl)

# Crawler
  --ids-per-discovery 100    # IDs a descubrir por ciclo (default: 100) \
  --batch-size        10     # metadatos a descargar por ciclo (default: 10) \
  --pdf-batch         5      # PDFs a descargar por ciclo (default: 5) \
  --discovery-interval 120   # segundos entre ciclos de discovery (default: 120) \
  --download-interval  30    # segundos entre ciclos de metadatos (default: 30) \
  --pdf-interval       60    # segundos entre ciclos de PDF (default: 60)

# Indexing
  --pdf-threshold     10     # PDFs sin indexar para disparar indexación (default: 10) \
  --index-poll        30     # segundos entre sondeos del watcher (default: 30) \
  --index-field       full_text  # full_text | abstract | both (default: full_text)

# LSI
  --lsi-interval      3600   # segundos entre rebuilds del modelo (default: 3600) \
  --lsi-k             100    # componentes latentes del SVD (default: 100) \
  --lsi-min-docs      10     # mínimo de docs indexados para construir el modelo (default: 10)
```

---

## Arquitectura: 3 hilos + CLI

```
┌──────────────────────────────────────────────────────────────┐
│                       Orchestrator                           │
│                                                              │
│  Hilo 1: crawler          Hilo 2: indexing watcher           │
│  ─────────────────        ───────────────────────            │
│  run_forever()       →    sondea BD cada 30s                 │
│  descarga IDs,            si Δ ≥ pdf_threshold               │
│  metadatos y PDFs         → IndexingPipeline.run()           │
│                                                              │
│  Hilo 3: lsi_rebuild      Hilo main: CLI                     │
│  ───────────────────      ──────────────────                 │
│  duerme N segundos   →    input() interactivo                │
│  reconstruye modelo       query | status | index             │
│  actualiza retriever      rebuild | help | quit              │
└──────────────────────────────────────────────────────────────┘
```

### Hilo 1 — Crawler

Llama directamente a `Crawler.run_forever()`. Un hilo auxiliar (watchdog) vigila la señal de shutdown y llama a `crawler.stop()` para que el hilo termine limpiamente.

### Hilo 2 — Indexing watcher

Cada `index_poll_interval` segundos consulta cuántos documentos tienen `pdf_downloaded=1` e `indexed_tfidf_at IS NULL`. Si el número es mayor o igual a `pdf_threshold`, lanza `IndexingPipeline.run()` de forma incremental.

### Hilo 3 — LSI rebuild

Intenta construir el modelo al arrancar (si hay suficientes docs). Luego duerme `lsi_rebuild_interval` segundos y repite. Cuando el modelo está listo, actualiza el `LSIRetriever` compartido bajo un `RLock` y activa el evento `_lsi_ready` para que la CLI pueda empezar a responder queries.

### Hilo main — CLI

Bucle `input()` que despacha comandos. Lee del `LSIRetriever` compartido solo cuando `_lsi_ready` está activo.

---

## Sincronización entre hilos

| Mecanismo | Tipo | Propósito |
|---|---|---|
| `_shutdown` | `threading.Event` | Señal de parada limpia para todos los hilos |
| `_lsi_lock` | `threading.RLock` | Protege el `LSIRetriever` durante el swap al rebuildar |
| `_lsi_ready` | `threading.Event` | Indica que ya existe al menos un modelo cargado |
| `_retriever_holder` | `list[LSIRetriever]` | Contenedor mutable de un elemento para el swap bajo lock |

---

## CLI interactiva

Una vez arrancado el sistema, el hilo principal queda en modo interactivo:

```
══════════════════════════════════════════════════════════
  OmniRetrieve-Engine — Orquestador
══════════════════════════════════════════════════════════
  Escribe 'help' para ver los comandos disponibles.

query>
```

### Comandos disponibles

| Comando | Descripción |
|---|---|
| `query <texto>` | Busca los 10 artículos más relevantes |
| `<texto>` | Atajo: cualquier texto sin prefijo se trata como query |
| `status` | Muestra el estado actual del sistema |
| `index` | Fuerza una indexación incremental ahora |
| `rebuild` | Fuerza una reconstrucción del modelo LSI ahora |
| `help` | Muestra los comandos disponibles |
| `quit` / `exit` | Detiene el sistema y sale |

### Ejemplo de sesión

```
query> transformer attention mechanism
  Resultados para: 'transformer attention mechanism'
  ──────────────────────────────────────────────────────
   1. [2301.001]  score=0.9412
      Attention Is All You Need
      Vaswani et al.
      We propose a new network architecture based solely on attention...

   2. [2301.002]  score=0.8731
      BERT: Pre-training of Deep Bidirectional Transformers
      ...

query> status

  Estado del sistema  (2024-03-26 17:00:00 UTC)
  ─────────────────────────────────────────────────────
  Documentos en BD        1243
  PDFs descargados        876
  PDFs pendientes         367
  Pendientes de indexar   0
  Vocabulario (terms)     42891
  Postings totales        198432
  Docs en modelo LSI      876
  Modelo LSI              ✔  listo

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
)

orc = Orchestrator(cfg)
orc.start()       # arranca los 3 hilos de fondo

# Consulta directa sin pasar por la CLI
results = orc.query("graph neural networks", top_n=5)
snapshot = orc.status()

orc.stop()        # parada limpia
```

---

## Logs

El orquestador escribe en consola y en `backend/data/orchestrator.log`. Cada hilo identifica sus mensajes con un prefijo:

```
[crawler]   — hilo de adquisición
[indexing]  — watcher de indexación
[lsi]       — hilo de rebuild del modelo LSI
```

---

## Notas sobre rate limiting

arXiv impone límites informales de velocidad. Si el sistema recibe errores `HTTP 429` se recomienda aumentar los intervalos:

```bash
python -m backend.orchestrator \
  --download-interval 60 \
  --batch-size 5 \
  --pdf-interval 60 \
  --pdf-batch 3
```