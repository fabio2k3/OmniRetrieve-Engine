---
noteId: "5c4d1fb0296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Adquisición de Datos

Crawler que descubre artículos de IA/ML en arXiv, descarga sus metadatos y extrae el texto completo, almacenándolo en SQLite para su posterior indexación.

---

## Estructura de archivos

```
backend/
├── main.py                    ← entrypoint principal
├── crawler/
│   ├── arxiv_client.py        ← cliente API Atom de arXiv
│   ├── crawler.py             ← orquestador (3 hilos daemon)
│   ├── document.py            ← dataclass Document + CSV
│   ├── id_store.py            ← gestión thread-safe de IDs
│   ├── pdf_extractor.py       ← extracción de texto (HTML / PDF)
│   └── robots.py              ← respeto a robots.txt
├── database/
│   ├── schema.py              ← definición de tablas SQLite
│   ├── crawler_repository.py  ← operaciones CRUD del crawler
│   └── index_repository.py   ← operaciones CRUD del índice
├── tools/
│   └── inspect_db.py          ← inspector visual de la DB
└── tests/
    └── test_crawler.py       ← suite de tests
```

---

## Instalación

```bash
pip install certifi pymupdf
```

| Paquete | Para qué |
|---|---|
| `certifi` | Certificados SSL en Windows |
| `pymupdf` | Extraer texto de PDFs (fallback) |

---

## Cómo ejecutar

```bash
# Desde la carpeta raíz que contiene backend/
python -m backend.main
```

### Parámetros disponibles

```bash
python -m backend.main \
  --batch-size 10          # metadatos a descargar por ciclo (default: 10) \
  --pdf-batch 10           # textos a extraer por ciclo (default: 10) \
  --ids-per-discovery 100  # IDs a buscar por ciclo (default: 100) \
  --discovery-interval 120 # segundos entre ciclos de discovery (default: 120) \
  --download-interval 30   # segundos entre ciclos de metadatos (default: 30) \
  --pdf-interval 2         # segundos entre extracciones individuales (default: 2)
```

---

## Arquitectura: los 3 hilos

El crawler ejecuta tres tareas en paralelo como hilos daemon:

```
┌─────────────────────────────────────────────────────────────┐
│                        Crawler                              │
│                                                             │
│  Hilo 1: Discovery          Hilo 2: Download               │
│  ─────────────────          ─────────────────               │
│  Busca IDs nuevos en   →    Descarga metadatos         →    │
│  arXiv cada 120s            de IDs pendientes               │
│  Guarda en                  Guarda en CSV + SQLite          │
│  ids_article.csv                                            │
│                                  ↓                          │
│                        Hilo 3: PDF/HTML                     │
│                        ────────────────                     │
│                        Extrae texto completo                │
│                        (HTML primero, PDF fallback)         │
│                        Guarda texto + chunks en SQLite      │
└─────────────────────────────────────────────────────────────┘
```

### Hilo 1 — Discovery (`_discovery_loop`)

- Consulta la API Atom de arXiv buscando papers en categorías de IA/ML: `cs.AI, cs.LG, cs.CV, cs.CL, cs.NE, stat.ML`
- Ordena por fecha de publicación descendente (papers más recientes primero)
- Guarda los IDs nuevos en `ids_article.csv` via `IdStore`
- El offset se inicializa al total de IDs ya conocidos para no re-escanear páginas viejas
- Cuando una página no devuelve IDs nuevos, resetea el offset a 0 para buscar papers recién publicados

### Hilo 2 — Downloader (`_download_loop`)

- Lee lotes de IDs pendientes de `IdStore`
- Descarga metadatos completos via la API de arXiv (título, autores, abstract, categorías, fechas, URL del PDF)
- Persiste en CSV (`documents.csv`) y en SQLite (`documents`) con upsert idempotente
- Marca los IDs como descargados en `ids_article.csv`

### Hilo 3 — PDF/HTML (`_pdf_loop`)

- Consulta SQLite para obtener documentos con `pdf_downloaded = 0` (pendientes)
- Para cada uno intenta dos métodos en orden:
  1. **HTML** — `arxiv.org/html/{id}` (~100–500 KB, sin dependencias extra)
  2. **PDF** — `arxiv.org/pdf/{id}` (~2–15 MB, requiere pymupdf)
- Rechaza PDFs mayores de 15 MB para no bloquear la cola
- Guarda el texto limpio y los chunks en SQLite

---

## Extracción de texto

### Método 1: HTML (preferido)

arXiv genera HTML para casi todos los papers usando **LaTeXML**, que produce una estructura fija con clases CSS `ltx_*`:

```
ltx_document
  ├── ltx_title          → título del paper             ✅ incluido
  ├── ltx_authors        → autores/afiliaciones         ❌ skip (ya en metadata)
  ├── ltx_abstract       → abstract                     ✅ incluido
  ├── ltx_section        → Introduction, Method...      ✅ incluido
  │     ├── ltx_title_section  → heading de sección     ✅ incluido
  │     └── ltx_para / ltx_p  → párrafos                ✅ incluido
  ├── ltx_figure         → figuras                      ❌ skip
  ├── ltx_table          → tablas                       ❌ skip
  ├── ltx_equation       → ecuaciones LaTeX             ❌ skip
  └── ltx_bibliography   → referencias                  ❌ skip
```

El resultado es texto limpio con la estructura del paper: título → abstract → secciones.

### Método 2: PDF (fallback)

Si el HTML no está disponible, descarga el PDF y extrae el texto usando **PyMuPDF** (`fitz`). Se aplica la misma limpieza posterior (números de página, espacios múltiples, líneas en blanco excesivas).

### Chunking

Después de la extracción, el texto se divide en fragmentos de ~1000 caracteres respetando párrafos. Estos chunks se guardan en la tabla `chunks` y están preparados para generar embeddings en la fase 2.

---

## Políticas de crawling

### robots.txt

El módulo `robots.py` verifica robots.txt antes de cada request. Los dominios de arXiv están en una **allowlist** (`arxiv.org`, `export.arxiv.org`) porque su API pública está explícitamente documentada y permitida para uso programático.

Para cualquier otro dominio, se respeta el robots.txt normalmente con caché TTL de 3600 segundos.

### Rate limiting

| Componente | Delay mínimo |
|---|---|
| API de metadatos (`arxiv_client.py`) | 3.5s entre requests |
| Extractor HTML/PDF (`pdf_extractor.py`) | 3.0s entre requests |
| Entre extracciones individuales (`crawler.py`) | 2.0s |

### Límite de tamaño

Los PDFs mayores de **15 MB** se rechazan automáticamente antes de descargar más bytes. Se marcan con `pdf_downloaded = 2` (error) y el crawler continúa con el siguiente documento.

---

## Herramientas

### Inspector de la DB

```bash
# Estado general de la DB
python -m backend.tools.inspect_db

# Actualización en tiempo real cada 5s
python -m backend.tools.inspect_db --watch

# Últimos N documentos guardados
python -m backend.tools.inspect_db --docs 10

# Detalle completo de un documento
python -m backend.tools.inspect_db --doc 2301.12345

# Ver el texto literal guardado en la DB
python -m backend.tools.inspect_db --text 2301.12345
python -m backend.tools.inspect_db --text 2301.12345 --chars 8000

# Ver un chunk concreto
python -m backend.tools.inspect_db --chunk 2301.12345 --idx 3
```

## Tests

```bash
# Tests sin red (unit tests locales)
python -m backend.tests.test_crawler --skip-network

# Solo verificar integridad de la DB real
python -m backend.tests.test_crawler --only-db

# Suite completa (requiere conexión a internet)
python -m backend.tests.test_crawler
```

---

## Datos generados

```
backend/data/
├── ids_article.csv     ← todos los IDs descubiertos
│                          columnas: arxiv_id, discovered_at, downloaded
├── documents.csv       ← metadatos de cada artículo (backup plano)
├── app.log             ← log completo de ejecución
└── db/
    └── documents.db    ← base de datos SQLite principal
```

`ids_article.csv` y `documents.csv` están en `.gitignore` por su tamaño.