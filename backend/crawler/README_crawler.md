---
noteId: "5c4d1fb0296511f1b0a22758fc0c48d3"
tags: []

---

# OmniRetrieve — Módulo de Adquisición de Datos

Crawler que descubre artículos de IA/ML en arXiv, descarga sus metadatos y extrae el texto completo, almacenándolo en SQLite para su posterior indexación y vectorización.

---

## Estructura de archivos

```
backend/
├── main.py                    <- entrypoint principal
├── crawler/
│   ├── arxiv_client.py        <- cliente API Atom de arXiv
│   ├── crawler.py             <- orquestador (3 hilos daemon)
│   ├── document.py            <- dataclass Document + CSV
│   ├── id_store.py            <- gestión thread-safe de IDs
│   ├── pdf_extractor.py       <- extracción de texto + chunking
│   └── robots.py              <- respeto a robots.txt
├── database/
│   ├── schema.py              <- definición de tablas SQLite
│   ├── crawler_repository.py  <- operaciones CRUD de documentos
│   └── chunk_repository.py    <- operaciones CRUD de chunks
└── tools/
    ├── inspect_db.py          <- inspector visual de la BD
    ├── rebuild_chunks.py      <- reconstruye todos los chunks con el nuevo algoritmo
    └── embed_chunks.py        <- embediza todos los chunks pendientes
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
  --batch-size 10          \
  --pdf-batch 10           \
  --ids-per-discovery 100  \
  --discovery-interval 120 \
  --download-interval 30   \
  --pdf-interval 2         \
  --chunk-size 1000        \
  --overlap 2
```

| Parámetro | Default | Descripción |
|---|---|---|
| `--batch-size` | 10 | Metadatos a descargar por ciclo |
| `--pdf-batch` | 10 | Textos a extraer por ciclo |
| `--ids-per-discovery` | 100 | IDs a buscar por ciclo |
| `--discovery-interval` | 120 | Segundos entre ciclos de discovery |
| `--download-interval` | 30 | Segundos entre ciclos de metadatos |
| `--pdf-interval` | 2 | Segundos entre extracciones individuales |
| `--chunk-size` | 1000 | Tamaño máximo de cada chunk en caracteres |
| `--overlap` | 2 | Oraciones de solapamiento entre chunks consecutivos |

---

## Arquitectura: los 3 hilos

El crawler ejecuta tres tareas en paralelo como hilos daemon:

```
+-------------------------------------------------------------+
|                        Crawler                              |
|                                                             |
|  Hilo 1: Discovery          Hilo 2: Download               |
|  -----------------          -----------------              |
|  Busca IDs nuevos en   ->   Descarga metadatos         ->  |
|  arXiv cada 120s            de IDs pendientes              |
|  Guarda en                  Guarda en CSV + SQLite         |
|  ids_article.csv                                           |
|                                  |                          |
|                        Hilo 3: PDF/HTML                     |
|                        ----------------                     |
|                        Extrae texto completo               |
|                        (HTML primero, PDF fallback)        |
|                        Aplica chunking con overlap          |
|                        Guarda texto + chunks en SQLite     |
+-------------------------------------------------------------+
```

### Hilo 1 — Discovery (`_discovery_loop`)

- Consulta la API Atom de arXiv buscando papers en categorías de IA/ML: `cs.AI, cs.LG, cs.CV, cs.CL, cs.NE, stat.ML`
- Ordena por fecha de publicación descendente (papers más recientes primero)
- Guarda los IDs nuevos en `ids_article.csv` via `IdStore`
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
- Aplica el algoritmo de chunking con solapamiento semántico
- Guarda el texto limpio y los chunks en SQLite via `chunk_repository`

---

## Extracción de texto

### Método 1: HTML (preferido)

arXiv genera HTML para casi todos los papers usando **LaTeXML**, que produce una estructura fija con clases CSS `ltx_*`:

```
ltx_document
  +-- ltx_title          -> titulo del paper             [incluido]
  +-- ltx_authors        -> autores/afiliaciones         [skip]
  +-- ltx_abstract       -> abstract                     [incluido]
  +-- ltx_section        -> Introduction, Method...      [incluido]
  |     +-- ltx_title_section  -> heading de seccion     [incluido]
  |     +-- ltx_para / ltx_p  -> parrafos                [incluido]
  +-- ltx_figure         -> figuras                      [skip]
  +-- ltx_table          -> tablas                       [skip]
  +-- ltx_equation       -> ecuaciones LaTeX             [skip]
  +-- ltx_bibliography   -> referencias                  [skip]
```

### Método 2: PDF (fallback)

Si el HTML no está disponible, descarga el PDF y extrae el texto usando **PyMuPDF** (`fitz`). Se aplica la misma limpieza posterior.

---

## Algoritmo de chunking

El texto extraído se divide en chunks usando un **sliding window semántico a nivel de oración**.

### Detección de oraciones

El texto se parte por fronteras lingüísticas, no por caracteres arbitrarios:

```python
# Punto/!/? solo si lo sigue mayuscula o digito (evita "Fig. 3", "et al.")
r'(?<=[.!?])\s+(?=[A-Z\"\'(0-9])'
# Punto y coma siempre separa
r'|(?<=;)\s+'
```

Las oraciones muy cortas (< 20 chars) se fusionan con la siguiente para evitar que abreviaturas sueltas queden como oraciones independientes.

### Solapamiento entre chunks

Los párrafos actúan como fronteras duras — nunca hay solapamiento entre párrafos distintos. Dentro de un párrafo, las últimas `overlap_sentences` oraciones del chunk emitido se incluyen al inicio del siguiente:

```
Oraciones del parrafo: [A] [B] [C] [D] [E] [F] [G] [H]

chunk_size=300, overlap=2:

Chunk 1 -> A B C D
Chunk 2 -> C D E F      <- C y D repetidas como contexto
Chunk 3 -> E F G H      <- E y F repetidas como contexto
```

Esto garantiza que ningún concepto queda partido sin que al menos un chunk lo contenga completo, mejorando significativamente la calidad de la búsqueda semántica.

### Parámetros de chunking en `CrawlerConfig`

| Campo | Default | Descripción |
|---|---|---|
| `chunk_size` | 1000 | Tamaño máximo de cada chunk en caracteres |
| `overlap_sentences` | 2 | Oraciones compartidas entre chunks consecutivos |

---

## Políticas de crawling

### robots.txt

El módulo `robots.py` verifica robots.txt antes de cada request. Los dominios de arXiv están en una **allowlist** porque su API pública está explícitamente documentada y permitida.

### Rate limiting

| Componente | Delay mínimo |
|---|---|
| API de metadatos (`arxiv_client.py`) | 3.5s entre requests |
| Extractor HTML/PDF (`pdf_extractor.py`) | 3.0s entre requests |
| Entre extracciones individuales (`crawler.py`) | 2.0s |

### Límite de tamaño

Los PDFs mayores de **15 MB** se rechazan automáticamente. Se marcan con `pdf_downloaded = 2` (error) y el crawler continúa con el siguiente documento.

---

## Herramientas

### Inspector de la BD

```bash
python -m backend.tools.inspect_db                     # resumen completo
python -m backend.tools.inspect_db --watch             # refresco automatico
python -m backend.tools.inspect_db --docs 10           # ultimos N documentos
python -m backend.tools.inspect_db --doc 2301.12345    # detalle de un doc
python -m backend.tools.inspect_db --text 2301.12345   # texto guardado en BD
python -m backend.tools.inspect_db --chunk 2301.12345 --idx 3  # un chunk concreto
python -m backend.tools.inspect_db --errors            # documentos con error
python -m backend.tools.inspect_db --categories        # distribucion por categoria
python -m backend.tools.inspect_db --crawl-log         # historial del crawler
```

### Reconstrucción de chunks

Útil tras cambiar el algoritmo de chunking o los parámetros `chunk_size` / `overlap`:

```bash
# Simular sin modificar nada
python -m backend.tools.rebuild_chunks --dry-run

# Reconstruir con nuevos parametros
python -m backend.tools.rebuild_chunks --chunk-size 800 --overlap 3

# Sin confirmacion interactiva (para scripts)
python -m backend.tools.rebuild_chunks --yes
```

Después de reconstruir los chunks hay que re-embedizar:

```bash
python -m backend.tools.embed_chunks --reembed
```

---

## Tests

```bash
# Tests sin red (unit tests locales)
python -m backend.tests.test_crawler --skip-network

# Solo verificar integridad de la BD real
python -m backend.tests.test_crawler --only-db

# Suite completa (requiere conexion a internet)
python -m backend.tests.test_crawler
```

---

## Datos generados

```
backend/data/
+-- ids_article.csv     <- todos los IDs descubiertos
+-- documents.csv       <- metadatos de cada articulo (backup plano)
+-- app.log             <- log completo de ejecucion
+-- db/
    +-- documents.db    <- base de datos SQLite principal
```