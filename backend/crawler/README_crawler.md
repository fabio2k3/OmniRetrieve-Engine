# Módulo `crawler` — Adquisición de Datos

Crawler multi-fuente que descubre artículos, descarga metadatos y extrae texto
completo, almacenando todo en SQLite para su posterior indexación y vectorización.

---

## Estructura del módulo

```
crawler/
├── clients/
│   ├── base_client.py       ← contrato que todo cliente debe implementar
│   ├── arxiv_client.py      ← cliente arXiv: política + descarga + extracción
│   └── __init__.py
├── arxiv_client.py          ← re-exporta ArxivClient (compatibilidad)
├── chunker.py               ← algoritmo de chunking genérico
├── crawler.py               ← orquestador (3 hilos daemon)
├── document.py              ← dataclass Document + persistencia CSV
├── id_store.py              ← store thread-safe de IDs compuestos
└── robots.py                ← verificador de robots.txt genérico
```

---

## Diseño: separación de responsabilidades

El módulo distingue tres capas independientes que no se mezclan entre sí:

```
┌─────────────────────────────────────────────────────────────────┐
│  POLÍTICA DE CRAWLING          cada cliente la declara           │
│    request_delay, trusted_domains                                │
│    ArxivClient: delay=15s, trusted={arxiv.org, export...}       │
├─────────────────────────────────────────────────────────────────┤
│  TRANSPORTE Y EXTRACCIÓN       cada cliente la implementa        │
│    download_text(local_id)                                       │
│    ArxivClient: HTML (LaTeXML) → PDF (PyMuPDF) fallback          │
├─────────────────────────────────────────────────────────────────┤
│  FRAGMENTACIÓN (CHUNKING)      genérica, sin saber de fuente     │
│    chunker.make_chunks(text, chunk_size, overlap_sentences)      │
│    compartida por todos los clientes                             │
└─────────────────────────────────────────────────────────────────┘
```

El `Crawler` une estas tres capas: llama a `client.download_text()` para obtener el
texto y luego a `chunker.make_chunks()` para fragmentarlo.
Ninguna de las dos capas conoce a la otra.

---

## Formato de IDs

Todos los IDs del sistema usan formato compuesto:

```
{fuente}:{id_local}
```

Ejemplos: `arxiv:2301.12345`, `semantic_scholar:abc123`

Este ID se almacena en el campo `arxiv_id` de SQLite y en `doc_id` del CSV.
El resto del sistema (indexing, embedding, retrieval) lo trata como un string
opaco sin saber de qué fuente proviene.

---

## Cómo funciona el Crawler

Ejecuta tres hilos daemon que se coordinan a través de SQLite e `IdStore`:

```
  cliente A ──┐
  cliente B ──┤──► Crawler
  cliente N ──┘
               │
     ┌─────────┴──────────┬──────────────────┐
     │                    │                  │
  Hilo 1              Hilo 2             Hilo 3
  Discovery          Downloader           Text
     │                    │                  │
  fetch_ids()        fetch_docs()      download_text()
     │                    │                  │
  IdStore  ──────────► SQLite  ─────────► make_chunks()
  (CSV)              (metadatos)            │
                                        SQLite
                                        (chunks)
```

### Hilo 1 — Discovery

- Llama a `client.fetch_ids()` en cada cliente registrado
- Construye IDs compuestos: `client.make_doc_id(local_id)` → `"arxiv:2301.12345"`
- Persiste los nuevos en `IdStore` (CSV thread-safe)
- Avanza el offset en cada ciclo; continúa hacia contenido más antiguo
  si la página ya era conocida

### Hilo 2 — Downloader

- Lee lotes de IDs pendientes de `IdStore`, los agrupa por fuente
- Llama a `client.fetch_documents(local_ids)` para cada fuente
- Guarda metadatos en SQLite con upsert idempotente
  (no borra el estado de texto existente si ya estaba indexado)
- Marca los IDs como descargados en `IdStore`

### Hilo 3 — Text

- Lee de SQLite los documentos con `pdf_downloaded = 0`
- Enruta al cliente correcto por el prefijo del ID compuesto
- Llama a `client.download_text(local_id)` — el cliente decide cómo obtener el texto
- Fragmenta con `chunker.make_chunks(text, chunk_size, overlap_sentences)`
- Persiste texto completo y chunks en SQLite
- Si falla, marca el documento con `pdf_downloaded = 2` (error) y continúa

---

## Políticas de crawling

### Dónde se declaran

Cada cliente declara su política directamente en su propia clase. `robots.py` es
completamente genérico y no contiene ningún dominio hardcodeado.

```python
class ArxivClient(BaseClient):

    @property
    def request_delay(self) -> float:
        return 15.0    # igual al Crawl-delay: 15 del robots.txt de arXiv

    @property
    def trusted_domains(self) -> FrozenSet[str]:
        return frozenset({"arxiv.org", "export.arxiv.org"})
        # robots.txt tiene Disallow: /api, pero la API Atom esta autorizada
        # por ToS -> bypass solo en allowed(), nunca en crawl_delay()
```

### Cómo se aplica el delay

```python
effective_delay = max(client.request_delay, checker.crawl_delay(url))
```

Para arXiv: `max(15.0, 15.0) = 15.0 s`. Si arXiv endurece su política en el
futuro, el valor del `robots.txt` tomará precedencia automáticamente sin tocar
el código del cliente.

### Thread-safety del rate-limiter

`_rate_lock` y `_last_request` son **variables de clase** en `ArxivClient`
(no de instancia). Todas las instancias, en cualquier hilo, comparten el mismo
contador. Si el `Crawler` tiene 3 hilos activos o se crean múltiples instancias,
ninguno puede saltarse el delay:

```
Hilo discovery:   with ArxivClient._rate_lock → espera → lanza petición
Hilo downloader:  with ArxivClient._rate_lock → bloqueado hasta que discovery termine
Hilo text:        with ArxivClient._rate_lock → bloqueado hasta que downloader termine
```

El `time.sleep()` ocurre **dentro** del lock para que el siguiente hilo empiece
su cuenta desde el momento en que el anterior reservó su turno, no desde que
terminó de descargar.

### `robots.py` — genérico y sin bypass de delay

| Método | Usa `trusted_domains` | Por qué |
|---|---|---|
| `allowed(url, trusted_domains)` | Sí (lo pasa el cliente) | Evita falsos negativos en Disallow |
| `crawl_delay(url)` | Nunca | El delay siempre se lee del robots.txt real |

---

## Algoritmo de chunking (`chunker.py`)

Módulo independiente de cualquier fuente. Recibe texto limpio y devuelve
una lista de fragmentos.

### División en oraciones

Respeta texto científico (evita partir en "Fig. 3", "et al.", etc.):

- Punto/!/? seguido de mayúscula o dígito
- Punto y coma como frontera blanda
- Oraciones menores de 20 caracteres se fusionan con la siguiente

### Solapamiento semántico

Los párrafos (`\n\n`) son fronteras duras. Dentro de un párrafo, las últimas
`N` oraciones de un chunk se repiten al inicio del siguiente:

```
Oraciones: [A] [B] [C] [D] [E] [F] [G] [H]   (chunk_size=300, overlap=2)

Chunk 1 ->  A  B  C  D
Chunk 2 ->  C  D  E  F     <- C y D repetidas como contexto de transición
Chunk 3 ->  E  F  G  H     <- E y F repetidas como contexto de transición
```

Ningún concepto queda partido entre chunks sin que al menos uno lo contenga completo.

---

## Añadir un nuevo cliente

Solo hay que crear un archivo nuevo e implementar `BaseClient`.
El resto del sistema no necesita ningún cambio:

```python
# crawler/clients/semantic_scholar_client.py

from typing import FrozenSet, List
from ..clients.base_client import BaseClient
from ..document import Document


class SemanticScholarClient(BaseClient):

    @property
    def source_name(self) -> str:
        return "semantic_scholar"

    @property
    def request_delay(self) -> float:
        return 5.0   # Crawl-delay del robots.txt de Semantic Scholar

    @property
    def trusted_domains(self) -> FrozenSet[str]:
        return frozenset({"api.semanticscholar.org"})

    def fetch_ids(self, max_results: int = 100, start: int = 0) -> List[str]:
        # llamar a la API y devolver IDs locales (sin prefijo)
        ...

    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        # descargar metadatos y devolver Documents con doc_id compuesto
        # doc_id = self.make_doc_id(local_id)  ->  "semantic_scholar:abc123"
        ...

    def download_text(self, local_id: str, **kwargs) -> str:
        # descargar y devolver el texto limpio del documento
        # el Crawler se encarga del chunking despues
        ...
```

Registrar el cliente en el Crawler:

```python
from backend.crawler.crawler import Crawler, CrawlerConfig
from backend.crawler.clients.arxiv_client import ArxivClient
from backend.crawler.clients.semantic_scholar_client import SemanticScholarClient

crawler = Crawler(
    config=CrawlerConfig(chunk_size=1000, overlap_sentences=2),
    clients=[ArxivClient(), SemanticScholarClient()],
)
crawler.run_forever()
```

---

## Arrancar el Crawler

```python
from backend.crawler import Crawler, CrawlerConfig, ArxivClient

crawler = Crawler(
    config=CrawlerConfig(
        ids_per_discovery  = 500,
        batch_size         = 20,
        pdf_batch_size     = 5,
        discovery_interval = 120.0,   # segundos entre ciclos de discovery
        download_interval  = 30.0,    # segundos entre ciclos de metadatos
        pdf_interval       = 15.0,    # pausa entre documentos de texto
        chunk_size         = 1000,    # caracteres maximos por chunk
        overlap_sentences  = 2,       # oraciones de solapamiento
    ),
    clients=[ArxivClient()],
)
crawler.run_forever()
```

O desde el orquestador:

```bash
python -m backend.orchestrator.main
```

---

## Tests

```bash
# Suite principal (sin red, ~1s)
python -m pytest backend/tests/test_crawler.py -v -m "not network"

# Con tests de red (requiere internet, ~5min por los delays de 15s)
python -m pytest backend/tests/test_crawler.py -v -m "network"

```

| Clase de test | Tests | Qué verifica |
|---|---|---|
| `TestDocument` | 13 | `doc_id`, alias `arxiv_id`, CSV nuevo y legado |
| `TestIdStore` | 9 | IDs compuestos, columna `doc_id`, retrocompat |
| `TestRobots` | 6 | Genérico, sin dominios hardcodeados, `crawl_delay` sin bypass |
| `TestBaseClient` | 11 | Propiedades abstractas, `make_doc_id`, `parse_doc_id` |
| `TestArxivClient` | 18 | Política (delay=15s, trusted_domains), thread-safety, parseo XML |
| `TestChunker` | 12 | `clean_text`, solapamiento, fronteras de párrafo |
| `TestArxivTextExtraction` | 5 | LaTeXML extractor, skip bibliografía/autores |
| `TestCrawlerRouting/Discovery/Text` | 11 | Routing multi-cliente, pipeline con SQLite |
| `TestBackwardCompatImports` | 4 | Imports desde rutas antiguas |
| `TestEndToEndFakeClient` | 2 | Pipeline completo mono y multi-fuente |

---

## Herramientas

### Inspector de la BD

```bash
python -m backend.tools.inspect_db                           # resumen completo
python -m backend.tools.inspect_db --watch                   # refresco automatico
python -m backend.tools.inspect_db --doc arxiv:2301.12345    # detalle de un doc
python -m backend.tools.inspect_db --text arxiv:2301.12345   # texto guardado
python -m backend.tools.inspect_db --errors                  # docs con error
python -m backend.tools.inspect_db --categories              # distribución por categoría
python -m backend.tools.inspect_db --crawl-log               # historial de ejecuciones
```

### Renombrado de IDs (`rename_ids.py`)

Para migrar IDs del formato antiguo (`2301.12345`) al nuevo (`arxiv:2301.12345`):

```bash
# Ver qué cambiaría sin tocar nada (dry-run por defecto)
python -m backend.tools.rename_ids --prefix arxiv

# Aplicar con copia de seguridad y verificación
python -m backend.tools.rename_ids --prefix arxiv --backup --verify --apply

# Solo algunos IDs (filtro glob)
python -m backend.tools.rename_ids --prefix arxiv --filter "2301.*" --apply --yes

# Renombrar un ID concreto
python -m backend.tools.rename_ids --rename 2301.12345 arxiv:2301.12345 --apply

# Lote desde CSV (columnas: old_id, new_id)
python -m backend.tools.rename_ids --mapping migration.csv --apply --yes --verify
```

Actualiza en cascada y de forma atómica las tres tablas afectadas:
`documents.arxiv_id`, `chunks.arxiv_id` y `postings.doc_id`.

### Reconstrucción de chunks (`rebuild_chunks.py`)

Tras cambiar `chunk_size` u `overlap_sentences`:

```bash
python -m backend.tools.rebuild_chunks --dry-run
python -m backend.tools.rebuild_chunks --chunk-size 800 --overlap 3
python -m backend.tools.rebuild_chunks --yes
```

Después hay que re-embedizar:

```bash
python -m backend.tools.embed_chunks --reembed
```

---

## Datos generados

```
backend/data/
├── ids_article.csv      ← IDs compuestos descubiertos (col: doc_id)
├── documents.csv        ← metadatos en plano (col: doc_id)
├── app.log              ← log de ejecución
└── db/
    └── documents.db     ← SQLite principal
                            ├── documents  (metadatos + texto completo)
                            ├── chunks     (fragmentos + embeddings)
                            ├── crawl_log  (historial de ejecuciones)
                            ├── terms      (vocabulario índice TF)
                            └── postings   (frecuencias por término/doc)
```

---

## Decisiones de diseño clave

| Decisión | Por qué |
|---|---|
| `pdf_extractor.py` eliminado | El Crawler llama a `client.download_text()` directamente. Sin intermediarios. |
| `_rate_lock` a nivel de clase | Coordina múltiples instancias e hilos sin shared state implícito. |
| `crawl_delay()` sin bypass | El delay es una restricción de frecuencia que se cumple siempre, aunque el dominio esté en `trusted_domains`. |
| `trusted_domains` por cliente | `robots.py` permanece genérico. Cada cliente conoce sus propias excepciones de ToS. |
| IDs compuestos | El resto del sistema trata el ID como string opaco — cero cambios al añadir nuevas fuentes. |
| Chunking separado del transporte | Un cliente nuevo solo implementa `download_text()`. El algoritmo de fragmentación no se duplica. |