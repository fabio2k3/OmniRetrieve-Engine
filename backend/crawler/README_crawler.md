# OmniRetrieve — Módulo `crawler`

Adquisición continua de artículos científicos. Ejecuta **tres hilos daemon**
coordinados que descubren IDs, descargan metadatos y obtienen el texto
completo, fragmentándolo en chunks listos para el módulo de embedding.

Diseñado para ser extensible a múltiples fuentes: cualquier fuente nueva
solo necesita implementar `BaseClient`.

---

## Estructura de archivos

```
backend/crawler/
├── crawler.py               ← Crawler: orquestador, arranca/detiene los 3 hilos
├── config.py                ← CrawlerConfig: todos los parámetros de comportamiento
├── document.py              ← Document: dataclass de un artículo descargado
├── id_store.py              ← IdStore: CSV thread-safe de IDs conocidos
├── robots.py                ← RobotsChecker: robots.txt genérico (singleton)
├── http.py                  ← fetch_bytes, SSL context, USER_AGENT compartidos
├── chunker.py               ← clean_text + make_chunks (agnostico de fuente)
├── _routing.py              ← client_for(), local_id() (interno, usado por los loops)
├── loops/
│   ├── discovery.py         ← DiscoveryLoop: descubre IDs nuevos periódicamente
│   ├── downloader.py        ← DownloaderLoop: descarga metadatos en lotes
│   └── text.py              ← TextLoop: descarga texto completo y genera chunks
└── clients/
    ├── base_client.py       ← BaseClient: interfaz abstracta para cualquier fuente
    └── arxiv/
        ├── client.py        ← ArxivClient: HTTP + rate-limiting + extracción de texto
        ├── api.py           ← Parseo XML Atom (parse_ids, parse_entries)
        ├── constants.py     ← URLs base, namespace Atom, parámetros por defecto
        └── extractors/
            ├── html.py      ← _LaTeXMLParser + extract() para HTML arXiv
            └── pdf.py       ← extract() para PDF via PyMuPDF
```

---

## Instalación

```bash
# Mínimo (HTML LaTeXML como fuente de texto)
pip install certifi

# Con extracción de PDF como fallback
pip install pymupdf
```

`certifi` es opcional pero recomendado: sin él el SSL context deshabilita
la verificación de hostname.

---

## Arquitectura de tres hilos

```
Crawler.start()
    │
    ├── Thread("discovery")  ─── DiscoveryLoop.run()
    │       ↓ cada discovery_interval (120 s)
    │   client.fetch_ids() → IDs locales
    │   client.make_doc_id() → IDs compuestos
    │   IdStore.add_ids()
    │
    ├── Thread("downloader") ─── DownloaderLoop.run()
    │       ↓ espera 10 s al arranque, luego cada download_interval (30 s)
    │   IdStore.get_pending_batch()
    │   client.fetch_documents() → list[Document]
    │   Document.save() → documents.csv
    │   repo.upsert_document() → SQLite
    │   IdStore.mark_downloaded()
    │
    └── Thread("text")       ─── TextLoop.run()
            ↓ espera 20 s al arranque, luego itera con pdf_interval (2 s) entre docs
        repo.get_pending_pdf_ids()
        client.download_text() → texto limpio
        chunker.make_chunks()  → list[str]
        repo.save_pdf_text()   → SQLite
        save_chunks()          → SQLite
```

Los hilos son `daemon=True`: terminan automáticamente con el proceso principal.
El `threading.Event` (`_stop`) coordina la parada limpia con timeout de 15 s.

---

## IDs compuestos

Todos los IDs tienen formato `"{source}:{local_id}"`:

```
arxiv:2301.12345          ← arXiv
semantic_scholar:abc123   ← Semantic Scholar (futura fuente)
```

`BaseClient.make_doc_id("2301.12345")` → `"arxiv:2301.12345"`  
`BaseClient.parse_doc_id("arxiv:2301.12345")` → `("arxiv", "2301.12345")`

El módulo `_routing.py` expone `client_for(doc_id, client_map)` y
`local_id(doc_id)`, usadas por los tres loops para enrutar cada ID a su
cliente sin duplicar la lógica.

---

## `BaseClient` — interfaz para nuevas fuentes

```python
from backend.crawler.clients.base_client import BaseClient
from backend.crawler.document import Document

class MiFuenteClient(BaseClient):

    @property
    def source_name(self) -> str:
        return "mi_fuente"

    @property
    def request_delay(self) -> float:
        return 5.0   # segundos entre peticiones HTTP

    @property
    def trusted_domains(self) -> frozenset:
        return frozenset({"api.mi_fuente.org"})

    def fetch_ids(self, max_results=100, start=0) -> list[str]:
        ...   # devuelve IDs locales, SIN prefijo

    def fetch_documents(self, local_ids: list[str]) -> list[Document]:
        ...   # devuelve Documents con doc_id = self.make_doc_id(lid)

    def download_text(self, local_id: str, **kwargs) -> str:
        ...   # devuelve texto limpio, listo para chunking
```

Para registrar la fuente, pasar la instancia a `Crawler(clients=[...])`.

---

## Política de crawling y robots.txt

El diseño separa completamente **quién declara la política** de **quién la aplica**:

| Componente | Responsabilidad |
|---|---|
| `robots.py` (`RobotsChecker`) | Motor genérico; no sabe nada de ninguna fuente |
| `BaseClient` | Declara `request_delay` y `trusted_domains` |
| `ArxivClient` | Implementa la política concreta para arXiv |

### `RobotsChecker`

- `allowed(url, trusted_domains)` → `True` si el acceso está permitido.  
  Si el host está en `trusted_domains`, devuelve `True` sin consultar robots.txt.
- `crawl_delay(url)` → segundos del `Crawl-delay` declarado en robots.txt.  
  **Nunca hace bypass**, ni siquiera para dominios de confianza.

El delay efectivo real siempre es:

```
effective_delay = max(client.request_delay, checker.crawl_delay(url))
```

Ante errores de red al leer robots.txt, el checker es **fail-open** (permite el acceso).

### Singleton compartido

```python
from backend.crawler.robots import checker   # instancia global del paquete
```

### `ArxivClient` — política declarada

| Propiedad | Valor | Motivo |
|---|---|---|
| `request_delay` | `15.0 s` | Igual al `Crawl-delay: 15` del robots.txt de arXiv |
| `trusted_domains` | `{"arxiv.org", "export.arxiv.org"}` | API Atom y HTML/PDF autorizados por ToS aunque robots.txt incluya `Disallow: /api` |

---

## `ArxivClient` — adquisición

### `fetch_ids()`

Consulta la API Atom de `export.arxiv.org/api/query` con paginación
(`start`, `max_results`). Delega el parseo XML a `api.parse_ids()`, que
extrae el ID local eliminando la versión (`v2`, `v3`…).

Búsqueda por defecto: categorías `cs.AI`, `cs.LG`, `cs.CL`, `stat.ML`.

### `fetch_documents()`

Agrupa los IDs locales en lotes de 20 y llama a `api.parse_entries()` para
cada lote. Aplica rate-limiting compartido entre instancias via
`_rate_lock` y `_last_request` (variables de clase).

### `download_text()`

Intenta dos estrategias en orden:

1. **HTML LaTeXML** (`export.arxiv.org/html/{local_id}`)  
   Analiza el HTML con `_LaTeXMLParser`, que solo extrae el cuerpo del
   artículo, ignorando clases CSS `ltx_authors`, `ltx_bibliography`,
   `ltx_figure`, `ltx_table`, `ltx_equation`, `ltx_pagination` y otras
   secciones no textuales.
   
2. **PDF** (`arxiv.org/pdf/{local_id}`)  
   Descarga el PDF y extrae texto página a página con PyMuPDF (`fitz`).
   Requiere `pip install pymupdf`. Si no está instalado, esta estrategia
   se omite silenciosamente.

El texto obtenido por cualquiera de las dos rutas se pasa por `clean_text()`
antes de devolverlo.

---

## `http.py` — HTTP compartido

M�dulo utilitario usado internamente por `robots.py` y `ArxivClient`:

| Nombre | Descripción |
|---|---|
| `USER_AGENT` | `"SRI-Crawler/1.0"` — valor por defecto del User-Agent |
| `_SSL_CTX` | Contexto SSL único; usa `certifi` si está instalado, degrada a sin verificación si no |
| `fetch_bytes(url, timeout, accept)` | GET minimalista → `bytes \| None`. Para descargas de bajo volumen (robots.txt, metadatos pequeños) |

Para descargas grandes (PDFs, HTML de artículos) `ArxivClient._get()` añade
streaming, control de tamaño máximo (`max_size_mb`) y rate-limiting.

---

## `chunker.py` — fragmentación de texto

API pública:

```python
from backend.crawler.chunker import make_chunks, clean_text

chunks = make_chunks(
    text,
    chunk_size=1000,          # caracteres máximos por chunk
    overlap_sentences=2,      # oraciones compartidas entre chunks consecutivos
    min_chunk_chars=100,      # descarta chunks más cortos
    min_sent_chars=20,        # fusiona oraciones muy cortas con la siguiente
)
```

### `clean_text(text)`

1. Colapsa más de 2 saltos de línea consecutivos (`\n{3,}` → `\n\n`)
2. Elimina líneas que solo contienen números (números de página)
3. Colapsa espacios/tabulaciones múltiples a uno solo

### `make_chunks()` — algoritmo

1. **`clean_text()`** — normaliza el texto crudo
2. **Split por párrafos** (`\n\n`) — fronteras duras; nunca se genera un chunk que cruce un párrafo
3. **Split por oraciones** dentro de cada párrafo usando regex lingüístico  
   (punto + mayúscula/dígito, `!`, `?`, `;`) con fusión de oraciones cortas
4. **Acumulación con solapamiento**:

```
Oraciones del párrafo: [A  B  C  D  E  F  G  H]
Chunk 1 → A B C D
Chunk 2 → C D E F    ← C y D repetidas (overlap_sentences=2)
Chunk 3 → E F G H    ← E y F repetidas
```

El solapamiento preserva contexto de transición entre chunks para que el
modelo de embedding no pierda información en los bordes.

---

## `IdStore` — almacén thread-safe de IDs

CSV en memoria con flush a disco en cada mutación. Permite operaciones
concurrentes de los tres hilos sin race conditions.

```
ids_article.csv:
doc_id,discovered_at,downloaded
arxiv:2301.12345,2024-01-01T10:00:00,True
arxiv:2302.00001,2024-01-01T10:05:00,False
```

- Columna `doc_id` (retrocompat: lee también CSVs con columna `arxiv_id`)
- `add_ids()` deduplica y devuelve el número de IDs **nuevos** añadidos
- `get_pending_batch(n)` → IDs con `downloaded=False`
- `mark_downloaded(ids)` → marca como descargados y persiste

---

## `Document` — persistencia dual CSV + SQLite

```python
doc = Document(
    doc_id="arxiv:2301.12345",
    title="Attention Is All You Need",
    authors="Vaswani et al.",
    abstract="...", categories="cs.CL",
    published="2017-06-12T00:00:00Z",
    updated="2017-06-12T00:00:00Z",
    pdf_url="https://arxiv.org/pdf/2301.12345",
)

doc.save(csv_path)                    # append a documents.csv
Document.load_all(csv_path)          # → list[Document]
Document.load_ids(csv_path)          # → set[str] (lectura rápida de IDs)
```

`doc.arxiv_id` es un alias de solo lectura de `doc.doc_id` para compatibilidad
retroactiva con los módulos `database`, `indexing` y `retrieval`.

---

## `CrawlerConfig` — parámetros

```python
from backend.crawler.config import CrawlerConfig

config = CrawlerConfig(
    discovery_interval = 120.0,  # s entre ciclos de descubrimiento
    download_interval  = 30.0,   # s entre ciclos de metadatos
    pdf_interval       = 2.0,    # s entre documentos en el ciclo de texto
    ids_per_discovery  = 100,    # IDs a pedir por ciclo
    batch_size         = 10,     # docs por lote de metadatos
    pdf_batch_size     = 10,     # docs por lote de texto
    chunk_size         = 1000,   # chars máximos por chunk
    overlap_sentences  = 2,      # oraciones de solapamiento
)
```

---

## Uso programático

```python
from backend.crawler import Crawler, CrawlerConfig
from backend.crawler.clients.arxiv import ArxivClient

crawler = Crawler(
    config=CrawlerConfig(ids_per_discovery=50),
    clients=[ArxivClient()],
)
crawler.run_forever()   # bloquea hasta Ctrl-C
```

```python
# Arranque y parada controlados
crawler.start()
time.sleep(300)
crawler.stop()
```

---

## Cómo añadir una nueva fuente

1. Crear `backend/crawler/clients/mi_fuente/client.py` implementando `BaseClient`
2. Declarar `source_name`, `request_delay` y `trusted_domains`
3. Implementar `fetch_ids`, `fetch_documents` y `download_text`
4. Pasar la instancia en `Crawler(clients=[ArxivClient(), MiFuenteClient()])`

El orquestador, los loops y el routing funcionan sin ningún otro cambio.

---

## Tests

Los tests están en `backend/tests/crawler/`.

```bash
pytest backend/tests/crawler/ -v -m "not network"   # excluye tests de red
pytest backend/tests/crawler/ -v                     # todos

pytest backend/tests/crawler/test_document.py        -v
pytest backend/tests/crawler/test_id_store.py        -v
pytest backend/tests/crawler/test_robots.py          -v
pytest backend/tests/crawler/test_base_client.py     -v
pytest backend/tests/crawler/test_arxiv_client.py    -v
pytest backend/tests/crawler/test_chunker.py         -v
pytest backend/tests/crawler/test_routing.py         -v
pytest backend/tests/crawler/test_backward_compat.py -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_document.py` | `doc_id`, alias `arxiv_id`, CSV save/load/round-trip, keys legacy, hash y equality |
| `test_id_store.py` | add/dedup, mark_downloaded, persistencia CSV, thread-safety con 5 hilos concurrentes |
| `test_robots.py` | `allowed()` y `crawl_delay()`, bypass de trusted_domains, fail-open en error de red |
| `test_base_client.py` | Clase abstracta no instanciable, `make_doc_id`, `parse_doc_id` válidos e inválidos |
| `test_arxiv_client.py` | Rate-limit thread-safe, `parse_ids()` strips versión, `_entry_to_document`, `_LaTeXMLParser` skips bibliography/autores |
| `test_chunker.py` | `clean_text`, `make_chunks` con overlap, párrafos como fronteras duras, texto vacío |
| `test_routing.py` | `client_for()` por source, None para fuente desconocida, `local_id()`, discovery y text loop end-to-end |
| `test_backward_compat.py` | Imports retrocompat + pipeline completo con `FakeClient` (discovery → metadata → chunks) |

Los tests marcados `@pytest.mark.network` hacen peticiones reales a arXiv
y se excluyen con `-m "not network"`.

El `conftest.py` de la carpeta define `FakeClient` (implementación completa de
`BaseClient` con datos sintéticos) y `tmp_db` (BD SQLite en `tmp_path`).