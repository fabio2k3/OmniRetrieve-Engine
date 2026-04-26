---
noteId: "c2261e103c0f11f19f111fe4db9ea2b5"
tags: []

---

# Módulo `crawler` — Documentación completa

## Índice

1. [Visión general](#1-visión-general)
2. [Estructura de ficheros](#2-estructura-de-ficheros)
3. [Flujo completo de trabajo](#3-flujo-completo-de-trabajo)
4. [Módulos — referencia detallada](#4-módulos--referencia-detallada)
   - [config.py](#41-configpy)
   - [crawler.py](#42-crawlerpy)
   - [http.py](#43-httpy)
   - [robots.py](#44-robotspy)
   - [_routing.py](#45-_routingpy)
   - [document.py](#46-documentpy)
   - [id_store.py](#47-id_storepy)
   - [chunker.py](#48-chunkerpy)
   - [loops/discovery.py](#49-loopsdiscoverypy)
   - [loops/downloader.py](#410-loopsdownloaderpy)
   - [loops/text.py](#411-loopstextpy)
   - [clients/base_client.py](#412-clientsbase_clientpy)
   - [clients/arxiv/constants.py](#413-clientsarxivconstantspy)
   - [clients/arxiv/api.py](#414-clientsarxivapipy)
   - [clients/arxiv/extractors/html.py](#415-clientsarxivextractorshtmlpy)
   - [clients/arxiv/extractors/pdf.py](#416-clientsarxivextractorspdfpy)
   - [clients/arxiv/client.py](#417-clientsarxivclientpy)
5. [Librerías utilizadas](#5-librerías-utilizadas)
6. [Formato de IDs compuestos](#6-formato-de-ids-compuestos)
7. [Cómo añadir un nuevo cliente](#7-cómo-añadir-un-nuevo-cliente)

---

## 1. Visión general

El módulo `crawler` es el responsable de la **adquisición de documentos científicos** desde fuentes externas. Opera de forma continua mediante tres hilos daemon que trabajan en pipeline:

```
Fuente externa
     │
     ▼
[DiscoveryLoop]  ──→  IdStore (CSV)
                            │
                            ▼
                    [DownloaderLoop]  ──→  Document (CSV + SQLite)
                                                    │
                                                    ▼
                                           [TextLoop]  ──→  Chunks (SQLite)
```

Cada hilo tiene **una única responsabilidad** y se comunica con los otros a través de estructuras de datos compartidas (thread-safe).

---

## 2. Estructura de ficheros

```
crawler/
│
├── __init__.py             Exports públicos del paquete.
├── config.py               CrawlerConfig — todos los parámetros de comportamiento.
├── crawler.py              Crawler — orquestador de los tres hilos.
├── http.py                 Utilidades HTTP compartidas (USER_AGENT, SSL, fetch_bytes).
├── robots.py               RobotsChecker — verificación de robots.txt con caché TTL.
├── _routing.py             Funciones internas de enrutamiento doc_id → cliente.
├── document.py             Document — dataclass + persistencia CSV.
├── id_store.py             IdStore — almacén thread-safe de IDs conocidos.
├── chunker.py              Algoritmo de fragmentación de texto (chunking).
│
├── loops/
│   ├── __init__.py
│   ├── discovery.py        DiscoveryLoop — descubre IDs nuevos en cada fuente.
│   ├── downloader.py       DownloaderLoop — descarga metadatos de artículos.
│   └── text.py             TextLoop — descarga texto completo y genera chunks.
│
└── clients/
    ├── __init__.py
    ├── base_client.py      BaseClient — interfaz abstracta para todos los clientes.
    └── arxiv/
        ├── __init__.py
        ├── constants.py    URLs, categorías y límites de arXiv.
        ├── api.py          Parseo de respuestas XML Atom de arXiv.
        ├── client.py       ArxivClient — HTTP + rate-limiting + download_text.
        └── extractors/
            ├── __init__.py
            ├── html.py     Extractor LaTeXML (HTML de arXiv).
            └── pdf.py      Extractor PDF con PyMuPDF.
```

---

## 3. Flujo completo de trabajo

### Paso 1 — Descubrimiento de IDs (`DiscoveryLoop`)

- Llama a `client.fetch_ids(max_results, start)` en cada cliente registrado.
- Los IDs locales devueltos se convierten en IDs compuestos (`"arxiv:2301.12345"`) con `client.make_doc_id()`.
- Los IDs nuevos se persisten en `IdStore` (CSV). Los duplicados se ignoran.
- Avanza el offset de paginación (`discovery_start`) en cada ciclo.
- Duerme `discovery_interval` segundos entre ciclos.

### Paso 2 — Descarga de metadatos (`DownloaderLoop`)

- Espera 10 s al arranque para dar tiempo al `DiscoveryLoop` a poblar el `IdStore`.
- Toma un lote de hasta `batch_size` IDs pendientes del `IdStore`.
- Agrupa los IDs por fuente (`source_name`) para hacer una única llamada por cliente.
- Llama a `client.fetch_documents(local_ids)` → lista de `Document`.
- Persiste cada `Document` en CSV (`documents.csv`) y en SQLite (tabla `documents`).
- Marca los IDs como `downloaded=True` en el `IdStore` solo si la fuente respondió correctamente.
- Si una fuente falla, sus IDs se reintentarán en el siguiente ciclo.
- Duerme `download_interval` segundos entre ciclos.

### Paso 3 — Descarga de texto y chunking (`TextLoop`)

- Espera 20 s al arranque para que SQLite tenga metadatos.
- Consulta SQLite (`get_pending_pdf_ids`) para obtener documentos sin texto aún.
- Para cada documento:
  1. Llama a `client.download_text(local_id, pdf_url=...)` → texto limpio.
  2. Llama a `chunker.make_chunks(text, chunk_size, overlap_sentences)` → lista de fragmentos.
  3. Persiste el texto completo en SQLite (`save_pdf_text`).
  4. Persiste los chunks en SQLite (`save_chunks`).
  5. Si falla, registra el error en SQLite (`save_pdf_error`) y continúa.
- Hace una pausa de `pdf_interval` segundos entre documentos (respeto al servidor).

---

## 4. Módulos — referencia detallada

### 4.1 `config.py`

**Propósito:** centralizar todos los parámetros de comportamiento del Crawler.

**Clase:** `CrawlerConfig` (dataclass)

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `discovery_interval` | `float` | `120.0` | Segundos entre ciclos de descubrimiento. |
| `download_interval` | `float` | `30.0` | Segundos entre ciclos de metadatos. |
| `pdf_interval` | `float` | `2.0` | Pausa en segundos entre documentos en el ciclo de texto. |
| `ids_per_discovery` | `int` | `100` | IDs a solicitar por ciclo de descubrimiento. |
| `batch_size` | `int` | `10` | Documentos por ciclo de metadatos. |
| `pdf_batch_size` | `int` | `10` | Documentos por ciclo de texto. |
| `chunk_size` | `int` | `1000` | Tamaño máximo de cada chunk en caracteres. |
| `overlap_sentences` | `int` | `2` | Oraciones compartidas entre chunks consecutivos. |
| `ids_csv` | `Path` | `data/ids_article.csv` | Ruta al CSV de IDs. |
| `documents_csv` | `Path` | `data/documents.csv` | Ruta al CSV de metadatos. |
| `discovery_start` | `int` | `0` | Offset de paginación (actualizado por `DiscoveryLoop`). |

---

### 4.2 `crawler.py`

**Propósito:** orquestar el ciclo de vida de los tres hilos daemon.

**Clase:** `Crawler`

#### `__init__(config, clients, client)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `config` | `CrawlerConfig \| None` | Configuración. Si es `None` usa defaults. |
| `clients` | `List[BaseClient] \| None` | Lista de clientes a usar. |
| `client` | `BaseClient \| None` | Alias de compatibilidad (un único cliente). Si se omiten ambos, usa `ArxivClient()`. |

**Métodos públicos:**

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `start()` | — | `None` | Arranca los tres hilos daemon. No bloquea. |
| `stop()` | — | `None` | Señaliza la parada y espera a que los hilos terminen (timeout 15 s). |
| `run_forever()` | — | `None` | Llama a `start()` y bloquea hasta `KeyboardInterrupt`. |

---

### 4.3 `http.py`

**Propósito:** centralizar los recursos HTTP compartidos por todo el paquete.

**Contenido:**

| Nombre | Tipo | Descripción |
|---|---|---|
| `USER_AGENT` | `str` | Cadena de identificación del crawler (`"SRI-Crawler/1.0"`). |
| `_SSL_CTX` | `ssl.SSLContext` | Contexto SSL compartido. Usa `certifi` si está disponible, degrada si no. |
| `fetch_bytes(url, timeout, accept)` | función | GET minimalista; devuelve `bytes` o `None` si falla. |

#### `fetch_bytes(url, timeout=15, accept="*/*") -> Optional[bytes]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `url` | `str` | URL a descargar. |
| `timeout` | `int` | Timeout en segundos. |
| `accept` | `str` | Valor del header `Accept`. |

**Salida:** `bytes` con el contenido de la respuesta, o `None` ante cualquier excepción.

> **Nota:** Para descargas grandes con rate-limiting y control de tamaño, usar `ArxivClient._get()`. `fetch_bytes` está pensado para robots.txt y otras peticiones de bajo volumen.

---

### 4.4 `robots.py`

**Propósito:** verificar si el crawler puede acceder a una URL según robots.txt.

**Librerías:** `urllib.robotparser` (stdlib), `threading` (stdlib).

**Clase:** `RobotsChecker`

El módulo expone un **singleton** `checker` que todos los clientes deben usar. No instanciar directamente.

#### `allowed(url, trusted_domains=frozenset()) -> bool`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `url` | `str` | URL a comprobar. |
| `trusted_domains` | `FrozenSet[str]` | Dominios con acceso garantizado por ToS. Si el host está aquí, devuelve `True` sin consultar robots.txt. |

**Salida:** `True` si el acceso está permitido; `True` también en caso de error de red (fail-open).

#### `crawl_delay(url) -> float`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `url` | `str` | URL para la que se quiere conocer el delay. |

**Salida:** `float` con los segundos de `Crawl-delay` de robots.txt, o `0.0` si no está definido.

> **Importante:** `crawl_delay` nunca hace bypass, ni para `trusted_domains`. El delay es una restricción de frecuencia independiente del acceso.

---

### 4.5 `_routing.py`

**Propósito:** funciones internas para traducir IDs compuestos al cliente correspondiente.

**Uso:** importado por los tres loops. No forma parte de la API pública del paquete.

#### `client_for(doc_id, client_map) -> Optional[BaseClient]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `doc_id` | `str` | ID compuesto `"source:local_id"`. |
| `client_map` | `Dict[str, BaseClient]` | Mapa `source_name → cliente`. |

**Salida:** el cliente registrado para la fuente del ID, o `None` si el ID es inválido o la fuente no está registrada.

#### `local_id(doc_id) -> str`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `doc_id` | `str` | ID compuesto `"source:local_id"`. |

**Salida:** la parte local del ID (sin prefijo de fuente).

---

### 4.6 `document.py`

**Propósito:** modelo de datos de un artículo descargado.

**Clase:** `Document` (dataclass)

| Campo | Tipo | Descripción |
|---|---|---|
| `doc_id` | `str` | ID compuesto `"source:local_id"`. |
| `title` | `str` | Título del artículo. |
| `authors` | `str` | Autores separados por comas. |
| `abstract` | `str` | Resumen del artículo. |
| `categories` | `str` | Categorías separadas por comas. |
| `published` | `str` | Fecha de publicación ISO-8601. |
| `updated` | `str` | Fecha de última actualización ISO-8601. |
| `pdf_url` | `str` | URL del PDF (puede estar vacía). |
| `fetched_at` | `str` | Timestamp UTC de descarga (auto-generado). |

**Métodos públicos:**

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `save(csv_path)` | `Path` | `None` | Añade el documento al CSV. Crea el fichero con cabecera si no existe. |
| `load_all(csv_path)` | `Path` | `List[Document]` | Carga todos los documentos del CSV. |
| `load_ids(csv_path)` | `Path` | `set[str]` | Devuelve el conjunto de `doc_id` ya persistidos (lectura rápida). |
| `from_dict(data)` | `dict` | `Document` | Reconstruye un `Document` desde un diccionario (acepta cabeceras antiguas). |

**Compatibilidad retroactiva:** la propiedad `arxiv_id` es un alias de `doc_id` para no romper módulos externos.

---

### 4.7 `id_store.py`

**Propósito:** almacén thread-safe de IDs conocidos respaldado en CSV.

**Clase:** `IdStore`

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `add_ids(ids)` | `List[str]` | `int` | Añade los IDs nuevos y devuelve cuántos se añadieron realmente. |
| `get_pending_batch(batch_size)` | `int` | `List[str]` | Devuelve hasta `batch_size` IDs pendientes de descarga de metadatos. |
| `mark_downloaded(ids)` | `List[str]` | `None` | Marca los IDs dados como descargados y persiste en CSV. |

**Propiedades:**

| Propiedad | Tipo | Descripción |
|---|---|---|
| `total` | `int` | Total de IDs conocidos. |
| `pending_count` | `int` | IDs pendientes de descarga. |
| `downloaded_count` | `int` | IDs ya descargados. |

**Thread-safety:** toda mutación está protegida por un `threading.Lock` interno. El CSV se reescribe completamente en cada mutación para garantizar durabilidad.

---

### 4.8 `chunker.py`

**Propósito:** fragmentación de texto en chunks con solapamiento semántico. Completamente agnóstico de fuente.

**Librerías:** `re` (stdlib).

**API pública:**

#### `make_chunks(text, chunk_size=1000, overlap_sentences=2) -> List[str]`

Punto de entrada único. Aplica limpieza y fragmenta el texto.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `text` | `str` | Texto crudo del documento. |
| `chunk_size` | `int` | Tamaño máximo de cada chunk en caracteres. |
| `overlap_sentences` | `int` | Oraciones de contexto compartidas entre chunks consecutivos del mismo párrafo. |

**Salida:** `List[str]` con los fragmentos listos para embedding. Los chunks menores de 100 caracteres se descartan.

#### `clean_text(text) -> str`

Normaliza el texto eliminando ruido tipográfico.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `text` | `str` | Texto crudo. |

**Salida:** `str` con saltos de línea normalizados, números de página eliminados y espacios múltiples colapsados.

> **Nota:** `clean_text` también es usado por los extractores de arXiv (`extractors/html.py` y `extractors/pdf.py`) para no duplicar lógica.

**Algoritmo de chunking:**
1. Divide el texto por párrafos (`\n\n`) — fronteras duras.
2. Dentro de cada párrafo, divide por oraciones (regex que respeta abreviaciones científicas).
3. Acumula oraciones hasta alcanzar `chunk_size` caracteres.
4. Al emitir un chunk, las últimas `overlap_sentences` oraciones se reutilizan como prefijo del siguiente para mantener contexto de transición.

---

### 4.9 `loops/discovery.py`

**Propósito:** descubrir IDs nuevos en cada fuente registrada.

**Clase:** `DiscoveryLoop`

#### `__init__(config, clients, id_store, stop)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `config` | `CrawlerConfig` | Configuración compartida. |
| `clients` | `List[BaseClient]` | Lista de clientes a interrogar. |
| `id_store` | `IdStore` | Almacén donde persistir los IDs nuevos. |
| `stop` | `threading.Event` | Evento de parada del hilo. |

#### `run() -> None`

Bucle principal del hilo. En cada iteración:
1. Recorre todos los clientes.
2. Llama a `client.fetch_ids(max_results, start)`.
3. Convierte IDs locales en compuestos con `client.make_doc_id()`.
4. Añade los nuevos al `IdStore`.
5. Avanza el offset `discovery_start`.
6. Espera `discovery_interval` segundos o hasta que `stop` se active.

---

### 4.10 `loops/downloader.py`

**Propósito:** descargar metadatos de artículos pendientes.

**Clase:** `DownloaderLoop`

#### `__init__(config, client_map, id_store, repo, db_path, stop)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `config` | `CrawlerConfig` | Configuración compartida. |
| `client_map` | `Dict[str, BaseClient]` | Mapa `source_name → cliente`. |
| `id_store` | `IdStore` | Almacén de IDs. |
| `repo` | módulo | `crawler_repository` (inyectado por `Crawler`). |
| `db_path` | `Path` | Ruta al fichero SQLite. |
| `stop` | `threading.Event` | Evento de parada. |

#### `run() -> None`

Bucle principal del hilo. En cada iteración:
1. Toma un lote de `batch_size` IDs pendientes del `IdStore`.
2. Agrupa los IDs por `source_name`.
3. Llama a `client.fetch_documents(local_ids)` por cada fuente.
4. Persiste los `Document` en CSV y SQLite.
5. Marca como descargados solo los IDs de fuentes que respondieron.
6. Espera `download_interval` segundos.

**Comportamiento ante fallos:** si una fuente no devuelve documentos, sus IDs no se marcan como descargados y se reintentarán en el siguiente ciclo.

---

### 4.11 `loops/text.py`

**Propósito:** descargar texto completo, fragmentarlo y persistir los chunks.

**Clase:** `TextLoop`

#### `__init__(config, client_map, repo, save_chunks, db_path, stop)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `config` | `CrawlerConfig` | Configuración compartida. |
| `client_map` | `Dict[str, BaseClient]` | Mapa `source_name → cliente`. |
| `repo` | módulo | `crawler_repository`. |
| `save_chunks` | `Callable` | `save_chunks(doc_id, chunks, db_path)`. |
| `db_path` | `Path` | Ruta al fichero SQLite. |
| `stop` | `threading.Event` | Evento de parada. |

#### `run() -> None`

Bucle principal del hilo. Para cada documento pendiente:
1. `client.download_text(local_id, pdf_url=...)` → texto limpio.
2. `chunker.make_chunks(text, chunk_size, overlap_sentences)` → `List[str]`.
3. `repo.save_pdf_text(doc_id, text)` y `save_chunks(doc_id, chunks)` → SQLite.
4. Ante error: `repo.save_pdf_error(doc_id, error_msg)` y continúa.

---

### 4.12 `clients/base_client.py`

**Propósito:** interfaz abstracta que todos los clientes deben implementar.

**Clase abstracta:** `BaseClient`

**Propiedades abstractas:**

| Propiedad | Tipo | Descripción |
|---|---|---|
| `source_name` | `str` | Identificador único de la fuente (`"arxiv"`, etc.). Alfanumérico, sin espacios. |
| `request_delay` | `float` | Pausa mínima en segundos entre peticiones HTTP. |
| `trusted_domains` | `FrozenSet[str]` | Dominios con acceso garantizado por ToS. |

**Métodos abstractos:**

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `fetch_ids(max_results, start)` | `int, int` | `List[str]` | Descubre IDs locales en la fuente. |
| `fetch_documents(local_ids)` | `List[str]` | `List[Document]` | Descarga metadatos. El `doc_id` de cada `Document` debe ser compuesto. |
| `download_text(local_id, **kwargs)` | `str` | `str` | Descarga y devuelve el texto completo limpio. |

**Métodos concretos:**

| Método | Entrada | Salida | Descripción |
|---|---|---|---|
| `make_doc_id(local_id)` | `str` | `str` | Construye el ID compuesto `"source:local_id"`. |
| `parse_doc_id(doc_id)` | `str` | `tuple[str, str]` | Descompone el ID compuesto en `(source, local_id)`. |

---

### 4.13 `clients/arxiv/constants.py`

**Propósito:** centralizar todos los literales del cliente arXiv.

| Constante | Valor | Descripción |
|---|---|---|
| `BASE_URL` | `https://export.arxiv.org/api/query` | Endpoint de la API Atom. |
| `ARXIV_HTML_URL` | template con `{local_id}` | URL del HTML LaTeXML. |
| `ARXIV_PDF_URL` | template con `{local_id}` | URL del PDF. |
| `ATOM_NS` | `http://www.w3.org/2005/Atom` | Namespace XML del feed Atom. |
| `AI_ML_CATEGORIES` | lista de strings | Categorías arXiv de IA/ML. |
| `DEFAULT_SEARCH_QUERY` | string | Query por defecto para `fetch_ids`. |
| `MAX_SIZE_MB` | `15` | Límite de descarga en MB. |
| `CHUNK_BYTES` | `65536` | Tamaño de cada lectura en streaming. |
| `LOG_EVERY_KB` | `512` | Frecuencia de log de progreso en KB. |

---

### 4.14 `clients/arxiv/api.py`

**Propósito:** parsear las respuestas XML Atom de la API de arXiv. No realiza ninguna petición HTTP.

**Librerías:** `xml.etree.ElementTree` (stdlib).

#### `parse_ids(xml_text) -> List[str]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `xml_text` | `str` | Texto XML completo de la respuesta. |

**Salida:** `List[str]` con IDs locales arXiv sin prefijo (p.ej. `["2301.12345"]`). Lista vacía si el XML es inválido.

#### `parse_entries(xml_text, make_doc_id) -> List[Document]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `xml_text` | `str` | Texto XML completo. |
| `make_doc_id` | `Callable[[str], str]` | Función que convierte un ID local en compuesto. |

**Salida:** `List[Document]` con metadatos completos y `doc_id` compuesto. Las entradas que fallen se omiten con warning.

---

### 4.15 `clients/arxiv/extractors/html.py`

**Propósito:** extraer texto limpio del HTML LaTeXML generado por arXiv.

**Librerías:** `html.parser.HTMLParser` (stdlib), `re` (stdlib).

#### `extract(raw) -> str`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `raw` | `bytes` | Bytes crudos del HTML descargado. |

**Salida:** `str` con el texto del artículo normalizado.

**Raises:** `ValueError` si el texto extraído es menor de 500 caracteres incluso con el fallback genérico.

**Estrategia:**
1. Parsea el HTML buscando el contenedor `ltx_document` (estructura LaTeXML).
2. Incluye: secciones, párrafos, headings.
3. Excluye: autores, bibliografía, figuras, tablas, ecuaciones, scripts, estilos.
4. Si no encuentra `ltx_document`, cae a un extractor genérico por regex.
5. Aplica `chunker.clean_text()` al resultado final.

---

### 4.16 `clients/arxiv/extractors/pdf.py`

**Propósito:** extraer texto de PDFs de arXiv.

**Librerías:** `PyMuPDF` (`fitz`) — **dependencia externa, no incluida en stdlib**.

#### `extract(pdf_bytes) -> str`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `pdf_bytes` | `bytes` | Bytes crudos del PDF descargado. |

**Salida:** `str` con el texto normalizado de todas las páginas concatenadas.

**Raises:**
- `ImportError`: si `pymupdf` no está instalado.
- `RuntimeError`: si PyMuPDF no puede procesar el fichero.

---

### 4.17 `clients/arxiv/client.py`

**Propósito:** gestionar la capa HTTP de arXiv con rate-limiting thread-safe.

**Clase:** `ArxivClient(BaseClient)`

#### `__init__(search_query, timeout=30)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `search_query` | `str` | Consulta Atom para `fetch_ids`. Por defecto: categorías AI/ML. |
| `timeout` | `int` | Timeout HTTP global en segundos. |

#### `_get(url, timeout, accept) -> bytes` *(interno)*

GET con rate-limiting thread-safe y comprobación de robots.txt.

- Comprueba `robots.allowed(url, trusted_domains)` antes de cada petición.
- Calcula `effective_delay = max(request_delay, robots.crawl_delay(url))`.
- Serializa el acceso entre hilos con `_rate_lock` (variable de clase compartida por todas las instancias).
- Descarga en streaming con chunks de `CHUNK_BYTES` bytes.
- Aborta si el fichero supera `MAX_SIZE_MB`.

#### `fetch_ids(max_results, start, sort_by, sort_order) -> List[str]`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `max_results` | `int` | `100` | Máximo de resultados. |
| `start` | `int` | `0` | Offset de paginación. |
| `sort_by` | `str` | `"submittedDate"` | Campo de ordenación. |
| `sort_order` | `str` | `"descending"` | Dirección de la ordenación. |

**Salida:** `List[str]` con IDs locales (sin prefijo). Lista vacía si la petición falla.

#### `fetch_documents(local_ids) -> List[Document]`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `local_ids` | `List[str]` | IDs locales arXiv. |

**Salida:** `List[Document]` con metadatos y `doc_id` compuesto (`"arxiv:…"`). Procesa en lotes de 20.

#### `download_text(local_id, **kwargs) -> str`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `local_id` | `str` | ID local arXiv. |
| `pdf_url` | `str` (kwarg) | URL alternativa del PDF (opcional). |

**Salida:** `str` con el texto completo limpio del artículo.

**Estrategia:**
1. Intenta HTML LaTeXML → `extractors.extract_html(raw)`.
2. Si falla o no está disponible → PDF → `extractors.extract_pdf(pdf_bytes)`.

**Raises:** `RuntimeError` si ambos métodos fallan.

---

## 5. Librerías utilizadas

| Librería | Origen | Dónde se usa | Propósito |
|---|---|---|---|
| `threading` | stdlib | `robots.py`, `id_store.py`, `clients/arxiv/client.py`, loops | Hilos y locks. |
| `urllib.request` | stdlib | `http.py`, `clients/arxiv/client.py` | Peticiones HTTP. |
| `urllib.robotparser` | stdlib | `robots.py` | Parseo de `robots.txt`. |
| `urllib.parse` | stdlib | `clients/arxiv/client.py` | Construcción de query strings. |
| `xml.etree.ElementTree` | stdlib | `clients/arxiv/api.py` | Parseo de XML Atom. |
| `html.parser.HTMLParser` | stdlib | `clients/arxiv/extractors/html.py` | Parseo de HTML LaTeXML. |
| `ssl` | stdlib | `http.py` | Contexto SSL para HTTPS. |
| `csv` | stdlib | `document.py`, `id_store.py` | Persistencia en CSV. |
| `dataclasses` | stdlib | `document.py`, `config.py` | Dataclasses. |
| `re` | stdlib | `chunker.py`, `clients/arxiv/extractors/html.py` | Expresiones regulares. |
| `pathlib` | stdlib | `document.py`, `id_store.py`, `config.py` | Rutas del sistema de ficheros. |
| `certifi` | **externa** | `http.py` | Certificados SSL actualizados. Opcional: degrada sin él. |
| `pymupdf` (`fitz`) | **externa** | `clients/arxiv/extractors/pdf.py` | Extracción de texto de PDFs. Necesario para el fallback PDF. |

---

## 6. Formato de IDs compuestos

Todos los IDs almacenados en `IdStore`, `Document` y SQLite siguen el formato:

```
{source_name}:{local_id}
```

**Ejemplos:**
```
arxiv:2301.12345
semantic_scholar:abc123def456
openalex:W2963403868
```

Este diseño permite al Crawler enrutar cada operación al cliente correcto (`_routing.client_for`) sin que ningún otro módulo del sistema necesite conocer el concepto de "fuente".

---

## 7. Cómo añadir un nuevo cliente

1. Crear el subpaquete `clients/nueva_fuente/` con al menos `client.py` y `__init__.py`.
2. Implementar una clase que extienda `BaseClient` con las tres propiedades abstractas y los tres métodos abstractos.
3. Registrar el cliente en `clients/__init__.py`.
4. Instanciar y pasar el nuevo cliente al `Crawler`:

```python
from crawler import Crawler, CrawlerConfig
from crawler.clients.arxiv    import ArxivClient
from crawler.clients.mi_fuente import MiFuenteClient

crawler = Crawler(
    config  = CrawlerConfig(ids_per_discovery=50),
    clients = [ArxivClient(), MiFuenteClient()],
)
crawler.run_forever()
```

El Crawler enruta automáticamente cada ID al cliente correcto gracias al prefijo de fuente en el ID compuesto. No es necesario modificar ningún otro fichero del paquete.
