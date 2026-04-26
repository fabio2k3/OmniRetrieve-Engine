# OmniRetrieve — Módulo de Búsqueda Web

Módulo de búsqueda web que amplía la recuperación de información activándose automáticamente cuando los documentos locales no son suficientes para responder una consulta. Usa **Tavily** como motor de búsqueda principal y **DuckDuckGo** como respaldo automático. Los documentos encontrados se persisten en la base de datos y se indexan para uso futuro.

---

## Estructura de archivos

```
backend/web_search/
├── searcher.py          ← cliente Tavily con fallback a DuckDuckGo
├── fallback_searcher.py ← buscador de respaldo usando DuckDuckGo
├── sufficiency.py       ← decide si activar la búsqueda web
├── web_repository.py    ← persiste documentos web en la BD
├── pipeline.py          ← orquestador, punto de entrada principal
└── __init__.py          ← exports públicos
```

---

## Instalación

```bash
pip install tavily-python python-dotenv duckduckgo-search
```

| Paquete | Para qué |
|---|---|
| `tavily-python` | Cliente oficial de la API de Tavily |
| `python-dotenv` | Carga la API key desde el archivo `.env` |
| `duckduckgo-search` | Búsqueda de respaldo sin API key |

---

## Configuración

Crea un archivo `.env` en la raíz del proyecto:

```
TAVILY_API_KEY=tvly-tu-key-aqui
```

> **Importante:** Añade `.env` al `.gitignore`. Nunca subas tu API key al repositorio.

```bash
echo ".env" >> .gitignore
```

---

## Flujo del módulo

```
retriever_results  (list[dict] con score)
        ↓
SufficiencyChecker → ¿hay docs locales suficientes?
    │
    ├── SÍ → devuelve resultados locales sin cambios
    │
    └── NO → WebSearcher.search(query)
                │
                ├── Tavily OK  → resultados Tavily   (source="web")
                └── Tavily FAIL → DuckDuckGoSearcher  (source="web_fallback")
                        ↓
              web_repository.save_web_results()
                        ↓
              IndexingPipeline (indexación automática)
                        ↓
              combina resultados locales + web
                        ↓
              devuelve lista combinada → módulo RAG
```

---

## Descripción de cada archivo

### `sufficiency.py`

Decide si los documentos recuperados por el retriever son suficientes para responder la consulta o si es necesario activar la búsqueda web.

**Criterio:** se considera suficiente si al menos `min_docs` documentos superan el umbral de score `threshold`.

```python
from backend.web_search.sufficiency import SufficiencyChecker

checker = SufficiencyChecker(threshold=0.15, min_docs=1)

if not checker.is_sufficient(retriever_results):
    # activar búsqueda web

# Obtener explicación de la decisión
reason = checker.get_reason(retriever_results)
print(reason)
# → "Solo 0/1 documento(s) superan el umbral (threshold=0.15, mejor score=0.03). Activando búsqueda web."
```

| Parámetro | Default | Descripción |
|---|---|---|
| `threshold` | `0.15` | Score mínimo de similitud coseno para considerar un doc relevante |
| `min_docs` | `1` | Número mínimo de docs que deben superar el threshold |

> Los valores por defecto están calibrados para similitud coseno LSI. Pueden ajustarse en la configuración del orquestador.

---

### `fallback_searcher.py`

Buscador de respaldo usando DuckDuckGo. No requiere API key. Se activa automáticamente cuando Tavily falla o no está disponible.

Devuelve el mismo formato de resultados que `WebSearcher` para ser completamente intercambiable en el pipeline.

```python
from backend.web_search.fallback_searcher import DuckDuckGoSearcher

searcher = DuckDuckGoSearcher(max_results=5)
results  = searcher.search("fairness in machine learning")
```

| Campo devuelto | Tipo | Descripción |
|---|---|---|
| `title` | `str` | Título del resultado |
| `url` | `str` | URL de la página |
| `content` | `str` | Fragmento de texto del resultado |
| `score` | `float` | Valor neutro `0.5` — DuckDuckGo no provee score propio |
| `source` | `str` | Siempre `"web_fallback"` para identificar el origen |

---

### `searcher.py`

Cliente principal de búsqueda web. Usa Tavily como motor principal y activa DuckDuckGo automáticamente si Tavily falla.

Lee la API key del archivo `.env` si no se pasa como parámetro.

```python
from backend.web_search.searcher import WebSearcher

# Lee TAVILY_API_KEY del .env automáticamente
searcher = WebSearcher()
results  = searcher.search("bias detection NLP", max_results=5)

# Pasar la key explícitamente (útil para tests)
searcher = WebSearcher(api_key="tvly-...")

# Desactivar fallback (no recomendado)
searcher = WebSearcher(use_fallback=False)
```

| Parámetro | Default | Descripción |
|---|---|---|
| `api_key` | `None` | Key de Tavily. Si es `None` la lee del `.env` |
| `max_results` | `5` | Máximo de resultados por búsqueda |
| `search_depth` | `"basic"` | `"basic"` (rápido) o `"advanced"` (más preciso, más créditos) |
| `use_fallback` | `True` | Si `True`, usa DuckDuckGo cuando Tavily falla |

**Comportamiento ante fallos:**

| Escenario | Resultado |
|---|---|
| Tavily OK | Devuelve resultados con `source="web"` |
| Tavily FAIL + `use_fallback=True` | Activa DuckDuckGo, devuelve `source="web_fallback"` |
| Tavily FAIL + `use_fallback=False` | Devuelve lista vacía |
| Ambos fallan | Devuelve lista vacía sin romper el sistema |

---

### `web_repository.py`

Persiste en la base de datos los documentos encontrados por la búsqueda web para que estén disponibles en futuras consultas sin necesidad de volver a buscar.

Los documentos web se guardan en la tabla `documents` con un ID sintético basado en la URL (`web_<hash_md5>`), de forma que el indexador los procese igual que cualquier otro documento.

```python
from backend.web_search.web_repository import save_web_results, get_web_documents

# Guardar resultados web en la BD
saved = save_web_results(query="fairness ML", results=web_results)
print(f"{saved} documentos nuevos guardados")

# Consultar documentos web almacenados
docs = get_web_documents(limit=10)
```

**Estructura del documento guardado:**

| Campo | Valor |
|---|---|
| `arxiv_id` | `web_<hash>` — ID único generado desde la URL |
| `title` | Título del resultado web |
| `abstract` | Primeros 500 caracteres del contenido |
| `full_text` | Contenido completo |
| `categories` | `"web"` — identifica el origen |
| `pdf_downloaded` | `1` — marcado como listo para indexar |

Adicionalmente registra cada búsqueda en la tabla `web_search_log` para auditoría.

---

### `pipeline.py`

Orquestador del módulo. Une todos los componentes en un flujo completo y expone la clase `WebSearchPipeline` para uso programático y una CLI para pruebas desde terminal.

```python
from backend.web_search.pipeline import WebSearchPipeline

# Uso básico — lee TAVILY_API_KEY del .env
pipeline = WebSearchPipeline()
output   = pipeline.run(
    query="fairness in machine learning",
    retriever_results=lsi_results,
)

# Resultado
print(output["web_activated"])   # True / False
print(output["indexed"])         # docs web indexados automáticamente
for r in output["results"]:
    print(f"{r['score']:.4f}  {r['title']}  [{r.get('source')}]")
```

**Parámetros del constructor:**

| Parámetro | Default | Descripción |
|---|---|---|
| `api_key` | `None` | Key de Tavily. Si es `None` la lee del `.env` |
| `threshold` | `0.15` | Score mínimo para considerar un doc suficiente |
| `min_docs` | `1` | Docs mínimos que deben superar el threshold |
| `max_results` | `5` | Máximo de resultados a pedir a Tavily |
| `search_depth` | `"basic"` | Profundidad de búsqueda Tavily |
| `use_fallback` | `True` | Usar DuckDuckGo si Tavily falla |
| `auto_index` | `True` | Indexar automáticamente los docs web guardados |
| `db_path` | `DB_PATH` | Ruta a la base de datos SQLite |

**Formato del dict devuelto por `run()`:**

| Key | Tipo | Descripción |
|---|---|---|
| `results` | `list[dict]` | Lista combinada de docs locales + web ordenada por fuente |
| `web_activated` | `bool` | `True` si se activó la búsqueda web |
| `web_results` | `list[dict]` | Solo los resultados web (vacío si no se activó) |
| `reason` | `str` | Explicación de la decisión de suficiencia |
| `query` | `str` | Query original del usuario |
| `indexed` | `int` | Número de docs web indexados (`0` si `auto_index=False`) |

---

## Cómo ejecutar

### Uso programático (integración con el retriever)

```python
from backend.retrieval import LSIRetriever
from backend.web_search.pipeline import WebSearchPipeline

# Fase de recuperación local
retriever = LSIRetriever()
retriever.load()
lsi_results = retriever.retrieve("transformer attention mechanisms", top_n=10)

# Módulo de búsqueda web
pipeline = WebSearchPipeline()
output   = pipeline.run(
    query="transformer attention mechanisms",
    retriever_results=lsi_results,
)

# Pasar resultados al módulo RAG
rag_input = output["results"]
```

### CLI — prueba directa desde terminal

```bash
# Búsqueda básica (lee la key del .env)
python -m backend.web_search.pipeline --query "fairness in machine learning"

# Con parámetros personalizados
python -m backend.web_search.pipeline \
  --query "bias detection NLP" \
  --threshold 0.2        \
  --min-docs 2           \
  --top 8                \
  --depth advanced

# Desactivar fallback y auto-indexación
python -m backend.web_search.pipeline \
  --query "AI ethics" \
  --no-fallback       \
  --no-index
```

### Parámetros CLI disponibles

```bash
python -m backend.web_search.pipeline \
  --query      "consulta"     # consulta de búsqueda (obligatorio) \
  --api-key    "tvly-..."     # key de Tavily (opcional, lee del .env si no se pasa) \
  --threshold  0.15           # score mínimo para docs locales (default: 0.15) \
  --min-docs   1              # docs mínimos que deben superar threshold (default: 1) \
  --top        5              # máximo de resultados web (default: 5) \
  --depth      basic          # basic | advanced (default: basic) \
  --no-fallback               # desactiva DuckDuckGo como respaldo \
  --no-index                  # no indexa automáticamente los docs web \
  --db         ruta/a/db      # ruta a la BD (default: data/db/documents.db)
```

---

## Tablas de la base de datos

El módulo añade una tabla al esquema existente. Añadir en `schema.py` antes de los índices:

```sql
-- ── Módulo de búsqueda web ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS web_search_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    searched_at    TEXT    NOT NULL,
    query          TEXT    NOT NULL,
    results_found  INTEGER DEFAULT 0,   -- resultados devueltos por Tavily/DDG
    results_saved  INTEGER DEFAULT 0    -- documentos nuevos guardados en BD
);
```

---

## Tests

```bash
# Todos los tests del módulo
python -m pytest backend/tests/test_web_search/ -v

# Por archivo
python -m pytest backend/tests/test_web_search/test_sufficiency.py -v
python -m pytest backend/tests/test_web_search/test_searcher.py -v
python -m pytest backend/tests/test_web_search/test_fallback_searcher.py -v
python -m pytest backend/tests/test_web_search/test_pipeline.py -v
```

---

## Notas importantes

- **La API key nunca va en el código.** Siempre en `.env` y `.env` siempre en `.gitignore`.
- **DuckDuckGo no provee score propio.** Los resultados de fallback tienen `score=0.5` (valor neutro) para que el pipeline pueda tratarlos uniformemente.
- **Los docs web quedan indexados.** Gracias a `auto_index=True`, los documentos encontrados en la web se indexan en el siguiente ciclo y pasan a estar disponibles para futuras consultas locales sin necesidad de volver a buscar en la web.
- **El ID sintético es reproducible.** El `web_<hash_md5>` se genera desde la URL, por lo que la misma URL siempre produce el mismo ID y no se duplican documentos en la BD.
- **`search_depth="advanced"` gasta más créditos de Tavily.** Usar solo cuando se necesite mayor precisión.
