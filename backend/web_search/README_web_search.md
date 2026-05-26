# OmniRetrieve — Módulo `web_search`

Amplía la recuperación activándose automáticamente cuando los resultados
del retriever local no superan el umbral de relevancia. Consulta la web
vía **Tavily** (motor principal) o **SiteSearcher** (fallback académico con
DuckDuckGo restringido a dominios de confianza), guarda los resultados en
la BD y los combina con los locales para el pipeline RAG.

---

## Estructura de archivos

```
backend/web_search/
├── pipeline.py          ← WebSearchPipeline: orquestador principal
├── sufficiency.py       ← SufficiencyChecker: decide si activar búsqueda web
├── searcher.py          ← WebSearcher: Tavily + SiteSearcher como fallback
├── site_searcher.py     ← SiteSearcher: DuckDuckGo restringido a dominios académicos
├── sites.py             ← DEFAULT_SEED_DOMAINS y get_site_filter()
├── web_repository.py    ← persiste resultados en web_search_results
└── __init__.py          ← exports: WebSearchPipeline, WebSearcher, SufficiencyChecker
```

---

## Instalación

```bash
pip install tavily-python duckduckgo-search python-dotenv
```

| Paquete | Para qué |
|---|---|
| `tavily-python` | Motor de búsqueda principal (requiere `TAVILY_API_KEY`) |
| `duckduckgo-search` | Usado por `SiteSearcher` como fallback académico |
| `python-dotenv` | Lee `TAVILY_API_KEY` del archivo `.env` |

Crear `.env` en la raíz del proyecto:
```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Flujo

```
retriever_results (list con 'score')
        ↓
SufficiencyChecker.is_sufficient()
   ├── SÍ  → devuelve retriever_results sin cambios
   └── NO  → WebSearcher.search(query)
               ├── Tavily API     ← intenta primero
               └── SiteSearcher  ← fallback si Tavily falla o sin API key
                   (DuckDuckGo restringido a dominios académicos de sites.py)
                         ↓
              save_web_results() → tabla web_search_results
                         ↓
              normalizar al formato del retriever
                         ↓
              combinar: local + web → lista unificada
```

---

## Uso programático

```python
from backend.web_search.pipeline import WebSearchPipeline

pipeline = WebSearchPipeline(
    threshold=0.15,       # score mínimo para considerar un resultado relevante
    min_docs=1,           # docs mínimos que deben superar el threshold
    max_results=5,        # máximo de resultados a pedir a Tavily/SiteSearcher
    use_fallback=True,    # activar SiteSearcher si Tavily no está disponible
    seed_domains=None,    # None → usa DEFAULT_SEED_DOMAINS de sites.py
    fetch_content=True,   # SiteSearcher descarga el contenido completo de la página
)

output = pipeline.run(
    query="fairness in machine learning",
    retriever_results=lsi_results,   # list[dict] con clave 'score'
)
```

`run()` devuelve siempre un `dict` con estas claves:

| Clave | Tipo | Descripción |
|---|---|---|
| `results` | `list[dict]` | Lista combinada (locales + web si se activó) |
| `web_activated` | `bool` | `True` si se activó la búsqueda web |
| `web_results` | `list[dict]` | Solo los resultados web normalizados |
| `reason` | `str` | Explicación de la decisión de suficiencia |
| `query` | `str` | Query original |

Los resultados web normalizados tienen el mismo formato que los locales:
`score`, `title`, `authors` (`"Web Search"`), `abstract` (contenido),
`url`, `source` (`"web"` o `"web_search"`).

---

## `SufficiencyChecker`

Criterio: hay suficiente información si al menos `min_docs` resultados
superan `threshold`.

```python
from backend.web_search.sufficiency import SufficiencyChecker

checker = SufficiencyChecker(threshold=0.15, min_docs=1)
if not checker.is_sufficient(results):
    print(checker.get_reason(results))
```

Acepta tanto `list[dict]` (con clave `score`) como `list[RetrievalResult]`
(con atributo `.score`).

---

## `WebSearcher` + `SiteSearcher`

`WebSearcher` prueba Tavily primero. Si no hay API key o falla, activa
`SiteSearcher` automáticamente.

```python
from backend.web_search.searcher import WebSearcher

searcher = WebSearcher(
    api_key="tvly-...",    # o None para leer del .env
    max_results=5,
    search_depth="basic",  # "basic" | "advanced" (Tavily)
    use_fallback=True,     # activar SiteSearcher si Tavily no está disponible
    seed_domains=None,     # None → DEFAULT_SEED_DOMAINS
    fetch_content=True,
)
results = searcher.search("attention mechanisms transformers")
```

`SiteSearcher` construye queries restringidas a dominios académicos:

```
"{query} (site:semanticscholar.org OR site:paperswithcode.com OR ...)"
```

El campo `source` en los resultados indica el origen:
- `"web"` → Tavily
- `"web_search"` → SiteSearcher

---

## `sites.py` — dominios semilla

Define los dominios académicos donde busca `SiteSearcher`.
Organizados en cuatro categorías:

| Categoría | Ejemplos |
|---|---|
| `ACADEMIC_SEARCH` | semanticscholar.org, paperswithcode.com, openreview.net, aclanthology.org |
| `CONFERENCE_PROCEEDINGS` | proceedings.neurips.cc, proceedings.mlr.press, dl.acm.org |
| `AI_ETHICS_RESEARCH` | ainowinstitute.org, algorithmwatch.org, fairmlbook.org |
| `SPECIALIZED_PUBLICATIONS` | distill.pub, technologyreview.com, nature.com |

Para añadir dominios propios pasar `seed_domains=[...]` a `WebSearchPipeline`
o a `WebSearcher`.

---

## Almacenamiento de resultados web

Los resultados se persisten en `web_search_results` (tabla separada de
`documents`). Esta separación evita contaminar el corpus científico con
resultados web. La tabla sirve para:
- Auditoría y monitoreo (`inspect_db --web N`)
- Caché de URLs ya visitadas (`get_cached_result(url)`)

Los resultados **no se indexan** en el índice TF ni en el modelo LSI.

---

## CLI

```bash
python -m backend.web_search.pipeline \
  --query "fairness in machine learning" \
  --threshold 0.15 \
  --top 5 \
  --depth basic \
  --no-fallback        # desactivar SiteSearcher
```

---

## Tests

Los tests están en `backend/tests/test_web_search/`.

```bash
pytest backend/tests/test_web_search/ -v

pytest backend/tests/test_web_search/test_sufficiency.py      -v
pytest backend/tests/test_web_search/test_searcher.py         -v
pytest backend/tests/test_web_search/test_pipeline.py         -v
```

### Qué cubre cada archivo

| Archivo | Qué verifica |
|---|---|
| `test_sufficiency.py` | `is_sufficient()`, `get_reason()`, con `list[dict]` y `list[RetrievalResult]` |
| `test_searcher.py` | Tavily mock, normalización de resultados, fallback a SiteSearcher |
| `test_pipeline.py` | Suficiencia → sin búsqueda; insuficiencia → activación + combinación + claves del dict devuelto |

Los tests usan `patch.object(pipeline.searcher, "search")` para evitar
llamadas reales a Tavily o DuckDuckGo.