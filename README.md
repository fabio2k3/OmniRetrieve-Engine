# OmniRetrieve-Engine

Motor de búsqueda semántica y generación aumentada por recuperación (RAG)
sobre artículos académicos de arXiv. Combina LSI, embeddings densos (FAISS),
reranking con CrossEncoder, búsqueda web y generación con Ollama.

---

## Estructura del proyecto

```
OmniRetrieve-Engine/
├── backend/
│   ├── crawler/        — descarga continua de metadatos y texto de arXiv
│   ├── database/       — esquema SQLite único y repositorios por módulo
│   ├── embedding/      — embeddings incrementales con sentence-transformers + FAISS
│   ├── eval/           — evaluación automática de retrieval y RAG
│   ├── indexing/       — índice invertido de frecuencias crudas (TF)
│   ├── orchestrator/   — coordinador de hilos y API pública
│   ├── qrf/            — refinamiento de consultas (LCE + BRF + Rocchio + MMR)
│   ├── rag/            — generación grounded con Ollama
│   ├── retrieval/      — LSI, FAISS, Hybrid y CrossEncoder
│   ├── tools/          — utilidades de mantenimiento y monitorización
│   ├── web_search/     — búsqueda web (Tavily + SiteSearcher académico)
│   └── requirements.txt
├── frontend/
│   └── app.py          — interfaz Streamlit
├── .streamlit/
│   └── config.toml     — configuración de Streamlit
└── .env                — variables de entorno (API keys)
```

Cada módulo tiene su propio `README_<modulo>.md` con detalles de arquitectura,
parámetros y ejemplos de uso.

---

## Requisitos previos

- **Python 3.10 o superior**
- **[Ollama](https://ollama.com/download)** — instalado y corriendo
- (Opcional) cuenta en [Tavily](https://tavily.com) para búsqueda web enriquecida

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/fabio2k3/OmniRetrieve-Engine.git
cd OmniRetrieve-Engine
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r backend/requirements.txt
pip install streamlit
```

### 4. Descargar el modelo LLM con Ollama

Asegúrate de que Ollama esté corriendo (icono en la bandeja del sistema), luego:

```bash
ollama pull llama3.2:3b
```

> Modelos alternativos (mayor calidad, más VRAM):
> ```bash
> ollama pull llama3.1:8b
> ollama pull mistral:7b
> ```
> Si cambias el modelo actualiza `rag_llm_model` en `backend/orchestrator/config.py`.

Verifica que Ollama responda en `http://localhost:11434` — debe mostrar `Ollama is running`.

### 5. Variables de entorno

Crea `.env` en la raíz del proyecto:

```env
# Búsqueda web enriquecida (opcional)
# Sin esta key el sistema usa SiteSearcher (DuckDuckGo restringido a dominios académicos)
TAVILY_API_KEY=tvly-tu-key-aqui
```

Obtén tu key gratuita en [app.tavily.com](https://app.tavily.com).

### 6. Configurar Streamlit

Crea `.streamlit/config.toml`:

```toml
[server]
fileWatcherType = "none"
```

---

## Ejecución

Desde la raíz del proyecto:

```bash
streamlit run frontend/app.py
```

O bien usando el lanzador del orquestador, que acepta flags para ajustar
la configuración sin editar código:

```bash
python -m backend.orchestrator.main
```

Ambos comandos abren la interfaz en `http://localhost:8501`.

### Opciones del lanzador

```bash
python -m backend.orchestrator.main --port 8080
python -m backend.orchestrator.main --no-browser
python -m backend.orchestrator.main --lsi-k 200 --lsi-interval 3600
python -m backend.orchestrator.main --embed-model all-mpnet-base-v2
python -m backend.orchestrator.main --web-threshold 0.4
python -m backend.orchestrator.main --lsi-min-df 5   # corpus pequeño
```

---

## Primer arranque

El primer arranque construye todos los modelos desde cero siguiendo este orden:

```
1. LSI rebuild    — 1–2 min la primera vez (construye el modelo semántico)
2. Embedding      — vectoriza chunks pendientes y construye el índice FAISS
3. QRF + RAG      — carga los pipelines de búsqueda y generación
4. Crawler        — descarga artículos de arXiv en segundo plano
5. Indexing       — indexa nuevos documentos en el índice TF
```

El **sidebar** muestra el estado de cada componente en tiempo real.
En arranques posteriores el modelo LSI carga desde disco en segundos.

---

## Modelos utilizados

| Componente | Modelo | Descarga |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Automática (sentence-transformers) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Automática (sentence-transformers) |
| Generación LLM | `llama3.2:3b` | Manual: `ollama pull llama3.2:3b` |

---

## Tests

```bash
# Todos los tests (excluye tests de red que requieren arXiv)
pytest backend/tests/ -v -m "not network"

# Por módulo
pytest backend/tests/crawler/     -v -m "not network"
pytest backend/tests/embedding/   -v
pytest backend/tests/indexing/    -v
pytest backend/tests/retrieval/   -v
pytest backend/tests/qrf/         -v
pytest backend/tests/rag/         -v
pytest backend/tests/eval/        -v
pytest backend/tests/integration/ -v

# Con cobertura
pytest backend/tests/ -v -m "not network" --cov=backend --cov-report=term-missing
```

---

## Agregar Ollama al PATH (solo Windows)

Si `ollama pull` devuelve *"El término 'ollama' no se reconoce..."*:

1. Inicio → busca *"Variables de entorno del sistema"*
2. Variables del sistema → `Path` → **Editar** → **Nuevo**:
   ```
   C:\Users\<tu-usuario>\AppData\Local\Programs\Ollama
   ```
3. Acepta y reabre la terminal.

---

## Resolución de problemas

**`No module named 'backend'`**  
Lanza siempre desde la raíz del proyecto, no desde subcarpetas.

**`Search unavailable: Orchestrator not ready`**  
El modelo LSI aún está cargando. Espera a que el sidebar muestre 🟢 en LSI.

**`RAG generation falló`**  
Verifica que Ollama esté corriendo (`http://localhost:11434`) y que el modelo
esté descargado (`ollama list`).

**Advertencias `Accessing __path__` o `No module named 'torchvision'`**  
Son inofensivas. El archivo `.streamlit/config.toml` con `fileWatcherType = "none"` las suprime.

**`Vocabulario vacío tras filtrado (min_df=20)`**  
El corpus es demasiado pequeño para los filtros por defecto.
Usa `--lsi-min-df 1` hasta tener al menos 500 documentos indexados.