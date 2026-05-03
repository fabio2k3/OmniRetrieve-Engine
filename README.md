# OmniRetrieve-Engine

Motor de búsqueda semántica e IA generativa sobre artículos académicos de arXiv.  
Combina LSI, embeddings densos (FAISS), reranking con CrossEncoder, búsqueda web y RAG con Ollama.

---

## Estructura del proyecto

```
OmniRetrieve-Engine/
├── backend/
│   └── backend/
│       ├── crawler/          — descarga de metadatos y PDFs de arXiv
│       ├── database/         — esquema SQLite y repositorios
│       ├── embedding/        — embeddings incrementales + FAISS
│       ├── indexing/         — índice invertido de frecuencias crudas (TF)
│       ├── orchestrator/     — coordinador de todos los módulos
│       ├── qrf/              — expansión de consultas
│       ├── rag/              — generación con Ollama
│       ├── retrieval/        — LSI retriever, embedding retriever, hybrid retriever y reranker
│       ├── tools/            — utilidades de mantenimiento
│       └── web_search/       — búsqueda web (Tavily + DuckDuckGo fallback)
├── frontend/
│   └── frontend/
│       └── app.py            — interfaz Streamlit
├── .streamlit/
│   └── config.toml           — configuración de Streamlit
└── .env                      — variables de entorno (API keys)
```

Cada módulo tiene su propio README con detalles de arquitectura, parámetros y ejemplos de uso.

---

## Requisitos previos

- **Python 3.10 o superior**
- **[Ollama](https://ollama.com/download)** — app de escritorio instalada y corriendo
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

### 3. Instalar dependencias Python

```bash
pip install -r backend/backend/requirements.txt
pip install streamlit
```

### 4. Descargar el modelo LLM con Ollama

Asegúrate de que la app de escritorio de Ollama esté abierta (icono en la bandeja del sistema), luego ejecuta:

```bash
ollama pull llama3.2:3b
```

> **Modelos alternativos** (mayor calidad, más VRAM):
> ```bash
> ollama pull llama3.1:8b
> ollama pull mistral:7b
> ```
> Si cambias el modelo, actualiza `rag_llm_model` en `orchestrator/config.py`.

Verifica que Ollama esté respondiendo en: `http://localhost:11434`  
Debe mostrar: `Ollama is running`

### 5. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Búsqueda web enriquecida (opcional — sin esta key usa DuckDuckGo gratis)
TAVILY_API_KEY=tvly-tu-key-aqui
```

> Obtén tu key gratuita en [app.tavily.com](https://app.tavily.com).  
> Sin esta key el sistema funciona igual pero usa DuckDuckGo como fallback.

### 6. Configurar Streamlit

Crea el archivo `.streamlit/config.toml` en la raíz del proyecto:

```toml
[server]
fileWatcherType = "none"
```

---

## Agregar Ollama al PATH (solo Windows)

Si al ejecutar `ollama pull` recibes *"El término 'ollama' no se reconoce..."*:

1. Abre **Inicio** → busca *"Variables de entorno del sistema"*
2. **Variables de entorno** → selecciona `Path` en Variables del sistema → **Editar**
3. Clic en **Nuevo** → pega:
   ```
   C:\Users\<tu-usuario>\AppData\Local\Programs\Ollama
   ```
4. Acepta todo y **reabre la terminal**

---

## Ejecución

### Opción A — Lanzador principal (recomendado)

Desde la raíz del proyecto:

```bash
python -m backend.orchestrator
```

Abre Streamlit automáticamente en `http://localhost:8501`.

Opciones útiles:

```bash
python -m backend.orchestrator --port 8080
python -m backend.orchestrator --no-browser
python -m backend.orchestrator --lsi-interval 3600
python -m backend.orchestrator --lsi-k 200
```

### Opción B — Streamlit directo

```bash
streamlit run frontend/frontend/app.py
```

---

## Primer arranque

El primer arranque construye todos los modelos desde cero. El proceso sigue este orden:

```
1. LSI rebuild   — puede tardar 1-2 min la primera vez
2. Embedding     — genera embeddings pendientes + construye FAISS
3. QRF + RAG     — carga los pipelines de búsqueda y generación
4. Crawler       — descarga artículos de arXiv en segundo plano
5. Indexing      — indexa nuevos documentos en la BD
```

El **sidebar** muestra el estado de cada componente en tiempo real. En arranques posteriores, LSI carga el modelo guardado en disco en segundos.

---


## Modelos utilizados

| Componente | Modelo | Descarga |
|-----------|--------|----------|
| Embeddings | `all-MiniLM-L6-v2` | Automática (sentence-transformers) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Automática (sentence-transformers) |
| Generación LLM | `llama3.2:3b` | Manual: `ollama pull llama3.2:3b` |

---

## Resolución de problemas

**`No module named 'backend'`**  
Lanza siempre desde la raíz del proyecto, no desde subcarpetas.

**`Search unavailable: Orchestrator not ready`**  
El modelo LSI aún está cargando. Observa el sidebar — cuando LSI muestre 🟢 la búsqueda estará disponible.

**`RAG generation falló`**  
Verifica que Ollama esté corriendo (`http://localhost:11434`) y que el modelo esté descargado (`ollama list`).

**Advertencias `Accessing __path__` o `No module named 'torchvision'` en consola**  
Son inofensivas. El archivo `.streamlit/config.toml` con `fileWatcherType = "none"` las suprime.