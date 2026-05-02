# OmniRetrieve — Interfaz de usuario (Streamlit)

Aplicación web interactiva para búsqueda semántica y preguntas contextuales sobre la colección de papers de IA y Ética. Proporciona dos modos principales: **Search** (recuperación híbrida local + web) y **Ask AI** (respuesta generativa con RAG y enriquecimiento opcional desde la web). Es el punto de entrada visual del sistema OmniRetrieve.

---

## Estructura de archivos

frontend/
├── app_advanced.py   ← aplicación principal Streamlit (todo el código)
└── (no assets externos: CSS inline + fuentes Google)

La UI **no contiene lógica de recuperación ni modelos**. Se conecta al backend exclusivamente a través de tres funciones marcadas en el código con `← CONECTAR`.

---

## Instalación y dependencias

### Requisitos previos

- Python 3.10+
- El **backend completo** debe estar instalado y funcionando (índice LSI construido, base de datos con chunks, etc.)
- (Opcional) API key de búsqueda web en `.env` para `WebSearchPipeline`

### Dependencias específicas de la UI

pip install streamlit

El resto de paquetes (`sentence-transformers`, `faiss-cpu`, `numpy`, `sqlite3`, etc.) ya son requeridos por el backend.

---

## Ejecución

Desde la raíz del proyecto:

streamlit run frontend/app_advanced.py

La aplicación abrirá automáticamente en el puerto por defecto (`8501`). Se puede cambiar el puerto con `--server.port`.

---

## Modos de uso

La interfaz tiene dos modos, seleccionables mediante botones en la parte superior:

Modo | Acción | Backend utilizado
-----|--------|------------------
🔍 Search | Búsqueda de papers por similitud (LSI) + resultados web complementarios | LSIRetriever + WebSearchPipeline
💬 Ask AI | Genera una respuesta fundamentada en fragmentos relevantes (RAG) + enriquece opcionalmente con web | RAGPipeline + WebSearchPipeline

Ambos modos ejecutan automáticamente una búsqueda web, si el pipeline web está disponible, para completar o enriquecer los resultados sin interrumpir la experiencia principal.

---

## Flujo de trabajo

### 1) Modo Search

1. El usuario escribe una consulta, por ejemplo: fairness in machine learning.
2. Se llama a `retriever.retrieve(query, top_n=10)` y se obtienen papers locales ordenados por score.
3. Automáticamente se invoca `WebSearchPipeline.run(query, local_results)`.
4. Se fusionan resultados locales y web, si existen.
5. Los resultados provenientes de web se marcan con `source: "web"`.
6. Se muestran:
   - barra de información con número de resultados, tiempo en ms e indicador de “Web sources included”;
   - tarjetas de cada resultado, con etiqueta “Research paper” o “Web” y enlace al artículo.

### 2) Modo Ask AI

1. El usuario escribe una pregunta, por ejemplo: How does bias affect LLM fairness?
2. Se llama a `rag.ask(query, top_k=10, candidate_k=50, max_chunks=5, max_chars=400)`.
3. Internamente se recuperan fragmentos relevantes y se genera una respuesta con un modelo generativo.
4. Los fragmentos usados (`sources`) se extraen del resultado del RAG.
5. Se lanza `WebSearchPipeline.run(query, normalized_sources)` para obtener contenido web complementario.
6. Se muestra:
   - caja de respuesta con el texto generado y etiquetas de fuentes locales;
   - aviso “Web sources included” si se activó la web;
   - lista de papers relacionados, combinando fuentes del RAG y resultados web, hasta 8.

---

## Puntos de conexión con el backend

Las funciones `@st.cache_resource`:

- load_retriever()
- load_rag()
- load_web()

Si `load_web()` falla, la app sigue funcionando sin la parte web.

---

## Funciones auxiliares

- `_normalize(r)`: estandariza resultados
- `render_result(r)`: renderiza tarjetas

---

## Gestión de estado

Solo se usa `mode` en `st.session_state`.

---

## Manejo de errores

Errores en retriever/rag detienen la app. Web es opcional.

---

## Interfaz visual

Diseño oscuro, tarjetas, tipografías Syne y DM Mono, totalmente responsiva.

---

## Personalización

Editar CSS, parámetros de top_k/top_n o desactivar web.

---

## Notas

Arquitectura modular, stateless, optimizada con cache.

---

## Tests

streamlit run frontend/app_advanced.py
