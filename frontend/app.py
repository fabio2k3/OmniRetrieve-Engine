"""
app.py
======
OmniRetrieve — Interfaz de usuario conectada al Orquestador.

Ejecutar (desde la raíz del proyecto):
    streamlit run frontend/app.py

ARQUITECTURA
------------
Este fichero es el punto de entrada y coordinador principal. Toda la lógica
de presentación está delegada en los módulos de ``frontend/components/``:

    components/styles.py  → inject_styles()
    components/sidebar.py → render_sidebar()
    components/results.py → render_result(), render_paginated()

PUNTO DE CONEXIÓN ÚNICO
------------------------
``get_orchestrator()`` crea y arranca el Orchestrator una sola vez por
proceso gracias a ``@st.cache_resource``. Todos los hilos daemon (crawler,
LSI, FAISS, embedding, QRF/RAG) se inician automáticamente en el primer uso.

FLUJOS DE CONSULTA
------------------
  Modo Search  → orc.query(text)         LSI local (rápido)
  Modo Ask AI  → orc.pipeline_ask(text)  QRF expand → Hybrid → Web → Rerank → RAG
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Asegurar que el paquete 'backend' sea importable ─────────────────────────
def _find_backend_root() -> Path | None:
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / "backend" / "__init__.py").exists():
            return p
        if (p / "backend" / "backend" / "__init__.py").exists():
            return p / "backend"
        p = p.parent
    return None

_BACKEND = _find_backend_root()
if _BACKEND and str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniRetrieve",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Componentes UI ────────────────────────────────────────────────────────────
# Las importaciones de componentes van DESPUÉS de set_page_config
from components.styles  import inject_styles
from components.sidebar import render_sidebar
from components.results import render_result, render_paginated, normalize_result

inject_styles()


# ═══════════════════════════════════════════════════════════════════════════════
# CONEXIÓN AL ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_orchestrator():
    """
    Crea y arranca el Orchestrator una sola vez por proceso.

    Returns
    -------
    tuple[Orchestrator | None, str | None]
        (orchestrator, None) en caso de éxito.
        (None, mensaje_error) si la inicialización falla.
    """
    try:
        import os, json
        from backend.orchestrator.orchestrator import Orchestrator
        from backend.orchestrator.config import OrchestratorConfig

        raw = os.environ.get("OMNIRETRIEVE_CONFIG")
        if raw:
            overrides = json.loads(raw)
            cfg = OrchestratorConfig(
                **{k: v for k, v in overrides.items() if hasattr(OrchestratorConfig(), k)}
            )
        else:
            cfg = OrchestratorConfig()

        orc = Orchestrator(cfg)
        orc.start()
        return orc, None
    except Exception as e:
        return None, str(e)


# ── Arranque inmediato del Orchestrator ───────────────────────────────────────
_orc, _orc_err = get_orchestrator()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    render_sidebar(_orc, _orc_err)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_rag_output(raw) -> tuple[str, list]:
    """
    Extrae (answer, sources) de la salida de pipeline_ask() de forma defensiva.

    Tolera dicts estándar, claves alternativas, raw None/str y otros tipos.
    """
    if raw is None:
        return "No answer generated.", []
    if isinstance(raw, str):
        return raw or "No answer generated.", []
    if not isinstance(raw, dict):
        return "No answer generated.", []

    answer = (
        raw.get("answer")
        or raw.get("response")
        or raw.get("text")
        or raw.get("result")
        or "No answer generated. Try rephrasing your question."
    )
    if not isinstance(answer, str):
        answer = str(answer)

    sources = raw.get("sources") or raw.get("references") or raw.get("docs") or []
    if not isinstance(sources, list):
        sources = []

    return answer, sources


def _run_web_search(query: str, base_results: list) -> tuple[bool, list, str]:
    """
    Ejecuta búsqueda web a través del Orchestrator (query_with_web).

    Returns
    -------
    (web_activated, web_results, elapsed_label)
    """
    try:
        orc, err = get_orchestrator()
        if err or orc is None:
            return False, [], ""
        t0  = time.monotonic()
        out = orc.query_with_web(query)
        elapsed     = f"{(time.monotonic() - t0) * 1000:.0f}ms"
        all_results = out.get("results", [])
        web_results = [r for r in all_results if r.get("source") in ("web", "web_fallback")]
        return out.get("web_activated", False), web_results, elapsed
    except Exception:
        return False, [], ""


def _reset_pagination() -> None:
    """Elimina todas las claves de paginación de la sesión."""
    for k in [k for k in st.session_state if k.startswith("page_")]:
        del st.session_state[k]


# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════════

_SESSION_DEFAULTS: dict = {
    "mode":          "search",
    "last_query":    None,
    "search_results": None,
    "ask_output":    None,
    "web_activated": None,
    "web_results":   None,
    "web_elapsed":   None,
}
for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⬡ &nbsp;AI Research Assistant</div>
        <div class="hero-logo">⬡</div>
        <div class="hero-title">OmniRetrieve</div>
        <div class="hero-desc">
            Explore AI &amp; Ethics research papers with intelligent retrieval,<br>
            contextual answers, and enriched web-assisted insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE TOGGLE
# ═══════════════════════════════════════════════════════════════════════════════

_, col_search, col_ask, _ = st.columns([2, 1, 1, 2])

with col_search:
    if st.button(
        "🔍  Search", use_container_width=True, key="btn_search",
        type="primary" if st.session_state.mode == "search" else "secondary",
    ):
        st.session_state.mode = "search"
        st.rerun()

with col_ask:
    if st.button(
        "💬  Ask AI", use_container_width=True, key="btn_ask",
        type="primary" if st.session_state.mode == "ask" else "secondary",
    ):
        st.session_state.mode = "ask"
        st.rerun()

mode = st.session_state.mode

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT DE CONSULTA
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

placeholder = (
    "Ask anything about AI & Ethics research…"
    if mode == "ask"
    else "Search papers on fairness, bias, transparency…"
)

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input("q", placeholder=placeholder, label_visibility="collapsed")
with col_btn:
    clicked = st.button(
        "Ask →" if mode == "ask" else "Go →",
        use_container_width=True, type="primary", key="btn_go",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if clicked and query.strip():

    # Resetear paginación si la query cambió
    if query.strip() != st.session_state.last_query:
        _reset_pagination()
        st.session_state.last_query = query.strip()

    orc, err = get_orchestrator()

    # ── SEARCH ────────────────────────────────────────────────────────────────
    if mode == "search":
        if err or orc is None:
            st.error(f"Search unavailable: {err or 'Orchestrator not ready'}")
            st.stop()
        if not orc._lsi_ready.is_set():
            st.warning("LSI model is still loading — try again in a moment.")
            st.stop()

        t0 = time.monotonic()
        with st.spinner("Searching papers…"):
            try:
                local_results = orc.query(query.strip(), top_n=10)
            except Exception as e:
                st.error(f"Search error: {e}")
                st.stop()
        search_elapsed = (time.monotonic() - t0) * 1000

        with st.spinner("Checking web sources…"):
            web_activated, web_results, web_elapsed = _run_web_search(
                query.strip(), local_results
            )

        all_results = local_results + web_results if web_activated else local_results

        # Info bar
        web_time_html = (
            f"<div class='info-item'>web <b>{web_elapsed}</b></div>"
            if web_activated and web_elapsed else ""
        )
        web_tag_html = (
            "<div class='info-item'>⬡ <b>Web sources included</b></div>"
            if web_activated else ""
        )
        st.markdown(
            f"""
            <div class="info-bar">
                <div class="info-item"><b>{len(all_results)}</b> results</div>
                <div class="info-item">search <b>{search_elapsed:.0f}ms</b></div>
                {web_time_html}{web_tag_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if all_results:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            render_paginated(all_results, page_key=query.strip())
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-icon">⬡</div>
                    <div class="empty-hint">
                        No results found.<br>
                        Try different keywords or switch to <i>Ask AI</i> mode.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── ASK AI ────────────────────────────────────────────────────────────────
    else:
        if err or orc is None:
            st.error(f"AI assistant unavailable: {err or 'Orchestrator not ready'}")
            st.stop()
        if not orc._pipeline_ready.is_set():
            st.warning(
                "Pipeline is still warming up (FAISS + LSI + QRF + RAG). "
                "Check the sidebar for current status."
            )
            st.stop()

        with st.spinner("Building answer…"):
            try:
                raw_out = orc.pipeline_ask(query.strip())
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        answer, sources = _safe_rag_output(raw_out)
        web_activated   = (
            raw_out.get("web_activated", False)
            if isinstance(raw_out, dict) else False
        )

        if web_activated:
            st.markdown(
                '<div class="web-notice">⬡ &nbsp; Web sources included for a more complete answer</div>',
                unsafe_allow_html=True,
            )

        chips = "".join(
            f'<span class="source-chip">'
            f'{s.get("title", s.get("arxiv_id", "Source"))[:48]}'
            f'</span>'
            for s in sources
        )

        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-label">⬡ &nbsp; Answer</div>
                <div class="answer-text">{answer}</div>
                <div class="answer-sources">
                    <span class="sources-label">Sources &nbsp;</span>
                    {chips or '<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:var(--text-muted)">No sources available</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_docs = [normalize_result(s) for s in sources]
        if all_docs:
            st.markdown('<div class="section-label">Related papers</div>', unsafe_allow_html=True)
            render_paginated(all_docs, page_key=f"ask_{query.strip()}")


elif clicked and not query.strip():
    st.warning("Please enter a query.")


# ── Estado inicial (sin consulta activa) ──────────────────────────────────────
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-hint">
                Try searching for:<br><br>
                <i>fairness in machine learning</i> &nbsp;·&nbsp; <i>bias in NLP models</i><br>
                <i>AI transparency and accountability</i> &nbsp;·&nbsp; <i>explainability methods</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )