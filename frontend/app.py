"""
app_advanced.py
===============

OmniRetrieve — Interfaz de usuario.

Ejecutar (desde la raíz del proyecto):
    streamlit run frontend/app_advanced.py

PUNTOS DE CONEXIÓN PARA EL ORQUESTADOR
---------------------------------------
Las tres funciones marcadas con  ← CONECTAR  son los únicos lugares donde
esta UI toca el backend. El orquestador debe asegurarse de que los módulos
correspondientes estén listos antes de que el usuario haga la primera consulta.

    load_retriever()  ← LSIRetriever cargado con modelo y BD
    load_rag()        ← RAGPipeline con retriever inyectado
    load_web()        ← WebSearchPipeline con API key del .env
"""

from __future__ import annotations

import time
import streamlit as st

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniRetrieve",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #050814;
    --card: rgba(14, 20, 38, 0.76);
    --border: rgba(148, 163, 184, 0.16);
    --accent: #2f80ff;
    --accent2: #00d4ff;
    --accent3: #2ee59d;
    --text: #f6f9ff;
    --text-muted: #9aa7ba;
    --radius: 18px;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.36);
    --shadow-soft: 0 10px 28px rgba(0, 0, 0, 0.22);
}

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Fondo ── */
.stApp {
    background:
        radial-gradient(ellipse 70% 50% at 15% 10%, rgba(47,128,255,0.07), transparent),
        radial-gradient(ellipse 55% 45% at 85% 8%,  rgba(0,212,255,0.05),   transparent),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(46,229,157,0.04),  transparent),
        #050814;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

/* Líneas de cuadrícula tenue — sólo en el tercio central */
.bg-grid {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 60% 55% at 50% 50%, black, transparent);
    -webkit-mask-image: radial-gradient(ellipse 60% 55% at 50% 50%, black, transparent);
}

/* Dos puntos de luz difusos — muy suaves */
.bg-glow {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
}

.bg-glow::before {
    content: "";
    position: absolute;
    width: 480px; height: 480px;
    top: -140px; right: -160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(47,128,255,0.09), transparent 68%);
    filter: blur(18px);
}

.bg-glow::after {
    content: "";
    position: absolute;
    width: 420px; height: 420px;
    bottom: -140px; left: -140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(46,229,157,0.07), transparent 68%);
    filter: blur(18px);
}

.block-container {
    position: relative; z-index: 1;
    padding-top: 2.2rem; padding-bottom: 2.8rem;
    max-width: 940px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 1.2rem 1rem 1.1rem;
    margin-bottom: 0.5rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    font-size: 0.72rem;
    color: rgba(219,234,254,0.7);
    margin-bottom: 1.15rem;
    letter-spacing: 0.04em;
}

.hero-logo {
    width: 72px; height: 72px;
    margin: 0 auto 1rem;
    display: flex; align-items: center; justify-content: center;
    border-radius: 20px; font-size: 1.9rem; color: white;
    background: linear-gradient(135deg, rgba(47,128,255,0.9), rgba(0,212,255,0.9));
    box-shadow: 0 14px 36px rgba(47,128,255,0.18);
    border: 1px solid rgba(255,255,255,0.10);
    position: relative; overflow: hidden;
}

.hero-logo::after {
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.16), transparent 52%);
    pointer-events: none;
}

.hero-title {
    font-size: 3rem; font-weight: 800;
    letter-spacing: -0.06em; line-height: 1; margin-bottom: 0.65rem;
    background: linear-gradient(90deg, #ffffff 10%, #dbeafe 40%, #60a5fa 72%, #22d3ee 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.hero-desc {
    max-width: 560px; margin: 0 auto;
    font-family: 'DM Mono', monospace; font-size: 0.82rem;
    line-height: 1.9; color: var(--text-muted);
}

/* ── Botones ── */
[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; border-radius: 999px !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important; padding: 0.7rem 1.4rem !important;
}
[data-testid="stButton"] > button:hover  { transform: translateY(-1px) !important; }
[data-testid="stButton"] > button:active { transform: scale(0.99) !important; }

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    box-shadow: 0 10px 22px rgba(47,128,255,0.22) !important;
    border-color: rgba(255,255,255,0.10) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 14px 28px rgba(0,212,255,0.20) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(16,22,40,0.80) !important;
    color: var(--text-muted) !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    backdrop-filter: blur(10px);
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(0,212,255,0.28) !important;
    color: var(--text) !important;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: rgba(12,18,36,0.86) !important;
    border: 1px solid rgba(148,163,184,0.14) !important;
    border-radius: 999px !important; color: var(--text) !important;
    font-family: 'Syne', sans-serif !important; font-size: 1rem !important;
    padding: 0.95rem 1.35rem !important; transition: all 0.2s ease !important;
    backdrop-filter: blur(12px);
}
[data-testid="stTextInput"] input:hover  { border-color: rgba(0,212,255,0.22) !important; }
[data-testid="stTextInput"] input:focus  {
    border-color: rgba(47,128,255,0.65) !important;
    box-shadow: 0 0 0 4px rgba(47,128,255,0.10) !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(154,167,186,0.65) !important; }
[data-testid="stTextInput"] label { display: none !important; }

/* ── Info bar ── */
.info-bar {
    display: flex; justify-content: center; gap: 1rem;
    padding: 0.7rem 1.1rem;
    background: rgba(12,18,36,0.70);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 999px; margin: 1.15rem 0; flex-wrap: wrap;
    backdrop-filter: blur(12px);
}
.info-item { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }
.info-item b { color: var(--text); }

/* ── Web notice ── */
.web-notice {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.65rem 1rem; margin: 0.8rem auto 1.15rem; width: fit-content;
    background: rgba(46,229,157,0.06);
    border: 1px solid rgba(46,229,157,0.16);
    border-radius: 999px;
    font-family: 'DM Mono', monospace; font-size: 0.71rem; color: #8ff0c8;
    backdrop-filter: blur(10px);
}

/* ── Answer box ── */
.answer-box {
    background: rgba(14,20,38,0.88);
    border: 1px solid rgba(47,128,255,0.20);
    border-radius: 20px; padding: 1.5rem 1.7rem;
    margin: 1.15rem 0; position: relative; overflow: hidden;
    box-shadow: var(--shadow); backdrop-filter: blur(14px);
}
.answer-box::before {
    content: ''; position: absolute; inset: 0 auto 0 0; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent3));
    opacity: 0.7;
}
.answer-label {
    position: relative; z-index: 1;
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    color: var(--accent2); text-transform: uppercase; letter-spacing: 0.16em;
    margin-bottom: 0.75rem; opacity: 0.85;
}
.answer-text { position: relative; z-index: 1; font-size: 0.97rem; line-height: 1.85; color: var(--text); }
.answer-sources {
    position: relative; z-index: 1;
    margin-top: 1rem; padding-top: 0.85rem;
    border-top: 1px solid rgba(148,163,184,0.10);
    display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center;
}
.sources-label {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em;
}
.source-chip {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    background: rgba(47,128,255,0.10); color: #d6e7ff;
    border: 1px solid rgba(47,128,255,0.18);
    border-radius: 999px; padding: 0.18rem 0.6rem; white-space: nowrap;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    color: rgba(154,167,186,0.6); text-transform: uppercase; letter-spacing: 0.16em;
    margin: 1.25rem 0 0.85rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0.10), transparent);
}

/* ── Result cards ── */
.result-card {
    background: rgba(13,19,36,0.80);
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: var(--radius); padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
    backdrop-filter: blur(10px);
}
.result-card::before {
    content: ''; position: absolute; inset: 0 auto 0 0; width: 3px;
    background: linear-gradient(180deg, rgba(47,128,255,0.6), rgba(46,229,157,0.5));
}
.result-card:hover {
    border-color: rgba(47,128,255,0.28);
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}
.result-card.is-web::before {
    background: linear-gradient(180deg, rgba(46,229,157,0.6), rgba(0,212,255,0.5));
}

.result-title   { font-size: 0.97rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; line-height: 1.35; }
.result-abstract { font-size: 0.82rem; color: rgba(154,167,186,0.85); line-height: 1.7; font-family: 'DM Mono', monospace; font-weight: 300; }
.result-footer  { display: flex; gap: 0.6rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; }

.result-tag {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    padding: 0.14rem 0.55rem; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.tag-local { background: rgba(47,128,255,0.10); color: #d6e7ff; border: 1px solid rgba(47,128,255,0.18); }
.tag-web   { background: rgba(46,229,157,0.08); color: #8ff0c8; border: 1px solid rgba(46,229,157,0.16); }

.result-link { font-family: 'DM Mono', monospace; font-size: 0.67rem; color: #93c5fd; text-decoration: none; }
.result-link:hover { text-decoration: underline; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 2.5rem 1rem;
    background: rgba(12,18,36,0.40);
    border: 1px dashed rgba(148,163,184,0.14);
    border-radius: 20px; backdrop-filter: blur(10px);
}
.empty-icon {
    width: 60px; height: 60px; border-radius: 18px;
    margin: 0 auto 0.85rem;
    display: grid; place-items: center; font-size: 1.75rem; color: #d6e7ff;
    background: rgba(47,128,255,0.10);
    border: 1px solid rgba(47,128,255,0.14);
}
.empty-hint { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--text-muted); line-height: 2; }
.empty-hint i { color: var(--accent2); font-style: normal; }

hr { border-color: rgba(148,163,184,0.10) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Fondo minimalista: sólo cuadrícula tenue + dos puntos de luz difusos
st.markdown(
    """
<div class="bg-grid"></div>
<div class="bg-glow"></div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTOS DE CONEXIÓN — el orquestador conecta aquí
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_retriever():
    """
    ← CONECTAR: retorna (LSIRetriever listo, None) o (None, mensaje_error).
    El retriever debe tener .retrieve(query: str, top_n: int) -> list[dict]
    donde cada dict contiene: score, arxiv_id, title, authors, abstract, url.
    """
    try:
        from backend.retrieval.lsi_retriever import LSIRetriever
        r = LSIRetriever()
        r.load()
        return r, None
    except FileNotFoundError:
        return None, "Modelo LSI no encontrado. Ejecuta: python -m backend.retrieval.build_lsi"
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_rag():
    """
    ← CONECTAR: retorna (RAGPipeline listo, None) o (None, mensaje_error).
    El pipeline debe tener .ask(query, top_k, candidate_k, max_chunks, max_chars)
    -> dict con claves: answer (str), sources (list[dict]).
    """
    try:
        from backend.rag.pipeline import RAGPipeline
        ret, err = load_retriever()
        if err:
            return None, err
        return RAGPipeline(retriever=ret), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_web():
    """
    ← CONECTAR: retorna (WebSearchPipeline listo, None) o (None, mensaje_error).
    El pipeline debe tener .run(query, retriever_results) -> dict con claves:
    results (list[dict]), web_activated (bool), web_results (list[dict]).
    Si falla, la app continúa sin búsqueda web (no es error crítico).
    """
    try:
        from backend.web_search.pipeline import WebSearchPipeline
        return WebSearchPipeline(), None
    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE RENDERIZADO
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(r) -> dict:
    """
    Garantiza que cualquier resultado sea un dict plano con las claves
    que necesita render_result().
    Cubre: dict del LSIRetriever, dict de build_sources() del RAG,
    y cualquier objeto con __dict__.
    """
    if not isinstance(r, dict):
        r = vars(r) if hasattr(r, "__dict__") else {}
    return {
        "title":    r.get("title") or r.get("document_title") or r.get("arxiv_id", "Untitled"),
        "abstract": r.get("abstract") or r.get("text", ""),
        "url":      r.get("url") or r.get("pdf_url", ""),
        "source":   r.get("source", "local"),
        "score":    float(r.get("score", 0.0)),
    }


def render_result(r: dict) -> None:
    """Renderiza una tarjeta de resultado (paper local o fuente web)."""
    is_web   = r.get("source", "local") in ("web", "web_fallback")
    card_cls = "result-card is-web" if is_web else "result-card"
    tag      = (
        '<span class="result-tag tag-web">Web</span>'
        if is_web else
        '<span class="result-tag tag-local">Research paper</span>'
    )
    url      = r.get("url", "")
    link     = f'<a class="result-link" href="{url}" target="_blank">↗ Read paper</a>' if url else ""
    abstract = (r.get("abstract") or "")[:220]
    if abstract:
        abstract += "…"

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div class="result-title">{r.get('title', 'Untitled')}</div>
            <div class="result-abstract">{abstract or 'No description available.'}</div>
            <div class="result-footer">{tag}{link}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SESIÓN Y MODO
# ═══════════════════════════════════════════════════════════════════════════════

if "mode" not in st.session_state:
    st.session_state.mode = "search"

# ── Hero ──────────────────────────────────────────────────────────────────────
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

# ── Mode toggle ───────────────────────────────────────────────────────────────
_, col_search, col_ask, _ = st.columns([2, 1, 1, 2])

with col_search:
    if st.button(
        "🔍  Search",
        use_container_width=True,
        type="primary" if st.session_state.mode == "search" else "secondary",
        key="btn_search",
    ):
        st.session_state.mode = "search"
        st.rerun()

with col_ask:
    if st.button(
        "💬  Ask AI",
        use_container_width=True,
        type="primary" if st.session_state.mode == "ask" else "secondary",
        key="btn_ask",
    ):
        st.session_state.mode = "ask"
        st.rerun()

mode = st.session_state.mode

# ── Input ─────────────────────────────────────────────────────────────────────
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
        use_container_width=True,
        type="primary",
        key="btn_go",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FLUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if clicked and query.strip():
    t0 = time.monotonic()

    # ── SEARCH ────────────────────────────────────────────────────────────────
    if mode == "search":
        with st.spinner("Searching…"):
            retriever, err = load_retriever()
            if err:
                st.error(f"⚠️ Search unavailable: {err}")
                st.stop()
            try:
                local_results = retriever.retrieve(query.strip(), top_n=10)
            except Exception as e:
                st.error(f"Search error: {e}")
                st.stop()

        # Web search automático (silencioso si falla)
        output = {"results": local_results, "web_activated": False, "web_results": []}
        try:
            pipeline, _ = load_web()
            if pipeline:
                output = pipeline.run(query.strip(), local_results)
        except Exception:
            pass

        elapsed = (time.monotonic() - t0) * 1000
        results = output["results"]

        st.markdown(
            f"""
            <div class="info-bar">
                <div class="info-item"><b>{len(results)}</b> results</div>
                <div class="info-item"><b>{elapsed:.0f}ms</b></div>
                {"<div class='info-item'>⬡ <b>Web sources included</b></div>" if output["web_activated"] else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if results:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            for r in results:
                render_result(_normalize(r))
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
        with st.spinner("Thinking…"):
            rag, err = load_rag()
            if err:
                st.error(f"⚠️ AI assistant unavailable: {err}")
                st.stop()
            try:
                out = rag.ask(
                    query=query.strip(),
                    top_k=10,
                    candidate_k=50,
                    max_chunks=5,
                    max_chars=400,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        sources = out.get("sources", [])

        # Web search automático sobre las fuentes del RAG (silencioso si falla)
        web_activated = False
        web_extra: list[dict] = []
        try:
            pipeline, _ = load_web()
            if pipeline:
                normalized_sources = [_normalize(s) for s in sources]
                web_out       = pipeline.run(query.strip(), normalized_sources)
                web_activated = web_out.get("web_activated", False)
                web_extra     = web_out.get("web_results", [])
        except Exception:
            pass

        if web_activated:
            st.markdown(
                """
                <div class="web-notice">
                    ⬡ &nbsp; Web sources included for a more complete answer
                </div>
                """,
                unsafe_allow_html=True,
            )

        chips = "".join(
            f'<span class="source-chip">{s.get("title", s.get("arxiv_id", "Source"))[:48]}</span>'
            for s in sources
        )

        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-label">⬡ &nbsp; Answer</div>
                <div class="answer-text">{out.get("answer", "No answer generated. Try rephrasing.")}</div>
                <div class="answer-sources">
                    <span class="sources-label">Sources &nbsp;</span>
                    {chips or '<span style="font-family:DM Mono,monospace;font-size:0.7rem;color:var(--text-muted)">No sources available</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_docs = [_normalize(s) for s in sources] + web_extra
        if all_docs:
            st.markdown('<div class="section-label">Related papers</div>', unsafe_allow_html=True)
            for r in all_docs[:8]:
                render_result(r)

elif clicked and not query.strip():
    st.warning("Please enter a query.")

# ── Estado inicial ─────────────────────────────────────────────────────────────
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