"""
app_advanced.py
===============
OmniRetrieve — Interfaz de usuario final.

Dos modos:
  · Search  — devuelve documentos relevantes ordenados por relevancia.
  · Ask AI  — genera una respuesta en lenguaje natural con fuentes citadas.

Todo lo demás (web search, fallback, indexación) ocurre automáticamente
por debajo sin que el usuario tenga que configurar nada.

Ejecutar (desde la raíz del proyecto):
    streamlit run frontend/app_advanced.py
"""

from __future__ import annotations

# ── Path fix: permite ejecutar desde cualquier directorio ─────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st

# ── DB init (idempotente — seguro llamar siempre) ─────────────────────────────
from backend.main import setup as _db_setup
try:
    _db_setup(verbose=False)
except Exception:
    pass  # si la BD no está disponible la app lo manejará por módulo

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniRetrieve",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --bg: #050814;
    --bg2: #0a1020;
    --bg3: #111a2e;
    --card: rgba(14, 20, 38, 0.76);
    --card-strong: rgba(18, 26, 48, 0.92);
    --border: rgba(148, 163, 184, 0.16);
    --border-strong: rgba(47, 128, 255, 0.34);

    --accent: #2f80ff;
    --accent2: #00d4ff;
    --accent3: #2ee59d;
    --accent4: #ffd166;

    --text: #f6f9ff;
    --text-muted: #9aa7ba;
    --radius: 18px;
    --shadow: 0 18px 50px rgba(0, 0, 0, 0.36);
    --shadow-soft: 0 10px 28px rgba(0, 0, 0, 0.22);
}

/* ── Fondo con movimiento ── */
.bg-motion {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.55;
    background:
        repeating-linear-gradient(
            120deg,
            rgba(255, 255, 255, 0.028) 0px,
            rgba(255, 255, 255, 0.028) 1px,
            transparent 1px,
            transparent 28px
        ),
        repeating-linear-gradient(
            60deg,
            rgba(255, 255, 255, 0.018) 0px,
            rgba(255, 255, 255, 0.018) 1px,
            transparent 1px,
            transparent 36px
        );
    animation: driftLines 22s linear infinite;
    mask-image: radial-gradient(circle at center, black 35%, transparent 100%);
    -webkit-mask-image: radial-gradient(circle at center, black 35%, transparent 100%);
}

@keyframes driftLines {
    from { transform: translate3d(0, 0, 0); }
    to   { transform: translate3d(-80px, 60px, 0); }
}

/* ── App base ── */
.stApp {
    background:
        radial-gradient(circle at 14% 18%, rgba(47, 128, 255, 0.18), transparent 26%),
        radial-gradient(circle at 82% 12%, rgba(0, 212, 255, 0.12), transparent 24%),
        radial-gradient(circle at 78% 84%, rgba(46, 229, 157, 0.10), transparent 22%),
        linear-gradient(180deg, #040711 0%, #080d18 50%, #050814 100%);
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    filter: blur(72px);
    opacity: 0.16;
    pointer-events: none;
    z-index: 0;
}

.stApp::before {
    top: -120px;
    right: -140px;
    background: radial-gradient(circle, rgba(47, 128, 255, 0.95), transparent 68%);
}

.stApp::after {
    bottom: -160px;
    left: -160px;
    background: radial-gradient(circle, rgba(46, 229, 157, 0.82), transparent 68%);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    position: relative;
    z-index: 1;
    padding-top: 2.2rem;
    padding-bottom: 2.8rem;
    max-width: 940px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 1.6rem 1rem 1.1rem;
    margin-bottom: 0.4rem;
}

.hero-logo {
    width: 72px;
    height: 72px;
    margin: 0 auto 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 22px;
    font-size: 1.9rem;
    color: white;
    background: linear-gradient(135deg, rgba(47,128,255,1), rgba(0,212,255,1));
    box-shadow: 0 16px 40px rgba(47, 128, 255, 0.25);
    border: 1px solid rgba(255,255,255,0.12);
    position: relative;
    overflow: hidden;
}

.hero-logo::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.20), transparent 55%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.65rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1;
    margin-bottom: 0.55rem;
    background: linear-gradient(90deg, #ffffff 10%, #cfe6ff 35%, #2f80ff 68%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-desc {
    max-width: 560px;
    margin: 0 auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.8;
    color: var(--text-muted);
    opacity: 0.95;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 999px !important;
    border: 1px solid transparent !important;
    transition: all 0.22s ease !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.4rem !important;
    box-shadow: none !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) scale(0.99) !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    box-shadow: 0 12px 24px rgba(47, 128, 255, 0.26) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 16px 30px rgba(0, 212, 255, 0.24) !important;
}

[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(18, 24, 42, 0.84) !important;
    color: var(--text-muted) !important;
    border: 1px solid rgba(148, 163, 184, 0.14) !important;
    backdrop-filter: blur(10px);
}

[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(0, 212, 255, 0.35) !important;
    color: var(--text) !important;
    background: rgba(22, 29, 50, 0.94) !important;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: rgba(14, 20, 38, 0.88) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 999px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.95rem 1.35rem !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(12px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

[data-testid="stTextInput"] input:hover {
    border-color: rgba(0, 212, 255, 0.26) !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(47, 128, 255, 0.72) !important;
    box-shadow: 0 0 0 4px rgba(47, 128, 255, 0.12) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: rgba(154, 167, 186, 0.78) !important;
}

[data-testid="stTextInput"] label { display: none !important; }

/* ── Search button ── */
.search-btn > div > div > div > button {
    background: linear-gradient(135deg, #2f80ff, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.95rem 1.2rem !important;
    width: 100% !important;
    box-shadow: 0 12px 24px rgba(47, 128, 255, 0.28) !important;
}

.search-btn > div > div > div > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 16px 30px rgba(0, 212, 255, 0.28) !important;
}

/* ── Info bar ── */
.info-bar {
    display: flex;
    justify-content: center;
    gap: 1rem;
    padding: 0.75rem 1.1rem;
    background: rgba(14, 20, 38, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 999px;
    margin: 1.15rem 0;
    flex-wrap: wrap;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(12px);
}

.info-item {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
}

.info-item b { color: var(--text); }

/* ── Web notice ── */
.web-notice {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.7rem 1rem;
    margin: 0.8rem auto 1.15rem;
    width: fit-content;
    background: rgba(46, 229, 157, 0.08);
    border: 1px solid rgba(46, 229, 157, 0.22);
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
    color: #8ff0c8;
    box-shadow: 0 10px 22px rgba(0,0,0,0.18);
    backdrop-filter: blur(12px);
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(180deg, rgba(18, 26, 48, 0.94), rgba(13, 17, 31, 0.92));
    border: 1px solid rgba(47, 128, 255, 0.24);
    border-radius: 22px;
    padding: 1.5rem 1.7rem;
    margin: 1.15rem 0;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
}

.answer-box::before {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 5px;
    background: linear-gradient(180deg, var(--accent), var(--accent3));
}

.answer-box::after {
    content: "";
    position: absolute;
    inset: -1px;
    background: radial-gradient(circle at top right, rgba(0,212,255,0.10), transparent 35%);
    pointer-events: none;
}

.answer-label {
    position: relative;
    z-index: 1;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.75rem;
}

.answer-text {
    position: relative;
    z-index: 1;
    font-size: 0.98rem;
    line-height: 1.85;
    color: var(--text);
}

.answer-sources {
    position: relative;
    z-index: 1;
    margin-top: 1rem;
    padding-top: 0.9rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}

.sources-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.source-chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    background: rgba(47, 128, 255, 0.12);
    color: #d6e7ff;
    border: 1px solid rgba(47, 128, 255, 0.22);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    white-space: nowrap;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin: 1.25rem 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0.14), transparent);
}

/* ── Result cards ── */
.result-card {
    background: linear-gradient(180deg, rgba(16, 24, 44, 0.88), rgba(12, 16, 30, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: var(--radius);
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.85rem;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    backdrop-filter: blur(10px);
}

.result-card::before {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent3));
}

.result-card:hover {
    border-color: rgba(47, 128, 255, 0.42);
    transform: translateY(-2px);
    box-shadow: 0 16px 32px rgba(0,0,0,0.25);
}

.result-card.is-web::before {
    background: linear-gradient(180deg, var(--accent3), var(--accent2));
}

.result-title {
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.45rem;
    line-height: 1.35;
}

.result-abstract {
    font-size: 0.83rem;
    color: rgba(154, 167, 186, 0.92);
    line-height: 1.7;
    font-family: 'DM Mono', monospace;
    font-weight: 300;
}

.result-footer {
    display: flex;
    gap: 0.65rem;
    align-items: center;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}

.result-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    padding: 0.16rem 0.6rem;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.tag-local {
    background: rgba(47, 128, 255, 0.12);
    color: #d6e7ff;
    border: 1px solid rgba(47, 128, 255, 0.22);
}

.tag-web {
    background: rgba(46, 229, 157, 0.1);
    color: #8ff0c8;
    border: 1px solid rgba(46, 229, 157, 0.2);
}

.result-link {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #d6e7ff;
    text-decoration: none;
}

.result-link:hover { text-decoration: underline; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 2.7rem 1rem;
    background: rgba(14, 20, 38, 0.46);
    border: 1px dashed rgba(148, 163, 184, 0.2);
    border-radius: 22px;
    backdrop-filter: blur(10px);
}

.empty-icon {
    width: 64px;
    height: 64px;
    border-radius: 22px;
    margin: 0 auto 0.9rem;
    display: grid;
    place-items: center;
    font-size: 2rem;
    color: #d6e7ff;
    background: linear-gradient(135deg, rgba(47,128,255,0.18), rgba(46,229,157,0.14));
    border: 1px solid rgba(47,128,255,0.18);
    box-shadow: 0 14px 30px rgba(0,0,0,0.18);
}

.empty-hint {
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: var(--text-muted);
    line-height: 1.95;
}

.empty-hint i {
    color: var(--accent2);
    font-style: normal;
}

hr { border-color: rgba(148, 163, 184, 0.14) !important; }
</style>
""", unsafe_allow_html=True)

# Fondo animado decorativo
st.markdown('<div class="bg-motion"></div>', unsafe_allow_html=True)

# ── Cache ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_retriever():
    """
    Carga el LSIRetriever con rutas por defecto.
    Devuelve (retriever, None) si OK, (None, mensaje_error) si falla.
    """
    try:
        from backend.retrieval.lsi_retriever import LSIRetriever
        r = LSIRetriever()
        r.load()          # usa MODEL_PATH y DB_PATH por defecto de lsi_retriever.py
        return r, None
    except FileNotFoundError:
        return None, (
            "El modelo LSI no existe todavía. "
            "Ejecuta el crawler y luego: python -m backend.retrieval.build_lsi"
        )
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def load_rag():
    """
    Carga el RAGPipeline inyectando el LSIRetriever ya cacheado.
    Devuelve (pipeline, None) si OK, (None, mensaje_error) si falla.
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
    Carga el WebSearchPipeline (lee TAVILY_API_KEY del .env automáticamente).
    Si no hay API key ni duckduckgo disponible devuelve (None, error) y la app
    continúa sin búsqueda web — no es un error crítico.
    """
    try:
        from backend.web_search.pipeline import WebSearchPipeline
        return WebSearchPipeline(), None
    except Exception as e:
        return None, str(e)


# ── Session state ─────────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "search"

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-logo">⬡</div>
    <div class="hero-title">OmniRetrieve</div>
    <div class="hero-desc">
        Search across AI &amp; Ethics research papers.<br>
        Ask questions and get intelligent answers.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode toggle ───────────────────────────────────────────────────────────────

col_l, col_search, col_ask, col_r = st.columns([2, 1, 1, 2])

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

# ── Search input ──────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_result(r) -> dict:
    """
    Garantiza que cualquier resultado (dict o dataclass) sea un dict
    plano con las claves que espera render_result().
    Cubre los tres casos posibles:
      - dict estándar del LSIRetriever  → pass-through
      - dict de build_sources() del RAG → añade claves faltantes
      - objeto con atributos            → convierte a dict
    """
    if not isinstance(r, dict):
        # Por si algún retriever devuelve un dataclass en el futuro
        r = vars(r) if hasattr(r, "__dict__") else {}

    return {
        "title":    r.get("title") or r.get("document_title") or r.get("arxiv_id", "Untitled"),
        "abstract": r.get("abstract") or r.get("text", ""),
        "url":      r.get("url") or r.get("pdf_url", ""),
        "source":   r.get("source", "local"),
        # conservamos score para el pipeline web
        "score":    r.get("score", 0.0),
    }


def render_result(r: dict) -> None:
    """Renderiza una tarjeta de resultado (local o web)."""
    src    = r.get("source", "local")
    is_web = src in ("web", "web_fallback")
    card_cls = "result-card is-web" if is_web else "result-card"
    tag = (
        '<span class="result-tag tag-web">Web</span>'
        if is_web else
        '<span class="result-tag tag-local">Research paper</span>'
    )
    url  = r.get("url") or r.get("pdf_url", "")
    link = f'<a class="result-link" href="{url}" target="_blank">↗ Read paper</a>' if url else ""

    abstract = (r.get("abstract") or r.get("text", ""))[:220]
    if abstract:
        abstract += "…"

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-title">{r.get('title', 'Untitled')}</div>
        <div class="result-abstract">{abstract or 'No description available.'}</div>
        <div class="result-footer">{tag}{link}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main logic ────────────────────────────────────────────────────────────────

if clicked and query.strip():
    t0 = time.monotonic()

    # ════════════════════════════════════════
    # SEARCH
    # ════════════════════════════════════════
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

        # Web search automático — invisible para el usuario
        output = {
            "results":       local_results,
            "web_activated": False,
            "web_results":   [],
        }
        try:
            pipeline, _ = load_web()
            if pipeline:
                output = pipeline.run(query.strip(), local_results)
        except Exception:
            pass  # fallback silencioso a resultados locales

        elapsed = (time.monotonic() - t0) * 1000
        results = output["results"]

        # Info bar
        st.markdown(f"""
        <div class="info-bar">
            <div class="info-item"><b>{len(results)}</b> results</div>
            <div class="info-item"><b>{elapsed:.0f}ms</b></div>
            {"<div class='info-item'>⬡ <b>Web sources included</b></div>" if output["web_activated"] else ""}
        </div>
        """, unsafe_allow_html=True)

        if results:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            for r in results:
                render_result(_normalize_result(r))
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">⬡</div>
                <div class="empty-hint">
                    No results found.<br>
                    Try different keywords or switch to <i>Ask AI</i> mode.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # ASK AI
    # ════════════════════════════════════════
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

        # Web search automático sobre los resultados del RAG
        # `sources` viene de build_sources() → lista de dicts con title, arxiv_id, score, etc.
        sources = out.get("sources", [])
        web_activated = False
        web_extra: list[dict] = []

        try:
            pipeline, _ = load_web()
            if pipeline:
                # Normalizamos sources antes de pasarlos al pipeline web
                # para que SufficiencyChecker pueda leer el campo "score"
                normalized_sources = [_normalize_result(s) for s in sources]
                web_out       = pipeline.run(query.strip(), normalized_sources)
                web_activated = web_out.get("web_activated", False)
                web_extra     = web_out.get("web_results", [])
        except Exception:
            pass

        elapsed = (time.monotonic() - t0) * 1000

        # Aviso web discreto
        if web_activated:
            st.markdown("""
            <div class="web-notice">
                ⬡ &nbsp; Web sources included for a more complete answer
            </div>
            """, unsafe_allow_html=True)

        # Chips de fuentes
        chips = "".join(
            f'<span class="source-chip">{s.get("title", s.get("arxiv_id", "Source"))[:48]}</span>'
            for s in sources
        )
        fallback_sources = (
            '<span style="font-family:DM Mono, monospace; '
            'font-size:0.7rem; color:var(--text-muted)">No sources available</span>'
        )
        sources_html = chips if chips else fallback_sources

        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">⬡ &nbsp; Answer</div>
            <div class="answer-text">{out.get("answer", "I couldn't generate an answer. Please try rephrasing.")}</div>
            <div class="answer-sources">
                <span class="sources-label">Sources &nbsp;</span>
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Documentos fuente + web extras
        # Normalizamos cada source para que render_result siempre reciba un dict completo
        all_docs = [_normalize_result(s) for s in sources] + web_extra
        if all_docs:
            st.markdown('<div class="section-label">Related papers</div>', unsafe_allow_html=True)
            for r in all_docs[:8]:
                render_result(r)

elif clicked and not query.strip():
    st.warning("Please enter a query.")

# ── Estado inicial ─────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <div class="empty-hint">
            Try searching for:<br><br>
            <i>fairness in machine learning</i> &nbsp;·&nbsp; <i>bias in NLP models</i><br>
            <i>AI transparency and accountability</i> &nbsp;·&nbsp; <i>explainability methods</i>
        </div>
    </div>
    """, unsafe_allow_html=True)