"""
app_advanced.py
===============
OmniRetrieve — Interfaz avanzada (modo desarrollador/investigador).
Muestra todos los controles técnicos y métricas del sistema.

Ejecutar:
    streamlit run frontend/app_advanced.py
"""

from __future__ import annotations
import time
import streamlit as st

st.set_page_config(
    page_title="OmniRetrieve · Advanced",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --bg:         #0a0a0f;
    --bg2:        #111118;
    --bg3:        #1a1a24;
    --border:     #2a2a38;
    --accent:     #7c6bff;
    --accent2:    #ff6b9d;
    --accent3:    #6bffca;
    --accent4:    #ffb86b;
    --text:       #e8e8f0;
    --text-muted: #6b6b80;
    --card-bg:    #13131c;
    --radius:     12px;
}

.stApp { background: var(--bg); font-family: 'Syne', sans-serif; color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

.omni-header {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.omni-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 1.4rem; flex-shrink: 0;
}
.omni-title {
    font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--text) 0%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1;
}
.omni-subtitle {
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    color: var(--text-muted); letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 0.25rem;
}
.mode-badge {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    background: rgba(255,184,107,0.15); color: var(--accent4);
    border: 1px solid rgba(255,184,107,0.3);
    border-radius: 20px; padding: 0.2rem 0.75rem;
    text-transform: uppercase; letter-spacing: 0.1em; margin-left: auto;
}

[data-testid="stTextInput"] input {
    background: var(--bg3) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text) !important;
    font-family: 'Syne', sans-serif !important; font-size: 1.05rem !important;
    padding: 1rem 1.25rem !important; transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,107,255,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }
[data-testid="stTextInput"] label {
    color: var(--text-muted) !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), #5a4fcf) !important;
    color: white !important; border: none !important;
    border-radius: var(--radius) !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important; transition: all 0.2s !important; width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,107,255,0.35) !important;
}

/* ── Sección de estadísticas del sistema ── */
.sys-stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.sys-stat-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem 1.25rem;
}
.sys-stat-value { font-size: 1.25rem; font-weight: 800; color: var(--text); }
.sys-stat-label {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem;
}
.sys-stat-sub {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: var(--accent); margin-top: 0.15rem;
}

/* ── Debug panel ── */
.debug-panel {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1rem 1.25rem;
    margin-top: 1rem; font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: var(--text-muted); line-height: 1.8;
}
.debug-panel b { color: var(--accent); }

.rag-answer-box {
    background: linear-gradient(135deg, #13131c, #1a1428);
    border: 1px solid rgba(124,107,255,0.35);
    border-radius: var(--radius); padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.rag-answer-box::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.rag-answer-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: var(--accent); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;
}
.rag-answer-text { font-size: 0.97rem; line-height: 1.75; color: var(--text); }
.rag-sources {
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 0.5rem;
}
.rag-source-chip {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    background: rgba(124,107,255,0.1); color: var(--accent);
    border: 1px solid rgba(124,107,255,0.25); border-radius: 20px; padding: 0.2rem 0.65rem;
}

.result-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.25rem 1.5rem;
    margin-bottom: 0.85rem; transition: border-color 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent2)); border-radius: 3px 0 0 3px;
}
.result-card:hover { border-color: var(--accent); transform: translateX(3px); }
.result-card.web-result::before { background: linear-gradient(180deg, var(--accent3), #4bf0b8); }
.result-card.rag-source::before { background: linear-gradient(180deg, var(--accent4), #ff8c2a); }
.result-rank { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.12em; margin-bottom: 0.4rem; }
.result-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.5rem; line-height: 1.3; }
.result-meta { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.75rem; align-items: center; }
.result-score { font-family: 'DM Mono', monospace; font-size: 0.72rem; background: rgba(124,107,255,0.15); color: var(--accent); padding: 0.2rem 0.55rem; border-radius: 20px; border: 1px solid rgba(124,107,255,0.3); }
.result-score.web  { background: rgba(107,255,202,0.1); color: var(--accent3); border-color: rgba(107,255,202,0.3); }
.result-score.rag  { background: rgba(255,184,107,0.1); color: var(--accent4); border-color: rgba(255,184,107,0.3); }
.result-source-badge { font-family: 'DM Mono', monospace; font-size: 0.65rem; padding: 0.2rem 0.5rem; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.08em; }
.badge-local    { background: rgba(124,107,255,0.1); color: var(--accent);  border: 1px solid rgba(124,107,255,0.2); }
.badge-web      { background: rgba(107,255,202,0.1); color: var(--accent3); border: 1px solid rgba(107,255,202,0.2); }
.badge-fallback { background: rgba(255,107,157,0.1); color: var(--accent2); border: 1px solid rgba(255,107,157,0.2); }
.badge-rag      { background: rgba(255,184,107,0.1); color: var(--accent4); border: 1px solid rgba(255,184,107,0.2); }
.result-abstract { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; font-family: 'DM Mono', monospace; font-weight: 300; }
.result-link { margin-top: 0.75rem; }
.result-link a { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--accent); text-decoration: none; }
.result-link a:hover { text-decoration: underline; }

.stats-bar {
    display: flex; gap: 1.5rem; padding: 0.9rem 1.25rem;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 1.5rem; flex-wrap: wrap;
}
.stat-item { display: flex; flex-direction: column; gap: 0.15rem; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

.web-banner { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.25rem; background: rgba(107,255,202,0.07); border: 1px solid rgba(107,255,202,0.25); border-radius: var(--radius); margin-bottom: 1rem; font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--accent3); }
.rag-banner { display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.25rem; background: rgba(124,107,255,0.07); border: 1px solid rgba(124,107,255,0.25); border-radius: var(--radius); margin-bottom: 1rem; font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--accent); }

.empty-state { text-align: center; padding: 4rem 2rem; color: var(--text-muted); }
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-muted); }
.empty-desc  { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--text-muted); opacity: 0.7; }

.section-title { font-size: 0.7rem; font-family: 'DM Mono', monospace; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
.section-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label { font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stCheckbox"] label { font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; color: var(--text-muted) !important; }
[data-testid="stSelectbox"] > div > div { background: var(--bg3) !important; border-color: var(--border) !important; color: var(--text) !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
[data-testid="stRadio"] label { font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; color: var(--text-muted) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Cache ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        from backend.retrieval.lsi_retriever import LSIRetriever
        r = LSIRetriever(); r.load()
        return r, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_rag():
    try:
        from backend.rag.pipeline import RAGPipeline
        ret, err = load_retriever()
        if err: return None, err
        return RAGPipeline(retriever=ret), None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_web(api_key, threshold, min_docs, max_results, use_fallback, auto_index):
    try:
        from backend.web_search.pipeline import WebSearchPipeline
        return WebSearchPipeline(
            api_key=api_key or None, threshold=threshold, min_docs=min_docs,
            max_results=max_results, use_fallback=use_fallback, auto_index=auto_index,
        ), None
    except Exception as e:
        return None, str(e)

def get_system_stats():
    """Obtiene estadísticas reales del sistema desde la BD y FAISS."""
    stats = {
        "total_docs": "—", "indexed_docs": "—",
        "vocab_size": "—", "total_chunks": "—",
        "embedded_chunks": "—", "faiss_vectors": "—",
        "faiss_type": "—", "lsi_k": "—",
    }
    try:
        from backend.database.schema import get_connection, DB_PATH
        conn = get_connection(DB_PATH)
        stats["total_docs"]    = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        stats["indexed_docs"]  = conn.execute("SELECT COUNT(*) FROM documents WHERE pdf_downloaded=1").fetchone()[0]
        stats["vocab_size"]    = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        stats["total_chunks"]  = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        stats["embedded_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
        conn.close()
    except Exception:
        pass
    try:
        from backend.retrieval.lsi_model import LSIModel, MODEL_PATH
        m = LSIModel(); m.load(MODEL_PATH)
        stats["lsi_k"]    = m.k
    except Exception:
        pass
    return stats

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
        <div style='font-family:"Syne",sans-serif;font-size:1.1rem;font-weight:800;color:#e8e8f0;letter-spacing:-0.02em;margin-bottom:0.25rem'>⬡ OmniRetrieve</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.6rem;color:#ffb86b;text-transform:uppercase;letter-spacing:0.12em'>Advanced · Developer Mode</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Search Mode</div>', unsafe_allow_html=True)
    search_mode = st.radio("Mode", ["Search", "Ask (RAG)"], index=0, label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:1.25rem">Retrieval — LSI</div>', unsafe_allow_html=True)
    top_n = st.slider("Top-N results", 1, 20, 10)
    candidate_k = max_chunks = max_chars = None
    if search_mode == "Ask (RAG)":
        candidate_k = st.slider("Candidate pool (reranking)", 10, 100, 50)
        max_chunks  = st.slider("Max chunks in context", 1, 10, 5)
        max_chars   = st.slider("Max chars per chunk", 100, 800, 400)
        show_debug  = st.checkbox("Show debug info (context + prompt)", value=False)
    else:
        show_debug = False

    st.markdown('<div class="section-title" style="margin-top:1.25rem">Web Search</div>', unsafe_allow_html=True)
    enable_web   = st.checkbox("Enable web search", value=True)
    threshold    = st.slider("Score threshold", 0.0, 1.0, 0.15, 0.01, help="Minimum cosine similarity to consider results sufficient")
    min_docs_val = st.slider("Min docs above threshold", 1, 5, 1)
    max_web      = st.slider("Max web results", 1, 10, 5)
    search_depth = st.selectbox("Search depth", ["basic", "advanced"], index=0)
    use_fallback = st.checkbox("DuckDuckGo fallback", value=True)
    auto_index   = st.checkbox("Auto-index web docs", value=True)

    st.markdown('<div class="section-title" style="margin-top:1.25rem">API Key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input("Tavily API Key", type="password", placeholder="tvly-... (or set in .env)", label_visibility="visible")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.65rem;color:#6b6b80;line-height:2'>
    <b style='color:#e8e8f0'>Active modules</b><br>
    ⬡ LSI Retriever<br>⬡ Web Search (Tavily)<br>
    ⬡ Fallback (DuckDuckGo)<br>⬡ RAG Pipeline<br>⬡ Auto-indexing
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="omni-header">
    <div class="omni-logo">⬡</div>
    <div>
        <div class="omni-title">OmniRetrieve</div>
        <div class="omni-subtitle">Semantic IR · LSI + RAG + Web Search</div>
    </div>
    <div class="mode-badge">⬡ Advanced Mode</div>
</div>
""", unsafe_allow_html=True)

# ── System stats ──────────────────────────────────────────────────────────────

with st.expander("⬡  System Statistics", expanded=False):
    sys = get_system_stats()
    st.markdown(f"""
    <div class="sys-stats-grid">
        <div class="sys-stat-card">
            <div class="sys-stat-value">{sys['total_docs']}</div>
            <div class="sys-stat-label">Total documents</div>
            <div class="sys-stat-sub">{sys['indexed_docs']} with full text</div>
        </div>
        <div class="sys-stat-card">
            <div class="sys-stat-value">{sys['vocab_size']}</div>
            <div class="sys-stat-label">Vocabulary size</div>
            <div class="sys-stat-sub">unique terms indexed</div>
        </div>
        <div class="sys-stat-card">
            <div class="sys-stat-value">{sys['total_chunks']}</div>
            <div class="sys-stat-label">Total chunks</div>
            <div class="sys-stat-sub">{sys['embedded_chunks']} embedded</div>
        </div>
        <div class="sys-stat-card">
            <div class="sys-stat-value">{sys['lsi_k']}</div>
            <div class="sys-stat-label">LSI components (k)</div>
            <div class="sys-stat-sub">latent dimensions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Search bar ────────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "QUERY",
        placeholder="e.g. What are the main challenges of AI fairness?" if search_mode == "Ask (RAG)" else "e.g. fairness in machine learning, bias detection NLP…",
        label_visibility="visible",
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    search_clicked = st.button("Ask →" if search_mode == "Ask (RAG)" else "Search →", use_container_width=True)

# ── Render helpers ─────────────────────────────────────────────────────────────

def render_card(r, rank, score_cls="result-score", card_cls="result-card"):
    source = r.get("source", "local")
    badge_map = {
        "web":          '<span class="result-source-badge badge-web">web · tavily</span>',
        "web_fallback": '<span class="result-source-badge badge-fallback">web · duckduckgo</span>',
        "rag":          '<span class="result-source-badge badge-rag">rag · source</span>',
    }
    badge   = badge_map.get(source, '<span class="result-source-badge badge-local">local · lsi</span>')
    url     = r.get("url") or r.get("pdf_url","")
    url_html = f'<div class="result-link"><a href="{url}" target="_blank">↗ {url[:65]}{"…" if len(url)>65 else ""}</a></div>' if url else ""
    abstract = (r.get("abstract") or r.get("text",""))[:280]
    arxiv_tag = f'<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:var(--text-muted)">{r.get("arxiv_id","")}</span>' if r.get("arxiv_id") else ""
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-rank">#{rank:02d}</div>
        <div class="result-title">{r.get('title','Untitled')}</div>
        <div class="result-meta">
            <span class="{score_cls}">score {r.get('score',0.0):.4f}</span>
            {badge} {arxiv_tag}
        </div>
        <div class="result-abstract">{(abstract+"…") if abstract else "No abstract available."}</div>
        {url_html}
    </div>
    """, unsafe_allow_html=True)

def render_list(items):
    if not items:
        st.markdown('<div class="empty-state"><div class="empty-icon">⬡</div><div class="empty-title">No results</div></div>', unsafe_allow_html=True)
        return
    for i, r in enumerate(items, 1):
        src = r.get("source","local")
        is_web = src in ("web","web_fallback")
        render_card(r, i, "result-score web" if is_web else "result-score", "result-card web-result" if is_web else "result-card")

# ── Main ──────────────────────────────────────────────────────────────────────

if search_clicked and query.strip():
    t0 = time.monotonic()

    if search_mode == "Ask (RAG)":
        with st.spinner("Loading RAG pipeline…"):
            rag, err = load_rag()
        if err:
            st.error(f"RAG error: {err}"); st.stop()

        with st.spinner("Retrieving, reranking and generating answer…"):
            try:
                out = rag.ask(
                    query=query.strip(), top_k=top_n,
                    candidate_k=candidate_k or 50,
                    max_chunks=max_chunks or 5,
                    max_chars=max_chars or 400,
                    include_debug=show_debug,
                )
            except Exception as e:
                st.error(f"RAG error: {e}"); st.stop()

        elapsed = (time.monotonic() - t0) * 1000
        sources = out.get("sources", [])

        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item"><span class="stat-value">{len(sources)}</span><span class="stat-label">Sources used</span></div>
            <div class="stat-item"><span class="stat-value">{candidate_k}</span><span class="stat-label">Candidate pool</span></div>
            <div class="stat-item"><span class="stat-value">{max_chunks}</span><span class="stat-label">Chunks in context</span></div>
            <div class="stat-item"><span class="stat-value">{elapsed:.0f}ms</span><span class="stat-label">Response time</span></div>
            <div class="stat-item"><span class="stat-value">LSI+RAG</span><span class="stat-label">Pipeline</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="rag-banner">⬡ &nbsp; RAG pipeline — retrieval · reranking · generation</div>', unsafe_allow_html=True)

        chips = "".join(f'<span class="rag-source-chip">⬡ {s.get("title",s.get("arxiv_id","src"))[:40]}</span>' for s in sources)
        st.markdown(f"""
        <div class="rag-answer-box">
            <div class="rag-answer-label">⬡ &nbsp; Generated Answer</div>
            <div class="rag-answer-text">{out.get("answer","No answer generated.")}</div>
            <div class="rag-sources">{chips}</div>
        </div>
        """, unsafe_allow_html=True)

        if show_debug and "context" in out:
            st.markdown('<div class="section-title">Debug — Context passed to LLM</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debug-panel">{out["context"][:2000]}…</div>', unsafe_allow_html=True)

        if sources:
            st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
            for i, s in enumerate(sources, 1):
                s["source"] = "rag"
                render_card(s, i, "result-score rag", "result-card rag-source")

    else:
        with st.spinner("Loading LSI model…"):
            retriever, err_ret = load_retriever()
        if err_ret:
            st.error(f"Retriever error: {err_ret}"); st.stop()

        with st.spinner("Searching…"):
            try:
                local_results = retriever.retrieve(query.strip(), top_n=top_n)
            except Exception as e:
                st.error(f"Retrieval error: {e}"); st.stop()

        output = {"results": local_results, "web_activated": False, "web_results": [], "reason": "", "indexed": 0}

        if enable_web:
            with st.spinner("Web search check…"):
                try:
                    pipeline, err_pip = load_web(
                        api_key=api_key_input.strip() or None,
                        threshold=threshold, min_docs=min_docs_val,
                        max_results=max_web, use_fallback=use_fallback, auto_index=auto_index,
                    )
                    if err_pip:
                        st.warning(f"Web search unavailable: {err_pip}")
                    else:
                        output = pipeline.run(query.strip(), local_results)
                except Exception as e:
                    st.warning(f"Web search failed: {e}")

        elapsed = (time.monotonic() - t0) * 1000
        results = output["results"]
        local_n = sum(1 for r in results if r.get("source","local") == "local")
        web_n   = len(results) - local_n

        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item"><span class="stat-value">{len(results)}</span><span class="stat-label">Total results</span></div>
            <div class="stat-item"><span class="stat-value">{local_n}</span><span class="stat-label">Local (LSI)</span></div>
            <div class="stat-item"><span class="stat-value">{web_n}</span><span class="stat-label">Web</span></div>
            <div class="stat-item"><span class="stat-value">{elapsed:.0f}ms</span><span class="stat-label">Response time</span></div>
            <div class="stat-item"><span class="stat-value">{output.get('indexed',0)}</span><span class="stat-label">Indexed</span></div>
            <div class="stat-item"><span class="stat-value">{"✓" if output["web_activated"] else "—"}</span><span class="stat-label">Web activated</span></div>
        </div>
        """, unsafe_allow_html=True)

        if output["web_activated"]:
            st.markdown(f'<div class="web-banner">⬡ &nbsp; Web search activated — {output.get("reason","")}</div>', unsafe_allow_html=True)

        tab_all, tab_local, tab_web = st.tabs([f"ALL ({len(results)})", f"LOCAL ({local_n})", f"WEB ({web_n})"])
        with tab_all:
            st.markdown('<div class="section-title">All results</div>', unsafe_allow_html=True)
            render_list(results)
        with tab_local:
            st.markdown('<div class="section-title">Local documents (LSI)</div>', unsafe_allow_html=True)
            render_list([r for r in results if r.get("source","local") == "local"])
        with tab_web:
            st.markdown('<div class="section-title">Web documents</div>', unsafe_allow_html=True)
            render_list([r for r in results if r.get("source","local") != "local"])

elif search_clicked:
    st.warning("Please enter a query.")
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <div class="empty-title">Developer mode — full control</div>
        <div class="empty-desc">Adjust all parameters from the sidebar.<br>Switch to <b>Ask</b> for RAG with debug output.</div>
    </div>
    """, unsafe_allow_html=True)