"""
app.py
======
OmniRetrieve — Interfaz de usuario final.
Diseñada para el usuario que solo quiere buscar y obtener respuestas.
Sin parámetros técnicos expuestos.

Ejecutar:
    streamlit run frontend/app.py
"""

from __future__ import annotations
import time
import streamlit as st

st.set_page_config(
    page_title="OmniRetrieve",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
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
.block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 860px; margin: 0 auto; }

/* Ocultar sidebar toggle en modo usuario */
[data-testid="collapsedControl"] { display: none !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2.5rem;
}
.hero-logo {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 18px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 1.8rem; margin-bottom: 1.25rem;
}
.hero-title {
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.04em;
    background: linear-gradient(90deg, var(--text) 20%, var(--accent) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.5rem;
}
.hero-desc {
    font-family: 'DM Mono', monospace; font-size: 0.82rem;
    color: var(--text-muted); line-height: 1.7; max-width: 480px; margin: 0 auto;
}

/* ── Mode toggle (pills) ── */
.mode-pills {
    display: flex; gap: 0.5rem; justify-content: center;
    margin: 1.5rem 0 1rem;
}
.mode-pill {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 0.4rem 1.1rem; border-radius: 20px; cursor: pointer;
    text-transform: uppercase; letter-spacing: 0.1em; border: none;
    transition: all 0.2s;
}
.mode-pill.active {
    background: var(--accent); color: white;
}
.mode-pill.inactive {
    background: var(--bg3); color: var(--text-muted);
    border: 1px solid var(--border);
}

/* ── Search input ── */
[data-testid="stTextInput"] input {
    background: var(--bg3) !important; border: 1px solid var(--border) !important;
    border-radius: 50px !important; color: var(--text) !important;
    font-family: 'Syne', sans-serif !important; font-size: 1rem !important;
    padding: 1rem 1.5rem !important; transition: all 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,107,255,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--text-muted) !important; }
[data-testid="stTextInput"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), #5a4fcf) !important;
    color: white !important; border: none !important; border-radius: 50px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; padding: 0.75rem 2rem !important;
    transition: all 0.2s !important; width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,107,255,0.35) !important;
}

/* ── RAG answer ── */
.answer-box {
    background: linear-gradient(135deg, #13131c, #1a1428);
    border: 1px solid rgba(124,107,255,0.3);
    border-radius: 18px; padding: 1.75rem 2rem;
    margin: 1.5rem 0; position: relative; overflow: hidden;
}
.answer-box::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.answer-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.answer-text {
    font-size: 1rem; line-height: 1.8; color: var(--text); font-weight: 400;
}
.answer-sources {
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
}
.sources-label {
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em; margin-right: 0.25rem;
}
.source-chip {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    background: rgba(124,107,255,0.1); color: var(--accent);
    border: 1px solid rgba(124,107,255,0.2); border-radius: 20px;
    padding: 0.2rem 0.65rem;
}

/* ── Result cards ── */
.result-card {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.25rem 1.5rem;
    margin-bottom: 0.85rem; transition: border-color 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent2)); border-radius: 3px 0 0 3px;
}
.result-card:hover { border-color: var(--accent); transform: translateX(3px); }
.result-card.web::before { background: linear-gradient(180deg, var(--accent3), #4bf0b8); }
.result-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; line-height: 1.3; }
.result-abstract { font-size: 0.84rem; color: var(--text-muted); line-height: 1.65; font-family: 'DM Mono', monospace; font-weight: 300; }
.result-footer {
    display: flex; gap: 0.75rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap;
}
.result-tag {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    padding: 0.18rem 0.55rem; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.08em;
}
.tag-local   { background: rgba(124,107,255,0.1); color: var(--accent);  border: 1px solid rgba(124,107,255,0.2); }
.tag-web     { background: rgba(107,255,202,0.1); color: var(--accent3); border: 1px solid rgba(107,255,202,0.2); }
.tag-fallback { background: rgba(255,107,157,0.1); color: var(--accent2); border: 1px solid rgba(255,107,157,0.2); }
.result-link { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--accent); text-decoration: none; }
.result-link:hover { text-decoration: underline; }

/* ── Info bar ── */
.info-bar {
    display: flex; justify-content: center; gap: 2rem; padding: 0.75rem;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 50px; margin-bottom: 1.5rem; flex-wrap: wrap;
}
.info-item { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }
.info-item b { color: var(--text); }

/* ── Web notice ── */
.web-notice {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.65rem 1.1rem;
    background: rgba(107,255,202,0.07); border: 1px solid rgba(107,255,202,0.2);
    border-radius: 50px; margin-bottom: 1.25rem; width: fit-content; margin-left: auto; margin-right: auto;
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--accent3);
}

/* ── Empty / Hero state ── */
.empty-state { text-align: center; padding: 2rem 1rem; }
.empty-icon  { font-size: 2.5rem; opacity: 0.3; margin-bottom: 0.75rem; }
.empty-hint  { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--text-muted); line-height: 1.7; }

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.85rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

[data-testid="stRadio"] { display: none !important; }
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
def load_web():
    try:
        from backend.web_search.pipeline import WebSearchPipeline
        return WebSearchPipeline(), None
    except Exception as e:
        return None, str(e)

# ── State ─────────────────────────────────────────────────────────────────────

if "mode" not in st.session_state:
    st.session_state.mode = "search"

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-logo">⬡</div>
    <div class="hero-title">OmniRetrieve</div>
    <div class="hero-desc">
        Search across thousands of AI &amp; Ethics research papers.<br>
        Ask questions and get intelligent answers with sources.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode selector (pill buttons via columns) ──────────────────────────────────

col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    if st.button("🔍  Search", use_container_width=True,
                  type="primary" if st.session_state.mode == "search" else "secondary"):
        st.session_state.mode = "search"
        st.rerun()
with col3:
    if st.button("💬  Ask AI", use_container_width=True,
                  type="primary" if st.session_state.mode == "ask" else "secondary"):
        st.session_state.mode = "ask"
        st.rerun()

mode = st.session_state.mode

# ── Search input ──────────────────────────────────────────────────────────────

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

col_q, col_b = st.columns([5, 1])
with col_q:
    placeholder = (
        "Ask anything about AI & Ethics research…"
        if mode == "ask"
        else "Search papers on fairness, bias, transparency…"
    )
    query = st.text_input("q", placeholder=placeholder, label_visibility="collapsed")
with col_b:
    clicked = st.button(
        "Ask →" if mode == "ask" else "Search →",
        use_container_width=True,
    )

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Logic ─────────────────────────────────────────────────────────────────────

def render_result(r: dict, rank: int) -> None:
    src = r.get("source", "local")
    tag_map = {
        "web":          '<span class="result-tag tag-web">Web</span>',
        "web_fallback": '<span class="result-tag tag-fallback">Web</span>',
    }
    tag     = tag_map.get(src, '<span class="result-tag tag-local">Research paper</span>')
    card_cls = "result-card web" if src in ("web","web_fallback") else "result-card"
    url     = r.get("url") or r.get("pdf_url","")
    link_html = f'<a class="result-link" href="{url}" target="_blank">↗ Read paper</a>' if url else ""
    abstract  = (r.get("abstract") or r.get("text",""))[:220]
    if abstract: abstract += "…"
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-title">{r.get('title','Untitled')}</div>
        <div class="result-abstract">{abstract or "No description available."}</div>
        <div class="result-footer">
            {tag} {link_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


if clicked and query.strip():
    t0 = time.monotonic()

    # ── ASK mode ─────────────────────────────────────────────────────────────
    if mode == "ask":
        with st.spinner("Thinking…"):
            rag, err = load_rag()
            if err:
                st.error("The AI assistant is not available right now. Try Search mode instead.")
                st.stop()
            try:
                out = rag.ask(query=query.strip(), top_k=10, candidate_k=50, max_chunks=5, max_chars=400)
            except Exception as e:
                st.error(f"Something went wrong: {e}"); st.stop()

        elapsed = (time.monotonic() - t0) * 1000
        sources = out.get("sources", [])

        # Web search for extra sources if needed
        web_out = {"web_activated": False}
        try:
            pipeline, _ = load_web()
            if pipeline:
                web_out = pipeline.run(query.strip(), sources)
        except Exception:
            pass

        if web_out.get("web_activated"):
            st.markdown('<div class="web-notice">⬡ &nbsp; Web sources included for a more complete answer</div>', unsafe_allow_html=True)

        chips = "".join(
            f'<span class="source-chip">{s.get("title",s.get("arxiv_id","Source"))[:45]}</span>'
            for s in sources
        )
        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">⬡ &nbsp; Answer</div>
            <div class="answer-text">{out.get("answer","I couldn't generate an answer. Please try rephrasing your question.")}</div>
            <div class="answer-sources">
                <span class="sources-label">Sources</span>
                {chips if chips else '<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--text-muted)">No sources available</span>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show source papers
        all_results = sources + web_out.get("web_results", [])
        if all_results:
            st.markdown('<div class="section-label">Related papers</div>', unsafe_allow_html=True)
            for i, r in enumerate(all_results[:6], 1):
                render_result(r, i)

    # ── SEARCH mode ──────────────────────────────────────────────────────────
    else:
        with st.spinner("Searching…"):
            retriever, err_ret = load_retriever()
            if err_ret:
                st.error("Search is not available right now. Please try again later."); st.stop()
            try:
                local_results = retriever.retrieve(query.strip(), top_n=10)
            except Exception as e:
                st.error(f"Search error: {e}"); st.stop()

        output = {"results": local_results, "web_activated": False, "web_results": [], "indexed": 0}
        try:
            pipeline, _ = load_web()
            if pipeline:
                output = pipeline.run(query.strip(), local_results)
        except Exception:
            pass

        elapsed = (time.monotonic() - t0) * 1000
        results = output["results"]
        web_n   = len(output.get("web_results", []))

        st.markdown(f"""
        <div class="info-bar">
            <div class="info-item"><b>{len(results)}</b> results found</div>
            <div class="info-item"><b>{elapsed:.0f}ms</b> response time</div>
            {"<div class='info-item'>⬡ <b>Web results included</b></div>" if output["web_activated"] else ""}
        </div>
        """, unsafe_allow_html=True)

        if output["web_activated"]:
            st.markdown('<div class="web-notice">⬡ &nbsp; Additional web sources included</div>', unsafe_allow_html=True)

        if results:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            for i, r in enumerate(results, 1):
                render_result(r, i)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">⬡</div>
                <div class="empty-hint">No results found for your query.<br>Try different keywords or switch to <b>Ask AI</b> mode.</div>
            </div>
            """, unsafe_allow_html=True)

elif clicked:
    st.warning("Please enter a query.")

else:
    # Suggestions
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <div class="empty-hint">
            Try searching for:<br><br>
            <i>fairness in machine learning</i> &nbsp;·&nbsp;
            <i>bias in NLP models</i><br>
            <i>AI transparency and accountability</i> &nbsp;·&nbsp;
            <i>explainability methods</i>
        </div>
    </div>
    """, unsafe_allow_html=True)