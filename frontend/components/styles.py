"""
styles.py
=========
Todo el CSS global de OmniRetrieve en un único lugar.

``inject_styles()`` debe llamarse una sola vez al inicio de ``app.py``,
antes de renderizar cualquier otro componente.

Estructura
----------
  1. Variables CSS (--bg, --card, --accent, …)
  2. Reset de elementos Streamlit (header, footer, menu)
  3. Fondo: grid + glow
  4. Hero
  5. Botones
  6. Input de consulta
  7. Info-bar de resultados
  8. Web-notice
  9. Answer box
 10. Section label
 11. Result cards
 12. Paginación
 13. Empty state
 14. Sidebar personalizado
"""

from __future__ import annotations
import streamlit as st


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ═══════════════════════════════════
   1. Variables
   ═══════════════════════════════════ */
:root {
    --bg:          #050814;
    --card:        rgba(14, 20, 38, 0.76);
    --border:      rgba(148, 163, 184, 0.16);
    --accent:      #2f80ff;
    --accent2:     #00d4ff;
    --accent3:     #2ee59d;
    --accent4:     #f59e0b;
    --text:        #f6f9ff;
    --text-muted:  #9aa7ba;
    --radius:      18px;
    --shadow:      0 18px 50px rgba(0,0,0,0.36);
    --shadow-soft: 0 10px 28px rgba(0,0,0,0.22);
    --green:  #22c55e;
    --yellow: #f59e0b;
    --red:    #ef4444;
    --blue:   #3b82f6;
}

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* ═══════════════════════════════════
   2. Reset Streamlit
   ═══════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

/* ═══════════════════════════════════
   3. Fondo
   ═══════════════════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse 70% 50% at 15% 10%, rgba(47,128,255,0.07), transparent),
        radial-gradient(ellipse 55% 45% at 85% 8%,  rgba(0,212,255,0.05),  transparent),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(46,229,157,0.04), transparent),
        #050814;
    color: var(--text);
    font-family: 'Syne', sans-serif;
}

.bg-grid {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse 60% 55% at 50% 50%, black, transparent);
    -webkit-mask-image: radial-gradient(ellipse 60% 55% at 50% 50%, black, transparent);
}

.bg-glow { position: fixed; inset: 0; pointer-events: none; z-index: 0; }
.bg-glow::before {
    content: ""; position: absolute;
    width: 480px; height: 480px; top: -140px; right: -160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(47,128,255,0.09), transparent 68%);
    filter: blur(18px);
}
.bg-glow::after {
    content: ""; position: absolute;
    width: 420px; height: 420px; bottom: -140px; left: -140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(46,229,157,0.07), transparent 68%);
    filter: blur(18px);
}

.block-container {
    position: relative; z-index: 1;
    padding-top: 2.2rem; padding-bottom: 2.8rem;
    max-width: 940px;
}

/* ═══════════════════════════════════
   4. Hero
   ═══════════════════════════════════ */
.hero { text-align: center; padding: 1.2rem 1rem 1.1rem; margin-bottom: 0.5rem; }

.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.85rem; border-radius: 999px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    font-size: 0.72rem; color: rgba(219,234,254,0.7);
    margin-bottom: 1.15rem; letter-spacing: 0.04em;
}

.hero-logo {
    width: 72px; height: 72px; margin: 0 auto 1rem;
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

/* ═══════════════════════════════════
   5. Botones
   ═══════════════════════════════════ */
[data-testid="stButton"] > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 999px !important; border: 1px solid transparent !important;
    transition: all 0.2s ease !important; font-size: 0.9rem !important;
    padding: 0.7rem 1.4rem !important;
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
    background: rgba(16,22,40,0.80) !important; color: var(--text-muted) !important;
    border: 1px solid rgba(148,163,184,0.12) !important; backdrop-filter: blur(10px);
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: rgba(0,212,255,0.28) !important; color: var(--text) !important;
}

/* ═══════════════════════════════════
   6. Input
   ═══════════════════════════════════ */
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

/* ═══════════════════════════════════
   7. Info bar
   ═══════════════════════════════════ */
.info-bar {
    display: flex; justify-content: center; gap: 1rem;
    padding: 0.7rem 1.1rem; background: rgba(12,18,36,0.70);
    border: 1px solid rgba(148,163,184,0.12); border-radius: 999px;
    margin: 1.15rem 0; flex-wrap: wrap; backdrop-filter: blur(12px);
}
.info-item { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-muted); }
.info-item b { color: var(--text); }

/* ═══════════════════════════════════
   8. Web notice
   ═══════════════════════════════════ */
.web-notice {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.65rem 1rem; margin: 0.8rem auto 1.15rem; width: fit-content;
    background: rgba(46,229,157,0.06); border: 1px solid rgba(46,229,157,0.16);
    border-radius: 999px;
    font-family: 'DM Mono', monospace; font-size: 0.71rem; color: #8ff0c8;
    backdrop-filter: blur(10px);
}

/* ═══════════════════════════════════
   9. Answer box
   ═══════════════════════════════════ */
.answer-box {
    background: rgba(14,20,38,0.88); border: 1px solid rgba(47,128,255,0.20);
    border-radius: 20px; padding: 1.5rem 1.7rem; margin: 1.15rem 0;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow); backdrop-filter: blur(14px);
}
.answer-box::before {
    content: ''; position: absolute; inset: 0 auto 0 0; width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent3)); opacity: 0.7;
}
.answer-label {
    position: relative; z-index: 1;
    font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--accent2);
    text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 0.75rem; opacity: 0.85;
}
.answer-text { position: relative; z-index: 1; font-size: 0.97rem; line-height: 1.85; color: var(--text); }
.answer-sources {
    position: relative; z-index: 1; margin-top: 1rem; padding-top: 0.85rem;
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

/* ═══════════════════════════════════
   10. Section label
   ═══════════════════════════════════ */
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

/* ═══════════════════════════════════
   11. Result cards
   ═══════════════════════════════════ */
.result-card {
    background: rgba(13,19,36,0.80); border: 1px solid rgba(148,163,184,0.10);
    border-radius: var(--radius); padding: 1.1rem 1.3rem; margin-bottom: 0.75rem;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden; backdrop-filter: blur(10px);
}
.result-card::before {
    content: ''; position: absolute; inset: 0 auto 0 0; width: 3px;
    background: linear-gradient(180deg, rgba(47,128,255,0.6), rgba(46,229,157,0.5));
}
.result-card:hover {
    border-color: rgba(47,128,255,0.28); transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}
.result-card.is-web::before {
    background: linear-gradient(180deg, rgba(46,229,157,0.6), rgba(0,212,255,0.5));
}

.result-title    { font-size: 0.97rem; font-weight: 700; color: var(--text); margin-bottom: 0.4rem; line-height: 1.35; }
.result-abstract { font-size: 0.82rem; color: rgba(154,167,186,0.85); line-height: 1.7; font-family: 'DM Mono', monospace; font-weight: 300; }
.result-footer   { display: flex; gap: 0.6rem; align-items: center; margin-top: 0.75rem; flex-wrap: wrap; }

.result-tag {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    padding: 0.14rem 0.55rem; border-radius: 999px;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.tag-local { background: rgba(47,128,255,0.10); color: #d6e7ff; border: 1px solid rgba(47,128,255,0.18); }
.tag-web   { background: rgba(46,229,157,0.08); color: #8ff0c8; border: 1px solid rgba(46,229,157,0.16); }

.result-link { font-family: 'DM Mono', monospace; font-size: 0.67rem; color: #93c5fd; text-decoration: none; }
.result-link:hover { text-decoration: underline; }

/* ═══════════════════════════════════
   12. Paginación
   ═══════════════════════════════════ */
.page-info {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: var(--text-muted); text-align: center; margin-top: 0.5rem;
}

/* ═══════════════════════════════════
   13. Empty state
   ═══════════════════════════════════ */
.empty-state {
    text-align: center; padding: 2.5rem 1rem;
    background: rgba(12,18,36,0.40); border: 1px dashed rgba(148,163,184,0.14);
    border-radius: 20px; backdrop-filter: blur(10px);
}
.empty-icon {
    width: 60px; height: 60px; border-radius: 18px; margin: 0 auto 0.85rem;
    display: grid; place-items: center; font-size: 1.75rem; color: #d6e7ff;
    background: rgba(47,128,255,0.10); border: 1px solid rgba(47,128,255,0.14);
}
.empty-hint { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--text-muted); line-height: 2; }
.empty-hint i { color: var(--accent2); font-style: normal; }

hr { border-color: rgba(148,163,184,0.10) !important; }

/* ═══════════════════════════════════
   14. Sidebar personalizado
   ═══════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(6, 10, 22, 0.96) !important;
    border-right: 1px solid rgba(148,163,184,0.08) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] > div { padding: 1.2rem 0.85rem; }

/* Scrollbar del sidebar */
[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.15); border-radius: 999px;
}

/* ── Cabecera del sidebar ── */
.sb-header {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.5rem 0 1rem;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    margin-bottom: 1rem;
}
.sb-logo {
    width: 32px; height: 32px; border-radius: 10px;
    display: grid; place-items: center; font-size: 1rem;
    background: linear-gradient(135deg, rgba(47,128,255,0.85), rgba(0,212,255,0.85));
    flex-shrink: 0;
}
.sb-title   { font-size: 0.9rem; font-weight: 700; color: var(--text); line-height: 1; }
.sb-version { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--text-muted); }

/* ── Secciones del sidebar ── */
.sb-section {
    margin-bottom: 1rem;
}
.sb-section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: rgba(154,167,186,0.5);
    margin-bottom: 0.55rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(148,163,184,0.06);
}

/* ── Filas de módulo ── */
.sb-module-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.38rem 0.5rem; border-radius: 8px;
    margin-bottom: 0.22rem;
    transition: background 0.15s ease;
}
.sb-module-row:hover { background: rgba(255,255,255,0.025); }
.sb-module-name {
    font-size: 0.78rem; font-weight: 500; color: var(--text-muted);
    display: flex; align-items: center; gap: 0.45rem;
}
.sb-module-name .icon { font-size: 0.7rem; opacity: 0.7; }

.sb-pill {
    font-family: 'DM Mono', monospace; font-size: 0.55rem; font-weight: 500;
    padding: 0.14rem 0.5rem; border-radius: 999px; letter-spacing: 0.04em;
    white-space: nowrap;
}
.pill-ready   { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.22); }
.pill-loading { background: rgba(245,158,11,0.10); color: #fbbf24; border: 1px solid rgba(245,158,11,0.20); }
.pill-stopped { background: rgba(239,68,68,0.10);  color: #f87171; border: 1px solid rgba(239,68,68,0.18); }
.pill-alive   { background: rgba(59,130,246,0.10); color: #93c5fd; border: 1px solid rgba(59,130,246,0.18); }

/* ── Stats del sidebar ── */
.sb-stat-row {
    display: flex; align-items: baseline; justify-content: space-between;
    padding: 0.3rem 0.5rem; border-radius: 6px;
    margin-bottom: 0.12rem;
}
.sb-stat-row:hover { background: rgba(255,255,255,0.02); }
.sb-stat-label {
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: var(--text-muted);
}
.sb-stat-label.sub {
    padding-left: 0.85rem; font-size: 0.63rem;
    color: rgba(154,167,186,0.55);
}
.sb-stat-value {
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    font-weight: 500; color: var(--text);
}
.sb-stat-value.muted { color: var(--text-muted); }
.sb-stat-value.pending { color: #fbbf24; }

/* ── Progress bar ── */
.sb-progress-wrap {
    padding: 0.25rem 0.5rem 0.6rem;
}
.sb-progress-label {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace; font-size: 0.6rem;
    color: var(--text-muted); margin-bottom: 0.3rem;
}
.sb-progress-track {
    height: 3px; border-radius: 999px;
    background: rgba(148,163,184,0.10);
    overflow: hidden;
}
.sb-progress-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.4s ease;
}

/* ── Config badges ── */
.sb-config-row {
    display: flex; align-items: flex-start; justify-content: space-between;
    padding: 0.3rem 0.5rem;
}
.sb-config-key {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    color: rgba(154,167,186,0.55);
}
.sb-config-val {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    color: var(--text-muted); text-align: right; max-width: 60%;
    word-break: break-all;
}

/* ── Footer del sidebar ── */
.sb-footer {
    margin-top: 0.75rem; padding-top: 0.75rem;
    border-top: 1px solid rgba(148,163,184,0.07);
}
.sb-timestamp {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    color: rgba(154,167,186,0.35); text-align: center;
    margin-bottom: 0.6rem;
}
.sb-uptime {
    font-family: 'DM Mono', monospace; font-size: 0.62rem;
    color: rgba(154,167,186,0.45); text-align: center;
    margin-bottom: 0.75rem;
}

/* ── Botón de refresh en sidebar ── */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    font-size: 0.75rem !important;
    padding: 0.45rem 0.9rem !important;
    border-radius: 8px !important;
}
"""


def inject_styles() -> None:
    """Inyecta el CSS global en la aplicación Streamlit."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="bg-grid"></div><div class="bg-glow"></div>',
        unsafe_allow_html=True,
    )