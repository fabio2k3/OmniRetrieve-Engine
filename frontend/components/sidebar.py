"""
sidebar.py
==========
Panel lateral de estado del sistema para OmniRetrieve.

``render_sidebar(orc, err)`` debe llamarse dentro de un bloque ``with st.sidebar:``.

Secciones renderizadas
----------------------
  1. Cabecera (logo + nombre)
  2. Pipeline Modules — estado de LSI, FAISS, QRF, RAG, Full Pipeline
  3. Background Threads — estado is_alive() de cada hilo daemon
  4. Knowledge Base — stats de docs, chunks, FAISS, vocab
  5. Configuration — modelo de embedding, parámetros web
  6. Footer — uptime, timestamp, botón de refresh
"""

from __future__ import annotations

import streamlit as st


# ── Helpers internos ──────────────────────────────────────────────────────────

def _pill(text: str, cls: str) -> str:
    return f'<span class="sb-pill {cls}">{text}</span>'


def _ready_pill(ready: bool, label_on: str = "Ready", label_off: str = "Loading") -> str:
    if ready:
        return _pill(label_on, "pill-ready")
    return _pill(label_off, "pill-loading")


def _alive_pill(alive: bool) -> str:
    if alive:
        return _pill("Running", "pill-alive")
    return _pill("Stopped", "pill-stopped")


def _module_row(icon: str, name: str, pill_html: str) -> str:
    return f"""
    <div class="sb-module-row">
        <span class="sb-module-name">
            <span class="icon">{icon}</span>{name}
        </span>
        {pill_html}
    </div>"""


def _stat_row(label: str, value: str, cls: str = "", sub: bool = False) -> str:
    label_cls = "sb-stat-label sub" if sub else "sb-stat-label"
    value_cls = f"sb-stat-value {cls}".strip()
    return f"""
    <div class="sb-stat-row">
        <span class="{label_cls}">{label}</span>
        <span class="{value_cls}">{value}</span>
    </div>"""


def _progress_bar(label: str, value: int, total: int) -> str:
    if total <= 0:
        pct = 0
    else:
        pct = min(100, int(value / total * 100))
    return f"""
    <div class="sb-progress-wrap">
        <div class="sb-progress-label">
            <span>{label}</span>
            <span>{pct}%</span>
        </div>
        <div class="sb-progress-track">
            <div class="sb-progress-fill" style="width:{pct}%"></div>
        </div>
    </div>"""


def _config_row(key: str, val: str) -> str:
    return f"""
    <div class="sb-config-row">
        <span class="sb-config-key">{key}</span>
        <span class="sb-config-val">{val}</span>
    </div>"""


def _fmt_number(n: int | float) -> str:
    """Formatea números grandes con separador de miles."""
    if n < 0:
        return "—"
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"


def _fmt_uptime(seconds: int) -> str:
    """Convierte segundos a formato legible: '2h 15m 8s'."""
    if seconds < 0:
        return "—"
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    parts  = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ── Secciones ─────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown(
        """
        <div class="sb-header">
            <div class="sb-logo">⬡</div>
            <div>
                <div class="sb-title">OmniRetrieve</div>
                <div class="sb-version">Engine Monitor</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_modules(s: dict) -> None:
    rows = (
        _module_row("◈", "LSI Model",      _ready_pill(s.get("lsi_model_ready", False))) +
        _module_row("◉", "FAISS Index",    _ready_pill(s.get("faiss_ready",     False))) +
        _module_row("⊕", "QRF Pipeline",   _ready_pill(s.get("qrf_ready",       False))) +
        _module_row("◎", "RAG Pipeline",   _ready_pill(s.get("rag_ready",       False))) +
        _module_row("⬡", "Full Pipeline",  _ready_pill(s.get("pipeline_ready",  False)))
    )
    st.markdown(
        f'<div class="sb-section">'
        f'<div class="sb-section-title">Pipeline Modules</div>'
        f'{rows}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_thread_statuses(s: dict) -> None:
    thread_statuses: dict = s.get("thread_statuses", {})
    if not thread_statuses:
        return

    # Orden canónico de visualización
    order = ["crawler", "indexing", "lsi_rebuild", "embedding", "qrf_rag"]
    icons = {
        "crawler":     "↺",
        "indexing":    "⊟",
        "lsi_rebuild": "◈",
        "embedding":   "◉",
        "qrf_rag":     "⬡",
    }

    rows = ""
    for key in order:
        if key not in thread_statuses:
            continue
        info  = thread_statuses[key]
        label = info.get("label", key)
        alive = info.get("alive", False)
        icon  = icons.get(key, "·")
        rows += _module_row(icon, label, _alive_pill(alive))

    if rows:
        st.markdown(
            f'<div class="sb-section">'
            f'<div class="sb-section-title">Background Threads</div>'
            f'{rows}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_knowledge_base(s: dict) -> None:
    docs_total  = s.get("docs_total",       0)
    docs_idx    = s.get("docs_pdf_indexed", 0)
    docs_pend   = s.get("docs_pdf_pending", 0)
    recent      = s.get("recent_docs_24h",  0)
    total_ch    = s.get("total_chunks",     0)
    emb_ch      = s.get("embedded_chunks",  0)
    pend_ch     = s.get("pending_chunks",   0)
    faiss_vecs  = s.get("faiss_vectors",    0)
    vocab       = s.get("vocab_size",       0)
    postings    = s.get("total_postings",   0)
    lsi_docs    = s.get("lsi_docs_in_model",0)

    docs_pct_html  = _progress_bar("Indexed", docs_idx, max(docs_total, 1))
    chunks_pct_html = _progress_bar("Embedded", emb_ch, max(total_ch, 1))

    recent_cls = "" if recent <= 0 else ""
    pend_cls   = "pending" if pend_ch > 0 else "muted"

    html = (
        f'<div class="sb-section">'
        f'<div class="sb-section-title">Knowledge Base</div>'
        + _stat_row("Documents",   _fmt_number(docs_total))
        + _stat_row("├ Indexed",   _fmt_number(docs_idx),  sub=True)
        + _stat_row("└ Pending",   _fmt_number(docs_pend), cls=("pending" if docs_pend > 0 else "muted"), sub=True)
        + docs_pct_html
        + _stat_row("New (24h)",   _fmt_number(recent) if recent >= 0 else "—")
        + _stat_row("Chunks",      _fmt_number(total_ch))
        + _stat_row("├ Embedded",  _fmt_number(emb_ch),   sub=True)
        + _stat_row("└ Pending",   _fmt_number(pend_ch),  cls=pend_cls, sub=True)
        + chunks_pct_html
        + _stat_row("FAISS Vectors", _fmt_number(faiss_vecs))
        + _stat_row("LSI Docs",      _fmt_number(lsi_docs))
        + _stat_row("Vocab Size",    _fmt_number(vocab))
        + _stat_row("BM25 Postings", _fmt_number(postings))
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_configuration(s: dict) -> None:
    embed_model   = s.get("embed_model",    "—")
    web_threshold = s.get("web_threshold",  "—")
    web_min_docs  = s.get("web_min_docs",   "—")
    faiss_type    = s.get("faiss_index_type", "—")

    html = (
        f'<div class="sb-section">'
        f'<div class="sb-section-title">Configuration</div>'
        + _config_row("Embed model",    embed_model)
        + _config_row("FAISS type",     faiss_type)
        + _config_row("Web threshold",  str(web_threshold))
        + _config_row("Web min docs",   str(web_min_docs))
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_footer(s: dict) -> None:
    timestamp      = s.get("timestamp",      "")
    uptime_seconds = s.get("uptime_seconds", 0)
    uptime_str     = _fmt_uptime(uptime_seconds)

    st.markdown(
        f"""
        <div class="sb-footer">
            <div class="sb-uptime">⏱ uptime &nbsp; {uptime_str}</div>
            <div class="sb-timestamp">{timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── API pública ───────────────────────────────────────────────────────────────

def render_sidebar(orc, err: str | None) -> None:
    """
    Renderiza el sidebar completo de estado del sistema.

    Debe llamarse dentro de ``with st.sidebar:``.

    Parámetros
    ----------
    orc : instancia de Orchestrator (o None si no está disponible).
    err : mensaje de error de inicialización, o None si todo fue bien.
    """
    _render_header()

    # ── Error de inicialización ───────────────────────────────────────────────
    if err or orc is None:
        st.error(f"Orchestrator unavailable{f': {err}' if err else '.'}")
        if st.button("Retry", use_container_width=True, type="secondary"):
            st.rerun()
        return

    # ── Obtener status ────────────────────────────────────────────────────────
    try:
        s = orc.status()
    except Exception as exc:
        st.warning(f"Status unavailable: {exc}")
        if st.button("Refresh", use_container_width=True, type="secondary"):
            st.rerun()
        return

    # ── Secciones ─────────────────────────────────────────────────────────────
    _render_pipeline_modules(s)
    _render_thread_statuses(s)
    _render_knowledge_base(s)
    _render_configuration(s)
    _render_footer(s)

    # ── Botón de refresh ──────────────────────────────────────────────────────
    if st.button("⟳  Refresh", use_container_width=True, type="secondary", key="sb_refresh"):
        st.rerun()