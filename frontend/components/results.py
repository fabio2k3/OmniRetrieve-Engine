"""
results.py
==========
Componentes de renderizado de resultados de búsqueda para OmniRetrieve.

API pública
-----------
normalize_result(r)           — normaliza cualquier resultado a dict plano.
render_result(r)              — renderiza una tarjeta de resultado.
render_paginated(results, key)— renderiza resultados con paginación.
"""

from __future__ import annotations

import streamlit as st

# Resultados por página en modo Search
PAGE_SIZE: int = 5


def normalize_result(r) -> dict:
    """
    Convierte cualquier resultado a un dict plano con las claves
    que necesita ``render_result()``.

    Cubre dicts del LSIRetriever, dicts de ``build_sources()`` del RAG
    y objetos con ``__dict__``.
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
    """
    Renderiza una tarjeta de resultado (paper local o fuente web).

    Parámetros
    ----------
    r : dict normalizado (salida de ``normalize_result``).
    """
    is_web   = r.get("source", "local") in ("web", "web_fallback")
    card_cls = "result-card is-web" if is_web else "result-card"
    tag      = (
        '<span class="result-tag tag-web">Web</span>'
        if is_web else
        '<span class="result-tag tag-local">Research paper</span>'
    )
    url      = r.get("url", "")
    link     = (
        f'<a class="result-link" href="{url}" target="_blank">↗ Read paper</a>'
        if url else ""
    )
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


def render_paginated(results: list, page_key: str) -> None:
    """
    Muestra los resultados en páginas de ``PAGE_SIZE`` elementos.

    Parámetros
    ----------
    results  : lista de resultados (dicts crudos o normalizados).
    page_key : clave única por consulta para que la paginación no se
               comparta entre búsquedas distintas.
    """
    if not results:
        return

    total_pages = max(1, -(-len(results) // PAGE_SIZE))  # ceil division

    state_key = f"page_{page_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    page = st.session_state[state_key]
    page = max(0, min(page, total_pages - 1))  # clamp

    start = page * PAGE_SIZE
    for r in results[start : start + PAGE_SIZE]:
        render_result(normalize_result(r))

    # Controles de paginación (solo si hay más de una página)
    if total_pages > 1:
        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        col_prev, col_info, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button(
                "← Prev", key=f"prev_{page_key}",
                disabled=(page == 0),
                use_container_width=True, type="secondary",
            ):
                st.session_state[state_key] -= 1
                st.rerun()

        with col_info:
            st.markdown(
                f'<div class="page-info">Page {page + 1} of {total_pages} '
                f'· {len(results)} results</div>',
                unsafe_allow_html=True,
            )

        with col_next:
            if st.button(
                "Next →", key=f"next_{page_key}",
                disabled=(page >= total_pages - 1),
                use_container_width=True, type="secondary",
            ):
                st.session_state[state_key] += 1
                st.rerun()