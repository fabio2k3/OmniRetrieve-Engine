"""
components
==========
Módulos de UI reutilizables para la interfaz Streamlit de OmniRetrieve.

    styles.py    → inject_styles()  — inyecta todo el CSS global + sidebar
    sidebar.py   → render_sidebar() — panel lateral de estado del sistema
    results.py   → render_result(), render_paginated() — tarjetas de resultados
"""

from .styles  import inject_styles
from .sidebar import render_sidebar
from .results import render_result, render_paginated, normalize_result

__all__ = [
    "inject_styles",
    "render_sidebar",
    "render_result",
    "render_paginated",
    "normalize_result",
]