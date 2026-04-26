"""Subpaquete extractors — extracción de texto de arXiv (HTML y PDF)."""
from .html import extract as extract_html
from .pdf  import extract as extract_pdf

__all__ = ["extract_html", "extract_pdf"]
