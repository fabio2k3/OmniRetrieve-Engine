"""
web_search/sites.py
===================
Registro de sitios web académicos de confianza para búsqueda web dirigida.

Estos dominios actúan como semillas: el sistema solo busca y recupera
contenido de estas fuentes, evitando resultados irrelevantes de búsqueda
general (redes sociales, noticias, etc.).

El corpus objetivo del proyecto es IA, ML y ética de la IA, por lo que
los sitios están seleccionados para cubrir:
  · Repositorios de preprints y papers (excludiendo arXiv — ya lo cubre el crawler)
  · Conferencias y actas de ML/NLP/IA
  · Centros de investigación en ética de IA
  · Think tanks y publicaciones especializadas en política de IA

Uso
---
    from backend.web_search.sites import DEFAULT_SEED_DOMAINS, get_site_filter

    # Genera el filtro site: para DuckDuckGo
    filter_str = get_site_filter(DEFAULT_SEED_DOMAINS)
    # → "site:semanticscholar.org OR site:paperswithcode.com OR ..."

Añadir sitios nuevos
--------------------
Basta con añadir el dominio a DEFAULT_SEED_DOMAINS o pasarlo como
parámetro a WebSearchPipeline vía cfg.web_seed_domains.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dominios por categoría (documentación interna)
# ---------------------------------------------------------------------------

#: Repositorios y motores de búsqueda académica
ACADEMIC_SEARCH = [
    "semanticscholar.org",    # motor de búsqueda académica general
    "paperswithcode.com",     # papers ML con código reproducible
    "openreview.net",         # revisiones abiertas: NeurIPS, ICLR, ICML
    "aclanthology.org",       # actas de conferencias NLP (ACL, EMNLP, NAACL)
]

#: Actas de conferencias de ML
CONFERENCE_PROCEEDINGS = [
    "proceedings.neurips.cc",  # NeurIPS
    "proceedings.mlr.press",   # PMLR: ICML, AISTATS, UAI, COLT
    "dl.acm.org",              # ACM Digital Library: FAccT, AIES
    "ieeexplore.ieee.org",     # IEEE: CVPR, ICCV, ECCV (parcialmente)
]

#: Institutos de investigación en ética y política de IA
AI_ETHICS_RESEARCH = [
    "ainowinstitute.org",       # AI Now Institute — impacto social de IA
    "partnershiponai.org",      # Partnership on AI
    "algorithmwatch.org",       # vigilancia de sistemas algorítmicos
    "fairmlbook.org",           # Fairness and Machine Learning (libro abierto)
    "montreal.ai",              # Montreal AI Ethics Institute
]

#: Publicaciones especializadas
SPECIALIZED_PUBLICATIONS = [
    "distill.pub",              # artículos ML explicables e interactivos
    "technologyreview.com",     # MIT Technology Review
    "brookings.edu",            # Brookings Institution — política de IA
    "nature.com",               # Nature: Machine Intelligence y afines
]

# ---------------------------------------------------------------------------
# Lista por defecto — usada si cfg.web_seed_domains está vacío
# ---------------------------------------------------------------------------

DEFAULT_SEED_DOMAINS: list[str] = (
    ACADEMIC_SEARCH
    + CONFERENCE_PROCEEDINGS
    + AI_ETHICS_RESEARCH
    + SPECIALIZED_PUBLICATIONS
)


def get_site_filter(domains: list[str]) -> str:
    """
    Genera el filtro 'site:' para DuckDuckGo a partir de una lista de dominios.

    Ejemplo
    -------
    >>> get_site_filter(["semanticscholar.org", "paperswithcode.com"])
    'site:semanticscholar.org OR site:paperswithcode.com'
    """
    if not domains:
        return ""
    return " OR ".join(f"site:{d}" for d in domains)