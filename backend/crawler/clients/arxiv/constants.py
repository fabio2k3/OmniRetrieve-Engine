"""
clients/arxiv/constants.py
==========================
Constantes del cliente arXiv.

Centraliza todos los valores literales del cliente para que ningún otro
módulo del subpaquete los duplique o hardcodee.
"""

# URLs de la API Atom y descarga de contenido
BASE_URL       = "https://export.arxiv.org/api/query"
ARXIV_HTML_URL = "https://arxiv.org/html/{local_id}"
ARXIV_PDF_URL  = "https://arxiv.org/pdf/{local_id}"

# Namespace XML de la API Atom
ATOM_NS = "http://www.w3.org/2005/Atom"

# Categorías de IA / ML usadas por defecto en las búsquedas
AI_ML_CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "stat.ML",
]
DEFAULT_SEARCH_QUERY = " OR ".join(f"cat:{c}" for c in AI_ML_CATEGORIES)

# Límites de descarga
MAX_SIZE_MB  = 15
CHUNK_BYTES  = 65_536    # tamaño de cada lectura en streaming
LOG_EVERY_KB = 512       # frecuencia de log de progreso
