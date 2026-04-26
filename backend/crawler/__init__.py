"""backend/crawler — adquisición y fragmentación de documentos."""
from .document import Document
from .id_store import IdStore
from .chunker  import make_chunks
from .config   import CrawlerConfig
from .crawler  import Crawler
from .robots   import checker as robots_checker
from .clients.base_client import BaseClient
from .clients.arxiv       import ArxivClient

__all__ = [
    # Modelo de datos
    "Document",
    "IdStore",
    # Configuración
    "CrawlerConfig",
    # Clientes
    "BaseClient",
    "ArxivClient",
    # Orquestador
    "Crawler",
    # Utilidades
    "make_chunks",
    "robots_checker",
]
