"""backend/crawler — adquisición y fragmentación de documentos."""

from .document import Document
from .id_store import IdStore
from .chunker import make_chunks
from .clients.base_client import BaseClient
from .clients.arxiv_client import ArxivClient
from .crawler import Crawler, CrawlerConfig
from .robots import checker as robots_checker

__all__ = [
    # Modelo de datos
    "Document",
    "IdStore",
    # Clientes
    "BaseClient",
    "ArxivClient",
    # Crawler
    "Crawler",
    "CrawlerConfig",
    # Utilidades
    "make_chunks",
    "robots_checker",
]
