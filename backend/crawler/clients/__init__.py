"""backend/crawler/clients — fuentes de datos del crawler."""
from .base_client import BaseClient
from .arxiv_client import ArxivClient

__all__ = ["BaseClient", "ArxivClient"]
