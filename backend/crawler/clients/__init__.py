"""backend/crawler/clients — fuentes de datos del crawler."""
from .base_client  import BaseClient
from .arxiv        import ArxivClient

__all__ = ["BaseClient", "ArxivClient"]
