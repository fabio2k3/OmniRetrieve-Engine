"""
base_client.py
==============
Interfaz abstracta que deben implementar todos los clientes de fuentes de datos.

Cada cliente es responsable de:
  1. Descubrir IDs locales de su fuente (fetch_ids).
  2. Descargar metadatos y devolver objetos Document (fetch_documents).
  3. Descargar el texto completo de un documento (download_text).

El texto crudo devuelto por download_text será convertido en chunks por el
Crawler usando el algoritmo centralizado de chunker.py, de modo que la lógica
de chunking no se duplique en cada cliente.

Formato de IDs compuestos
--------------------------
El Crawler construye un ID global único combinando la fuente y el ID local:

    doc_id = f"{client.source_name}:{local_id}"

Ejemplos:
    arxiv:2301.12345
    semantic_scholar:abc123def456

Este ID compuesto se almacena en el campo arxiv_id de Document y de la DB,
lo que mantiene compatibilidad con el resto de módulos (indexing, embedding,
retrieval) sin requerir ningún cambio en ellos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..document import Document


class BaseClient(ABC):
    """
    Interfaz que deben implementar todos los clientes de fuentes de datos.

    Implementación mínima requerida
    --------------------------------
    - source_name  → propiedad (str) que identifica la fuente, p.ej. "arxiv".
    - fetch_ids    → descubre IDs locales en la fuente.
    - fetch_documents → obtiene metadatos y devuelve Documents con ID compuesto.
    - download_text   → descarga el texto completo del documento.
    """

    # ── Identificador de fuente ───────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Nombre corto y estable de la fuente, p.ej. 'arxiv', 'semantic_scholar'.

        Se usa como prefijo del ID compuesto.  Debe ser un identificador
        alfanumérico sin espacios ni dos-puntos.
        """

    # ── Construcción del ID compuesto ─────────────────────────────────────────

    def make_doc_id(self, local_id: str) -> str:
        """
        Construye el ID compuesto a partir del ID local de la fuente.

        Formato: ``{source_name}:{local_id}``
        Ejemplo: ``arxiv:2301.12345``
        """
        return f"{self.source_name}:{local_id}"

    @staticmethod
    def parse_doc_id(doc_id: str) -> tuple[str, str]:
        """
        Descompone un ID compuesto en (source_name, local_id).

        Lanza ValueError si el formato no es válido.
        """
        parts = doc_id.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"ID compuesto inválido {doc_id!r}. "
                "Formato esperado: 'fuente:id_local'"
            )
        return parts[0], parts[1]

    # ── API obligatoria ───────────────────────────────────────────────────────

    @abstractmethod
    def fetch_ids(
        self,
        max_results: int = 100,
        start: int = 0,
    ) -> List[str]:
        """
        Descubre IDs locales disponibles en la fuente (sin prefijo).

        Parámetros
        ----------
        max_results : int
            Número máximo de IDs a devolver por llamada.
        start : int
            Offset de paginación.

        Returns
        -------
        List[str]
            IDs locales (sin prefijo de fuente).  El Crawler añadirá
            el prefijo usando make_doc_id().
        """

    @abstractmethod
    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        """
        Descarga los metadatos de los IDs locales indicados.

        El campo ``arxiv_id`` de cada Document devuelto debe contener el
        **ID compuesto** (``make_doc_id(local_id)``), no el local.

        Parámetros
        ----------
        local_ids : List[str]
            IDs sin prefijo de fuente (tal como los devuelve fetch_ids).

        Returns
        -------
        List[Document]
            Documentos con metadatos completos e ID compuesto.
        """

    @abstractmethod
    def download_text(self, local_id: str, **kwargs) -> str:
        """
        Descarga y devuelve el texto completo del documento.

        El cliente decide cómo obtener el texto (HTML, PDF, API, etc.).
        El Crawler llamará a este método y luego aplicará el chunking
        por separado usando chunker.make_chunks().

        Parámetros
        ----------
        local_id : str
            ID local del documento (sin prefijo de fuente).
        **kwargs
            Parámetros adicionales específicos de la fuente,
            p.ej. ``pdf_url`` para ArxivClient.

        Returns
        -------
        str
            Texto limpio del documento, listo para ser fragmentado.

        Raises
        ------
        RuntimeError
            Si no es posible descargar el texto por ningún método disponible.
        """
