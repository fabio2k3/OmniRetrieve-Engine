"""
base_client.py
==============
Interfaz abstracta que deben implementar todos los clientes de fuentes de datos.

Responsabilidades de un cliente
--------------------------------
Cada cliente es responsable de declarar y encapsular:

  1. Su política de crawling:
       request_delay    — pausa mínima entre peticiones HTTP (en segundos).
       trusted_domains  — dominios con acceso garantizado por ToS/API oficial,
                          incluso si robots.txt contiene Disallow para alguna ruta.

  2. Su lógica de adquisición:
       fetch_ids        — descubre IDs locales en la fuente.
       fetch_documents  — descarga metadatos y devuelve Documents con ID compuesto.
       download_text    — descarga el texto completo del documento.

Separación de responsabilidades
---------------------------------
  robots.py     → motor genérico de robots.txt (sin conocimiento de ninguna fuente)
  BaseClient    → declara la política (qué dominios confiar, qué delay aplicar)
  ArxivClient   → implementa la política para arXiv
  RobotsChecker → aplica la política que le pasa el cliente en cada llamada

Delay efectivo
--------------
El delay real entre peticiones es siempre el máximo de dos valores:

    effective_delay = max(client.request_delay, checker.crawl_delay(url))

- ``client.request_delay``  es el compromiso mínimo del cliente.
- ``checker.crawl_delay()`` es lo que exige robots.txt (siempre se lee).

Así se garantiza que nunca se viola ninguno de los dos límites.

Formato de IDs compuestos
--------------------------
    doc_id = f"{source_name}:{local_id}"

Ejemplos:
    arxiv:2301.12345
    semantic_scholar:abc123def456
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet, List

from ..document import Document


class BaseClient(ABC):
    """
    Interfaz que todos los clientes de fuentes de datos deben implementar.

    Implementación mínima obligatoria
    ----------------------------------
    Propiedades abstractas:
        source_name      → identificador único de la fuente ('arxiv', 'ss', …)
        request_delay    → segundos mínimos entre peticiones HTTP
        trusted_domains  → dominios con acceso por ToS (puede ser frozenset vacío)

    Métodos abstractos:
        fetch_ids        → lista de IDs locales disponibles en la fuente
        fetch_documents  → metadatos de una lista de IDs locales
        download_text    → texto completo de un documento

    Métodos concretos (no sobreescribir salvo causa justificada):
        make_doc_id      → construye el ID compuesto "source:local_id"
        parse_doc_id     → descompone un ID compuesto en (source, local_id)
    """

    # ── Identificador de fuente ───────────────────────────────────────────────

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Nombre corto y estable de la fuente, p.ej. 'arxiv'.
        Alfanumérico, sin espacios ni dos-puntos.
        Se usa como prefijo del ID compuesto.
        """

    # ── Política de crawling ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def request_delay(self) -> float:
        """
        Pausa mínima en segundos entre peticiones HTTP consecutivas.

        Este valor es el compromiso mínimo del cliente con la fuente.
        El delay efectivo real será:

            max(self.request_delay, checker.crawl_delay(url))

        Por tanto, si robots.txt exige más, se respeta el valor mayor.
        Nunca debe ser inferior al Crawl-delay del robots.txt de la fuente.
        """

    @property
    @abstractmethod
    def trusted_domains(self) -> FrozenSet[str]:
        """
        Conjunto de dominios cuyo acceso programático está explícitamente
        autorizado por sus Términos de Servicio o documentación de API,
        incluso si robots.txt contiene Disallow para alguna ruta concreta.

        Este conjunto se pasa a ``RobotsChecker.allowed()`` como override
        para evitar falsos negativos en la comprobación de acceso.

        IMPORTANTE: declarar un dominio aquí NO exime del Crawl-delay.
        ``RobotsChecker.crawl_delay()`` siempre lee robots.txt, sin bypass.

        Devuelve frozenset() si la fuente no tiene ningún dominio especial.

        Ejemplo (ArxivClient):
            frozenset({"arxiv.org", "export.arxiv.org"})

            Motivo: robots.txt de arXiv tiene 'Disallow: /api', pero su
            API en export.arxiv.org está explícitamente permitida por ToS.
        """

    # ── Construcción de IDs compuestos ────────────────────────────────────────

    def make_doc_id(self, local_id: str) -> str:
        """
        Construye el ID compuesto ``{source_name}:{local_id}``.
        Ejemplo: ``arxiv:2301.12345``
        """
        return f"{self.source_name}:{local_id}"

    @staticmethod
    def parse_doc_id(doc_id: str) -> tuple[str, str]:
        """
        Descompone ``"source:local_id"`` en ``(source, local_id)``.
        Lanza ValueError si el formato no es válido.
        """
        parts = doc_id.split(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"ID compuesto inválido {doc_id!r}. "
                "Formato esperado: 'fuente:id_local'"
            )
        return parts[0], parts[1]

    # ── API de adquisición ────────────────────────────────────────────────────

    @abstractmethod
    def fetch_ids(self, max_results: int = 100, start: int = 0) -> List[str]:
        """
        Descubre IDs locales disponibles en la fuente (sin prefijo de fuente).

        Returns
        -------
        List[str]
            IDs locales.  El Crawler añade el prefijo con make_doc_id().
        """

    @abstractmethod
    def fetch_documents(self, local_ids: List[str]) -> List[Document]:
        """
        Descarga metadatos de los IDs locales indicados.

        El campo ``doc_id`` de cada Document devuelto debe ser el ID
        compuesto (``make_doc_id(local_id)``), no el ID local.
        """

    @abstractmethod
    def download_text(self, local_id: str, **kwargs) -> str:
        """
        Descarga el texto completo del documento y lo devuelve limpio.

        El cliente decide cómo obtenerlo (HTML, PDF, API REST, etc.).
        El Crawler fragmentará el texto después con chunker.make_chunks().

        Raises
        ------
        RuntimeError
            Si no es posible descargar el texto por ningún método.
        """
