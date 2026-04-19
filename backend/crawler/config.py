"""
config.py
=========
Configuración del Crawler.

Todas las constantes de comportamiento del Crawler viven aquí.
Ningún otro módulo del paquete debe hardcodear estos valores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .document import DOCUMENTS_CSV
from .id_store import IDS_CSV


@dataclass
class CrawlerConfig:
    """
    Parámetros de comportamiento del Crawler.

    Intervalos de hilos
    -------------------
    discovery_interval : segundos entre ciclos de descubrimiento de IDs.
    download_interval  : segundos entre ciclos de descarga de metadatos.
    pdf_interval       : pausa en segundos entre documentos en el ciclo de texto.

    Tamaños de lote
    ---------------
    ids_per_discovery : IDs a solicitar por ciclo de descubrimiento.
    batch_size        : documentos a procesar por ciclo de metadatos.
    pdf_batch_size    : documentos a procesar por ciclo de texto.

    Chunking
    --------
    chunk_size        : tamaño máximo de cada chunk en caracteres.
    overlap_sentences : oraciones compartidas entre chunks consecutivos.

    Rutas
    -----
    ids_csv       : ruta al CSV de IDs conocidos.
    documents_csv : ruta al CSV de metadatos.

    Estado
    ------
    discovery_start : offset de paginación para el descubrimiento.
    """

    # Intervalos
    discovery_interval: float = 120.0
    download_interval:  float = 30.0
    pdf_interval:       float = 2.0

    # Lotes
    ids_per_discovery: int = 100
    batch_size:        int = 10
    pdf_batch_size:    int = 10

    # Chunking
    chunk_size:        int = 1000
    overlap_sentences: int = 2

    # Rutas
    ids_csv:       Path = field(default_factory=lambda: IDS_CSV)
    documents_csv: Path = field(default_factory=lambda: DOCUMENTS_CSV)

    # Estado mutable (actualizado por DiscoveryLoop)
    discovery_start: int = 0
