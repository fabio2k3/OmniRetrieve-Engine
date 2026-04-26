"""Subpaquete loops — hilos daemon del Crawler."""
from .discovery  import DiscoveryLoop
from .downloader import DownloaderLoop
from .text       import TextLoop

__all__ = ["DiscoveryLoop", "DownloaderLoop", "TextLoop"]
