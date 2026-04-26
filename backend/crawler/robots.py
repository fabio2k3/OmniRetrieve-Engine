"""
robots.py
=========
Verificador de robots.txt genérico, completamente agnóstico de fuente.

Diseño
------
Este módulo no sabe nada de arXiv ni de ninguna otra fuente.
Toda la política de crawling (dominios de confianza, delay mínimo) la declara
cada cliente en su propia clase a través de las propiedades de ``BaseClient``:

    client.trusted_domains  → dominios con acceso permitido por ToS.
    client.request_delay    → delay mínimo que el cliente se compromete a respetar.

``RobotsChecker`` expone dos métodos ortogonales e independientes:

    allowed(url, trusted_domains)
        True si el User-Agent puede acceder a la URL.
        Si el host está en ``trusted_domains``, devuelve True sin consultar
        robots.txt (permite declarar acceso a APIs autorizadas por ToS aunque
        robots.txt incluya un Disallow genérico).

    crawl_delay(url)
        Segundos de Crawl-delay declarados en robots.txt.
        Nunca hace bypass, ni siquiera para dominios de confianza.

El delay efectivo real lo calcula el cliente:
    effective = max(self.request_delay, checker.crawl_delay(url))

Singleton
---------
``checker`` es la instancia compartida por todo el paquete.
Los clientes no deben instanciar ``RobotsChecker`` directamente.
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.robotparser
from typing import FrozenSet
from urllib.parse import urlparse

from .http import USER_AGENT, fetch_bytes

logger = logging.getLogger(__name__)

ROBOTS_CACHE_TTL = 3600  # segundos


# ---------------------------------------------------------------------------
# RobotsChecker
# ---------------------------------------------------------------------------

class RobotsChecker:
    """
    Verificador genérico de robots.txt con caché TTL thread-safe.

    No contiene ningún dominio hardcodeado. La política de confianza
    (qué dominios se consideran seguros pese a robots.txt) la aporta
    cada cliente en cada llamada a ``allowed()``.

    Parámetros del constructor
    --------------------------
    ttl : segundos de validez de cada entrada en el caché (por defecto 3600).
    """

    def __init__(self, ttl: float = ROBOTS_CACHE_TTL) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}
        self._lock  = threading.Lock()

    # ── Helpers internos ─────────────────────────────────────────────────────

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        """Devuelve ``(scheme://host, host)`` para una URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}", parsed.netloc

    def _get_parser(self, origin: str) -> urllib.robotparser.RobotFileParser:
        """
        Devuelve el parser de robots.txt para ``origin``, usando el caché
        si el TTL no ha expirado. El fetch se hace fuera del lock para no
        bloquear otros hilos durante la petición de red.
        """
        with self._lock:
            now    = time.monotonic()
            cached = self._cache.get(origin)
            if cached:
                parser, fetched_at = cached
                if now - fetched_at < self._ttl:
                    return parser

        robots_url = f"{origin}/robots.txt"
        logger.debug("Obteniendo %s", robots_url)

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        raw = fetch_bytes(robots_url)
        if raw is not None:
            parser.parse(raw.decode("utf-8", errors="replace").splitlines())
        else:
            logger.warning(
                "robots.txt inaccesible para %s — se asume acceso permitido.", origin
            )
            parser.parse(["User-agent: *", "Allow: /"])

        with self._lock:
            self._cache[origin] = (parser, time.monotonic())

        return parser

    # ── API pública ───────────────────────────────────────────────────────────

    def allowed(
        self,
        url: str,
        trusted_domains: FrozenSet[str] = frozenset(),
    ) -> bool:
        """
        Indica si ``USER_AGENT`` puede acceder a ``url``.

        Parámetros
        ----------
        url             : URL a comprobar.
        trusted_domains : dominios cuyo acceso está sancionado por ToS o API
                          oficial, incluso si robots.txt tiene Disallow.

        Returns
        -------
        bool
            ``True`` si el acceso está permitido o si el dominio es de confianza.
            ``True`` también ante errores de red (fail-open).
        """
        _, host = self._origin(url)
        if host in trusted_domains:
            return True
        try:
            origin, _ = self._origin(url)
            parser    = self._get_parser(origin)
            result    = parser.can_fetch(USER_AGENT, url)
            if not result:
                logger.info("robots.txt prohíbe: %s", url)
            return result
        except Exception as exc:
            logger.warning("Error comprobando robots.txt para %s: %s — permitiendo.", url, exc)
            return True

    def crawl_delay(self, url: str) -> float:
        """
        Segundos de Crawl-delay declarados en robots.txt (``0.0`` si no definido).

        Nunca hace bypass: el delay se lee siempre del robots.txt real,
        independientemente de ``trusted_domains``.

        Returns
        -------
        float
            Delay en segundos, o ``0.0`` si no está declarado o hay error.
        """
        try:
            origin, _ = self._origin(url)
            parser    = self._get_parser(origin)
            delay     = parser.crawl_delay(USER_AGENT)
            return float(delay) if delay is not None else 0.0
        except Exception:
            return 0.0


# Singleton compartido por todo el paquete crawler
checker = RobotsChecker()
