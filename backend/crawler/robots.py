"""
robots.py
=========
Verificador de robots.txt genérico, completamente agnóstico de fuente.

Diseño
------
Este módulo no sabe nada de arXiv ni de ninguna otra fuente.
Toda la política de crawling (dominios de confianza, delay mínimo)
la declara cada cliente en su propia clase, a través de las propiedades
que define BaseClient:

    client.trusted_domains  → set de dominios con acceso permitido por ToS
    client.request_delay    → delay mínimo que el cliente se compromete a respetar

RobotsChecker expone dos métodos ortogonales e independientes:

    allowed(url, trusted_domains)
        True si el User-Agent puede acceder a la URL.
        - Para URLs cuyo host está en trusted_domains: siempre True.
          (Permite que un cliente declare que su API está autorizada por ToS
          aunque robots.txt incluya un Disallow genérico para esa ruta.)
        - Para el resto: consulta robots.txt con caché TTL.

    crawl_delay(url)
        Segundos de Crawl-delay declarados en robots.txt.
        NUNCA hace bypass, ni siquiera para trusted_domains.
        El delay es una restricción de frecuencia independiente del acceso;
        si robots.txt lo declara, hay que respetarlo siempre.

El delay efectivo real lo calcula cada cliente:
    effective = max(self.request_delay, checker.crawl_delay(url))

Singleton
---------
``checker`` es la instancia compartida por todo el paquete.
Los clientes no deben instanciar RobotsChecker directamente.
"""

from __future__ import annotations

import logging
import ssl
import threading
import time
import urllib.request
import urllib.robotparser
from typing import FrozenSet, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

USER_AGENT       = "SRI-Crawler/1.0"
ROBOTS_CACHE_TTL = 3600  # segundos


# ---------------------------------------------------------------------------
# SSL context compartido
# ---------------------------------------------------------------------------
def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        return ctx

_SSL_CTX = _build_ssl_context()


def _fetch_robots_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return resp.read()
    except Exception as exc:
        logger.warning("No se pudo obtener robots.txt de %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# RobotsChecker
# ---------------------------------------------------------------------------
class RobotsChecker:
    """
    Verificador genérico de robots.txt con caché TTL thread-safe.

    No contiene ningún dominio hardcodeado.  La política de confianza
    (qué dominios se consideran seguros pese a robots.txt) la aporta
    cada cliente en cada llamada a allowed().
    """

    def __init__(self, ttl: float = ROBOTS_CACHE_TTL) -> None:
        self._ttl        = ttl
        self._cache: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}
        self._cache_lock = threading.Lock()   # protege _cache en entornos multi-hilo

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        """Devuelve (scheme://host, host) para una URL."""
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}", p.netloc

    def _get_parser(self, origin: str) -> urllib.robotparser.RobotFileParser:
        # Consulta rápida bajo lock para ver si el caché es válido
        with self._cache_lock:
            now    = time.monotonic()
            cached = self._cache.get(origin)
            if cached:
                parser, fetched_at = cached
                if now - fetched_at < self._ttl:
                    return parser

        # Fetch fuera del lock para no bloquear otros hilos durante la red.
        # Puede haber una carrera benigna (dos hilos fetchean a la vez),
        # pero el resultado es idempotente: ambos obtienen el mismo robots.txt.
        robots_url = f"{origin}/robots.txt"
        logger.debug("Obteniendo %s", robots_url)

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        raw = _fetch_robots_bytes(robots_url)
        if raw is not None:
            parser.parse(raw.decode("utf-8", errors="replace").splitlines())
        else:
            # No se pudo obtener robots.txt → fail-open: permitir todo
            logger.warning(
                "robots.txt inaccesible para %s — se asume acceso permitido.", origin
            )
            parser.parse(["User-agent: *", "Allow: /"])

        # Actualizar caché bajo lock
        with self._cache_lock:
            self._cache[origin] = (parser, time.monotonic())

        return parser

    # ── API pública ───────────────────────────────────────────────────────────

    def allowed(
        self,
        url: str,
        trusted_domains: FrozenSet[str] = frozenset(),
    ) -> bool:
        """
        True si USER_AGENT puede acceder a *url*.

        Parámetros
        ----------
        url : str
            URL a comprobar.
        trusted_domains : FrozenSet[str]
            Dominios cuyo acceso está explícitamente sancionado por ToS o API
            oficial, incluso si robots.txt contiene Disallow para alguna ruta.
            Lo aporta el cliente que realiza la petición.

        Nota: el Crawl-delay de estos dominios se sigue respetando;
        solo se saltea la comprobación de Disallow.
        """
        _, host = self._origin(url)

        if host in trusted_domains:
            return True

        try:
            origin, _ = self._origin(url)
            parser    = self._get_parser(origin)
            result    = parser.can_fetch(USER_AGENT, url)
            if not result:
                logger.info("robots.txt prohibe: %s", url)
            return result
        except Exception as exc:
            logger.warning(
                "Error comprobando robots.txt para %s: %s — permitiendo.", url, exc
            )
            return True  # fail-open

    def crawl_delay(self, url: str) -> float:
        """
        Segundos de Crawl-delay declarados en robots.txt (0.0 si no definido).

        Nunca hace bypass: el delay se lee siempre del robots.txt real,
        independientemente de si el dominio está en trusted_domains.
        El Crawl-delay es una restricción de frecuencia que debe respetarse
        aunque el dominio tenga acceso garantizado por ToS.
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
