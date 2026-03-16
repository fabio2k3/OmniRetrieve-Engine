"""
robots.py
Verifica robots.txt antes de hacer cualquier request.

Nota sobre arXiv
----------------
arXiv provee explícitamente su API REST (export.arxiv.org/api/) y sus PDFs
para acceso programático según su API User Manual. Sin embargo, su robots.txt
puede bloquear ciertos user agents o paths. Para estos dominios oficialmente
permitidos usamos una allowlist que evita falsos positivos.
"""

from __future__ import annotations

import logging
import ssl
import time
import urllib.request
import urllib.robotparser
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT      = "SRI-Crawler/1.0"
ROBOTS_CACHE_TTL = 3600  # segundos

# ---------------------------------------------------------------------------
# Dominios con acceso programático explícitamente permitido por sus ToS/API
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS = {
    "export.arxiv.org",   # API oficial de arXiv
    "arxiv.org",          # PDFs y páginas de artículos
}


# ---------------------------------------------------------------------------
# SSL context
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
    Verificador de robots.txt con caché TTL.

    Para dominios en ALLOWED_DOMAINS devuelve True directamente,
    ya que su acceso programático está explícitamente sancionado.
    Para el resto respeta robots.txt normalmente.
    """

    def __init__(self, ttl: float = ROBOTS_CACHE_TTL) -> None:
        self._ttl   = ttl
        self._cache: dict[str, tuple[urllib.robotparser.RobotFileParser, float]] = {}

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        """Devuelve (scheme://host, host) para una URL."""
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}", p.netloc

    def _get_parser(self, origin: str) -> urllib.robotparser.RobotFileParser:
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

        raw = _fetch_robots_bytes(robots_url)
        if raw is not None:
            parser.parse(raw.decode("utf-8", errors="replace").splitlines())
        else:
            logger.warning(
                "robots.txt inaccesible para %s — se asume acceso permitido.", origin
            )

        self._cache[origin] = (parser, now)
        return parser

    # ── API pública ──────────────────────────────────────────────────────────

    def allowed(self, url: str) -> bool:
        """
        True si USER_AGENT puede acceder a *url*.
        Para dominios en ALLOWED_DOMAINS siempre devuelve True.
        """
        _, host = self._origin(url)

        # Dominios con API pública explícitamente permitida
        if host in ALLOWED_DOMAINS:
            return True

        try:
            origin, _ = self._origin(url)
            parser     = self._get_parser(origin)
            result     = parser.can_fetch(USER_AGENT, url)
            if not result:
                logger.info("robots.txt prohíbe: %s", url)
            return result
        except Exception as exc:
            logger.warning("Error comprobando robots.txt para %s: %s — permitiendo.", url, exc)
            return True  # fail-open

    def crawl_delay(self, url: str) -> float:
        """
        Devuelve el Crawl-delay indicado en robots.txt (0.0 si no está definido).
        Para dominios en ALLOWED_DOMAINS devuelve 0.0 (respetamos nuestro propio delay).
        """
        _, host = self._origin(url)
        if host in ALLOWED_DOMAINS:
            return 0.0
        try:
            origin, _ = self._origin(url)
            parser     = self._get_parser(origin)
            delay      = parser.crawl_delay(USER_AGENT)
            return float(delay) if delay is not None else 0.0
        except Exception:
            return 0.0


# Singleton compartido por todo el paquete crawler
checker = RobotsChecker()