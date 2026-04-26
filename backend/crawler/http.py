"""
http.py
=======
Utilidades HTTP compartidas por todo el paquete crawler.

Centraliza tres recursos que antes estaban duplicados o acoplados:

    USER_AGENT  — cadena de identificación del crawler en todas las peticiones.
    _SSL_CTX    — contexto SSL único (usa certifi cuando está disponible,
                  degrada a verificación desactivada si no lo está).
    fetch_bytes — GET minimalista que devuelve bytes crudos o None si falla.

Módulos que consumen este fichero
----------------------------------
    robots.py             — usa USER_AGENT, _SSL_CTX y fetch_bytes.
    clients/arxiv/client  — usa USER_AGENT y _SSL_CTX para sus peticiones HTTP.
"""

from __future__ import annotations

import logging
import ssl
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

USER_AGENT = "SRI-Crawler/1.0"


# ---------------------------------------------------------------------------
# Contexto SSL compartido
# ---------------------------------------------------------------------------

def _build_ssl_context() -> ssl.SSLContext:
    """
    Construye el contexto SSL.

    Intenta usar los certificados de ``certifi`` (más actualizados).
    Si ``certifi`` no está instalado, crea un contexto sin verificación
    de hostname (menos seguro, pero operativo en entornos sin certifi).

    Returns
    -------
    ssl.SSLContext
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


_SSL_CTX: ssl.SSLContext = _build_ssl_context()


# ---------------------------------------------------------------------------
# Helper de descarga
# ---------------------------------------------------------------------------

def fetch_bytes(
    url: str,
    timeout: int = 15,
    accept: str = "*/*",
) -> Optional[bytes]:
    """
    Realiza un GET y devuelve los bytes de la respuesta, o ``None`` si falla.

    Diseñado para usos de bajo volumen (robots.txt, metadatos).
    Para descargas grandes con rate-limiting y control de tamaño
    usa ``ArxivClient._get()``, que añade lógica de streaming y pausa.

    Parámetros
    ----------
    url     : URL a descargar.
    timeout : tiempo máximo de espera en segundos.
    accept  : valor del header ``Accept``.

    Returns
    -------
    bytes | None
        Contenido completo de la respuesta, o ``None`` ante cualquier
        excepción de red o HTTP.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return resp.read()
    except Exception as exc:
        logger.warning("fetch_bytes falló para %s: %s", url, exc)
        return None
