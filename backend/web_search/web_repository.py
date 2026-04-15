"""
web_repository.py
=================
Persiste en la base de datos los documentos encontrados por la búsqueda web,
para que estén disponibles en futuras consultas sin necesidad de volver a buscar.

Diseño
------
Los documentos web se guardan en la tabla `documents` igual que los del crawler,
usando un arxiv_id sintético basado en la URL (prefijo "web_").
Esto permite que el indexador los procese en el siguiente ciclo sin cambios.

Adicionalmente se mantiene una tabla `web_search_log` para auditoría.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database.schema import DB_PATH, get_connection

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _url_to_id(url: str) -> str:
    """
    Genera un ID único y reproducible a partir de una URL.
    Formato: web_<primeros 16 chars del hash MD5>
    """
    return "web_" + hashlib.md5(url.encode()).hexdigest()[:16]


def save_web_results(
    query: str,
    results: list[dict[str, Any]],
    db_path: Path = DB_PATH,
) -> int:
    """
    Guarda los resultados de búsqueda web en la tabla `documents`.

    Solo guarda documentos nuevos (no existentes previamente).
    Devuelve el número de documentos nuevos insertados.

    Parámetros
    ----------
    query   : consulta original que generó estos resultados
    results : lista de dicts de WebSearcher.search()
    db_path : ruta a la BD
    """
    conn = get_connection(db_path)
    saved = 0

    try:
        for result in results:
            url     = result.get("url", "")
            title   = result.get("title", "Sin título")
            content = result.get("content", "")

            if not url or not content:
                continue

            doc_id = _url_to_id(url)

            # Verificar si ya existe
            exists = conn.execute(
                "SELECT 1 FROM documents WHERE arxiv_id = ?", (doc_id,)
            ).fetchone()

            if exists:
                log.debug("[WebRepo] Documento ya existe: %s", doc_id)
                continue

            # Insertar como documento nuevo
            conn.execute(
                """
                INSERT OR IGNORE INTO documents
                    (arxiv_id, title, authors, abstract, categories,
                     published, updated, pdf_url, fetched_at,
                     full_text, text_length, pdf_downloaded, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    title,
                    "Web Search",          # authors — fuente web
                    content[:500],         # abstract — primeros 500 chars
                    "web",                 # categories
                    _now(),                # published
                    _now(),                # updated
                    url,                   # pdf_url — guardamos la URL original
                    _now(),                # fetched_at
                    content,               # full_text — contenido completo
                    len(content),          # text_length
                    1,                     # pdf_downloaded = 1 (listo para indexar)
                    _now(),                # indexed_at
                ),
            )
            saved += 1

        conn.commit()

        # Registrar en el log de búsquedas web
        _log_web_search(conn, query, len(results), saved)
        conn.commit()

        log.info(
            "[WebRepo] Query='%s…' — %d resultados, %d nuevos guardados.",
            query[:40], len(results), saved,
        )

    finally:
        conn.close()

    return saved


def _log_web_search(
    conn,
    query: str,
    total_results: int,
    saved: int,
) -> None:
    """Registra la búsqueda web en web_search_log si la tabla existe."""
    try:
        conn.execute(
            """
            INSERT INTO web_search_log (searched_at, query, results_found, results_saved)
            VALUES (?, ?, ?, ?)
            """,
            (_now(), query, total_results, saved),
        )
    except Exception:
        # La tabla puede no existir aún — no es crítico
        pass


def get_web_documents(
    limit: int = 20,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """
    Devuelve documentos provenientes de búsquedas web (arxiv_id empieza por 'web_').
    Útil para monitorización.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT arxiv_id, title, pdf_url AS url, fetched_at
            FROM   documents
            WHERE  arxiv_id LIKE 'web_%'
            ORDER  BY fetched_at DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()