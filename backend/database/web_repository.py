"""
web_repository.py
=================
Persiste en la base de datos los documentos encontrados por la búsqueda web.

Cambio respecto a la versión anterior
--------------------------------------
Los resultados web se guardan en la tabla ``web_search_results`` en lugar
de en ``documents``. Esto evita dos problemas:

  1. Contaminación del corpus local: los docs web no deben mezclarse con
     los papers de arXiv en el índice TF ni en el modelo LSI.
  2. Indexación accidental: el watcher de indexación procesaba TODOS los
     docs pendientes en ``documents``, incluyendo los web, lo que causaba
     conteos incorrectos y ralentizaba la indexación real.

Los resultados web solo se usan de forma efímera durante el pipeline de
consulta. Se guardan en ``web_search_results`` únicamente para auditoría
y para evitar fetchear la misma URL varias veces en futuras búsquedas.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database.schema import DB_PATH, get_connection

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_web_results(
    query:   str,
    results: list[dict[str, Any]],
    db_path: Path = DB_PATH,
) -> int:
    """
    Guarda los resultados de búsqueda web en ``web_search_results``.

    Solo inserta URLs nuevas (UNIQUE constraint en url).
    Devuelve el número de filas nuevas insertadas.

    Parámetros
    ----------
    query   : consulta original que generó estos resultados.
    results : lista de dicts de WebSearcher / DuckDuckGoSearcher.
    db_path : ruta a la BD SQLite.
    """
    conn = get_connection(db_path)
    saved = 0

    try:
        for r in results:
            url     = r.get("url", "").strip()
            title   = r.get("title", "")
            content = r.get("content", "")
            score   = float(r.get("score", 0.5))
            source  = r.get("source", "web")

            if not url:
                continue

            conn.execute(
                """
                INSERT OR IGNORE INTO web_search_results
                    (searched_at, query, title, url, content, score, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_now(), query, title, url, content, score, source),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                saved += 1

        conn.commit()
        _log_web_search(conn, query, len(results), saved)
        conn.commit()

        log.info(
            "[WebRepo] query='%s…' — %d resultados, %d nuevos guardados en web_search_results.",
            query[:40], len(results), saved,
        )

    finally:
        conn.close()

    return saved


def get_cached_result(url: str, db_path: Path = DB_PATH) -> dict | None:
    """
    Devuelve el contenido cacheado de una URL si ya fue fetched antes.

    Útil para evitar fetchear la misma página en búsquedas futuras.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT title, url, content, score, source FROM web_search_results WHERE url = ?",
            (url,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_web_results(
    limit:  int  = 20,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """
    Devuelve los resultados web más recientes. Útil para monitorización.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT searched_at, query, title, url, score, source
            FROM   web_search_results
            ORDER  BY searched_at DESC
            LIMIT  ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _log_web_search(
    conn,
    query:         str,
    total_results: int,
    saved:         int,
) -> None:
    """Registra la búsqueda en web_search_log si la tabla existe."""
    try:
        conn.execute(
            """
            INSERT INTO web_search_log (searched_at, query, results_found, results_saved)
            VALUES (?, ?, ?, ?)
            """,
            (_now(), query, total_results, saved),
        )
    except Exception:
        pass