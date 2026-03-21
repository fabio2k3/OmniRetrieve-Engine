"""
bm25.py
=======
Motor de indexación BM25 (Best Match 25).

Sustituye a tfidf.py — mismas tablas `terms` y `postings`,
mismos campos, solo cambia la fórmula de cálculo del peso.

Fórmula BM25
------------
    IDF(t)    = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    TF_BM25   = (tf · (k1 + 1)) / (tf + k1 · (1 - b + b · |d| / avgdl))
    W(t, d)   = IDF(t) · TF_BM25

Parámetros estándar
-------------------
    k1 = 1.5  — controla la saturación de la frecuencia de término
    b  = 0.75 — controla la normalización por longitud de documento
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.database.schema import DB_PATH, init_db as _schema_init_db
from backend.new_indexing.preprocessor import TextPreprocessor

log = logging.getLogger(__name__)


class BM25Indexer:
    """
    Lee documentos de la BD, calcula pesos BM25 y escribe
    los resultados en las tablas `terms` y `postings`.

    Parámetros
    ----------
    db_path      : ruta a documents.db
    preprocessor : instancia de TextPreprocessor
    field        : "full_text" | "abstract" | "both"
    batch_size   : documentos procesados por lote
    k1           : parámetro de saturación de TF (default: 1.5)
    b            : parámetro de normalización por longitud (default: 0.75)
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        preprocessor: TextPreprocessor | None = None,
        field: str = "both",
        batch_size: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.db_path      = db_path
        self.preprocessor = preprocessor or TextPreprocessor()
        self.field        = field
        self.batch_size   = batch_size
        self.k1           = k1
        self.b            = b


    # Conexión 

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


    # Esquema 

    def init_schema(self) -> None:
        """Delega la creación de tablas a schema.py (fuente de verdad central)."""
        _schema_init_db(self.db_path)
        log.info("Esquema inicializado vía schema.py")


    # Consulta de documentos 

    def _get_documents(
        self,
        conn: sqlite3.Connection,
        reindex: bool,
    ) -> Iterator[tuple[str, str]]:
        """Genera tuplas (arxiv_id, texto) para los documentos pendientes."""
        if self.field == "full_text":
            text_expr = "COALESCE(full_text, '')"
            where_pdf = "pdf_downloaded = 1"
        elif self.field == "abstract":
            text_expr = "COALESCE(abstract, '')"
            where_pdf = "1=1"
        else:  # "both"
            text_expr = "COALESCE(full_text, abstract, '')"
            where_pdf = "1=1"

        where_indexed = (
            "1=1"
            if reindex
            else "arxiv_id NOT IN (SELECT DISTINCT doc_id FROM postings)"
        )

        sql = f"""
            SELECT arxiv_id, {text_expr} AS texto
            FROM   documents
            WHERE  {where_pdf}
              AND  {where_indexed}
              AND  {text_expr} != ''
            ORDER  BY published DESC
        """
        cursor = conn.execute(sql)
        while True:
            rows = cursor.fetchmany(self.batch_size)
            if not rows:
                break
            for row in rows:
                yield row["arxiv_id"], row["texto"]


    #  Construcción del índice

    def build(self, reindex: bool = False) -> dict:
        """
        Punto de entrada principal del indexador BM25.

        1. Lee documentos pendientes (o todo el corpus si reindex=True).
        2. Tokeniza y acumula frecuencias TF por documento.
        3. Calcula df global y longitud media del corpus (avgdl).
        4. Calcula pesos BM25 y escribe terms y postings en la BD.

        Devuelve un dict con estadísticas del proceso.
        """
        conn = self._connect()
        stats = {
            "docs_processed": 0,
            "terms_added":    0,
            "postings_added": 0,
            "started_at":     _now(),
        }

        try:
            if reindex:
                log.warning("Modo reindex: limpiando tablas terms y postings…")
                conn.execute("DELETE FROM postings")
                conn.execute("DELETE FROM terms")
                conn.commit()

            # Paso 1: tokenizar y acumular TF por documento
            log.info("Paso 1/3 — Tokenizando documentos…")
            doc_tfs: dict[str, Counter] = {}
            doc_lengths: dict[str, int] = {}  # |d| = número de tokens del doc

            for arxiv_id, texto in self._get_documents(conn, reindex):
                tokens = self.preprocessor.process(texto)
                if tokens:
                    doc_tfs[arxiv_id]     = Counter(tokens)
                    doc_lengths[arxiv_id] = len(tokens)
                stats["docs_processed"] += 1
                if stats["docs_processed"] % 50 == 0:
                    log.info("  … %d documentos leídos", stats["docs_processed"])

            if not doc_tfs:
                log.info("No hay documentos nuevos para indexar.")
                return stats

            log.info("Documentos a indexar: %d", len(doc_tfs))

            # Paso 2: calcular df y avgdl
            log.info("Paso 2/3 — Calculando df y longitud media del corpus (avgdl)…")
            df_map: dict[str, int] = defaultdict(int)
            for counter in doc_tfs.values():
                for term in counter:
                    df_map[term] += 1

            # avgdl incluye todos los docs del corpus (no solo los nuevos)
            existing_lengths = conn.execute(
                "SELECT SUM(text_length) AS total, COUNT(*) AS n FROM documents WHERE pdf_downloaded = 1"
            ).fetchone()
            existing_total_chars = existing_lengths["total"] or 0
            existing_n           = existing_lengths["n"] or 0

            # Longitud en tokens de los nuevos docs
            new_total_tokens = sum(doc_lengths.values())
            new_n            = len(doc_lengths)

            # avgdl como media de tokens del corpus completo
            # Aproximamos chars→tokens dividiendo por 5 (promedio en inglés)
            total_tokens = (existing_total_chars // 5) + new_total_tokens
            total_n      = existing_n + new_n
            avgdl        = total_tokens / total_n if total_n > 0 else 1.0

            # N total de documentos en el corpus
            n_docs_total = existing_n + new_n

            log.info("  avgdl=%.1f tokens | N=%d docs", avgdl, n_docs_total)

            # Paso 3: persistir con pesos BM25
            log.info("Paso 3/3 — Escribiendo índice BM25 en la base de datos…")
            self._persist(conn, doc_tfs, doc_lengths, df_map,
                          n_docs_total, avgdl, stats)

        finally:
            stats["finished_at"] = _now()
            self._save_meta(conn, stats)
            conn.close()

        log.info(
            "✅ Indexación BM25 completa — docs: %d | términos nuevos: %d | postings: %d",
            stats["docs_processed"],
            stats["terms_added"],
            stats["postings_added"],
        )
        return stats


    #  Persistencia 

    def _persist(
        self,
        conn: sqlite3.Connection,
        doc_tfs: dict[str, Counter],
        doc_lengths: dict[str, int],
        df_map: dict[str, int],
        n_docs_total: int,
        avgdl: float,
        stats: dict,
    ) -> None:
        # Términos existentes
        existing_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute("SELECT term_id, word FROM terms")
        }

        # Insertar términos nuevos
        new_terms = [(w, df) for w, df in df_map.items() if w not in existing_terms]
        if new_terms:
            conn.executemany(
                "INSERT OR IGNORE INTO terms (word, df) VALUES (?, ?)", new_terms
            )
            conn.commit()
            stats["terms_added"] = len(new_terms)

        # Actualizar df de términos existentes
        update_df = [(df_map[w], w) for w in df_map if w in existing_terms]
        if update_df:
            conn.executemany(
                "UPDATE terms SET df = df + ? WHERE word = ?", update_df
            )
            conn.commit()

        # Recargar mapa completo word → term_id
        all_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute("SELECT term_id, word FROM terms")
        }

        # Calcular y flushear postings con pesos BM25
        batch: list[tuple[int, str, float, float]] = []
        k1   = self.k1
        b    = self.b

        for arxiv_id, counter in doc_tfs.items():
            doc_len = doc_lengths.get(arxiv_id, avgdl)

            for term, freq in counter.items():
                if term not in all_terms:
                    continue

                df = df_map[term]

                # IDF Robertson (suavizado, siempre positivo)
                idf = math.log((n_docs_total - df + 0.5) / (df + 0.5) + 1)

                # TF normalizado por longitud de documento
                tf_norm = (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * doc_len / avgdl)
                )

                bm25_score = idf * tf_norm

                # Guardamos tf_norm en columna tf y bm25 en tfidf_weight
                # — mismas columnas, distintos valores, cero cambio en schema
                batch.append((all_terms[term], arxiv_id, tf_norm, bm25_score))

            if len(batch) >= self.batch_size * 50:
                self._flush_postings(conn, batch, stats)
                batch = []

        if batch:
            self._flush_postings(conn, batch, stats)
        conn.commit()

    def _flush_postings(
        self,
        conn: sqlite3.Connection,
        batch: list[tuple[int, str, float, float]],
        stats: dict,
    ) -> None:
        conn.executemany(
            """
            INSERT INTO postings (term_id, doc_id, tf, tfidf_weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term_id, doc_id) DO UPDATE SET
                tf           = excluded.tf,
                tfidf_weight = excluded.tfidf_weight
            """,
            batch,
        )
        conn.commit()
        stats["postings_added"] += len(batch)


    #  Metadatos 

    def _save_meta(self, conn: sqlite3.Connection, stats: dict) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [
                ("indexer",            "BM25"),
                ("last_run_at",        stats.get("finished_at", _now())),
                ("last_docs_indexed",  str(stats.get("docs_processed", 0))),
                ("last_terms_added",   str(stats.get("terms_added",    0))),
                ("last_postings_added",str(stats.get("postings_added", 0))),
            ],
        )
        conn.commit()



#  Utilidades 

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()