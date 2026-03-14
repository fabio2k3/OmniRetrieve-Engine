"""
tfidf.py
========
Motor de indexación TF-IDF.

Leer documentos de la base de datos, calcular los
pesos TF e IDF, y persistir los resultados en las tablas `terms` y `postings`.

Fórmulas
--------
    TF(t, d)  = log(1 + freq(t, d))          # suavizado logarítmico
    IDF(t)    = log((N + 1) / (df(t) + 1))   # suavizado para evitar div/0
    W(t, d)   = TF(t, d) * IDF(t)
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
from backend.indexing.preprocessor import TextPreprocessor

log = logging.getLogger(__name__)


class TFIDFIndexer:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        preprocessor: TextPreprocessor | None = None,
        field: str = "both",
        batch_size: int = 100,
    ) -> None:
        self.db_path      = db_path
        self.preprocessor = preprocessor or TextPreprocessor()
        self.field        = field
        self.batch_size   = batch_size


    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


    def init_schema(self) -> None:
        _schema_init_db(self.db_path)
        log.info("Esquema inicializado vía schema.py")


    def _get_documents(
        self,
        conn: sqlite3.Connection,
        reindex: bool,
    ) -> Iterator[tuple[str, str]]:

        if self.field == "full_text":
            text_expr = "COALESCE(full_text, '')"
            where_pdf = "pdf_downloaded = 1"
        elif self.field == "abstract":
            text_expr = "COALESCE(abstract, '')"
            where_pdf = "1=1"
        else: 
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

    def build(self, reindex: bool = False) -> dict:
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

            log.info("Paso 1/3 — Tokenizando documentos…")
            doc_tfs: dict[str, Counter] = {}

            for arxiv_id, texto in self._get_documents(conn, reindex):
                tokens = self.preprocessor.process(texto)
                if tokens:
                    doc_tfs[arxiv_id] = Counter(tokens)
                stats["docs_processed"] += 1
                if stats["docs_processed"] % 50 == 0:
                    log.info("  … %d documentos leídos", stats["docs_processed"])

            if not doc_tfs:
                log.info("No hay documentos nuevos para indexar.")
                return stats

            log.info("Documentos a indexar: %d", len(doc_tfs))

            log.info("Paso 2/3 — Calculando frecuencias de documento (df)…")
            df_map: dict[str, int] = defaultdict(int)
            for counter in doc_tfs.values():
                for term in counter:
                    df_map[term] += 1

            existing_count = conn.execute(
                "SELECT COUNT(*) AS n FROM postings"
            ).fetchone()["n"]
            n_docs_total = existing_count + len(doc_tfs)

            log.info("Paso 3/3 — Escribiendo índice en la base de datos…")
            self._persist(conn, doc_tfs, df_map, n_docs_total, stats)

        finally:
            stats["finished_at"] = _now()
            self._save_meta(conn, stats)
            conn.close()

        log.info(
            "✅ Indexación completa — docs: %d | términos nuevos: %d | postings: %d",
            stats["docs_processed"],
            stats["terms_added"],
            stats["postings_added"],
        )
        return stats

    def _persist(
        self,
        conn: sqlite3.Connection,
        doc_tfs: dict[str, Counter],
        df_map: dict[str, int],
        n_docs_total: int,
        stats: dict,
    ) -> None:
        existing_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute("SELECT term_id, word FROM terms")
        }

        new_terms = [(w, df) for w, df in df_map.items() if w not in existing_terms]
        if new_terms:
            conn.executemany(
                "INSERT OR IGNORE INTO terms (word, df) VALUES (?, ?)", new_terms
            )
            conn.commit()
            stats["terms_added"] = len(new_terms)

        update_df = [(df_map[w], w) for w in df_map if w in existing_terms]
        if update_df:
            conn.executemany(
                "UPDATE terms SET df = df + ? WHERE word = ?", update_df
            )
            conn.commit()

        all_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute("SELECT term_id, word FROM terms")
        }

        batch: list[tuple[int, str, float, float]] = []
        for arxiv_id, counter in doc_tfs.items():
            for term, freq in counter.items():
                if term not in all_terms:
                    continue
                tf    = math.log(1 + freq)
                idf   = math.log((n_docs_total + 1) / (df_map[term] + 1))
                batch.append((all_terms[term], arxiv_id, tf, tf * idf))

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


    def recalculate_idf(self) -> None:
        conn = self._connect()
        try:
            n_docs = conn.execute(
                "SELECT COUNT(DISTINCT doc_id) AS n FROM postings"
            ).fetchone()["n"]

            if n_docs == 0:
                log.warning("No hay postings; omitiendo recalculo de IDF.")
                return

            log.info("Recalculando IDF para N=%d documentos…", n_docs)
            conn.execute(f"""
                UPDATE postings
                SET tfidf_weight = tf * LOG(({n_docs} + 1.0) / (
                    SELECT df FROM terms WHERE terms.term_id = postings.term_id
                ) + 1.0)
            """)
            conn.commit()
            log.info("✅ IDF recalculado.")
        finally:
            conn.close()


    def _save_meta(self, conn: sqlite3.Connection, stats: dict) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [
                ("last_run_at",         stats.get("finished_at", _now())),
                ("last_docs_indexed",   str(stats.get("docs_processed", 0))),
                ("last_terms_added",    str(stats.get("terms_added",    0))),
                ("last_postings_added", str(stats.get("postings_added", 0))),
            ],
        )
        conn.commit()

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()