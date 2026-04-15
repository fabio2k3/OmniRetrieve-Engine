"""
bm25.py
=======
Motor de indexación — guarda frecuencias crudas en el índice invertido.

Responsabilidad única: tokenizar documentos y persistir en `postings`
la frecuencia cruda (freq) de cada término en cada documento.

El cálculo de pesos BM25 (o cualquier otro modelo) es responsabilidad
del módulo recuperador, que lee estas frecuencias y aplica la fórmula
que necesite sin que el indexador tenga que cambiar.

Referencia BM25 (para el recuperador)
--------------------------------------
    IDF(t)   = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    TF_norm  = (freq · (k1+1)) / (freq + k1·(1 - b + b·|d|/avgdl))
    BM25     = IDF(t) · TF_norm
    k1=1.5, b=0.75 — parámetros estándar

    Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance
    Framework: BM25 and Beyond. Foundations and Trends in IR, 3(4), 333-389.
"""

from __future__ import annotations

import logging
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
    Indexador que construye el índice invertido con frecuencias crudas.

    Lee documentos de la BD, tokeniza con TextPreprocessor y escribe
    en `terms` (vocabulario + df) y `postings` (freq cruda por término/doc).

    Parámetros
    ----------
    db_path      : ruta a documents.db
    preprocessor : instancia de TextPreprocessor
    field        : "full_text" | "abstract" | "both"
    batch_size   : documentos procesados por lote
    """

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

    # Construcción del índice 
    def build(self, reindex: bool = False) -> dict:
        """
        Construye el índice invertido con frecuencias crudas.

        1. Tokeniza cada documento.
        2. Cuenta la frecuencia cruda de cada término por documento.
        3. Escribe terms (vocabulario + df) y postings (freq) en la BD.

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

            # Paso 1: tokenizar y contar frecuencias crudas
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

            # Paso 2: calcular df por término
            log.info("Paso 2/3 — Calculando document frequency (df)…")
            df_map: dict[str, int] = defaultdict(int)
            for counter in doc_tfs.values():
                for term in counter:
                    df_map[term] += 1

            # Paso 3: persistir frecuencias crudas
            log.info("Paso 3/3 — Escribiendo índice en la base de datos…")
            self._persist(conn, doc_tfs, df_map, stats)

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

    # Persistencia 
    def _persist(
        self,
        conn: sqlite3.Connection,
        doc_tfs: dict[str, Counter],
        df_map: dict[str, int],
        stats: dict,
    ) -> None:
        # Términos existentes
        existing_terms: dict[str, int] = {
            r["word"]: r["term_id"]
            for r in conn.execute("SELECT term_id, word FROM terms")
        }

        # Insertar términos nuevos con su df
        new_terms = [(w, df) for w, df in df_map.items() if w not in existing_terms]
        if new_terms:
            conn.executemany(
                "INSERT OR IGNORE INTO terms (word, df) VALUES (?, ?)", new_terms
            )
            conn.commit()
            stats["terms_added"] = len(new_terms)

        # Actualizar df de términos ya existentes
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

        # Construir postings con frecuencia cruda únicamente
        batch: list[tuple[int, str, int]] = []
        for arxiv_id, counter in doc_tfs.items():
            for term, freq in counter.items():
                if term not in all_terms:
                    continue
                batch.append((all_terms[term], arxiv_id, freq))

            if len(batch) >= self.batch_size * 50:
                self._flush_postings(conn, batch, stats)
                batch = []

        if batch:
            self._flush_postings(conn, batch, stats)
        conn.commit()

    def _flush_postings(
        self,
        conn: sqlite3.Connection,
        batch: list[tuple[int, str, int]],
        stats: dict,
    ) -> None:
        conn.executemany(
            """
            INSERT INTO postings (term_id, doc_id, freq)
            VALUES (?, ?, ?)
            ON CONFLICT(term_id, doc_id) DO UPDATE SET
                freq = excluded.freq
            """,
            batch,
        )
        conn.commit()
        stats["postings_added"] += len(batch)

    # Metadatos 
    def _save_meta(self, conn: sqlite3.Connection, stats: dict) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            [
                ("indexer",            "frequency_index"),
                ("last_run_at",        stats.get("finished_at", _now())),
                ("last_docs_indexed",  str(stats.get("docs_processed", 0))),
                ("last_terms_added",   str(stats.get("terms_added",    0))),
                ("last_postings_added",str(stats.get("postings_added", 0))),
            ],
        )
        conn.commit()


# Utilidades 
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()