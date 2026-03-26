"""
indexer.py
==========
Motor de indexación: construye el índice invertido con frecuencias crudas.

Responsabilidad única
---------------------
Lee documentos de la base de datos, tokeniza el texto con TextPreprocessor,
cuenta la frecuencia de aparición de cada término por documento y persiste
el resultado en las tablas `terms` y `postings`.

No calcula ningún peso TF-IDF. Esa responsabilidad corresponde al módulo
retrieval, que leerá las frecuencias a través de index_repository.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.database.schema import DB_PATH, init_db as _init_db
from backend.database.index_repository import (
    clear_index,
    upsert_terms,
    flush_postings,
    save_index_meta,
    get_unindexed_documents,
    mark_documents_indexed,
)
from backend.indexing.preprocessor import TextPreprocessor

log = logging.getLogger(__name__)

_FLUSH_EVERY = 5_000   # postings acumulados antes de volcar a la BD


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TFIndexer:
    """
    Construye el índice invertido de frecuencias sobre el corpus arXiv.

    Parámetros
    ----------
    db_path      : ruta a la base de datos SQLite.
    preprocessor : instancia de TextPreprocessor (se crea una por defecto).
    field        : campo de texto a indexar — 'full_text', 'abstract' o 'both'.
    batch_size   : nº de documentos leídos por lote desde la BD.
    """

    def __init__(
        self,
        db_path:      Path                    = DB_PATH,
        preprocessor: TextPreprocessor | None = None,
        field:        str                     = "full_text",
        batch_size:   int                     = 100,
    ) -> None:
        self.db_path      = Path(db_path)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.field        = field
        self.batch_size   = batch_size

    def init_schema(self) -> None:
        """Inicializa el esquema completo de la BD (idempotente)."""
        _init_db(self.db_path)
        log.info("Esquema inicializado.")

    def build(self, reindex: bool = False) -> dict:
        """
        Ejecuta la indexación completa.

        Parámetros
        ----------
        reindex : si True, borra el índice actual y lo reconstruye desde cero.

        Flujo
        -----
        1. (Opcional) Limpia terms y postings si reindex=True.
        2. Itera documentos no indexados (o todos si reindex=True).
        3. Tokeniza con TextPreprocessor y cuenta frecuencias por doc.
        4. Calcula df agregado del lote.
        5. Persiste términos (upsert_terms) y postings (flush_postings).
        6. Guarda metadatos en index_meta.

        Devuelve
        --------
        dict con: docs_processed, terms_added, postings_added,
                  started_at, finished_at.
        """
        stats = {
            "docs_processed": 0,
            "terms_added":    0,
            "postings_added": 0,
            "started_at":     _now(),
        }

        if reindex:
            log.warning("Modo reindex: eliminando términos y postings previos…")
            clear_index(self.db_path)

        # ── Paso 1: tokenizar y contar ────────────────────────────────────
        log.info("Paso 1/3 — Tokenizando documentos…")
        doc_tfs: dict[str, Counter] = {}
        df_map:  dict[str, int]     = defaultdict(int)

        for arxiv_id, texto in get_unindexed_documents(
            field=self.field,
            batch_size=self.batch_size,
            reindex=reindex,
            db_path=self.db_path,
        ):
            tokens = self.preprocessor.process(texto)
            if tokens:
                counter = Counter(tokens)
                doc_tfs[arxiv_id] = counter
                for term in counter:
                    df_map[term] += 1

            stats["docs_processed"] += 1
            if stats["docs_processed"] % 50 == 0:
                log.info("  … %d documentos leídos", stats["docs_processed"])

        if not doc_tfs:
            log.info("No hay documentos nuevos para indexar.")
            stats["finished_at"] = _now()
            save_index_meta(stats, self.db_path)
            return stats

        log.info(
            "Documentos a indexar: %d | términos únicos en lote: %d",
            len(doc_tfs), len(df_map),
        )

        # ── Paso 2: persistir vocabulario ─────────────────────────────────
        log.info("Paso 2/3 — Persistiendo vocabulario (terms + df)…")
        all_terms = upsert_terms(df_map, self.db_path)
        stats["terms_added"] = len(
            [w for w in df_map if w in all_terms]
        )

        # ── Paso 3: persistir postings en lotes ───────────────────────────
        log.info("Paso 3/3 — Escribiendo postings…")
        batch: list[tuple[int, str, int]] = []

        for arxiv_id, counter in doc_tfs.items():
            for term, freq in counter.items():
                term_id = all_terms.get(term)
                if term_id is None:
                    continue
                batch.append((term_id, arxiv_id, freq))

                if len(batch) >= _FLUSH_EVERY:
                    stats["postings_added"] += flush_postings(batch, self.db_path)
                    batch = []

        if batch:
            stats["postings_added"] += flush_postings(batch, self.db_path)

        # ── Paso 4: marcar documentos como indexados ─────────────────
        mark_documents_indexed(list(doc_tfs.keys()), self.db_path)

        stats["finished_at"] = _now()
        save_index_meta(stats, self.db_path)

        log.info(
            "Indexación completa — docs: %d | términos: %d | postings: %d",
            stats["docs_processed"],
            stats["terms_added"],
            stats["postings_added"],
        )
        return stats