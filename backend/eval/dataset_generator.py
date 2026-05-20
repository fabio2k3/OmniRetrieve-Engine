"""
dataset_generator.py
====================
Genera automáticamente un EvalDataset a partir de chunks reales de la BD.

Tipos de casos generados
-------------------------
exact     — fragmento literal del chunk como query.
            Prueba retrieval léxico. Rápido, sin LLM.

semantic  — paráfrasis del fragmento (LLM).
            Prueba retrieval semántico (FAISS/dense).

generated — pregunta real de usuario generada por LLM a partir del chunk.
            Es el tipo más representativo para RAG: prueba si el sistema
            responde preguntas como las haría un usuario real, no si
            recupera fragmentos de texto.

Uso rápido
----------
>>> from backend.eval.dataset_generator import DatasetGenerator
>>> from pathlib import Path

>>> # Solo preguntas reales (recomendado para eval RAG)
>>> gen = DatasetGenerator(sample_size=50, include_generated=True,
...                        include_exact=False, include_semantic=False)
>>> ds = gen.generate()
>>> ds.save(Path("backend/data/eval/dataset_rag.json"))

>>> # Dataset completo para comparar los tres tipos
>>> gen = DatasetGenerator(sample_size=50, include_generated=True,
...                        include_exact=True, include_semantic=True)
>>> ds = gen.generate()
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from pathlib import Path

from backend.database.schema import DB_PATH, get_connection
from .paraphraser import Paraphraser
from .query_generator import QueryGenerator
from .schema import EvalCase, EvalDataset

log = logging.getLogger(__name__)

_MIN_CHUNK_CHARS = 200
_SENT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'\(0-9])'
    r'|(?<=;)\s+'
)


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _extract_fragment(text: str, n_sentences: int = 2) -> str | None:
    """Extrae oraciones del interior del chunk (evitando bordes)."""
    sents   = _split_sentences(text)
    margin  = 1
    interior = sents[margin: len(sents) - margin]
    if len(interior) < n_sentences:
        if len(sents) >= n_sentences:
            interior = sents[:n_sentences]
        else:
            return None
    start    = random.randint(0, max(0, len(interior) - n_sentences))
    fragment = " ".join(interior[start: start + n_sentences])
    return fragment if len(fragment) >= 40 else None


# ---------------------------------------------------------------------------
# Muestreo estratificado
# ---------------------------------------------------------------------------

def _sample_chunks_stratified(
    conn:        sqlite3.Connection,
    sample_size: int,
    min_chars:   int,
    seed:        int,
) -> list[sqlite3.Row]:
    rng = random.Random(seed)
    arxiv_ids: list[str] = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT arxiv_id FROM chunks
            WHERE char_count >= ? OR (char_count IS NULL AND length(text) >= ?)
            ORDER BY arxiv_id
            """,
            (min_chars, min_chars),
        ).fetchall()
    ]

    if not arxiv_ids:
        log.warning("[dataset_gen] No se encontraron chunks con char_count >= %d", min_chars)
        return []

    n_docs   = len(arxiv_ids)
    per_doc  = max(1, sample_size // n_docs)
    extra    = sample_size - per_doc * n_docs

    rng.shuffle(arxiv_ids)
    sampled: list[sqlite3.Row] = []

    for i, arxiv_id in enumerate(arxiv_ids):
        quota = per_doc + (1 if i < extra else 0)
        rows  = conn.execute(
            """
            SELECT c.id, c.arxiv_id, c.chunk_index, c.text,
                   c.char_count, d.title AS title
            FROM   chunks c
            JOIN   documents d USING (arxiv_id)
            WHERE  c.arxiv_id = ?
              AND  (c.char_count >= ? OR (c.char_count IS NULL AND length(c.text) >= ?))
            ORDER BY c.chunk_index
            """,
            (arxiv_id, min_chars, min_chars),
        ).fetchall()

        if not rows:
            continue
        sampled.extend(rng.sample(rows, min(quota, len(rows))))
        if len(sampled) >= sample_size:
            break

    rng.shuffle(sampled)
    return sampled[:sample_size]


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Genera un EvalDataset con casos exact, semantic y/o generated.

    Parámetros
    ----------
    sample_size         : chunks a muestrear. Define el máximo de casos
                          por tipo (uno por chunk si el LLM no falla).
    include_exact       : genera casos con fragmento literal como query.
    include_semantic    : genera casos con paráfrasis del fragmento (LLM).
    include_generated   : genera casos con queries reales de usuario (LLM).
                          Recomendado para eval RAG end-to-end.
    min_chunk_chars     : tamaño mínimo del chunk para ser elegible.
    fragment_sentences  : oraciones a extraer como semilla (exact/semantic).
    paraphrase_model    : modelo Ollama para paráfrasis.
    query_gen_model     : modelo Ollama para generación de queries.
    model_temperature   : temperatura compartida para LLM calls.
    max_retries         : intentos antes de descartar un caso LLM.
    seed                : semilla aleatoria.
    db_path             : ruta a la BD SQLite.
    """

    def __init__(
        self,
        sample_size:        int   = 50,
        include_exact:      bool  = True,
        include_semantic:   bool  = False,
        include_generated:  bool  = True,
        min_chunk_chars:    int   = _MIN_CHUNK_CHARS,
        fragment_sentences: int   = 2,
        paraphrase_model:   str   = "llama3.2:3b",
        query_gen_model:    str   = "llama3.2:3b",
        model_temperature:  float = 0.4,
        max_retries:        int   = 3,
        seed:               int   = 42,
        db_path:            Path  = DB_PATH,
    ) -> None:
        self.sample_size        = sample_size
        self.include_exact      = include_exact
        self.include_semantic   = include_semantic
        self.include_generated  = include_generated
        self.min_chunk_chars    = min_chunk_chars
        self.fragment_sentences = fragment_sentences
        self.paraphrase_model   = paraphrase_model
        self.query_gen_model    = query_gen_model
        self.model_temperature  = model_temperature
        self.max_retries        = max_retries
        self.seed               = seed
        self.db_path            = db_path

        self._paraphraser:     Paraphraser    | None = None
        self._query_generator: QueryGenerator | None = None

        if include_semantic:
            self._paraphraser = Paraphraser(
                model=paraphrase_model,
                temperature=model_temperature,
                max_retries=max_retries,
            )
        if include_generated:
            self._query_generator = QueryGenerator(
                model=query_gen_model,
                temperature=model_temperature,
                max_retries=max_retries,
            )

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def generate(self) -> EvalDataset:
        """
        Genera y devuelve el EvalDataset completo.

        Por cada chunk muestreado puede crear hasta 3 casos:
        · exact     — fragmento literal (sin LLM)
        · semantic  — paráfrasis del fragmento (LLM)
        · generated — query real de usuario (LLM)
        """
        random.seed(self.seed)
        conn = get_connection(self.db_path)
        try:
            chunks = _sample_chunks_stratified(
                conn, self.sample_size, self.min_chunk_chars, self.seed
            )
        finally:
            conn.close()

        if not chunks:
            log.error("[dataset_gen] Sin chunks. ¿Está la BD poblada?")
            return EvalDataset(cases=[], db_path=str(self.db_path),
                               generator_cfg=self._cfg())

        log.info("[dataset_gen] %d chunks muestreados → generando casos…", len(chunks))

        cases: list[EvalCase] = []
        counters = {"exact": 0, "semantic": 0, "generated": 0}
        skipped  = {"fragment": 0, "semantic": 0, "generated": 0}

        for row in chunks:
            chunk_id    = row["id"]
            arxiv_id    = row["arxiv_id"]
            chunk_index = row["chunk_index"]
            text        = row["text"]
            try:
                title = row["title"] or arxiv_id
            except (IndexError, KeyError):
                title = arxiv_id

            meta = {"title": title, "char_count": len(text)}

            # ── Exact ────────────────────────────────────────────────────
            if self.include_exact:
                fragment = _extract_fragment(text, self.fragment_sentences)
                if fragment is None:
                    skipped["fragment"] += 1
                else:
                    cases.append(EvalCase(
                        case_id=f"exact_{counters['exact']:04d}",
                        case_type="exact",
                        query=fragment,
                        expected_chunk_id=chunk_id,
                        expected_arxiv_id=arxiv_id,
                        expected_chunk_index=chunk_index,
                        source_text=text,
                        fragment_used=fragment,
                        metadata=meta,
                    ))
                    counters["exact"] += 1

            # ── Semantic ─────────────────────────────────────────────────
            if self.include_semantic and self._paraphraser:
                fragment = _extract_fragment(text, self.fragment_sentences)
                if fragment:
                    paraphrase = self._paraphraser.paraphrase(fragment)
                    if paraphrase is None:
                        skipped["semantic"] += 1
                    else:
                        cases.append(EvalCase(
                            case_id=f"semantic_{counters['semantic']:04d}",
                            case_type="semantic",
                            query=paraphrase,
                            expected_chunk_id=chunk_id,
                            expected_arxiv_id=arxiv_id,
                            expected_chunk_index=chunk_index,
                            source_text=text,
                            fragment_used=fragment,
                            paraphrase_model=self.paraphrase_model,
                            metadata=meta,
                        ))
                        counters["semantic"] += 1

            # ── Generated ────────────────────────────────────────────────
            if self.include_generated and self._query_generator:
                query = self._query_generator.generate(text)
                if query is None:
                    skipped["generated"] += 1
                    log.debug("[dataset_gen] chunk_id=%d query generada fallida.", chunk_id)
                else:
                    cases.append(EvalCase(
                        case_id=f"generated_{counters['generated']:04d}",
                        case_type="generated",
                        query=query,
                        expected_chunk_id=chunk_id,
                        expected_arxiv_id=arxiv_id,
                        expected_chunk_index=chunk_index,
                        source_text=text,
                        fragment_used=text[:500],  # extracto del chunk usado como contexto
                        paraphrase_model=self.query_gen_model,
                        metadata=meta,
                    ))
                    counters["generated"] += 1

        log.info(
            "[dataset_gen] Completado: exact=%d semantic=%d generated=%d "
            "skipped(frag=%d sem=%d gen=%d)",
            counters["exact"], counters["semantic"], counters["generated"],
            skipped["fragment"], skipped["semantic"], skipped["generated"],
        )

        return EvalDataset(
            cases=cases,
            db_path=str(self.db_path),
            generator_cfg=self._cfg(),
        )

    def _cfg(self) -> dict:
        return {
            "sample_size":       self.sample_size,
            "include_exact":     self.include_exact,
            "include_semantic":  self.include_semantic,
            "include_generated": self.include_generated,
            "min_chunk_chars":   self.min_chunk_chars,
            "fragment_sentences":self.fragment_sentences,
            "paraphrase_model":  self.paraphrase_model if self.include_semantic else None,
            "query_gen_model":   self.query_gen_model  if self.include_generated else None,
            "seed":              self.seed,
        }
