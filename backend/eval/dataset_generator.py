"""
dataset_generator.py
====================
Genera automáticamente un EvalDataset a partir de chunks reales de la BD.

Estrategia de muestreo
----------------------
1. Extrae chunks de la tabla `chunks` de forma estratificada por documento
   (``arxiv_id``), para que el dataset cubra diversidad temática y no quede
   dominado por un solo paper muy largo.
2. Por cada chunk muestreado crea dos casos:
   · **Exact**    → extrae 1-3 oraciones del cuerpo del chunk (evitando el
                    inicio/fin para no coincidir con la frontera de chunk) y
                    las usa directamente como query.
   · **Semantic** → toma el mismo fragmento y lo parafrasea con el LLM local
                    (Ollama).  Si la paráfrasis no supera el filtro de calidad
                    se omite ese caso semántico sin abortar la generación.
3. Guarda el dataset como JSON con ``EvalDataset.save()``.

Uso rápido
----------
>>> from backend.eval.dataset_generator import DatasetGenerator
>>> gen = DatasetGenerator(sample_size=50, include_semantic=True)
>>> dataset = gen.generate()
>>> dataset.save(Path("backend/data/eval/dataset_v1.json"))
>>> print(dataset)
EvalDataset(total=95, exact=50, semantic=45, generated_at=...)

Parámetros importantes
-----------------------
sample_size      : número de chunks a muestrear (casos exact = sample_size).
include_semantic : si False, solo genera casos exact (más rápido, sin LLM).
min_chunk_chars  : descarta chunks demasiado cortos para extraer un fragmento.
fragment_sentences: oraciones a extraer del chunk como semilla de la query.
seed             : semilla aleatoria para reproducibilidad.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from pathlib import Path
from typing import Iterator

from backend.database.schema import DB_PATH, get_connection
from .paraphraser import Paraphraser
from .schema import EvalCase, EvalDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_MIN_CHUNK_CHARS   = 200   # chunk mínimo para poder extraer un fragmento útil
_SENT_RE           = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'\(0-9])'
    r'|(?<=;)\s+'
)


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """División simple de oraciones (misma lógica que chunker.py)."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _extract_fragment(text: str, n_sentences: int = 2) -> str | None:
    """
    Extrae ``n_sentences`` oraciones del cuerpo del chunk (no del inicio ni
    del final) para que la query no coincida trivialmente con el borde del
    chunk.

    Devuelve None si el chunk tiene menos oraciones de las necesarias para
    elegir un fragmento interior.
    """
    sents = _split_sentences(text)
    # Necesitamos al menos 2 oraciones de margen a cada lado
    margin = 1
    interior = sents[margin: len(sents) - margin]
    if len(interior) < n_sentences:
        # Fallback: usar las primeras n_sentences si no hay interior
        if len(sents) >= n_sentences:
            interior = sents[:n_sentences]
        else:
            return None
    # Empezar en una posición aleatoria dentro del interior
    start = random.randint(0, max(0, len(interior) - n_sentences))
    fragment = " ".join(interior[start: start + n_sentences])
    return fragment if len(fragment) >= 40 else None


# ---------------------------------------------------------------------------
# Muestreo estratificado desde la BD
# ---------------------------------------------------------------------------

def _sample_chunks_stratified(
    conn:        sqlite3.Connection,
    sample_size: int,
    min_chars:   int,
    seed:        int,
) -> list[sqlite3.Row]:
    """
    Muestrea ``sample_size`` chunks de forma estratificada por ``arxiv_id``.

    Algoritmo
    ---------
    1. Obtiene todos los arxiv_ids distintos que tienen chunks con
       char_count >= min_chars.
    2. Calcula cuántos chunks tomar por documento (≈ sample_size / n_docs),
       con un mínimo de 1.
    3. Para cada documento toma una muestra aleatoria de sus chunks.
    """
    rng = random.Random(seed)

    arxiv_ids: list[str] = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT arxiv_id
            FROM chunks
            WHERE char_count >= ? OR (char_count IS NULL AND length(text) >= ?)
            ORDER BY arxiv_id
            """,
            (min_chars, min_chars),
        ).fetchall()
    ]

    if not arxiv_ids:
        log.warning("[dataset_gen] No se encontraron chunks en la BD con char_count >= %d", min_chars)
        return []

    n_docs          = len(arxiv_ids)
    per_doc         = max(1, sample_size // n_docs)
    extra           = sample_size - per_doc * n_docs  # los primeros `extra` docs toman uno más

    log.info(
        "[dataset_gen] %d documentos encontrados; ~%d chunks/doc (extra=%d)",
        n_docs, per_doc, extra,
    )

    sampled: list[sqlite3.Row] = []
    rng.shuffle(arxiv_ids)

    for i, arxiv_id in enumerate(arxiv_ids):
        quota = per_doc + (1 if i < extra else 0)
        rows: list[sqlite3.Row] = conn.execute(
            """
            SELECT id, arxiv_id, chunk_index, text, char_count,
                   d.title AS title
            FROM   chunks c
            JOIN   documents d USING (arxiv_id)
            WHERE  c.arxiv_id = ?
              AND  (c.char_count >= ? OR (c.char_count IS NULL AND length(c.text) >= ?))
            ORDER  BY c.chunk_index
            """,
            (arxiv_id, min_chars, min_chars),
        ).fetchall()

        if not rows:
            continue

        chosen = rng.sample(rows, min(quota, len(rows)))
        sampled.extend(chosen)

        if len(sampled) >= sample_size:
            break

    # Recortar si nos pasamos (puede ocurrir con el extra)
    rng.shuffle(sampled)
    return sampled[:sample_size]


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """
    Genera un EvalDataset de casos exact y/o semantic desde la BD.

    Parámetros
    ----------
    sample_size       : número de chunks a muestrear. Cada chunk produce
                        1 caso exact y (opcionalmente) 1 caso semantic.
    include_semantic  : si True, genera también casos de paráfrasis con LLM.
    min_chunk_chars   : tamaño mínimo del chunk para ser elegible.
    fragment_sentences: oraciones del chunk a usar como semilla de la query.
    paraphrase_model  : modelo Ollama para la paráfrasis.
    paraphrase_temp   : temperatura del modelo de paráfrasis.
    paraphrase_retries: intentos antes de descartar un caso semantic.
    seed              : semilla aleatoria.
    db_path           : ruta a la BD SQLite (por defecto la del sistema).
    """

    def __init__(
        self,
        sample_size:        int   = 50,
        include_semantic:   bool  = True,
        min_chunk_chars:    int   = _MIN_CHUNK_CHARS,
        fragment_sentences: int   = 2,
        paraphrase_model:   str   = "llama3.2:3b",
        paraphrase_temp:    float = 0.55,
        paraphrase_retries: int   = 3,
        seed:               int   = 42,
        db_path:            Path  = DB_PATH,
    ) -> None:
        self.sample_size        = sample_size
        self.include_semantic   = include_semantic
        self.min_chunk_chars    = min_chunk_chars
        self.fragment_sentences = fragment_sentences
        self.paraphrase_model   = paraphrase_model
        self.paraphrase_temp    = paraphrase_temp
        self.paraphrase_retries = paraphrase_retries
        self.seed               = seed
        self.db_path            = db_path

        self._paraphraser: Paraphraser | None = None
        if include_semantic:
            self._paraphraser = Paraphraser(
                model=paraphrase_model,
                temperature=paraphrase_temp,
                max_retries=paraphrase_retries,
            )

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def generate(self) -> EvalDataset:
        """
        Genera y devuelve el EvalDataset completo.

        Proceso
        -------
        1. Muestrea chunks de la BD (estratificado por documento).
        2. Por cada chunk:
           a. Extrae un fragmento interior como semilla.
           b. Crea un EvalCase de tipo ``exact``.
           c. Si ``include_semantic``, genera una paráfrasis y crea un
              EvalCase de tipo ``semantic`` (se omite si la paráfrasis falla).
        3. Construye y devuelve un EvalDataset.
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
            log.error("[dataset_gen] No hay chunks disponibles. ¿Está la BD poblada?")
            return EvalDataset(cases=[], db_path=str(self.db_path),
                               generator_cfg=self._cfg())

        log.info("[dataset_gen] %d chunks muestreados → generando casos…", len(chunks))

        cases: list[EvalCase] = []
        exact_counter    = 0
        semantic_counter = 0
        skipped_fragment = 0
        skipped_paraph   = 0

        for row in chunks:
            chunk_id    = row["id"]
            arxiv_id    = row["arxiv_id"]
            chunk_index = row["chunk_index"]
            text        = row["text"]
            try:
                title = row["title"] or arxiv_id
            except (IndexError, KeyError):
                title = arxiv_id

            fragment = _extract_fragment(text, self.fragment_sentences)
            if fragment is None:
                skipped_fragment += 1
                log.debug("[dataset_gen] chunk_id=%d sin fragmento interior; se omite.", chunk_id)
                continue

            # ── Caso exact ──────────────────────────────────────────────
            exact_id = f"exact_{exact_counter:04d}"
            cases.append(EvalCase(
                case_id=exact_id,
                case_type="exact",
                query=fragment,
                expected_chunk_id=chunk_id,
                expected_arxiv_id=arxiv_id,
                expected_chunk_index=chunk_index,
                source_text=text,
                fragment_used=fragment,
                paraphrase_model=None,
                metadata={"title": title, "char_count": len(text)},
            ))
            exact_counter += 1

            # ── Caso semantic ────────────────────────────────────────────
            if self._paraphraser is not None:
                paraphrase = self._paraphraser.paraphrase(fragment)
                if paraphrase is None:
                    skipped_paraph += 1
                    log.debug(
                        "[dataset_gen] chunk_id=%d paráfrasis fallida; se omite caso semantic.",
                        chunk_id,
                    )
                else:
                    sem_id = f"semantic_{semantic_counter:04d}"
                    cases.append(EvalCase(
                        case_id=sem_id,
                        case_type="semantic",
                        query=paraphrase,
                        expected_chunk_id=chunk_id,
                        expected_arxiv_id=arxiv_id,
                        expected_chunk_index=chunk_index,
                        source_text=text,
                        fragment_used=fragment,
                        paraphrase_model=self.paraphrase_model,
                        metadata={"title": title, "char_count": len(text)},
                    ))
                    semantic_counter += 1

        log.info(
            "[dataset_gen] Generación completada: exact=%d semantic=%d "
            "skipped_fragment=%d skipped_paraph=%d",
            exact_counter, semantic_counter, skipped_fragment, skipped_paraph,
        )

        return EvalDataset(
            cases=cases,
            db_path=str(self.db_path),
            generator_cfg=self._cfg(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cfg(self) -> dict:
        """Devuelve los parámetros de configuración para auditoría."""
        return {
            "sample_size":        self.sample_size,
            "include_semantic":   self.include_semantic,
            "min_chunk_chars":    self.min_chunk_chars,
            "fragment_sentences": self.fragment_sentences,
            "paraphrase_model":   self.paraphrase_model if self.include_semantic else None,
            "paraphrase_temp":    self.paraphrase_temp  if self.include_semantic else None,
            "paraphrase_retries": self.paraphrase_retries,
            "seed":               self.seed,
        }
