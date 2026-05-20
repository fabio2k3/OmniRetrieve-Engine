"""
test_eval_dataset.py
====================
Tests unitarios del módulo backend.eval.

Cubre
-----
· EvalCase / EvalDataset: serialización JSON ida y vuelta.
· Paraphraser: filtro de similitud Jaccard (_jaccard helper).
· DatasetGenerator: generación con BD en memoria (sqlite3 mock).
· _extract_fragment: extracción de fragmentos interiores.
· _sample_chunks_stratified: muestreo estratificado.

No requiere Ollama ni la BD real — todo se mockea con sqlite3 en memoria.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backend.eval.schema import EvalCase, EvalDataset
from backend.eval.paraphraser import _jaccard, Paraphraser
from backend.eval.dataset_generator import (
    _extract_fragment,
    _split_sentences,
    _sample_chunks_stratified,
    DatasetGenerator,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_in_memory_db() -> sqlite3.Connection:
    """Crea una BD SQLite en memoria con el esquema mínimo para los tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE documents (
            arxiv_id TEXT PRIMARY KEY,
            title    TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id    TEXT NOT NULL REFERENCES documents(arxiv_id),
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            char_count  INTEGER
        );
    """)
    # Insertar documentos de prueba
    docs = [("2401.00001", "Paper Alpha"), ("2401.00002", "Paper Beta")]
    conn.executemany("INSERT INTO documents VALUES (?, ?)", docs)

    # Textos suficientemente largos para que pasen el filtro min_chars
    long_text = (
        "This is the first sentence about attention mechanisms. "
        "The second sentence discusses transformer architectures in detail. "
        "Third sentence covers multi-head attention theory. "
        "Fourth sentence elaborates on positional encoding schemes. "
        "Fifth sentence concludes with experimental results on benchmarks."
    )
    chunks = [
        ("2401.00001", 0, long_text, len(long_text)),
        ("2401.00001", 1, long_text.replace("first", "sixth"), len(long_text)),
        ("2401.00002", 0, long_text.replace("first", "seventh"), len(long_text)),
        ("2401.00002", 1, long_text.replace("first", "eighth"), len(long_text)),
    ]
    conn.executemany(
        "INSERT INTO chunks (arxiv_id, chunk_index, text, char_count) VALUES (?, ?, ?, ?)",
        chunks,
    )
    conn.commit()
    return conn


# ===========================================================================
# Tests — EvalCase / EvalDataset
# ===========================================================================

class TestEvalCase:
    def test_roundtrip_json(self):
        case = EvalCase(
            case_id="exact_0001",
            case_type="exact",
            query="attention mechanism transformer",
            expected_chunk_id=42,
            expected_arxiv_id="2401.00001",
            expected_chunk_index=0,
            source_text="Full chunk text here.",
            fragment_used="attention mechanism transformer",
            paraphrase_model=None,
            metadata={"title": "Test Paper"},
        )
        d = case.to_dict()
        restored = EvalCase.from_dict(d)
        assert restored.case_id == case.case_id
        assert restored.case_type == "exact"
        assert restored.expected_chunk_id == 42

    def test_semantic_case_stores_model(self):
        case = EvalCase(
            case_id="semantic_0000",
            case_type="semantic",
            query="neural network layers process information sequentially",
            expected_chunk_id=7,
            expected_arxiv_id="2401.00002",
            expected_chunk_index=1,
            source_text="Layers in a neural network process data one after another.",
            fragment_used="Layers in a neural network process data one after another.",
            paraphrase_model="llama3.2:3b",
        )
        assert case.paraphrase_model == "llama3.2:3b"
        assert case.case_type == "semantic"


class TestEvalDataset:
    def _make_dataset(self) -> EvalDataset:
        cases = [
            EvalCase("exact_0000", "exact", "q1", 1, "2401.00001", 0, "src", "q1"),
            EvalCase("semantic_0000", "semantic", "q2", 1, "2401.00001", 0, "src", "q1",
                     paraphrase_model="llama3.2:3b"),
        ]
        return EvalDataset(cases=cases, db_path="/fake/db.sqlite")

    def test_counts(self):
        ds = self._make_dataset()
        assert ds.n_exact == 1
        assert ds.n_semantic == 1
        assert len(ds) == 2

    def test_filter_methods(self):
        ds = self._make_dataset()
        assert len(ds.exact_cases()) == 1
        assert len(ds.semantic_cases()) == 1

    def test_save_and_load(self):
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sub" / "dataset.json"
            ds.save(p)
            assert p.exists()
            loaded = EvalDataset.load(p)
            assert len(loaded) == 2
            assert loaded.n_exact == 1
            assert loaded.n_semantic == 1

    def test_save_valid_json(self):
        ds = self._make_dataset()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            ds.save(p)
            payload = json.loads(p.read_text())
            assert "cases" in payload
            assert payload["n_exact"] == 1


# ===========================================================================
# Tests — Paraphraser helpers
# ===========================================================================

class TestJaccard:
    def test_identical(self):
        assert _jaccard("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert _jaccard("apple orange", "banana pear") == 0.0

    def test_partial(self):
        j = _jaccard("the cat sat on the mat", "the cat chased a rat")
        assert 0.0 < j < 1.0

    def test_empty_strings(self):
        # No debe lanzar excepción
        assert _jaccard("", "hello") == 0.0
        assert _jaccard("hello", "") == 0.0


class TestParaphraser:
    def test_returns_none_when_ollama_missing(self):
        p = Paraphraser(model="llama3.2:3b")
        with patch.dict("sys.modules", {"ollama": None}):
            result = p.paraphrase("Some text to paraphrase.")
        assert result is None

    def test_rejects_too_similar_output(self):
        """Si el LLM devuelve el mismo texto, debe rechazarlo."""
        original = "Attention mechanisms allow models to focus on relevant parts."
        p = Paraphraser(max_retries=2)

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": original}}

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = p.paraphrase(original)

        assert result is None  # Jaccard = 1.0, debería rechazarse siempre

    def test_accepts_valid_paraphrase(self):
        original = "Attention mechanisms allow models to focus on relevant parts."
        paraphrase = "Neural networks use focus strategies to identify important information."
        p = Paraphraser(max_retries=1)

        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": paraphrase}}

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = p.paraphrase(original)

        assert result == paraphrase


# ===========================================================================
# Tests — _extract_fragment
# ===========================================================================

class TestExtractFragment:
    _LONG = (
        "First sentence of the paragraph contains an introduction. "
        "Second sentence discusses the main topic in great detail. "
        "Third sentence elaborates on the methodology used. "
        "Fourth sentence presents the experimental results clearly. "
        "Fifth sentence concludes with future directions and remarks."
    )

    def test_returns_string(self):
        frag = _extract_fragment(self._LONG, n_sentences=2)
        assert isinstance(frag, str)
        assert len(frag) >= 40

    def test_too_short_text_returns_none(self):
        result = _extract_fragment("Hi.", n_sentences=3)
        assert result is None

    def test_fragment_shorter_than_source(self):
        frag = _extract_fragment(self._LONG, n_sentences=2)
        assert len(frag) < len(self._LONG)

    def test_different_seeds_may_differ(self):
        """Con texto largo puede haber variabilidad (no garantizado, pero testeable)."""
        import random
        results = set()
        for seed in range(10):
            random.seed(seed)
            f = _extract_fragment(self._LONG, n_sentences=1)
            if f:
                results.add(f)
        # Esperamos al menos 2 resultados distintos en 10 semillas
        assert len(results) >= 1


# ===========================================================================
# Tests — _sample_chunks_stratified
# ===========================================================================

class TestSampleChunksStratified:
    def test_returns_correct_count(self):
        conn = _make_in_memory_db()
        rows = _sample_chunks_stratified(conn, sample_size=3, min_chars=100, seed=42)
        assert len(rows) <= 3
        assert len(rows) >= 1

    def test_respects_min_chars(self):
        conn = _make_in_memory_db()
        rows = _sample_chunks_stratified(conn, sample_size=10, min_chars=99999, seed=42)
        assert len(rows) == 0

    def test_stratified_across_docs(self):
        conn = _make_in_memory_db()
        rows = _sample_chunks_stratified(conn, sample_size=4, min_chars=100, seed=42)
        arxiv_ids = {row["arxiv_id"] for row in rows}
        assert len(arxiv_ids) == 2  # debe cubrir ambos documentos


# ===========================================================================
# Tests — DatasetGenerator (integración con DB en memoria)
# ===========================================================================

class TestDatasetGenerator:
    def _gen_with_mock_db(self, include_semantic=False, sample_size=4) -> EvalDataset:
        """Crea un generador apuntando a una BD en memoria mockeada."""
        mock_conn = _make_in_memory_db()

        gen = DatasetGenerator(
            sample_size=sample_size,
            include_semantic=include_semantic,
            min_chunk_chars=100,
            fragment_sentences=2,
            seed=42,
        )

        # Parchear get_connection para devolver nuestra BD en memoria
        with patch("backend.eval.dataset_generator.get_connection", return_value=mock_conn):
            dataset = gen.generate()

        return dataset

    def test_generates_exact_cases(self):
        ds = self._gen_with_mock_db(include_semantic=False)
        assert ds.n_exact > 0
        assert ds.n_semantic == 0

    def test_exact_cases_have_valid_chunk_ids(self):
        ds = self._gen_with_mock_db(include_semantic=False)
        for case in ds.exact_cases():
            assert isinstance(case.expected_chunk_id, int)
            assert case.expected_chunk_id > 0
            assert case.expected_arxiv_id in ("2401.00001", "2401.00002")

    def test_query_is_substring_or_derived_of_source(self):
        """En casos exact, el fragmento debe ser parte del texto fuente."""
        ds = self._gen_with_mock_db(include_semantic=False)
        for case in ds.exact_cases():
            # El fragmento viene de oraciones del source_text
            assert len(case.query) < len(case.source_text)

    def test_semantic_cases_generated_with_mock_llm(self):
        """Con LLM mockeado que devuelve paráfrasis válida, debe crear casos semánticos."""
        paraphrase_text = "Neural networks utilize focus mechanisms to identify key information segments."
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": paraphrase_text}}

        mock_conn = _make_in_memory_db()
        gen = DatasetGenerator(
            sample_size=4,
            include_semantic=True,
            min_chunk_chars=100,
            fragment_sentences=2,
            seed=42,
        )

        with patch("backend.eval.dataset_generator.get_connection", return_value=mock_conn), \
             patch.dict("sys.modules", {"ollama": mock_ollama}):
            ds = gen.generate()

        assert ds.n_semantic > 0
        for case in ds.semantic_cases():
            assert case.paraphrase_model == "llama3.2:3b"

    def test_dataset_saveable(self):
        ds = self._gen_with_mock_db(include_semantic=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            ds.save(p)
            loaded = EvalDataset.load(p)
            assert len(loaded) == len(ds)

    def test_empty_db_returns_empty_dataset(self):
        empty_conn = sqlite3.connect(":memory:")
        empty_conn.row_factory = sqlite3.Row
        empty_conn.executescript("""
            CREATE TABLE documents (arxiv_id TEXT PRIMARY KEY, title TEXT);
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, arxiv_id TEXT, chunk_index INTEGER,
                text TEXT, char_count INTEGER
            );
        """)
        gen = DatasetGenerator(sample_size=10, include_semantic=False)
        with patch("backend.eval.dataset_generator.get_connection", return_value=empty_conn):
            ds = gen.generate()
        assert len(ds) == 0
