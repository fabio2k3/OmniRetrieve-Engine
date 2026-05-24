"""Tests de DatasetGenerator, _extract_fragment y _sample_chunks_stratified."""
from __future__ import annotations
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from backend.eval.schema import EvalCase, EvalDataset
from backend.eval.dataset_generator import (
    _extract_fragment, _sample_chunks_stratified, DatasetGenerator,
)

_LONG = (
    "First sentence of the paragraph contains an introduction. "
    "Second sentence discusses the main topic in great detail. "
    "Third sentence elaborates on the methodology used. "
    "Fourth sentence presents the experimental results clearly. "
    "Fifth sentence concludes with future directions and remarks."
)


def _make_in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE documents (arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL);
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, arxiv_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, text TEXT NOT NULL, char_count INTEGER
        );
    """)
    conn.executemany("INSERT INTO documents VALUES (?, ?)",
                     [("2401.00001", "Paper Alpha"), ("2401.00002", "Paper Beta")])
    long_text = _LONG
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


# ── _extract_fragment ────────────────────────────────────────────────────────

def test_extract_fragment_returns_string():
    assert isinstance(_extract_fragment(_LONG, n_sentences=2), str)


def test_extract_fragment_too_short_returns_none():
    assert _extract_fragment("Hi.", n_sentences=3) is None


def test_extract_fragment_shorter_than_source():
    frag = _extract_fragment(_LONG, n_sentences=2)
    assert len(frag) < len(_LONG)


# ── _sample_chunks_stratified ────────────────────────────────────────────────

def test_sample_chunks_correct_count():
    conn = _make_in_memory_db()
    rows = _sample_chunks_stratified(conn, sample_size=3, min_chars=100, seed=42)
    assert 1 <= len(rows) <= 3


def test_sample_chunks_respects_min_chars():
    conn = _make_in_memory_db()
    rows = _sample_chunks_stratified(conn, sample_size=10, min_chars=99999, seed=42)
    assert len(rows) == 0


def test_sample_chunks_stratified_covers_both_docs():
    conn = _make_in_memory_db()
    rows = _sample_chunks_stratified(conn, sample_size=4, min_chars=100, seed=42)
    arxiv_ids = {row["arxiv_id"] for row in rows}
    assert len(arxiv_ids) == 2


# ── DatasetGenerator ─────────────────────────────────────────────────────────

def _gen_with_mock_db(include_semantic=False, sample_size=4):
    mock_conn = _make_in_memory_db()
    gen = DatasetGenerator(
        sample_size=sample_size, include_semantic=include_semantic,
        min_chunk_chars=100, fragment_sentences=2, seed=42,
    )
    with patch("backend.eval.dataset_generator.get_connection", return_value=mock_conn):
        return gen.generate()


def test_generates_exact_cases():
    ds = _gen_with_mock_db(include_semantic=False)
    assert ds.n_exact > 0
    assert ds.n_semantic == 0


def test_exact_cases_have_valid_chunk_ids():
    ds = _gen_with_mock_db(include_semantic=False)
    for case in ds.exact_cases():
        assert isinstance(case.expected_chunk_id, int)
        assert case.expected_chunk_id > 0
        assert case.expected_arxiv_id in ("2401.00001", "2401.00002")


def test_query_shorter_than_source():
    ds = _gen_with_mock_db(include_semantic=False)
    for case in ds.exact_cases():
        assert len(case.query) < len(case.source_text)


def test_semantic_cases_with_mock_llm():
    paraphrase = "Neural networks utilize focus mechanisms to identify key information segments."
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = {"message": {"content": paraphrase}}
    mock_conn = _make_in_memory_db()
    gen = DatasetGenerator(
        sample_size=4, include_semantic=True,
        min_chunk_chars=100, fragment_sentences=2, seed=42,
    )
    with patch("backend.eval.dataset_generator.get_connection", return_value=mock_conn), \
         patch.dict("sys.modules", {"ollama": mock_ollama}):
        ds = gen.generate()
    assert ds.n_semantic > 0


def test_dataset_saveable(tmp_path):
    ds = _gen_with_mock_db(include_semantic=False)
    p = tmp_path / "dataset.json"
    ds.save(p)
    loaded = EvalDataset.load(p)
    assert len(loaded) == len(ds)


def test_empty_db_returns_empty_dataset():
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
