"""Tests del módulo chunker — clean_text, make_chunks, overlap, edge cases."""
from __future__ import annotations
import pytest


def test_clean_text_collapses_newlines():
    from backend.crawler.chunker import clean_text
    result = clean_text("line1\n\n\n\n\nline2")
    assert "\n\n\n" not in result
    assert "line1" in result and "line2" in result


def test_clean_text_removes_page_numbers():
    from backend.crawler.chunker import clean_text
    result = clean_text("Intro text\n42\nMore text")
    assert "42" not in result.split()


def test_clean_text_collapses_spaces():
    from backend.crawler.chunker import clean_text
    result = clean_text("word1    word2\t\t\tword3")
    assert "  " not in result


def test_make_chunks_returns_list_of_strings():
    from backend.crawler.chunker import make_chunks
    chunks = make_chunks("Hello world. This is a test. " * 30)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
    assert len(chunks) > 0


def test_make_chunks_respects_chunk_size():
    from backend.crawler.chunker import make_chunks
    chunks = make_chunks(("A" * 50 + ". ") * 50, chunk_size=200)
    oversized = [c for c in chunks if len(c) > 400]
    assert not oversized


def test_make_chunks_overlap_shared_sentences():
    from backend.crawler.chunker import make_chunks
    sentences = [f"Sentence {i} is here and has enough text to count." for i in range(20)]
    chunks = make_chunks(" ".join(sentences), chunk_size=200, overlap_sentences=2)
    if len(chunks) >= 2:
        end_of_first = chunks[0].split()[-5:]
        assert any(w in chunks[1] for w in end_of_first)


def test_make_chunks_no_overlap_has_less_total_text():
    from backend.crawler.chunker import make_chunks
    text = ("Word word word. ") * 100
    chunks_ov = make_chunks(text, chunk_size=200, overlap_sentences=2)
    chunks_no = make_chunks(text, chunk_size=200, overlap_sentences=0)
    assert sum(len(c) for c in chunks_ov) >= sum(len(c) for c in chunks_no)


def test_make_chunks_empty_text():
    from backend.crawler.chunker import make_chunks
    assert make_chunks("") == []
    assert make_chunks("   \n\n   ") == []


def test_make_chunks_very_short_text():
    from backend.crawler.chunker import make_chunks
    result = make_chunks("Short text.")
    assert isinstance(result, list)


def test_make_chunks_paragraph_boundaries_respected():
    from backend.crawler.chunker import make_chunks
    p1 = "Alpha beta gamma delta epsilon zeta eta. " * 8
    p2 = "Omega theta iota kappa lambda mu nu xi. " * 8
    chunks = make_chunks(p1 + "\n\n" + p2, chunk_size=150, overlap_sentences=0)
    assert len(chunks) >= 2
    mixed = [c for c in chunks if "Alpha" in c and "Omega" in c]
    assert not mixed


def test_make_chunks_applies_clean_text_internally():
    from backend.crawler.chunker import make_chunks
    text = "Good sentence one here.\n\n\n\n   Good sentence two here."
    full = " ".join(make_chunks(text, chunk_size=500))
    assert "\n\n\n" not in full


def test_make_chunks_default_params_match_explicit():
    from backend.crawler.chunker import make_chunks
    text = "A complete sentence here. " * 100
    assert make_chunks(text) == make_chunks(text, chunk_size=1000, overlap_sentences=2)
