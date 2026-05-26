"""Tests de EvalCase y EvalDataset — serialización JSON ida y vuelta."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest

from backend.eval.schema import EvalCase, EvalDataset


def _make_case(**kwargs):
    defaults = dict(
        case_id="exact_0001", case_type="exact",
        query="attention mechanism transformer",
        expected_chunk_id=42, expected_arxiv_id="2401.00001",
        expected_chunk_index=0, source_text="Full chunk text here.",
        fragment_used="attention mechanism transformer",
        paraphrase_model=None, metadata={"title": "Test Paper"},
    )
    defaults.update(kwargs)
    return EvalCase(**defaults)


def test_evalcase_roundtrip_json():
    case = _make_case()
    data = case.to_dict()
    restored = EvalCase.from_dict(data)
    assert restored.case_id == case.case_id
    assert restored.query == case.query
    assert restored.expected_chunk_id == case.expected_chunk_id


def test_evaldataset_save_and_load(tmp_path):
    cases = [_make_case(case_id=f"exact_{i:04d}", expected_chunk_id=i) for i in range(3)]
    ds = EvalDataset(cases=cases)
    path = tmp_path / "dataset.json"
    ds.save(path)
    loaded = EvalDataset.load(path)
    assert len(loaded) == 3


def test_evaldataset_exact_cases():
    cases = [_make_case(case_type="exact"), _make_case(case_type="semantic", case_id="sem_0001")]
    ds = EvalDataset(cases=cases)
    exact = list(ds.exact_cases())
    semantic = list(ds.semantic_cases())
    assert len(exact) == 1
    assert len(semantic) == 1


def test_evaldataset_n_exact_and_n_semantic():
    cases = [
        _make_case(case_type="exact"),
        _make_case(case_type="exact", case_id="e2"),
        _make_case(case_type="semantic", case_id="s1"),
    ]
    ds = EvalDataset(cases=cases)
    assert ds.n_exact == 2
    assert ds.n_semantic == 1
