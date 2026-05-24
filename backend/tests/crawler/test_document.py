"""Tests del modelo Document — doc_id, arxiv_id alias, CSV round-trip."""
from __future__ import annotations
import csv
import pytest


def _make_doc(doc_id="arxiv:2301.12345", **kwargs):
    from backend.crawler.document import Document
    defaults = dict(
        title="Test Paper", authors="Alice, Bob",
        abstract="A great paper.", categories="cs.AI, cs.LG",
        published="2023-01-01T00:00:00Z", updated="2023-01-02T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2301.12345",
    )
    defaults.update(kwargs)
    return Document(doc_id=doc_id, **defaults)


def test_doc_id_field():
    doc = _make_doc()
    assert doc.doc_id == "arxiv:2301.12345"


def test_arxiv_id_alias_returns_doc_id():
    doc = _make_doc(doc_id="arxiv:9999.00001")
    assert doc.arxiv_id == "arxiv:9999.00001"
    assert doc.arxiv_id == doc.doc_id


def test_to_dict_uses_doc_id_key():
    from backend.crawler.document import DOCUMENT_FIELDS
    doc = _make_doc()
    d = doc.to_dict()
    assert "doc_id" in d
    assert "arxiv_id" not in d
    assert set(d.keys()) == set(DOCUMENT_FIELDS)


def test_from_dict_new_key():
    from backend.crawler.document import Document
    data = {
        "doc_id": "arxiv:1111.22222", "title": "T", "authors": "A",
        "abstract": "X", "categories": "cs.AI",
        "published": "2024-01-01", "updated": "2024-01-01",
        "pdf_url": "http://x", "fetched_at": "2024-01-01",
    }
    doc = Document.from_dict(data)
    assert doc.doc_id == "arxiv:1111.22222"


def test_from_dict_legacy_key():
    from backend.crawler.document import Document
    data = {
        "arxiv_id": "arxiv:legacy999", "title": "T", "authors": "A",
        "abstract": "X", "categories": "cs.AI",
        "published": "2024-01-01", "updated": "2024-01-01",
        "pdf_url": "http://x", "fetched_at": "2024-01-01",
    }
    doc = Document.from_dict(data)
    assert doc.doc_id == "arxiv:legacy999"


def test_equality_based_on_doc_id():
    doc_a = _make_doc(doc_id="arxiv:001")
    doc_b = _make_doc(doc_id="arxiv:001", title="Different")
    doc_c = _make_doc(doc_id="arxiv:002")
    assert doc_a == doc_b
    assert doc_a != doc_c


def test_hash_based_on_doc_id():
    doc_a = _make_doc(doc_id="arxiv:001")
    doc_b = _make_doc(doc_id="arxiv:001")
    assert hash(doc_a) == hash(doc_b)
    assert len({doc_a, doc_b}) == 1


def test_save_creates_csv_with_doc_id_column(tmp_path):
    csv_path = tmp_path / "docs.csv"
    _make_doc().save(csv_path)
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
    assert "doc_id" in header
    assert "arxiv_id" not in header


def test_save_and_load_all_round_trip(tmp_path):
    from backend.crawler.document import Document
    csv_path = tmp_path / "docs.csv"
    docs = [_make_doc(doc_id=f"arxiv:{i:04d}") for i in range(5)]
    for d in docs:
        d.save(csv_path)
    loaded = Document.load_all(csv_path)
    assert len(loaded) == 5
    assert {d.doc_id for d in loaded} == {f"arxiv:{i:04d}" for i in range(5)}


def test_load_all_from_legacy_csv(tmp_path):
    from backend.crawler.document import Document, DOCUMENT_FIELDS
    csv_path = tmp_path / "legacy.csv"
    legacy_fields = ["arxiv_id"] + DOCUMENT_FIELDS[1:]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=legacy_fields)
        w.writeheader()
        w.writerow({
            "arxiv_id": "arxiv:legacy001", "title": "Legacy",
            "authors": "A", "abstract": "X", "categories": "cs.AI",
            "published": "2024", "updated": "2024",
            "pdf_url": "http://x", "fetched_at": "2024",
        })
    docs = Document.load_all(csv_path)
    assert len(docs) == 1
    assert docs[0].doc_id == "arxiv:legacy001"


def test_load_ids_new_csv(tmp_path):
    csv_path = tmp_path / "docs.csv"
    doc = _make_doc(doc_id="arxiv:2301.12345")
    doc.save(csv_path)
    ids = type(doc).load_ids(csv_path)
    assert "arxiv:2301.12345" in ids


def test_load_ids_legacy_csv(tmp_path):
    from backend.crawler.document import Document, DOCUMENT_FIELDS
    csv_path = tmp_path / "legacy.csv"
    legacy_fields = ["arxiv_id"] + DOCUMENT_FIELDS[1:]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=legacy_fields)
        w.writeheader()
        w.writerow({
            "arxiv_id": "arxiv:old001", "title": "T", "authors": "A",
            "abstract": "X", "categories": "cs.AI",
            "published": "2024", "updated": "2024",
            "pdf_url": "http://x", "fetched_at": "2024",
        })
    ids = Document.load_ids(csv_path)
    assert "arxiv:old001" in ids


def test_composite_id_non_arxiv_source(tmp_path):
    from backend.crawler.document import Document
    doc = Document(
        doc_id="semantic_scholar:abc123",
        title="SS Paper", authors="X",
        abstract="Y", categories="cs.CL",
        published="2024-01-01", updated="2024-01-01",
        pdf_url="https://ss.org/paper/abc123",
    )
    assert doc.doc_id == "semantic_scholar:abc123"
    csv_path = tmp_path / "docs.csv"
    doc.save(csv_path)
    loaded = Document.load_all(csv_path)
    assert loaded[0].doc_id == "semantic_scholar:abc123"
