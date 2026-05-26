"""Tests de embedding_repository — faiss_log, embedding_meta, stats."""
from __future__ import annotations
import sqlite3
import pytest

from .conftest import total_chunks


def test_log_faiss_build(db_path):
    from backend.database.embedding_repository import log_faiss_build
    log_faiss_build({
        "n_vectors": 100, "index_type": "IndexIVFPQ",
        "model_name": "mock", "nlist": 10, "m": 8, "nbits": 8,
        "index_path": "/tmp/index.faiss", "id_map_path": "/tmp/id_map.npy",
    }, db_path=db_path)
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT * FROM faiss_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row is not None
    assert row[2] == 100
    assert row[3] == "IndexIVFPQ"


def test_save_and_get_embedding_meta(db_path):
    from backend.database.embedding_repository import save_embedding_meta, get_embedding_meta
    save_embedding_meta("model_name", "mock-model-v0", db_path=db_path)
    save_embedding_meta("last_run_at", "2024-01-01T00:00:00Z", db_path=db_path)

    assert get_embedding_meta("model_name", db_path=db_path) == "mock-model-v0"
    assert get_embedding_meta("last_run_at", db_path=db_path) == "2024-01-01T00:00:00Z"
    assert get_embedding_meta("nonexistent", db_path=db_path) is None


def test_upsert_embedding_meta(db_path):
    from backend.database.embedding_repository import save_embedding_meta, get_embedding_meta
    save_embedding_meta("model_name", "mock-model-v0", db_path=db_path)
    save_embedding_meta("model_name", "mock-model-v1", db_path=db_path)
    assert get_embedding_meta("model_name", db_path=db_path) == "mock-model-v1"


def test_get_embedding_stats(db_path):
    from backend.database.embedding_repository import log_faiss_build, get_embedding_stats
    log_faiss_build({
        "n_vectors": 100, "index_type": "IndexIVFPQ",
        "model_name": "mock", "nlist": 10, "m": 8, "nbits": 8,
        "index_path": "/tmp/i.faiss", "id_map_path": "/tmp/m.npy",
    }, db_path=db_path)
    s = get_embedding_stats(db_path=db_path)
    assert s["total_chunks"] == total_chunks()
    assert s["pending_chunks"] == total_chunks()
    assert s["last_index_type"] == "IndexIVFPQ"
    assert s["last_n_vectors"] == 100
