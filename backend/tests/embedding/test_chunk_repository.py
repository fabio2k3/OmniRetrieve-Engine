"""Tests de chunk_repository — save, get, conteos, iteradores, reset."""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pytest

from .conftest import SAMPLE_CHUNKS, DIM, total_chunks


def test_save_and_get_chunks(db_path):
    from backend.database.chunk_repository import save_chunks, get_chunks
    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        rows = get_chunks(arxiv_id, db_path=db_path)
        assert len(rows) == len(texts)
        for i, row in enumerate(rows):
            assert row["chunk_index"] == i
            assert row["text"] == texts[i]
            assert row["char_count"] == len(texts[i])


def test_save_chunks_replaces_existing(db_path):
    from backend.database.chunk_repository import save_chunks, get_chunks
    save_chunks("2401.00001", ["único chunk nuevo"], db_path=db_path)
    assert len(get_chunks("2401.00001", db_path=db_path)) == 1
    save_chunks("2401.00001", SAMPLE_CHUNKS["2401.00001"], db_path=db_path)
    assert len(get_chunks("2401.00001", db_path=db_path)) == 3


def test_get_chunk_count(db_path):
    from backend.database.chunk_repository import get_chunk_count
    assert get_chunk_count(db_path) == total_chunks()


def test_get_embedded_count_starts_zero(db_path):
    from backend.database.chunk_repository import get_embedded_count
    assert get_embedded_count(db_path) == 0


def test_get_chunk_stats(db_path):
    from backend.database.chunk_repository import get_chunk_stats
    stats = get_chunk_stats(db_path)
    assert stats["total_chunks"] == total_chunks()
    assert stats["embedded_chunks"] == 0
    assert stats["pending_chunks"] == total_chunks()


def test_get_unembedded_chunks_limit(db_path):
    from backend.database.chunk_repository import get_unembedded_chunks
    pending = get_unembedded_chunks(limit=5, db_path=db_path)
    assert len(pending) == 5
    assert all("id" in row.keys() and "text" in row.keys() for row in pending)


def test_save_chunk_embedding_individual(db_path):
    from backend.database.chunk_repository import (
        get_unembedded_chunks, save_chunk_embedding, get_embedded_count,
    )
    first_id = get_unembedded_chunks(limit=1, db_path=db_path)[0]["id"]
    vec = np.random.randn(DIM).astype(np.float32)
    save_chunk_embedding(first_id, vec.tobytes(), db_path=db_path)
    assert get_embedded_count(db_path) == 1
    remaining = get_unembedded_chunks(limit=total_chunks(), db_path=db_path)
    assert all(r["id"] != first_id for r in remaining)


def test_save_chunk_embeddings_batch(db_path):
    from backend.database.chunk_repository import (
        get_unembedded_chunks, save_chunk_embeddings_batch, get_embedded_count,
    )
    now = datetime.now(timezone.utc).isoformat()
    pending = get_unembedded_chunks(limit=total_chunks(), db_path=db_path)
    batch = [(np.random.randn(DIM).astype(np.float32).tobytes(), now, r["id"]) for r in pending]
    n = save_chunk_embeddings_batch(batch, db_path=db_path)
    assert n == len(pending)
    assert get_embedded_count(db_path) == total_chunks()


def test_get_unembedded_chunks_iter_no_duplicates(db_path):
    from backend.database.chunk_repository import get_unembedded_chunks_iter, reset_embeddings
    reset_embeddings(db_path=db_path)
    seen = []
    for batch in get_unembedded_chunks_iter(batch_size=3, db_path=db_path):
        assert len(batch) <= 3
        seen.extend(r["id"] for r in batch)
    assert len(seen) == total_chunks()
    assert len(set(seen)) == total_chunks()


def test_get_all_embeddings_iter_only_embedded(db_path):
    from backend.database.chunk_repository import (
        get_unembedded_chunks, save_chunk_embeddings_batch,
        get_all_embeddings_iter, reset_embeddings,
    )
    reset_embeddings(db_path=db_path)
    now = datetime.now(timezone.utc).isoformat()
    half = get_unembedded_chunks(limit=total_chunks() // 2, db_path=db_path)
    batch = [(np.random.randn(DIM).astype(np.float32).tobytes(), now, r["id"]) for r in half]
    save_chunk_embeddings_batch(batch, db_path=db_path)

    retrieved = []
    for batch_rows in get_all_embeddings_iter(batch_size=2, db_path=db_path):
        for row in batch_rows:
            assert row["embedding"] is not None
            vec_back = np.frombuffer(row["embedding"], dtype=np.float32)
            assert vec_back.shape == (DIM,)
            retrieved.append(row["id"])
    assert len(retrieved) == len(half)


def test_reset_embeddings(db_path):
    from backend.database.chunk_repository import (
        get_unembedded_chunks, save_chunk_embeddings_batch,
        reset_embeddings, get_embedded_count,
    )
    now = datetime.now(timezone.utc).isoformat()
    pending = get_unembedded_chunks(limit=total_chunks(), db_path=db_path)
    batch = [(np.random.randn(DIM).astype(np.float32).tobytes(), now, r["id"]) for r in pending]
    save_chunk_embeddings_batch(batch, db_path=db_path)

    n_reset = reset_embeddings(db_path=db_path)
    assert n_reset == total_chunks()
    assert get_embedded_count(db_path) == 0
