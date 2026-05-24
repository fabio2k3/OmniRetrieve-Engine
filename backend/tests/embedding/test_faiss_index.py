"""Tests de FaissIndexManager — add, search, save/load, rebuild, maybe_rebuild."""
from __future__ import annotations
import sqlite3
import threading
from datetime import datetime, timezone
import numpy as np
import pytest

from .conftest import DIM


def test_add_and_search(faiss_dir):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=10_000,
        index_path=faiss_dir / "index.faiss",
        id_map_path=faiss_dir / "id_map.npy",
    )
    vecs = np.random.randn(20, DIM).astype(np.float32)
    mgr.add(vecs, list(range(1, 21)))

    assert mgr.total_vectors == 20
    assert mgr.index_type == "IndexFlatL2"

    results = mgr.search(vecs[0], top_k=5)
    assert len(results) == 5
    assert results[0]["chunk_id"] == 1
    assert results[0]["score"] < 1e-4


def test_save_and_load(faiss_dir):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    index_path = faiss_dir / "index.faiss"
    id_map_path = faiss_dir / "id_map.npy"
    mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        index_path=index_path, id_map_path=id_map_path,
    )
    vecs = np.random.randn(20, DIM).astype(np.float32)
    mgr.add(vecs, list(range(1, 21)))
    mgr.save()

    mgr2 = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        index_path=index_path, id_map_path=id_map_path,
    )
    assert mgr2.load()
    assert mgr2.total_vectors == 20
    assert mgr2.search(vecs[0], top_k=3)[0]["chunk_id"] == 1


def test_rebuild_produces_ivfpq(tmp_path, db_path, faiss_dir):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    from backend.database.chunk_repository import (
        get_unembedded_chunks, save_chunk_embeddings_batch, reset_embeddings,
    )
    reset_embeddings(db_path=db_path)

    # Insert enough vectors for IVFPQ (need >=256 for nlist=4, nbits=8)
    conn = sqlite3.connect(str(db_path))
    for i in range(300):
        aid = f"2401.99{i:03d}"
        conn.execute(
            "INSERT OR IGNORE INTO documents "
            "(arxiv_id, title, abstract, full_text, categories, published, "
            "updated, pdf_url, fetched_at, pdf_downloaded) "
            "VALUES (?, '', '', '', '', '', '', '', '', 1)", (aid,)
        )
        conn.execute(
            "INSERT OR IGNORE INTO chunks "
            "(arxiv_id, chunk_index, text, char_count, created_at) "
            "VALUES (?, 0, 'test', 4, '2024-01-01')", (aid,)
        )
    conn.commit()
    conn.close()

    ts = datetime.now(timezone.utc).isoformat()
    pending = get_unembedded_chunks(limit=10_000, db_path=db_path)
    batch = [(np.random.randn(DIM).astype(np.float32).tobytes(), ts, r["id"]) for r in pending]
    save_chunk_embeddings_batch(batch, db_path=db_path)

    mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=10_000,
        index_path=faiss_dir / "index_ivfpq.faiss",
        id_map_path=faiss_dir / "id_map_ivfpq.npy",
    )
    stats = mgr.rebuild(db_path=db_path)
    assert mgr.index_type == "IndexIVFPQ"
    assert mgr.total_vectors >= 256
    assert stats["index_type"] == "IndexIVFPQ"
    assert len(mgr.search(np.random.randn(DIM).astype(np.float32), top_k=10)) == 10


def test_maybe_rebuild_triggers_at_threshold(faiss_dir, db_path):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    from backend.database.chunk_repository import get_unembedded_chunks
    mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=5,
        index_path=faiss_dir / "index_mr.faiss",
        id_map_path=faiss_dir / "id_map_mr.npy",
    )
    vecs = np.random.randn(5, DIM).astype(np.float32)
    ids5 = list(range(9000, 9005))
    mgr.add(vecs, ids5)
    assert mgr.maybe_rebuild(db_path=db_path)
    assert mgr._added_since_last_rebuild == 0


def test_add_concurrent_thread_safety(faiss_dir):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    mgr = FaissIndexManager(dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=10_000)
    lock = threading.Lock()
    errors = []

    def worker(wid):
        try:
            vecs = np.random.randn(10, DIM).astype(np.float32)
            with lock:
                mgr.add(vecs, [wid * 100 + i for i in range(10)])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not errors
    assert mgr.total_vectors == 50