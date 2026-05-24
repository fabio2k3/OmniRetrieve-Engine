"""Tests de EmbeddingPipeline end-to-end con embedder mock inyectado."""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pytest

from .conftest import DIM, total_chunks, create_test_db, MockEmbedder


def test_mock_embedder_encode_shape_and_normalized(mock_embedder):
    texts = ["hello world", "attention mechanism", "neural network"]
    vecs = mock_embedder.encode(texts)
    assert vecs.shape == (3, DIM)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_mock_embedder_encode_single(mock_embedder):
    single = mock_embedder.encode_single("test sentence")
    assert single.shape == (DIM,)
    assert single.dtype == np.float32


def test_mock_embedder_encode_empty(mock_embedder):
    empty = mock_embedder.encode([])
    assert empty.shape == (0, DIM)


def test_pipeline_end_to_end(tmp_path):
    from backend.embedding.faiss.index_manager import FaissIndexManager
    from backend.embedding.pipeline import EmbeddingPipeline
    from backend.database.chunk_repository import (
        get_unembedded_chunks_iter as _iter,
        save_chunk_embeddings_batch as _save,
        get_embedded_count,
    )

    db2 = tmp_path / "test2.db"
    faiss2 = tmp_path / "faiss"
    faiss2.mkdir()
    create_test_db(db2)

    pipeline = EmbeddingPipeline(
        db_path=db2, model_name="mock-model-v0",
        batch_size=4, rebuild_every=10_000,
        nlist=4, m=8, nbits=8,
        index_path=faiss2 / "index.faiss",
        id_map_path=faiss2 / "id_map.npy",
    )
    pipeline._embedder = MockEmbedder()
    pipeline._faiss_mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=10_000,
        index_path=faiss2 / "index.faiss",
        id_map_path=faiss2 / "id_map.npy",
    )

    n_processed = 0
    for batch_rows in _iter(batch_size=4, db_path=db2):
        texts = [r["text"] for r in batch_rows]
        chunk_ids = [r["id"] for r in batch_rows]
        vecs_batch = pipeline._embedder.encode(texts)
        ts = datetime.now(timezone.utc).isoformat()
        db_batch = [(v.astype(np.float32).tobytes(), ts, cid) for v, cid in zip(vecs_batch, chunk_ids)]
        _save(db_batch, db_path=db2)
        pipeline._faiss_mgr.add(vecs_batch, chunk_ids)
        n_processed += len(chunk_ids)

    assert n_processed == total_chunks()
    assert get_embedded_count(db_path=db2) == total_chunks()

    q = MockEmbedder().encode_single("attention transformer")
    results = pipeline._faiss_mgr.search(q, top_k=3)
    assert len(results) == 3
    assert all("chunk_id" in r for r in results)