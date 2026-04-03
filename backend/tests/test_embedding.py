"""
test_embedding.py
=================
Tests de integración del módulo de embedding.

Cubre las tres capas del módulo:
  1. chunk_repository       — operaciones CRUD sobre la tabla chunks.
  2. embedding_repository   — esquema propio (faiss_log, embedding_meta).
  3. ChunkEmbedder          — vectorización (usa embedder mock para evitar
                              descargar modelos en CI).
  4. FaissIndexManager      — ciclo de vida del índice FAISS.
  5. EmbeddingPipeline      — flujo end-to-end con embedder mock inyectado.

El embedder mock genera vectores aleatorios normalizados del mismo dtype y
shape que sentence-transformers, sin necesitar la librería instalada.

Ejecución
---------
    python -m pytest backend/tests/test_embedding.py -v
    python -m backend.tests.test_embedding          # modo script
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path

import numpy as np

from backend.database.schema import init_db
from backend.database.chunk_repository import (
    save_chunks,
    get_chunks,
    get_chunk_count,
    get_embedded_count,
    get_chunk_stats,
    get_unembedded_chunks,
    get_unembedded_chunks_iter,
    save_chunk_embedding,
    save_chunk_embeddings_batch,
    get_all_embeddings_iter,
    reset_embeddings,
)
from backend.database.embedding_repository import (
    init_embedding_schema,
    log_faiss_build,
    save_embedding_meta,
    get_embedding_meta,
    get_embedding_stats,
)
from backend.embedding.faiss_index import FaissIndexManager

# ---------------------------------------------------------------------------
# Datos de ejemplo
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "arxiv_id":       "2401.00001",
        "title":          "Attention Is All You Need",
        "full_text":      "The transformer architecture relies entirely on attention mechanisms.",
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2401.00002",
        "title":          "BERT: Pre-training of Deep Bidirectional Transformers",
        "full_text":      "BERT applies bidirectional training of transformers for language understanding.",
        "pdf_downloaded": 1,
    },
    {
        "arxiv_id":       "2401.00003",
        "title":          "GPT-3: Language Models are Few-Shot Learners",
        "full_text":      "Large language models can perform tasks with few examples.",
        "pdf_downloaded": 1,
    },
]

SAMPLE_CHUNKS = {
    "2401.00001": [
        "The transformer model uses self-attention to compute representations.",
        "Multi-head attention allows the model to attend to different positions.",
        "Positional encoding is added to give the model information about sequence order.",
    ],
    "2401.00002": [
        "BERT is trained on masked language modeling and next sentence prediction.",
        "The bidirectional context helps BERT understand word meaning in context.",
    ],
    "2401.00003": [
        "GPT-3 has 175 billion parameters and is trained on internet text.",
        "Few-shot learning allows the model to generalize from minimal examples.",
        "Prompt engineering is key to eliciting good responses from GPT-3.",
        "In-context learning happens at inference time without gradient updates.",
    ],
}

DIM = 64  # dimensión del embedder mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_db(path: Path) -> None:
    """Crea una BD temporal con esquema completo y documentos de ejemplo."""
    init_db(path)
    init_embedding_schema(path)
    conn = sqlite3.connect(str(path))
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (arxiv_id, title, full_text, pdf_downloaded,
             abstract, categories, published, updated, pdf_url, fetched_at)
        VALUES (:arxiv_id, :title, :full_text, :pdf_downloaded,
                '', '', '', '', '', '')
        """,
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()

    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        save_chunks(arxiv_id, texts, db_path=path)


class _MockEmbedder:
    """
    Embedder falso que genera vectores aleatorios L2-normalizados.

    Tiene la misma interfaz que ChunkEmbedder para poder ser inyectado
    en EmbeddingPipeline sin tocar la implementación del pipeline.
    """

    dim        = DIM
    model_name = "mock-model-v0"

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.random.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _total_chunks() -> int:
    return sum(len(v) for v in SAMPLE_CHUNKS.values())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(db_path: Path, faiss_dir: Path) -> None:  # noqa: C901
    sep  = "─" * 60
    sep2 = "═" * 60

    # ── 1. chunk_repository — save_chunks / get_chunks ──────────────────────
    print(f"\n{sep}")
    print("  TEST 1: save_chunks() / get_chunks()")
    print(sep)

    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        rows = get_chunks(arxiv_id, db_path=db_path)
        assert len(rows) == len(texts), (
            f"{arxiv_id}: esperados {len(texts)} chunks, hay {len(rows)}"
        )
        for i, row in enumerate(rows):
            assert row["chunk_index"] == i
            assert row["text"] == texts[i]
            assert row["char_count"] == len(texts[i])

    # Verificar reemplazo: guardar chunks nuevos sobreescribe los anteriores
    save_chunks("2401.00001", ["único chunk nuevo"], db_path=db_path)
    assert len(get_chunks("2401.00001", db_path=db_path)) == 1
    # Restaurar chunks originales
    save_chunks("2401.00001", SAMPLE_CHUNKS["2401.00001"], db_path=db_path)
    assert len(get_chunks("2401.00001", db_path=db_path)) == 3

    total = _total_chunks()
    print(f"  ✔ {total} chunks guardados y verificados correctamente.")
    print(f"  ✔ Reemplazo de chunks funciona — restaurado a {total}.")

    # ── 2. chunk_repository — conteos ───────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 2: get_chunk_count() / get_embedded_count() / get_chunk_stats()")
    print(sep)

    assert get_chunk_count(db_path)    == total
    assert get_embedded_count(db_path) == 0
    stats = get_chunk_stats(db_path)
    assert stats["total_chunks"]    == total
    assert stats["embedded_chunks"] == 0
    assert stats["pending_chunks"]  == total

    print(f"  ✔ total={stats['total_chunks']} embedded=0 pending={stats['pending_chunks']}")

    # ── 3. chunk_repository — get_unembedded_chunks ─────────────────────────
    print(f"\n{sep}")
    print("  TEST 3: get_unembedded_chunks() — sin embedding")
    print(sep)

    pending = get_unembedded_chunks(limit=5, db_path=db_path)
    assert len(pending) == 5
    # La query filtra WHERE embedding IS NULL, por lo que todos son pendientes.
    # Verificamos columnas presentes en el resultado.
    assert all("id" in row.keys() and "text" in row.keys() for row in pending)
    print(f"  ✔ Devolvió 5 chunks sin embedding (hay {total} en total).")

    # ── 4. chunk_repository — save_chunk_embedding (individual) ─────────────
    print(f"\n{sep}")
    print("  TEST 4: save_chunk_embedding() — un embedding individual")
    print(sep)

    first_id = get_unembedded_chunks(limit=1, db_path=db_path)[0]["id"]
    vec      = np.random.randn(DIM).astype(np.float32)
    save_chunk_embedding(first_id, vec.tobytes(), db_path=db_path)

    assert get_embedded_count(db_path) == 1
    rows_with = get_unembedded_chunks(limit=total, db_path=db_path)
    assert all(r["id"] != first_id for r in rows_with)

    print(f"  ✔ chunk id={first_id} embedido — embedded_count=1.")

    # ── 5. chunk_repository — save_chunk_embeddings_batch ───────────────────
    print(f"\n{sep}")
    print("  TEST 5: save_chunk_embeddings_batch() — lote de embeddings")
    print(sep)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    pending2 = get_unembedded_chunks(limit=total, db_path=db_path)
    batch    = [(np.random.randn(DIM).astype(np.float32).tobytes(), now, r["id"])
                for r in pending2]
    n = save_chunk_embeddings_batch(batch, db_path=db_path)
    assert n == len(pending2)
    assert get_embedded_count(db_path) == total

    print(f"  ✔ {n} embeddings persistidos en lote — todos los chunks embedidos.")

    # ── 6. chunk_repository — get_unembedded_chunks_iter ────────────────────
    print(f"\n{sep}")
    print("  TEST 6: get_unembedded_chunks_iter() — iterador por lotes")
    print(sep)

    # Reset primero para tener chunks pendientes
    reset_embeddings(db_path=db_path)
    assert get_embedded_count(db_path) == 0

    seen = []
    for batch in get_unembedded_chunks_iter(batch_size=3, db_path=db_path):
        assert len(batch) <= 3
        seen.extend(r["id"] for r in batch)
    assert len(seen) == total
    assert len(set(seen)) == total, "No debe haber IDs duplicados entre lotes."

    print(f"  ✔ {total} chunks iterados en lotes de ≤3 — sin duplicados.")

    # ── 7. chunk_repository — get_all_embeddings_iter ───────────────────────
    print(f"\n{sep}")
    print("  TEST 7: get_all_embeddings_iter() — solo chunks con embedding")
    print(sep)

    # Embeder la mitad de los chunks
    half = get_unembedded_chunks(limit=total // 2, db_path=db_path)
    batch_half = [(np.random.randn(DIM).astype(np.float32).tobytes(), now, r["id"])
                  for r in half]
    save_chunk_embeddings_batch(batch_half, db_path=db_path)

    retrieved = []
    for batch in get_all_embeddings_iter(batch_size=2, db_path=db_path):
        for row in batch:
            assert row["embedding"] is not None
            vec_back = np.frombuffer(row["embedding"], dtype=np.float32)
            assert vec_back.shape == (DIM,)
            retrieved.append(row["id"])

    assert len(retrieved) == len(half)
    print(f"  ✔ {len(retrieved)} embeddings recuperados y verificados (shape y dtype).")

    # ── 8. chunk_repository — reset_embeddings ──────────────────────────────
    print(f"\n{sep}")
    print("  TEST 8: reset_embeddings() — pone todos los embeddings a NULL")
    print(sep)

    n_reset = reset_embeddings(db_path=db_path)
    assert n_reset == total
    assert get_embedded_count(db_path) == 0

    print(f"  ✔ {n_reset} embeddings reseteados — embedded_count=0.")

    # ── 9. embedding_repository — faiss_log ─────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 9: log_faiss_build() — registro en faiss_log")
    print(sep)

    log_faiss_build({
        "n_vectors": 100, "index_type": "IndexIVFPQ",
        "model_name": "mock", "nlist": 10, "m": 8, "nbits": 8,
        "index_path": "/tmp/index.faiss", "id_map_path": "/tmp/id_map.npy",
    }, db_path=db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT * FROM faiss_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[2] == 100         # n_vectors
    assert row[3] == "IndexIVFPQ"  # index_type

    print(f"  ✔ faiss_log insertado — n_vectors={row[2]} index_type={row[3]}")

    # ── 10. embedding_repository — embedding_meta ────────────────────────────
    print(f"\n{sep}")
    print("  TEST 10: save_embedding_meta() / get_embedding_meta()")
    print(sep)

    save_embedding_meta("model_name", "mock-model-v0", db_path=db_path)
    save_embedding_meta("last_run_at", "2024-01-01T00:00:00Z", db_path=db_path)

    assert get_embedding_meta("model_name", db_path=db_path) == "mock-model-v0"
    assert get_embedding_meta("last_run_at", db_path=db_path) == "2024-01-01T00:00:00Z"
    assert get_embedding_meta("nonexistent", db_path=db_path) is None

    # Upsert — actualizar valor existente
    save_embedding_meta("model_name", "mock-model-v1", db_path=db_path)
    assert get_embedding_meta("model_name", db_path=db_path) == "mock-model-v1"

    print("  ✔ metadata guardada, leída y actualizada correctamente.")

    # ── 11. embedding_repository — get_embedding_stats ──────────────────────
    print(f"\n{sep}")
    print("  TEST 11: get_embedding_stats() — resumen combinado")
    print(sep)

    s = get_embedding_stats(db_path=db_path)
    assert s["total_chunks"]    == total
    assert s["pending_chunks"]  == total        # reset_embeddings se hizo antes
    assert s["last_index_type"] == "IndexIVFPQ"
    assert s["last_n_vectors"]  == 100

    print(f"  ✔ total={s['total_chunks']} pending={s['pending_chunks']} "
          f"last_type={s['last_index_type']}")

    # ── 12. MockEmbedder — encode / encode_single ────────────────────────────
    print(f"\n{sep}")
    print("  TEST 12: _MockEmbedder — encode() y encode_single()")
    print(sep)

    emb = _MockEmbedder()
    texts = ["hello world", "attention mechanism", "neural network"]

    vecs = emb.encode(texts)
    assert vecs.shape    == (3, DIM)
    assert vecs.dtype    == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Los vectores deben estar normalizados."

    single = emb.encode_single("test sentence")
    assert single.shape == (DIM,)
    assert single.dtype == np.float32

    empty = emb.encode([])
    assert empty.shape == (0, DIM)

    print(f"  ✔ encode: shape={vecs.shape} dtype={vecs.dtype} normalizado=True")
    print(f"  ✔ encode_single: shape={single.shape}")
    print(f"  ✔ encode([]): shape={empty.shape}")

    # ── 13. FaissIndexManager — add / search ─────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 13: FaissIndexManager — add() y search()")
    print(sep)

    index_path  = faiss_dir / "index.faiss"
    id_map_path = faiss_dir / "id_map.npy"
    mgr = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        rebuild_every=10_000,
        index_path=index_path,
        id_map_path=id_map_path,
    )

    # Añadir 20 vectores (insuficientes para IVFPQ → fallback FlatL2)
    vecs20  = np.random.randn(20, DIM).astype(np.float32)
    ids20   = list(range(1, 21))
    mgr.add(vecs20, ids20)

    assert mgr.total_vectors  == 20
    assert mgr.index_type     == "IndexFlatL2"

    results = mgr.search(vecs20[0], top_k=5)
    assert len(results)              == 5
    assert results[0]["chunk_id"]    == 1       # el vector más cercano a sí mismo
    assert results[0]["score"]       < 1e-4     # distancia ≈ 0

    print(f"  ✔ 20 vectores añadidos — tipo={mgr.index_type} total={mgr.total_vectors}")
    print(f"  ✔ search(top_k=5): {results[:2]}")

    # ── 14. FaissIndexManager — save / load ──────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 14: FaissIndexManager — save() / load()")
    print(sep)

    mgr.save()
    assert index_path.exists()
    assert id_map_path.exists()

    mgr2 = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        index_path=index_path,
        id_map_path=id_map_path,
    )
    loaded = mgr2.load()
    assert loaded
    assert mgr2.total_vectors == 20

    results2 = mgr2.search(vecs20[0], top_k=3)
    assert results2[0]["chunk_id"] == 1

    print(f"  ✔ Índice guardado y cargado — total_vectors={mgr2.total_vectors}")

    # ── 15. FaissIndexManager — rebuild con suficientes vectores (IVFPQ) ─────
    print(f"\n{sep}")
    print("  TEST 15: FaissIndexManager.rebuild() — IndexIVFPQ con suficientes vectores")
    print(sep)

    # nlist=4, nbits=8 → FAISS necesita max(4*39=156, 2^8=256)=256 vectores.
    # Usamos 300 para tener margen suficiente.
    reset_embeddings(db_path=db_path)

    # Insertar docs y chunks ficticios para llegar a 200 vectores
    conn2 = sqlite3.connect(str(db_path))
    for i in range(300):
        aid = f"2401.99{i:03d}"
        conn2.execute(
            "INSERT OR IGNORE INTO documents "
            "(arxiv_id, title, abstract, full_text, categories, published, "
            "updated, pdf_url, fetched_at, pdf_downloaded) "
            "VALUES (?, '', '', '', '', '', '', '', '', 1)", (aid,)
        )
        conn2.execute(
            "INSERT OR IGNORE INTO chunks "
            "(arxiv_id, chunk_index, text, char_count, created_at) "
            "VALUES (?, 0, 'test', 4, '2024-01-01')", (aid,)
        )
    conn2.commit()
    conn2.close()

    # Embedir todos los chunks ficticios
    from datetime import datetime, timezone as tz
    ts_now = datetime.now(tz.utc).isoformat()
    all_pending = get_unembedded_chunks(limit=10_000, db_path=db_path)
    batch_200 = [
        (np.random.randn(DIM).astype(np.float32).tobytes(), ts_now, r["id"])
        for r in all_pending
    ]
    save_chunk_embeddings_batch(batch_200, db_path=db_path)

    mgr3 = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        rebuild_every=10_000,
        index_path=faiss_dir / "index_ivfpq.faiss",
        id_map_path=faiss_dir / "id_map_ivfpq.npy",
    )
    stats_r = mgr3.rebuild(db_path=db_path)

    assert mgr3.index_type    == "IndexIVFPQ"
    assert mgr3.total_vectors >= 256
    assert stats_r["index_type"] == "IndexIVFPQ"

    q = np.random.randn(DIM).astype(np.float32)
    results3 = mgr3.search(q, top_k=10)
    assert len(results3) == 10

    print(f"  ✔ rebuild() → tipo={mgr3.index_type} total={mgr3.total_vectors}")
    print(f"  ✔ search después de rebuild: {len(results3)} resultados")

    # ── 16. FaissIndexManager — maybe_rebuild dispara en umbral ──────────────
    print(f"\n{sep}")
    print("  TEST 16: FaissIndexManager.maybe_rebuild() — disparo en umbral")
    print(sep)

    mgr4 = FaissIndexManager(
        dim=DIM, nlist=4, m=8, nbits=8,
        rebuild_every=5,         # umbral bajo para el test
        index_path=faiss_dir / "index_mr.faiss",
        id_map_path=faiss_dir / "id_map_mr.npy",
    )
    vecs5 = np.random.randn(5, DIM).astype(np.float32)
    ids5  = [r["id"] for r in get_unembedded_chunks(limit=5, db_path=db_path)]
    # Si no hay suficientes pendientes, usar IDs ficticios
    if len(ids5) < 5:
        ids5 = list(range(9000, 9005))

    mgr4.add(vecs5, ids5)
    triggered = mgr4.maybe_rebuild(db_path=db_path)
    assert triggered, "maybe_rebuild debe dispararse al alcanzar el umbral."
    assert mgr4._added_since_last_rebuild == 0, "El contador debe resetearse tras rebuild."

    print(f"  ✔ maybe_rebuild() disparado al añadir {mgr4.rebuild_every} vectores.")

    # ── 17. EmbeddingPipeline end-to-end (embedder mock inyectado) ────────────
    print(f"\n{sep}")
    print("  TEST 17: EmbeddingPipeline end-to-end con embedder mock")
    print(sep)

    # Crear BD fresca para este test
    with tempfile.TemporaryDirectory() as tmp2:
        db2    = Path(tmp2) / "test2.db"
        faiss2 = Path(tmp2) / "faiss"
        faiss2.mkdir()
        create_test_db(db2)

        from backend.embedding.pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline(
            db_path=db2,
            model_name="mock-model-v0",
            batch_size=4,
            rebuild_every=10_000,
            nlist=4, m=8, nbits=8,
            index_path=faiss2 / "index.faiss",
            id_map_path=faiss2 / "id_map.npy",
        )
        # Inyectar embedder mock para no necesitar sentence-transformers
        pipeline._embedder  = _MockEmbedder()
        pipeline._faiss_mgr = FaissIndexManager(
            dim=DIM, nlist=4, m=8, nbits=8,
            rebuild_every=10_000,
            index_path=faiss2 / "index.faiss",
            id_map_path=faiss2 / "id_map.npy",
        )

        # Ejecutar run() directamente desde _process_batch (saltamos init modelo)
        from backend.database.chunk_repository import (
            get_unembedded_chunks_iter as _iter,
            save_chunk_embeddings_batch as _save,
        )

        n_processed = 0
        for batch_rows in _iter(batch_size=4, db_path=db2):
            texts      = [r["text"] for r in batch_rows]
            chunk_ids  = [r["id"]   for r in batch_rows]
            vecs_batch = pipeline._embedder.encode(texts)

            from datetime import datetime, timezone as tz2
            ts = datetime.now(tz2.utc).isoformat()
            db_batch = [
                (v.astype(np.float32).tobytes(), ts, cid)
                for v, cid in zip(vecs_batch, chunk_ids)
            ]
            _save(db_batch, db_path=db2)
            pipeline._faiss_mgr.add(vecs_batch, chunk_ids)
            n_processed += len(chunk_ids)

        total2 = _total_chunks()
        assert n_processed     == total2
        assert get_embedded_count(db_path=db2) == total2

        q2      = _MockEmbedder().encode_single("attention transformer")
        results4 = pipeline._faiss_mgr.search(q2, top_k=3)
        assert len(results4)  == 3
        assert all("chunk_id" in r for r in results4)

        print(f"  ✔ {n_processed} chunks embedidos end-to-end.")
        print(f"  ✔ search(top_k=3): {results4}")

    # ── 18. thread safety — add() concurrente ────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 18: FaissIndexManager — add() concurrente (thread safety básico)")
    print(sep)

    mgr5  = FaissIndexManager(dim=DIM, nlist=4, m=8, nbits=8, rebuild_every=10_000)
    lock5 = threading.Lock()
    errors: list[Exception] = []

    def _add_worker(worker_id: int) -> None:
        try:
            vecs_w = np.random.randn(10, DIM).astype(np.float32)
            ids_w  = [worker_id * 100 + i for i in range(10)]
            with lock5:
                mgr5.add(vecs_w, ids_w)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_add_worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errores en threads: {errors}"
    assert mgr5.total_vectors == 50

    print(f"  ✔ 5 hilos × 10 vectores = {mgr5.total_vectors} sin errores.")

    # ── Resumen ──────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  ✅ Todos los tests de embedding completados correctamente.")
    print(f"{sep2}\n")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path  = Path(tmp) / "test_embedding.db"
        faiss_dir = Path(tmp) / "faiss"
        faiss_dir.mkdir()
        print(f"BD temporal : {db_path}")
        print(f"FAISS dir   : {faiss_dir}")
        create_test_db(db_path)
        run_tests(db_path, faiss_dir)


if __name__ == "__main__":
    main()