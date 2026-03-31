"""
test_embedding.py
=================
Tests de integracion del modulo de embeddings vectoriales (ChromaDB).

Cubre:
  1. chroma_store: get_collection, add_chunks, get_existing_chunk_ids,
     count, query.
  2. EmbeddingPipeline: incremental por embedded_at IS NULL, pending_count,
     deteccion de chunks nuevos.
  3. VectorRetriever: load, is_ready, retrieve con deduplicacion y top_n.
  4. Integracion end-to-end.

Requiere: pip install chromadb
Los tests se saltan si chromadb no esta instalado.
FakeEmbedder evita descargar cualquier modelo de red.

Nota Windows
------------
ChromaDB mantiene ficheros abiertos (chroma.sqlite3, data_level0.bin...)
mientras el cliente existe en memoria. Para evitar WinError 32 al limpiar
directorios temporales:
  - Se usa un unico TemporaryDirectory raiz con ignore_cleanup_errors=True.
  - Cada test usa un subdirectorio propio dentro de ese raiz.
  - close_client() / close_all_clients() liberan los ficheros antes del
    fin del proceso.

Ejecucion
---------
    python -m pytest backend/tests/test_embedding.py -v
    python -m backend.tests.test_embedding
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Skip si chromadb no esta disponible
# ---------------------------------------------------------------------------
try:
    import chromadb as _chroma_check  # noqa: F401
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

if not CHROMA_AVAILABLE:
    if __name__ == "__main__":
        print("SKIP: chromadb no instalado. Ejecuta: pip install chromadb")
        sys.exit(0)

try:
    import pytest as _pytest
    _pytest.importorskip("chromadb", reason="pip install chromadb")
except ImportError:
    pass  # ejecutando como script, CHROMA_AVAILABLE ya garantiza chromadb

from backend.database.schema import init_db
from backend.embedding.embedder import Embedder
from backend.embedding import chroma_store
from backend.embedding.chroma_store import close_client, close_all_clients
from backend.embedding.pipeline import EmbeddingPipeline
from backend.retrieval.vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------

class FakeEmbedder(Embedder):
    """
    Vectores float32 L2-normalizados deterministas a partir del hash del
    texto. Sin red, sin modelo descargado.
    """

    DIM = 16

    def __init__(self) -> None:
        self.model_name = "fake-model"
        self._model     = None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vecs = []
        for text in texts:
            seed = abs(hash(text)) % (2 ** 31)
            rng  = np.random.default_rng(seed)
            v    = rng.standard_normal(self.DIM).astype(np.float32)
            v   /= np.linalg.norm(v)
            vecs.append(v)
        return np.stack(vecs, axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self.DIM


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    ("2401.001", "Attention Mechanisms in Transformers",
     "Self-attention transformer architecture for natural language processing"),
    ("2401.002", "BERT and Bidirectional Language Models",
     "Pre-training bidirectional transformers for language understanding tasks"),
    ("2401.003", "Gradient Descent Optimization Methods",
     "Stochastic gradient descent with adaptive learning rate Adam optimizer"),
    ("2401.004", "Convolutional Networks for Image Classification",
     "Deep convolutional neural networks residual connections image recognition"),
    ("2401.005", "Reinforcement Learning Policy Gradient",
     "Actor critic proximal policy optimization deep reinforcement learning"),
]

# 2 chunks por documento para probar deduplicacion en VectorRetriever
SAMPLE_CHUNKS = [
    (arxiv_id, i, f"{title} chunk {i}: {text}")
    for arxiv_id, title, text in SAMPLE_DOCS
    for i in range(2)
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_test_db(path: Path) -> None:
    """BD temporal con esquema, documentos y chunks sin embedded_at."""
    init_db(path)
    conn = sqlite3.connect(str(path))
    conn.executemany(
        """
        INSERT INTO documents
            (arxiv_id, title, abstract, authors, full_text,
             text_length, pdf_downloaded, published, pdf_url)
        VALUES (?, ?, '', '', ?, ?, 1, '2024-01-01', '')
        """,
        [(arxiv_id, title, text, len(text))
         for arxiv_id, title, text in SAMPLE_DOCS],
    )
    conn.executemany(
        """
        INSERT INTO chunks (arxiv_id, chunk_index, text, char_count, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(arxiv_id, idx, text, len(text), _now())
         for arxiv_id, idx, text in SAMPLE_CHUNKS],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Suite de tests
# ---------------------------------------------------------------------------

def run_tests(db_path: Path, tmp_dir: Path) -> None:
    """
    Ejecuta todos los tests.

    Parametros
    ----------
    db_path  : BD SQLite de prueba ya creada con create_test_db().
    tmp_dir  : directorio temporal raiz. Todos los subdirectorios de Chroma
               se crean aqui para evitar TemporaryDirectory anidados
               (problema de bloqueo de ficheros en Windows).
    """
    sep      = "─" * 58
    embedder = FakeEmbedder()

    # Directorio Chroma principal para tests 1-5
    chroma_main = tmp_dir / "chroma_main"

    # ── TEST 1: get_collection() crea coleccion vacia ─────────────────────
    print(f"\n{sep}")
    print("  TEST 1: chroma_store -- get_collection() crea coleccion vacia")
    print(sep)

    col = chroma_store.get_collection(chroma_main)

    assert col is not None
    assert col.name == chroma_store.COLLECTION_NAME
    assert col.count() == 0,              "Coleccion nueva debe estar vacia"
    assert chroma_store.count(chroma_main) == 0

    print(f"  OK coleccion '{col.name}' creada | count=0")

    # ── TEST 2: add_chunks() persiste vectores y metadatos ────────────────
    print(f"\n{sep}")
    print("  TEST 2: chroma_store -- add_chunks() persiste vectores y metadatos")
    print(sep)

    chunk_ids   = [1, 2, 3]
    arxiv_ids_t = ["2401.001", "2401.001", "2401.002"]
    chunk_idxs  = [0, 1, 0]
    texts_t     = ["transformer attention text", "language model text", "bert bidirectional text"]
    char_counts = [25, 20, 22]
    vecs        = embedder.encode(texts_t)

    chroma_store.add_chunks(
        chunk_ids   = chunk_ids,
        arxiv_ids   = arxiv_ids_t,
        chunk_idxs  = chunk_idxs,
        texts       = texts_t,
        char_counts = char_counts,
        embeddings  = vecs,
        chroma_path = chroma_main,
    )

    assert chroma_store.count(chroma_main) == 3

    col2   = chroma_store.get_collection(chroma_main)
    result = col2.get(ids=["chunk_1"], include=["metadatas", "documents"])
    assert result["metadatas"][0]["arxiv_id"]    == "2401.001"
    assert result["metadatas"][0]["chunk_index"] == 0
    assert result["metadatas"][0]["char_count"]  == 25
    assert result["documents"][0]                == "transformer attention text"

    print("  OK 3 vectores insertados | metadatos y documentos correctos")

    # ── TEST 3: add_chunks() es idempotente (upsert) ──────────────────────
    print(f"\n{sep}")
    print("  TEST 3: chroma_store -- add_chunks() es idempotente (upsert)")
    print(sep)

    chroma_store.add_chunks(
        chunk_ids=chunk_ids, arxiv_ids=arxiv_ids_t, chunk_idxs=chunk_idxs,
        texts=texts_t, char_counts=char_counts, embeddings=vecs,
        chroma_path=chroma_main,
    )
    assert chroma_store.count(chroma_main) == 3, \
        "Upsert con IDs repetidos no debe cambiar el count"

    print("  OK upsert con IDs repetidos: count sigue siendo 3")

    # ── TEST 4: get_existing_chunk_ids() ──────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 4: chroma_store -- get_existing_chunk_ids() devuelve IDs correctos")
    print(sep)

    existing = chroma_store.get_existing_chunk_ids(chroma_main)
    assert existing == {1, 2, 3}, f"Esperado {{1, 2, 3}}, obtenido {existing}"
    assert all(isinstance(cid, int) for cid in existing), "Los IDs deben ser enteros"

    print(f"  OK existing_ids={existing}")

    # ── TEST 5: query() devuelve hits ordenados por distancia ─────────────
    print(f"\n{sep}")
    print("  TEST 5: chroma_store -- query() devuelve hits ordenados por distancia")
    print(sep)

    q_vec = embedder.encode_one("transformer attention mechanism")
    hits  = chroma_store.query(q_vec, n_results=3, chroma_path=chroma_main)

    assert len(hits) == 3
    for h in hits:
        assert "chunk_id"  in h and isinstance(h["chunk_id"], int)
        assert "arxiv_id"  in h and isinstance(h["arxiv_id"], str)
        assert "text"      in h
        assert "distance"  in h and isinstance(h["distance"], float)
        assert 0.0 <= h["distance"] <= 2.0, \
            f"Distancia coseno debe estar en [0, 2], obtenida {h['distance']}"

    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances), "Hits deben estar ordenados por distancia"

    print(f"  OK {len(hits)} hits ordenados ascendentemente:")
    for h in hits:
        print(f"    chunk_{h['chunk_id']}  dist={h['distance']:.4f}  '{h['text']}'")

    # ── TEST 6: query() en coleccion vacia devuelve [] ────────────────────
    # Usa un subdirectorio del tmp_dir raiz — sin TemporaryDirectory anidado
    # para evitar WinError 32 al intentar borrar ficheros abiertos por Chroma.
    print(f"\n{sep}")
    print("  TEST 6: chroma_store -- query() en coleccion vacia devuelve []")
    print(sep)

    chroma_empty = tmp_dir / "chroma_empty_t6"
    hits_empty   = chroma_store.query(q_vec, n_results=5, chroma_path=chroma_empty)
    assert hits_empty == [], f"Coleccion vacia debe devolver [], devolvio {hits_empty}"

    print("  OK query en coleccion vacia devuelve []")

    # ── TEST 7: EmbeddingPipeline procesa chunks pendientes ───────────────
    # Subdirectorio propio para aislar este grupo de tests (7-9)
    print(f"\n{sep}")
    print("  TEST 7: EmbeddingPipeline -- procesa todos los chunks pendientes")
    print(sep)

    chroma_pipe = tmp_dir / "chroma_pipe_t7"
    pipeline    = EmbeddingPipeline(
        db_path     = db_path,
        chroma_path = chroma_pipe,
        embedder    = embedder,
        batch_size  = 4,
    )

    pending_before = pipeline.pending_count()
    assert pending_before == len(SAMPLE_CHUNKS), \
        f"Antes: {len(SAMPLE_CHUNKS)} esperados, obtenidos {pending_before}"

    stats = pipeline.run()

    assert stats["chunks_embedded"] == len(SAMPLE_CHUNKS)
    assert "started_at"  in stats
    assert "finished_at" in stats
    assert pipeline.pending_count() == 0, "No deben quedar pendientes tras run()"
    assert chroma_store.count(chroma_pipe) == len(SAMPLE_CHUNKS)

    print(f"  OK chunks_embedded={stats['chunks_embedded']}  "
          f"pendientes_antes={pending_before}  pendientes_despues=0")

    # ── TEST 8: EmbeddingPipeline es incremental ──────────────────────────
    print(f"\n{sep}")
    print("  TEST 8: EmbeddingPipeline -- es incremental (no reprocesa)")
    print(sep)

    stats2 = pipeline.run()
    assert stats2["chunks_embedded"] == 0, \
        f"Segunda pasada debe embeber 0, embebia {stats2['chunks_embedded']}"
    assert chroma_store.count(chroma_pipe) == len(SAMPLE_CHUNKS)

    print("  OK segunda ejecucion: chunks_embedded=0")

    # ── TEST 9: EmbeddingPipeline detecta chunks nuevos ───────────────────
    print(f"\n{sep}")
    print("  TEST 9: EmbeddingPipeline -- detecta y procesa chunks nuevos")
    print(sep)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO chunks (arxiv_id, chunk_index, text, char_count, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2401.001", 99, "new chunk added after first run", 32, _now()),
    )
    conn.commit()
    conn.close()

    assert pipeline.pending_count() == 1, "Debe haber 1 chunk pendiente"

    stats3 = pipeline.run()
    assert stats3["chunks_embedded"] == 1
    assert pipeline.pending_count() == 0

    # Limpiar chunk extra
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM chunks WHERE chunk_index = 99")
    conn.commit()
    conn.close()

    # Resetear embedded_at del chunk 99 si quedo en SQLite
    print("  OK 1 chunk nuevo detectado y embebido correctamente")

    # ── TEST 10: VectorRetriever.load() con Chroma vacia ──────────────────
    print(f"\n{sep}")
    print("  TEST 10: VectorRetriever -- is_ready() False con Chroma vacia")
    print(sep)

    chroma_empty10 = tmp_dir / "chroma_empty_t10"
    ret_empty      = VectorRetriever(embedder=embedder)
    ret_empty.load(db_path=db_path, chroma_path=chroma_empty10)

    assert not ret_empty.is_ready(), \
        "is_ready() debe ser False cuando ChromaDB esta vacia"

    print("  OK is_ready()=False con Chroma vacia")

    # ── TEST 11: VectorRetriever.load() con datos ─────────────────────────
    print(f"\n{sep}")
    print("  TEST 11: VectorRetriever -- load() correcto con ChromaDB poblada")
    print(sep)

    retriever = VectorRetriever(embedder=embedder)
    retriever.load(db_path=db_path, chroma_path=chroma_pipe)

    assert retriever.is_ready()
    assert len(retriever._meta) == len(SAMPLE_DOCS), \
        f"Esperados {len(SAMPLE_DOCS)} docs en meta, cargados {len(retriever._meta)}"

    print(f"  OK is_ready()=True | meta={len(retriever._meta)} docs")

    # ── TEST 12: RuntimeError antes de load() ─────────────────────────────
    print(f"\n{sep}")
    print("  TEST 12: VectorRetriever -- RuntimeError si retrieve() sin load()")
    print(sep)

    ret_unloaded = VectorRetriever(embedder=embedder)
    assert not ret_unloaded.is_ready()

    error_raised = False
    try:
        ret_unloaded.retrieve("any query")
    except RuntimeError as exc:
        error_raised = True
        assert "load()" in str(exc)

    assert error_raised, "Debia lanzar RuntimeError"
    print("  OK RuntimeError lanzado correctamente antes de load()")

    # ── TEST 13: retrieve() deduplica y ordena por score ──────────────────
    print(f"\n{sep}")
    print("  TEST 13: VectorRetriever -- retrieve() deduplica y ordena")
    print(sep)

    results = retriever.retrieve("transformer attention language model", top_n=5)

    assert len(results) == len(SAMPLE_DOCS), \
        f"Deduplicacion: {len(SAMPLE_DOCS)} esperados, {len(results)} obtenidos"

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Orden descendente por score"

    arxiv_ids_res = [r["arxiv_id"] for r in results]
    assert len(arxiv_ids_res) == len(set(arxiv_ids_res)), \
        "Cada arxiv_id debe aparecer una sola vez"

    for r in results:
        assert isinstance(r["score"],    float)
        assert isinstance(r["arxiv_id"], str)
        assert "title"    in r
        assert "abstract" in r
        assert -1.0 <= r["score"] <= 1.0 + 1e-5, \
            f"score = 1 - distancia_coseno, debe estar en [-1, 1]: {r['score']}"

    print(f"  OK {len(results)} resultados unicos, ordenados:")
    for r in results:
        print(f"    [{r['arxiv_id']}]  score={r['score']:.4f}  {r['title']}")

    # ── TEST 14: top_n respetado ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 14: VectorRetriever -- top_n respetado")
    print(sep)

    results2 = retriever.retrieve("gradient optimization", top_n=2)
    assert len(results2) == 2
    assert results2[0]["score"] >= results2[1]["score"]

    print("  OK top_n=2 devuelve exactamente 2 resultados")

    # ── TEST 15: Integracion end-to-end con BD fresca ─────────────────────
    # Usa subdirectorios del tmp_dir raiz en lugar de TemporaryDirectory anidado
    print(f"\n{sep}")
    print("  TEST 15: Integracion end-to-end -- BD fresca -> pipeline -> retriever")
    print(sep)

    db2     = tmp_dir / "db_e2e.db"
    chroma2 = tmp_dir / "chroma_e2e"
    create_test_db(db2)

    pipe2  = EmbeddingPipeline(
        db_path=db2, chroma_path=chroma2, embedder=embedder, batch_size=10,
    )
    stats4 = pipe2.run()
    assert stats4["chunks_embedded"] == len(SAMPLE_CHUNKS)

    ret2 = VectorRetriever(embedder=embedder)
    ret2.load(db_path=db2, chroma_path=chroma2)
    assert ret2.is_ready()

    res = ret2.retrieve("convolutional image classification", top_n=3)
    assert len(res) == 3
    assert res[0]["score"] >= res[1]["score"] >= res[2]["score"]

    print(f"  OK {stats4['chunks_embedded']} chunks -> ChromaDB -> "
          f"top-3 resultados correctos")

    print(f"\n{'='*58}")
    print("  TODOS los tests de embedding (ChromaDB) completados.")
    print(f"{'='*58}\n")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def main() -> None:
    # ignore_cleanup_errors=True: en Windows ChromaDB puede dejar ficheros
    # bloqueados; el directorio se limpiara cuando el SO lo permita.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "test.db"
        print(f"SQLite:   {db_path}")
        print(f"Tmp dir:  {tmp_dir}")
        create_test_db(db_path)
        try:
            run_tests(db_path, tmp_dir)
        finally:
            close_all_clients()  # liberar ficheros antes de intentar limpiar


def test_all(tmp_path: Path) -> None:
    """Punto de entrada para pytest (tmp_path es gestionado por pytest)."""
    db_path = tmp_path / "test.db"
    create_test_db(db_path)
    try:
        run_tests(db_path, tmp_path)
    finally:
        close_all_clients()


if __name__ == "__main__":
    main()