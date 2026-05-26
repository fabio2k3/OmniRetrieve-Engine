"""Mocks y fixtures para los tests del módulo query refinement (qrf)."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

DIM = 32
K = 8

SAMPLE_DOCS = [
    {"arxiv_id": "2501.00001", "title": "Attention Mechanisms in Neural Networks",
     "abstract": "We study attention mechanisms in transformer models.",
     "full_text": "Transformers use self-attention to compute representations. "
                  "Multi-head attention allows attending to different positions. "
                  "The attention mechanism has revolutionized NLP tasks.",
     "pdf_downloaded": 1, "pdf_url": "https://arxiv.org/pdf/2501.00001"},
    {"arxiv_id": "2501.00002", "title": "Graph Neural Networks for Drug Discovery",
     "abstract": "Graph networks applied to molecular property prediction.",
     "full_text": "Graph neural networks process molecular graphs. "
                  "Node embeddings capture local chemical environments. "
                  "Message passing aggregates neighborhood information.",
     "pdf_downloaded": 1, "pdf_url": "https://arxiv.org/pdf/2501.00002"},
    {"arxiv_id": "2501.00003", "title": "Diffusion Models for Image Synthesis",
     "abstract": "Score-based generative models for high-quality images.",
     "full_text": "Diffusion models learn to reverse a noising process. "
                  "The denoising score matching objective trains the network. "
                  "Classifier-free guidance improves sample quality.",
     "pdf_downloaded": 1, "pdf_url": "https://arxiv.org/pdf/2501.00003"},
]

SAMPLE_CHUNKS = {
    "2501.00001": [
        "Transformers use self-attention to compute dense representations of text.",
        "Multi-head attention allows the model to attend to different subspaces.",
        "The attention mechanism replaces recurrent computation with parallelism.",
    ],
    "2501.00002": [
        "Graph neural networks aggregate neighbor features via message passing.",
        "Node-level embeddings capture local chemical graph structure.",
    ],
    "2501.00003": [
        "Diffusion models iteratively denoise a Gaussian-corrupted signal.",
        "Score matching trains a neural network to estimate the score function.",
    ],
}

_VOCAB_WORDS = [
    "attention", "neural", "learning", "transformer", "embedding",
    "graph", "network", "training", "model", "language",
    "diffusion", "latent", "retrieval", "semantic", "vector",
    "corpus", "query", "token", "layer", "gradient",
    "encoder", "decoder", "softmax", "kernel", "matrix",
    "cluster", "ranking", "document", "score", "feature",
    "sparse", "dense", "cosine", "distance", "similarity",
    "convolution", "pooling", "dropout", "batch", "epoch",
    "dataset", "benchmark", "evaluation", "precision", "recall",
    "probabilistic", "inference", "representation", "objective", "regularization",
]


class MockLSIModel:
    def __init__(self):
        np.random.seed(42)
        n_terms = 50
        self.k = K
        self.term_ids = list(range(n_terms))
        self.doc_ids = [d["arxiv_id"] for d in SAMPLE_DOCS]
        self.df_map = {i: max(1, i % 5) for i in range(n_terms)}

        components = np.random.randn(K, n_terms).astype(np.float32)
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        components = components / np.where(norms == 0, 1, norms)

        docs_latent = np.random.randn(len(SAMPLE_DOCS), K).astype(np.float32)
        norms_d = np.linalg.norm(docs_latent, axis=1, keepdims=True)
        self.docs_latent = docs_latent / np.where(norms_d == 0, 1, norms_d)

        # FIX: use self.components_ inside transform, not a closure variable
        _comp = components

        class _MockSVD:
            def __init__(self, c):
                self.components_ = c

            def transform(self, x):
                return x @ self.components_.T   # uses self, not closure

        self.svd = _MockSVD(_comp)
        self.normalizer = _L2Normalizer()

    def project_query(self, q_tfidf):
        q_svd = self.svd.transform(q_tfidf.reshape(1, -1))
        return self.normalizer(q_svd).flatten()


class _L2Normalizer:
    def transform(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms == 0, 1, norms)
    def __call__(self, x): return self.transform(x)


class MockEmbedder:
    dim = DIM
    model_name = "mock-model-v0"
    def encode(self, texts):
        rng = np.random.RandomState(abs(hash(str(texts))) % (2**31))
        vecs = rng.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)
    def encode_single(self, text): return self.encode([text])[0]


class MockFaissIndex:
    def __init__(self, dim=DIM):
        self._vectors, self._ids = [], []
        self.dim = dim
        self.index_type = "MockFlatL2"
    @property
    def total_vectors(self): return len(self._ids)
    def add(self, vectors, chunk_ids):
        for v, cid in zip(vectors, chunk_ids):
            self._vectors.append(v.astype(np.float32))
            self._ids.append(cid)
    def load(self): return True
    def search(self, query, top_k=10):
        if not self._vectors: return []
        matrix = np.stack(self._vectors)
        dists = np.linalg.norm(matrix - query.reshape(1, -1), axis=1)
        top_k = min(top_k, len(self._ids))
        idxs = np.argsort(dists)[:top_k]
        return [{"chunk_id": self._ids[i], "score": float(dists[i])} for i in idxs]


def build_mock_word_index(model):
    n = len(model.term_ids)
    words = [_VOCAB_WORDS[i % len(_VOCAB_WORDS)] for i in range(n)]
    word_index, idx_to_word = {}, {}
    for i, w in enumerate(words):
        if w not in word_index:
            word_index[w] = (i, model.df_map[i])
        idx_to_word[i] = w
    return word_index, idx_to_word


def create_qrf_db(path: Path) -> list:
    from backend.database.schema import init_db
    from backend.database.embedding_repository import init_embedding_schema
    from backend.database.chunk_repository import save_chunks, save_chunk_embeddings_batch

    init_db(path)
    init_embedding_schema(path)

    conn = sqlite3.connect(str(path))
    conn.executemany(
        "INSERT OR IGNORE INTO documents "
        "(arxiv_id, title, abstract, full_text, pdf_downloaded, pdf_url, categories, published, updated, fetched_at) "
        "VALUES (:arxiv_id, :title, :abstract, :full_text, :pdf_downloaded, :pdf_url, '', '', '', '')",
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()

    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        save_chunks(arxiv_id, texts, db_path=path)

    conn2 = sqlite3.connect(str(path))
    rows = conn2.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    conn2.close()
    chunk_ids = [r[0] for r in rows]

    ts = datetime.now(timezone.utc).isoformat()
    rng = np.random.RandomState(0)
    batch = []
    for cid in chunk_ids:
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        batch.append((v.tobytes(), ts, cid))
    save_chunk_embeddings_batch(batch, db_path=path)
    return chunk_ids


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_qrf.db"


@pytest.fixture
def db_with_chunks(db_path):
    chunk_ids = create_qrf_db(db_path)
    return db_path, chunk_ids


@pytest.fixture
def mock_model():
    return MockLSIModel()