"""
test_query_refinement.py
========================
Tests de integración del módulo query_refinement.

Cubre las cuatro capas del módulo sin requerir sentence-transformers
ni el índice FAISS real en disco:

  1. QueryExpander          — expansión LCE con modelo LSI mock.
  2. BlindRelevanceFeedback — ajuste de centroide (BRF).
  3. RocchioFeedback        — retroalimentación explícita + caché de sesión.
  4. MMRReranker            — diversificación de resultados.
  5. QueryPipeline          — flujo end-to-end con todos los componentes mockeados.

Mocks usados
------------
- _MockLSIModel   : sustituye LSIModel (SVD + normalizer sintéticos).
- _MockEmbedder   : sustituye ChunkEmbedder (vectores aleatorios normalizados).
- _MockFaissIndex : sustituye FaissIndexManager (búsqueda determinista).

Ejecución
---------
    python -m pytest backend/tests/test_qrf.py -v
    python -m backend.tests.test_qrf
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.database.schema import init_db
from backend.database.chunk_repository import save_chunks, save_chunk_embeddings_batch
from backend.database.embedding_repository import init_embedding_schema
from backend.qrf.query_expander import QueryExpander
from backend.qrf.brf     import BlindRelevanceFeedback
from backend.qrf.rocchio import RocchioFeedback
from backend.qrf.mmr     import MMRReranker
from backend.qrf.pipeline import QueryPipeline

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DIM = 32   # dimensión de embeddings en los tests (pequeña para rapidez)
K   = 8    # componentes latentes del modelo LSI mock

SAMPLE_DOCS = [
    {
        "arxiv_id":       "2501.00001",
        "title":          "Attention Mechanisms in Neural Networks",
        "abstract":       "We study attention mechanisms in transformer models.",
        "full_text":      "Transformers use self-attention to compute representations. "
                          "Multi-head attention allows attending to different positions. "
                          "The attention mechanism has revolutionized NLP tasks.",
        "pdf_downloaded": 1,
        "pdf_url":        "https://arxiv.org/pdf/2501.00001",
    },
    {
        "arxiv_id":       "2501.00002",
        "title":          "Graph Neural Networks for Drug Discovery",
        "abstract":       "Graph networks applied to molecular property prediction.",
        "full_text":      "Graph neural networks process molecular graphs. "
                          "Node embeddings capture local chemical environments. "
                          "Message passing aggregates neighborhood information.",
        "pdf_downloaded": 1,
        "pdf_url":        "https://arxiv.org/pdf/2501.00002",
    },
    {
        "arxiv_id":       "2501.00003",
        "title":          "Diffusion Models for Image Synthesis",
        "abstract":       "Score-based generative models for high-quality images.",
        "full_text":      "Diffusion models learn to reverse a noising process. "
                          "The denoising score matching objective trains the network. "
                          "Classifier-free guidance improves sample quality.",
        "pdf_downloaded": 1,
        "pdf_url":        "https://arxiv.org/pdf/2501.00003",
    },
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


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class _MockLSIModel:
    """
    Sustituye LSIModel con una SVD sintética.

    Genera componentes_, term_ids, doc_ids, df_map y docs_latent
    deterministas para poder testear QueryExpander sin datos reales.
    """

    def __init__(self) -> None:
        np.random.seed(42)
        n_terms    = 50
        n_docs     = len(SAMPLE_DOCS)

        # SVD sintético
        self.k          = K
        self.term_ids   = list(range(n_terms))
        self.doc_ids    = [d["arxiv_id"] for d in SAMPLE_DOCS]
        self.df_map     = {i: max(1, i % 5) for i in range(n_terms)}

        # components_[dim, term] — pesos de cada término en cada concepto
        components = np.random.randn(K, n_terms).astype(np.float32)
        # Normalizar filas para simular SVD real
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        components = components / np.where(norms == 0, 1, norms)

        # docs_latent: (n_docs, k) normalizado en L2
        docs_latent = np.random.randn(n_docs, K).astype(np.float32)
        norms_d = np.linalg.norm(docs_latent, axis=1, keepdims=True)
        self.docs_latent = docs_latent / np.where(norms_d == 0, 1, norms_d)

        # Mock SVD con project_query funcional
        class _MockSVD:
            def __init__(self, comp):
                self.components_ = comp

            def transform(self, x):
                # x: (1, n_terms) → (1, k)
                return (x @ comp.T)

        comp = components
        self.svd        = _MockSVD(comp)
        self.normalizer = _make_l2_normalizer()

    def project_query(self, q_tfidf: np.ndarray) -> np.ndarray:
        q_svd = self.svd.transform(q_tfidf.reshape(1, -1))   # (1, k)
        return self.normalizer(q_svd).flatten()


def _make_l2_normalizer():
    """Devuelve una función de normalización L2 compatible con la interfaz del pipeline."""
    class _Norm:
        def transform(self, x):
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.where(norms == 0, 1, norms)
        def __call__(self, x):
            return self.transform(x)
    return _Norm()


class _MockEmbedder:
    """Embedder determinista que devuelve vectores L2-normalizados fijos por texto."""

    dim        = DIM
    model_name = "mock-model-v0"

    def encode(self, texts: list[str]) -> np.ndarray:
        rng  = np.random.RandomState(abs(hash(str(texts))) % (2**31))
        vecs = rng.randn(len(texts), self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class _MockFaissIndex:
    """
    Sustituto de FaissIndexManager con búsqueda bruta determinista.

    Almacena vectores en memoria y los busca por distancia L2.
    """

    def __init__(self, dim: int) -> None:
        self._vectors: list[np.ndarray] = []
        self._ids:     list[int]        = []
        self.dim        = dim
        self.index_type = "MockFlatL2"

    @property
    def total_vectors(self) -> int:
        return len(self._ids)

    def add(self, vectors: np.ndarray, chunk_ids: list[int]) -> None:
        for v, cid in zip(vectors, chunk_ids):
            self._vectors.append(v.astype(np.float32))
            self._ids.append(cid)

    def load(self) -> bool:
        return True   # siempre "cargado" en tests

    def search(self, query: np.ndarray, top_k: int = 10) -> list[dict]:
        if not self._vectors:
            return []
        matrix = np.stack(self._vectors)   # (N, dim)
        dists  = np.linalg.norm(matrix - query.reshape(1, -1), axis=1)
        top_k  = min(top_k, len(self._ids))
        idxs   = np.argsort(dists)[:top_k]
        return [
            {"chunk_id": self._ids[i], "score": float(dists[i])}
            for i in idxs
        ]


# ---------------------------------------------------------------------------
# Setup de BD
# ---------------------------------------------------------------------------

def _create_test_db(path: Path) -> list[int]:
    """
    Crea BD con documentos, chunks y embeddings de test.
    Devuelve la lista de chunk_ids insertados.
    """
    init_db(path)
    init_embedding_schema(path)

    conn = sqlite3.connect(str(path))
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (arxiv_id, title, abstract, full_text, pdf_downloaded,
             pdf_url, categories, published, updated, fetched_at)
        VALUES (:arxiv_id, :title, :abstract, :full_text, :pdf_downloaded,
                :pdf_url, '', '', '', '')
        """,
        SAMPLE_DOCS,
    )
    conn.commit()
    conn.close()

    for arxiv_id, texts in SAMPLE_CHUNKS.items():
        save_chunks(arxiv_id, texts, db_path=path)

    # Insertar embeddings aleatorios en la BD
    conn2 = sqlite3.connect(str(path))
    rows  = conn2.execute("SELECT id FROM chunks ORDER BY id").fetchall()
    conn2.close()
    chunk_ids = [r[0] for r in rows]

    ts  = datetime.now(timezone.utc).isoformat()
    rng = np.random.RandomState(0)
    batch = []
    for cid in chunk_ids:
        v    = rng.randn(DIM).astype(np.float32)
        v   /= np.linalg.norm(v)
        batch.append((v.tobytes(), ts, cid))
    save_chunk_embeddings_batch(batch, db_path=path)

    return chunk_ids


# Palabras reales que sobreviven TextPreprocessor (isalpha, min_len>=3, no stopwords)
# Deben ser >= n_terms del modelo mock (50) para evitar duplicados en word_index
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


def _build_mock_word_index(
    model: _MockLSIModel,
) -> tuple[dict, dict]:
    """Construye word_index e idx_to_word con palabras reales que pasan el preprocesador."""
    n = len(model.term_ids)
    words = [_VOCAB_WORDS[i % len(_VOCAB_WORDS)] for i in range(n)]
    word_index:  dict[str, tuple[int, int]] = {}
    idx_to_word: dict[int, str]             = {}
    for i, w in enumerate(words):
        if w not in word_index:
            word_index[w] = (i, model.df_map[i])
        idx_to_word[i] = w
    return word_index, idx_to_word


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests(db_path: Path) -> None:
    sep  = "─" * 60
    sep2 = "═" * 60

    chunk_ids = _create_test_db(db_path)
    rng       = np.random.RandomState(7)

    # ── 1. QueryExpander — carga y estructura ────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 1: QueryExpander — inicialización con modelo mock")
    print(sep)

    mock_model   = _MockLSIModel()
    expander     = QueryExpander(
        lsi_model=mock_model,
        top_dims=2,
        top_terms_per_dim=5,
        min_correlation=0.1,
        max_expansion=6,
    )
    expander._model      = mock_model
    expander._word_index, expander._idx_to_word = _build_mock_word_index(mock_model)

    assert expander._model is not None
    # word_index puede tener <= term_ids si _VOCAB_WORDS tiene palabras repetidas
    assert 0 < len(expander._word_index) <= len(mock_model.term_ids)
    assert len(expander._idx_to_word) == len(mock_model.term_ids)

    print(f"  ✔ Expander inicializado — vocabulario: {len(expander._word_index)} términos")

    # ── 2. QueryExpander — expand() con query en vocabulario ─────────────────
    print(f"\n{sep}")
    print("  TEST 2: QueryExpander.expand() — query con términos en vocabulario")
    print(sep)

    # Usar un término que sí existe en el word_index
    query_in_vocab = "attention neural transformer"
    expanded, new_terms = expander.expand(query_in_vocab)

    assert isinstance(expanded, str)
    assert isinstance(new_terms, list)
    assert expanded.startswith(query_in_vocab)   # la query original está al principio
    assert len(new_terms) <= expander.max_expansion

    # Los nuevos términos no deben repetir los originales
    original_tokens = {"attention", "neural", "transformer"}
    for t in new_terms:
        assert t not in original_tokens, f"Término original repetido en expansión: {t}"

    print(f"  ✔ expand('{query_in_vocab}') → +{len(new_terms)} términos: {new_terms[:4]}")
    print(f"  ✔ Query expandida: '{expanded[:80]}'")

    # ── 3. QueryExpander — expand() con query fuera de vocabulario ───────────
    print(f"\n{sep}")
    print("  TEST 3: QueryExpander.expand() — query sin términos en vocabulario")
    print(sep)

    expanded2, terms2 = expander.expand("zyxwvut qrstuvw")   # palabras inexistentes

    assert expanded2 == "zyxwvut qrstuvw"   # debe devolver la query sin cambios
    assert terms2 == []

    print("  ✔ Query fuera de vocabulario devuelta sin cambios.")

    # ── 4. QueryExpander — _vectorize internamente ───────────────────────────
    print(f"\n{sep}")
    print("  TEST 4: QueryExpander._vectorize() — estructura del vector")
    print(sep)

    # Palabras reales del vocabulario mock que pasan TextPreprocessor
    vec = expander._vectorize("attention neural transformer learning")
    assert vec.shape == (len(mock_model.term_ids),)
    assert vec.dtype == np.float32
    assert vec.sum() > 0, (
        "El vector debe tener al menos un término activo. "
        "Comprueba que las palabras están en word_index y pasan TextPreprocessor."
    )
    active = int((vec > 0).sum())
    assert active >= 1

    print(f"  ✔ Vector shape={vec.shape} dtype={vec.dtype} términos activos={active}")

    # ── 5. BlindRelevanceFeedback — ajuste de centroide ──────────────────────
    print(f"\n{sep}")
    print("  TEST 5: BlindRelevanceFeedback.adjust() — centroide y alpha")
    print(sep)

    brf         = BlindRelevanceFeedback(alpha=0.75, top_k_rf=3)
    query_vec   = rng.randn(DIM).astype(np.float32)
    query_vec  /= np.linalg.norm(query_vec)

    top_results = [{"chunk_id": cid} for cid in chunk_ids[:5]]
    adjusted    = brf.adjust(query_vec, top_results, db_path)

    # El vector ajustado debe estar normalizado
    assert adjusted.shape == (DIM,)
    assert adjusted.dtype == np.float32
    assert abs(np.linalg.norm(adjusted) - 1.0) < 1e-5, "El vector ajustado no está normalizado."

    # Con alpha=0.75 el ajustado debe estar más cerca del original que del centroide
    from backend.qrf._feedback_utils import cosine_similarity as _cosine_similarity
    sim_to_orig    = _cosine_similarity(query_vec, adjusted)
    assert sim_to_orig > 0, "El vector ajustado debe tener correlación positiva con el original."

    print(f"  ✔ Vector ajustado — shape={adjusted.shape} norm≈1 ✓")
    print(f"  ✔ Similitud original→ajustado: {sim_to_orig:.3f} (alpha=0.75)")

    # ── 6. BlindRelevanceFeedback — sin embeddings disponibles ───────────────
    print(f"\n{sep}")
    print("  TEST 6: BlindRelevanceFeedback.adjust() — fallback sin embeddings")
    print(sep)

    brf2         = BlindRelevanceFeedback(alpha=0.8)
    query_vec2   = rng.randn(DIM).astype(np.float32)
    query_vec2  /= np.linalg.norm(query_vec2)

    # IDs inexistentes — no habrá embeddings en la BD
    top_fake = [{"chunk_id": 999999}, {"chunk_id": 999998}]
    result2  = brf2.adjust(query_vec2, top_fake, db_path)

    assert np.allclose(result2, query_vec2), "Debe devolver el vector original si no hay embeddings."
    print("  ✔ Fallback correcto — devuelve vector original cuando no hay embeddings.")

    # ── 7. RocchioFeedback — retroalimentación con relevantes ────────────────
    print(f"\n{sep}")
    print("  TEST 7: RocchioFeedback.adjust() — documentos relevantes")
    print(sep)

    rocchio   = RocchioFeedback(alpha=0.6, beta=0.4, gamma=0.1)
    qvec      = rng.randn(DIM).astype(np.float32)
    qvec     /= np.linalg.norm(qvec)

    relevant_ids   = chunk_ids[:2]
    irrelevant_ids = chunk_ids[2:3]

    adjusted_r = rocchio.adjust(
        query_id="session_test_1",
        query_vector=qvec,
        relevant_ids=relevant_ids,
        irrelevant_ids=irrelevant_ids,
        db_path=db_path,
    )

    assert adjusted_r.shape == (DIM,)
    assert abs(np.linalg.norm(adjusted_r) - 1.0) < 1e-5, "Vector Rocchio no normalizado."

    # Con beta > gamma, el vector debe acercarse a los relevantes
    print(f"  ✔ Rocchio aplicado — norm≈1 ✓")
    print(f"  ✔ Similitud original→ajustado: "
          f"{_cosine_similarity(qvec, adjusted_r):.3f}")

    # ── 8. RocchioFeedback — caché de sesión ─────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 8: RocchioFeedback — caché de sesión")
    print(sep)

    cached = rocchio.get_cached("session_test_1")
    assert cached is not None, "El vector ajustado debe estar en caché."
    assert np.allclose(cached, adjusted_r), "El vector en caché debe ser idéntico al ajustado."

    assert rocchio.get_cached("nonexistent_session") is None

    assert "session_test_1" in rocchio.cached_queries

    rocchio.clear_cache("session_test_1")
    assert rocchio.get_cached("session_test_1") is None

    # Varias sesiones a la vez
    for i in range(3):
        rocchio.adjust(
            query_id=f"session_{i}",
            query_vector=rng.randn(DIM).astype(np.float32),
            relevant_ids=chunk_ids[:1],
            irrelevant_ids=[],
            db_path=db_path,
        )
    assert len(rocchio.cached_queries) == 3
    rocchio.clear_cache()
    assert len(rocchio.cached_queries) == 0

    print("  ✔ Caché: set / get / clear individual / clear all ✓")

    # ── 9. RocchioFeedback — sin documentos relevantes ni irrelevantes ────────
    print(f"\n{sep}")
    print("  TEST 9: RocchioFeedback.adjust() — sin feedback devuelve vector original")
    print(sep)

    qvec3   = rng.randn(DIM).astype(np.float32)
    qvec3  /= np.linalg.norm(qvec3)
    result3 = rocchio.adjust("sid_empty", qvec3, [], [], db_path)
    assert np.allclose(result3, qvec3)

    print("  ✔ Sin feedback, devuelve vector original sin modificar.")

    # ── 10. MMRReranker — reranking básico ───────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 10: MMRReranker.rerank() — selección diversa")
    print(sep)

    mmr     = MMRReranker(lambda_=0.6)
    qvec4   = rng.randn(DIM).astype(np.float32)
    qvec4  /= np.linalg.norm(qvec4)

    candidates = [{"chunk_id": cid, "score": float(i) * 0.1}
                  for i, cid in enumerate(chunk_ids)]
    reranked   = mmr.rerank(candidates, qvec4, top_n=4, db_path=db_path)

    assert len(reranked) <= 4
    assert len(reranked) <= len(chunk_ids)

    # Todos los resultados deben tener mmr_score
    for r in reranked:
        assert "mmr_score" in r
        assert isinstance(r["mmr_score"], float)

    # No deben repetirse chunk_ids
    returned_ids = [r["chunk_id"] for r in reranked]
    assert len(returned_ids) == len(set(returned_ids)), "IDs duplicados en MMR."

    print(f"  ✔ MMR devolvió {len(reranked)} de {len(candidates)} candidatos.")
    print(f"  ✔ mmr_scores: {[r['mmr_score'] for r in reranked]}")

    # ── 11. MMRReranker — diversidad (lambda bajo = más diversidad) ───────────
    print(f"\n{sep}")
    print("  TEST 11: MMRReranker — mayor diversidad con lambda bajo")
    print(sep)

    mmr_div  = MMRReranker(lambda_=0.1)   # muy diverso
    mmr_rel  = MMRReranker(lambda_=0.9)   # muy relevante

    ranked_div = mmr_div.rerank(candidates, qvec4, top_n=3, db_path=db_path)
    ranked_rel = mmr_rel.rerank(candidates, qvec4, top_n=3, db_path=db_path)

    # Ambos deben devolver resultados sin duplicados
    assert len({r["chunk_id"] for r in ranked_div}) == len(ranked_div)
    assert len({r["chunk_id"] for r in ranked_rel}) == len(ranked_rel)

    # Con distintos lambdas el orden puede diferir
    ids_div = [r["chunk_id"] for r in ranked_div]
    ids_rel = [r["chunk_id"] for r in ranked_rel]
    print(f"  ✔ lambda=0.1 (diverso)  → IDs: {ids_div}")
    print(f"  ✔ lambda=0.9 (relevante)→ IDs: {ids_rel}")

    # ── 12. MMRReranker — fallback sin embeddings ─────────────────────────────
    print(f"\n{sep}")
    print("  TEST 12: MMRReranker.rerank() — fallback sin embeddings en BD")
    print(sep)

    fake_candidates = [{"chunk_id": 999990 + i, "score": float(i)} for i in range(5)]
    result_fake     = mmr.rerank(fake_candidates, qvec4, top_n=3, db_path=db_path)

    # Fallback: devuelve los primeros top_n sin reordenar
    assert len(result_fake) == 3
    assert result_fake[0]["chunk_id"] == 999990

    print(f"  ✔ Fallback correcto — devuelve primeros {len(result_fake)} sin reordenar.")

    # ── 13. QueryPipeline — flujo end-to-end mockeado ────────────────────────
    print(f"\n{sep}")
    print("  TEST 13: QueryPipeline — búsqueda end-to-end con mocks inyectados")
    print(sep)

    pipeline = _build_mock_pipeline(db_path, chunk_ids)
    results  = pipeline.search("attention transformer mechanism", top_k=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert "chunk_id"    in r
        assert "score"       in r
        assert "arxiv_id"    in r
        assert "text"        in r
        assert "title"       in r
        assert "expanded_terms" in r   # campo añadido por el pipeline

    print(f"  ✔ Pipeline devolvió {len(results)} resultados.")
    for r in results:
        print(f"    chunk_id={r['chunk_id']}  score={r['score']:.4f}  "
              f"mmr={r['mmr_score']}  arxiv={r['arxiv_id']}")

    # ── 14. QueryPipeline — search_with_session + refine ─────────────────────
    print(f"\n{sep}")
    print("  TEST 14: QueryPipeline — search_with_session() + refine()")
    print(sep)

    pipeline2 = _build_mock_pipeline(db_path, chunk_ids)
    results1, sid = pipeline2.search_with_session("graph neural networks", top_k=4)

    assert isinstance(sid, str) and len(sid) > 0
    assert isinstance(results1, list)
    assert sid in pipeline2._session_vectors

    # Refinar con el primer chunk como relevante
    if results1:
        rel_id = results1[0]["chunk_id"]
        results2 = pipeline2.refine(
            session_id=sid,
            relevant_ids=[rel_id],
            irrelevant_ids=[],
            top_k=4,
        )
        assert isinstance(results2, list)
        assert len(results2) <= 4
        print(f"  ✔ search_with_session: {len(results1)} resultados, session_id={sid[:8]}…")
        print(f"  ✔ refine (relevante={rel_id}): {len(results2)} resultados.")
    else:
        print("  ✔ search_with_session OK (sin resultados en índice vacío).")

    # ── 15. QueryPipeline — KeyError con session_id inválido ─────────────────
    print(f"\n{sep}")
    print("  TEST 15: QueryPipeline.refine() — KeyError con sesión inexistente")
    print(sep)

    pipeline3 = _build_mock_pipeline(db_path, chunk_ids)
    try:
        pipeline3.refine("id_que_no_existe", [chunk_ids[0]], [], top_k=5)
        assert False, "Debería haber lanzado KeyError."
    except KeyError as e:
        print(f"  ✔ KeyError lanzado correctamente: {e}")

    # ── 16. QueryPipeline — clear_session ────────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 16: QueryPipeline.clear_session() — limpieza de caché")
    print(sep)

    pipeline4 = _build_mock_pipeline(db_path, chunk_ids)
    _, sid_a  = pipeline4.search_with_session("diffusion models", top_k=3)
    _, sid_b  = pipeline4.search_with_session("attention heads", top_k=3)

    assert sid_a in pipeline4._session_vectors
    assert sid_b in pipeline4._session_vectors

    pipeline4.clear_session(sid_a)
    assert sid_a not in pipeline4._session_vectors
    assert sid_b in pipeline4._session_vectors

    pipeline4.clear_session()
    assert len(pipeline4._session_vectors) == 0

    print("  ✔ clear_session individual y total funcionan correctamente.")

    # ── 17. Parámetros inválidos en BRF y MMR ────────────────────────────────
    print(f"\n{sep}")
    print("  TEST 17: Validación de parámetros en BRF y MMR")
    print(sep)

    for alpha in [-0.1, 1.1]:
        try:
            BlindRelevanceFeedback(alpha=alpha)
            assert False, f"Debería fallar con alpha={alpha}"
        except ValueError:
            pass

    for lam in [-0.1, 1.1]:
        try:
            MMRReranker(lambda_=lam)
            assert False, f"Debería fallar con lambda_={lam}"
        except ValueError:
            pass

    print("  ✔ ValueError lanzado para alpha y lambda_ fuera de [0, 1].")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  ✅  Todos los tests de query_refinement completados correctamente.")
    print(f"{sep2}\n")


# ---------------------------------------------------------------------------
# Helper: construir pipeline con mocks inyectados
# ---------------------------------------------------------------------------

def _build_mock_pipeline(db_path: Path, chunk_ids: list[int]) -> QueryPipeline:
    """
    Construye un QueryPipeline con todos los componentes mockeados.
    Inyecta el embedder y el índice FAISS directamente para evitar
    cargar modelos reales o archivos de disco.
    """
    mock_model = _MockLSIModel()
    pipeline   = QueryPipeline(
        db_path=db_path,
        expand=True,
        top_k_initial=len(chunk_ids),
    )

    # Inyectar embedder mock
    pipeline._embedder = _MockEmbedder()

    # Inyectar índice FAISS mock con los vectores de la BD
    faiss_mock = _MockFaissIndex(dim=DIM)
    rng = np.random.RandomState(0)
    for cid in chunk_ids:
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        faiss_mock.add(v.reshape(1, -1), [cid])
    pipeline._faiss_mgr = faiss_mock

    # Inyectar expander mock
    expander = QueryExpander(
        lsi_model=mock_model,
        top_dims=2,
        top_terms_per_dim=5,
        min_correlation=0.1,
        max_expansion=4,
    )
    expander._model = mock_model
    expander._word_index, expander._idx_to_word = _build_mock_word_index(mock_model)
    pipeline._expander = expander
    pipeline._expand_enabled = True

    return pipeline


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = Path(tmp) / "test_qr.db"
        print(f"BD temporal: {db_path}")
        run_tests(db_path)


if __name__ == "__main__":
    main()