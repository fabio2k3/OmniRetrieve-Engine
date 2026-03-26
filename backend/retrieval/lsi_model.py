"""
lsi_model.py
============
Fase offline del módulo LSI.

Lee las frecuencias crudas del índice invertido a través de
index_repository, construye la matriz TF-IDF, aplica SVD truncada
para obtener el espacio semántico latente y persiste el modelo.

Separación de responsabilidades
--------------------------------
  indexing/   → cuenta frecuencias y las guarda en postings.
  retrieval/  → lee esas frecuencias, aplica la fórmula TF-IDF y
                descompone con SVD. El cálculo de pesos vive aquí.

Fórmulas aplicadas
------------------
    TF(t, d)  = log(1 + freq(t, d))          suavizado logarítmico
    IDF(t)    = log((N + 1) / (df(t) + 1))   suavizado Laplace
    W(t, d)   = TF(t, d) × IDF(t)

Referencia: Manning et al. (2008), Cap. 18, sec. 18.1–18.4
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from backend.database.schema import DB_PATH, get_connection
from backend.database.index_repository import get_postings_for_matrix

MODEL_DIR  = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "lsi_model.pkl"

log = logging.getLogger(__name__)


class LSIModel:
    """
    Construye y persiste el espacio semántico latente LSI.

    Atributos persistidos en el .pkl
    ---------------------------------
    svd          : TruncatedSVD ajustado (contiene components_ para proyectar queries).
    normalizer   : Normalizer L2 ajustado.
    docs_latent  : np.ndarray (n_docs × k) — vectores latentes de cada documento.
    doc_ids      : list[str] — arxiv_ids en el mismo orden que las columnas de la matriz.
    term_ids     : list[int] — term_ids en el mismo orden que las filas de la matriz.
    df_map       : dict[int, int] — term_id → df del corpus (para vectorizar queries).
    k            : int — número de componentes latentes.
    """

    def __init__(self, k: int = 100, n_iter: int = 10, random_state: int = 42) -> None:
        self.k            = k
        self.svd          = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state)
        self.normalizer   = Normalizer(copy=False)
        self.docs_latent: np.ndarray | None = None
        self.doc_ids:     list[str]  | None = None
        self.term_ids:    list[int]  | None = None
        self.df_map:      dict[int, int]    = {}

    # ------------------------------------------------------------------
    # Fase offline
    # ------------------------------------------------------------------

    def build(self, db_path: Path = DB_PATH, max_docs: int | None = None) -> dict:
        """
        Construye el espacio latente LSI desde el índice invertido.

        Flujo
        -----
        1. Lee postings (freq), df_map, doc_ids, term_ids desde index_repository.
        2. Construye la matriz TF-IDF sparse (n_terms × n_docs).
        3. Aplica TruncatedSVD sobre la transpuesta → (n_docs × k).
        4. Normaliza L2 para que similitud coseno = producto punto.
        5. Registra la sesión en lsi_log.

        Devuelve
        --------
        dict con n_docs, n_terms, k, var_explained, elapsed_s.
        """
        t0 = time.monotonic()
        log.info("[LSIModel] Leyendo índice desde BD: %s", db_path)

        postings, df_map, doc_ids, term_ids, n_docs_total = get_postings_for_matrix(
            max_docs=max_docs, db_path=db_path
        )

        if not doc_ids or not term_ids:
            raise RuntimeError(
                "El índice está vacío. Ejecuta el módulo indexing antes de build()."
            )

        log.info(
            "[LSIModel] Índice: %d docs | %d términos | %d postings",
            len(doc_ids), len(term_ids), len(postings),
        )

        # ── Construir matriz TF-IDF sparse (n_terms × n_docs) ────────────
        doc_idx  = {d: i for i, d in enumerate(doc_ids)}
        term_idx = {t: i for i, t in enumerate(term_ids)}
        N        = n_docs_total

        matrix = lil_matrix((len(term_ids), len(doc_ids)), dtype=np.float32)

        for term_id, doc_id, freq in postings:
            ti = term_idx.get(term_id)
            di = doc_idx.get(doc_id)
            if ti is None or di is None:
                continue
            df = df_map.get(term_id, 1)
            tf  = math.log(1 + freq)
            idf = math.log((N + 1) / (df + 1))
            matrix[ti, di] = tf * idf

        log.info("[LSIModel] Aplicando TruncatedSVD (k=%d, n_iter=%d)…",
                 self.k, self.svd.n_iter)

        # SVD espera (n_samples × n_features) → transponemos a (n_docs × n_terms)
        docs_svd        = self.svd.fit_transform(matrix.tocsr().T)
        self.docs_latent = self.normalizer.fit_transform(docs_svd)
        self.doc_ids     = doc_ids
        self.term_ids    = term_ids
        self.df_map      = df_map

        var     = float(self.svd.explained_variance_ratio_.sum())
        elapsed = time.monotonic() - t0

        log.info("[LSIModel] SVD completado. Varianza=%.2f%%  Tiempo=%.1fs",
                 var * 100, elapsed)

        stats = {
            "n_docs":        len(doc_ids),
            "n_terms":       len(term_ids),
            "k":             self.k,
            "var_explained": var,
            "elapsed_s":     round(elapsed, 2),
        }
        self._log_to_db(stats, db_path)
        return stats

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        """Persiste el modelo completo en un archivo .pkl."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "svd":         self.svd,
            "normalizer":  self.normalizer,
            "docs_latent": self.docs_latent,
            "doc_ids":     self.doc_ids,
            "term_ids":    self.term_ids,
            "df_map":      self.df_map,
            "k":           self.k,
        }, path)
        log.info("[LSIModel] Modelo guardado en %s (%d docs, k=%d)",
                 path, len(self.doc_ids), self.k)

    def load(self, path: Path = MODEL_PATH) -> None:
        """Carga el modelo desde un archivo .pkl."""
        data = joblib.load(Path(path))
        self.svd         = data["svd"]
        self.normalizer  = data["normalizer"]
        self.docs_latent = data["docs_latent"]
        self.doc_ids     = data["doc_ids"]
        self.term_ids    = data["term_ids"]
        self.df_map      = data["df_map"]
        self.k           = data["k"]
        log.info("[LSIModel] Modelo cargado: %d docs, k=%d",
                 len(self.doc_ids), self.k)

    # ------------------------------------------------------------------
    # Proyección de queries (usada por LSIRetriever)
    # ------------------------------------------------------------------

    def project_query(self, query_tfidf: np.ndarray) -> np.ndarray:
        """
        Proyecta un vector TF-IDF de query al espacio latente.

        Parámetros
        ----------
        query_tfidf : np.ndarray (n_terms,) — vector TF-IDF alineado con
                      el mismo orden de term_ids que se usó en build().

        Devuelve
        --------
        np.ndarray (k,) normalizado en L2.
        """
        if self.docs_latent is None:
            raise RuntimeError("Modelo no cargado. Llama a load() o build() primero.")
        q_svd = self.svd.transform(query_tfidf.reshape(1, -1))   # (1, k)
        return self.normalizer.transform(q_svd).flatten()

    # ------------------------------------------------------------------
    # Log en BD
    # ------------------------------------------------------------------

    def _log_to_db(self, stats: dict, db_path: Path) -> None:
        """Registra la sesión de construcción en la tabla lsi_log."""
        try:
            conn = get_connection(db_path)
            conn.execute(
                """
                INSERT INTO lsi_log (built_at, k, n_docs, n_terms, var_explained, model_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    stats["k"],
                    stats["n_docs"],
                    stats["n_terms"],
                    stats["var_explained"],
                    str(MODEL_PATH),
                ),
            )
            conn.commit()
            conn.close()
            log.info("[LSIModel] Sesión registrada en lsi_log.")
        except Exception as exc:
            log.warning("[LSIModel] No se pudo registrar en lsi_log: %s", exc)
