"""
lsi_model.py
============
Fase offline del modulo LSI.
Construye el espacio semantico latente LSI sobre el corpus de arXiv,
persiste el modelo y registra la sesion en la tabla lsi_log.

Referencia: Manning et al. (2008), Cap. 18, sec. 18.1-18.4
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from backend.database.schema import DB_PATH, get_connection

# El modelo se guarda en data/models/ (no en el mismo directorio que el codigo)
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "lsi_model.pkl"

log = logging.getLogger(__name__)  # "retrieval.lsi_model"


class LSIModel:
    """
    Construye y persiste el espacio semantico latente LSI.

    Flujo offline:
        model = LSIModel(k=100)
        model.build()  # lee BD, aplica Pipeline, guarda .pkl

    Flujo online:
        model = LSIModel()
        model.load()  # carga el .pkl previamente guardado
    """

    def __init__(self, k: int = 100):
        self.k = k
        # Pipeline: TF-IDF -> SVD truncada -> Normalizar L2.
        # fit_transform(texts) = fase offline: aprende todo.
        # transform([query]) = fase online: usa lo aprendido.
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                sublinear_tf=True,  # log(1+tf): variante SMART lnc
                strip_accents="unicode",
                stop_words="english",
                min_df=2,  # ignora terminos en < 2 docs
            )),
            ("svd", TruncatedSVD(
                n_components=k,
                n_iter=10,  # mejor calidad que n_iter=5 (default)
                random_state=42,  # reproducibilidad
            )),
            ("norm", Normalizer(copy=False)),  # L2: cos(a,b) = a.dot(b)
        ])
        self.docs_latent: np.ndarray | None = None
        self.doc_ids: list[str] | None = None

    # -----------------------------------------------------------------
    def _load_texts(self, db_path=DB_PATH):
        """ Lee titulo + texto completo (o abstract) de la BD. """
        log.info("[LSIModel] Cargando textos de la BD: %s", db_path)
        conn = get_connection(db_path)
        try:
            rows = conn.execute("""
                SELECT arxiv_id,
                       title,
                       COALESCE(full_text, abstract, '') AS body
                FROM documents
                WHERE pdf_downloaded = 1
                  AND COALESCE(full_text, abstract, '') != ''
                ORDER BY published DESC
            """).fetchall()
        finally:
            conn.close()

        doc_ids = [r["arxiv_id"] for r in rows]
        texts = [r["title"] + " " + r["body"] for r in rows]
        log.info("[LSIModel] %d documentos cargados", len(texts))
        return doc_ids, texts

    # -----------------------------------------------------------------
    def build(self, db_path=DB_PATH) -> dict:
        """ Fase offline: construye el espacio latente LSI. """
        t0 = time.monotonic()
        self.doc_ids, texts = self._load_texts(db_path)

        log.info("[LSIModel] Aplicando Pipeline (TF-IDF + SVD k=%d + L2)...", self.k)
        # fit_transform aplica los 3 pasos en orden con fit en cada uno:
        # 1. TfidfVectorizer.fit_transform(texts) -> (n_docs, vocab)
        # 2. TruncatedSVD.fit_transform(X) -> (n_docs, k)
        # 3. Normalizer.fit_transform(X) -> (n_docs, k) L2-norm
        self.docs_latent = self.pipeline.fit_transform(texts)

        var = self.pipeline.named_steps["svd"].explained_variance_ratio_.sum()
        elapsed = time.monotonic() - t0

        log.info("[LSIModel] SVD completado. Varianza=%.2f%% Tiempo=%.1fs",
                 var * 100, elapsed)

        stats = {
            "n_docs": len(texts),
            "k": self.k,
            "var_explained": var,
            "elapsed_s": round(elapsed, 2),
        }
        # Registrar en la tabla lsi_log de la BD
        self._log_to_db(stats, db_path)
        return stats

    def _log_to_db(self, stats: dict, db_path=DB_PATH) -> None:
        """ Registra la sesion de construccion en la tabla lsi_log. """
        try:
            conn = get_connection(db_path)
            conn.execute(
                "INSERT OR IGNORE INTO lsi_log "
                "(built_at, k, n_docs, var_explained, model_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    stats["k"],
                    stats["n_docs"],
                    stats["var_explained"],
                    str(MODEL_PATH),
                ),
            )
            conn.commit()
            conn.close()
            log.info("[LSIModel] Sesion registrada en lsi_log")
        except Exception as exc:
            # No falla si la tabla lsi_log no existe todavia
            log.warning("[LSIModel] No se pudo registrar en lsi_log: %s", exc)

    # -----------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        """ Persiste pipeline + docs_latent + doc_ids en un .pkl. """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "docs_latent": self.docs_latent,
            "doc_ids": self.doc_ids,
            "k": self.k,
        }, path)
        log.info("[LSIModel] Modelo guardado en %s (%d docs, k=%d)",
                 path, len(self.doc_ids), self.k)

    def load(self, path: Path = MODEL_PATH) -> None:
        """ Carga el modelo previamente guardado. """
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.docs_latent = data["docs_latent"]
        self.doc_ids = data["doc_ids"]
        self.k = data["k"]
        log.info("[LSIModel] Modelo cargado: %d docs, k=%d",
                 len(self.doc_ids), self.k)
