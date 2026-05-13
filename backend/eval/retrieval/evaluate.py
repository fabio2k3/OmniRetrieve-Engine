"""
evaluate.py
===========
CLI para ejecutar la evaluación de retrieval sobre un EvalDataset existente.

Uso
---
    python -m backend.eval.retrieval.evaluate [opciones]

Opciones
--------
  --dataset   PATH   Ruta al JSON generado por dataset_generator   (requerido)
  --retriever STR    Retriever a evaluar: hybrid | embedding        (default: hybrid)
  --top-k     INT    Ventana de evaluación                          (default: 10)
  --output    PATH   Ruta para guardar el reporte JSON              (opcional)
  --verbose          Activa logging DEBUG

Nota sobre LSI
--------------
LSI opera a nivel de documento, no de chunk. Sus chunk_ids son strings
sintéticos que nunca coinciden con los enteros de la tabla chunks, por lo
que evaluarlo en solitario siempre dará Hit@K=0. Dentro del hybrid sí
aporta como rama sparse en la fusión RRF.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def _build_faiss():
    """Carga FaissIndexManager con las rutas por defecto del sistema."""
    from backend.embedding.faiss.index_manager import FaissIndexManager
    from backend.embedding.pipeline import _INDEX_PATH, _ID_MAP_PATH
    from backend.embedding.embedder import DEFAULT_MODEL
    from sentence_transformers import SentenceTransformer

    dim = SentenceTransformer(DEFAULT_MODEL).get_sentence_embedding_dimension()
    mgr = FaissIndexManager(dim=dim, index_path=_INDEX_PATH, id_map_path=_ID_MAP_PATH)
    mgr.load()
    return mgr


def _build_lsi_adapter():
    """
    Carga LSIRetriever y lo envuelve en LSIRetrieverAdapter.

    LSIRetriever.retrieve() devuelve list[dict] (nivel documento).
    LSIRetrieverAdapter lo convierte a list[RetrievalResult] para ser
    compatible con HybridRetriever. Sus chunk_ids siguen siendo sintéticos
    ("arxiv_id__lsi__N") — nunca coinciden con IDs reales de chunks.
    """
    from backend.retrieval.lsi_retriever import LSIRetriever, LSIRetrieverAdapter
    r = LSIRetriever()
    r.load()
    return LSIRetrieverAdapter(r)


def _build_retriever(name: str):
    """
    Construye el retriever solicitado con la configuración real del sistema.
    """
    if name == "hybrid":
        from backend.retrieval.hybrid_retriever import HybridRetriever
        from backend.retrieval.embedding_retriever import EmbeddingRetriever
        return HybridRetriever(
            sparse=_build_lsi_adapter(),
            dense=EmbeddingRetriever(faiss_mgr=_build_faiss()),
        )

    if name == "embedding":
        from backend.retrieval.embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever(faiss_mgr=_build_faiss())

    if name == "lsi":
        print(
            "\n[eval] AVISO: LSI opera a nivel documento, no de chunk.\n"
            "       Hit@K siempre será 0. Usa 'hybrid' o 'embedding'.\n",
            flush=True,
        )
        return _build_lsi_adapter()

    raise ValueError(f"Retriever desconocido: {name!r}. Opciones: hybrid, embedding, lsi")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa la calidad de retrieval sobre un EvalDataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",   required=True,  type=Path,
                        help="Ruta al dataset JSON.")
    parser.add_argument("--retriever", default="hybrid",
                        choices=["hybrid", "embedding", "lsi"],
                        help="Retriever a evaluar.")
    parser.add_argument("--top-k",     default=10, type=int,
                        help="Ventana de evaluación.")
    parser.add_argument("--output",    default=None, type=Path,
                        help="Ruta para guardar el reporte JSON.")
    parser.add_argument("--verbose",   action="store_true",
                        help="Activa logging DEBUG.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Barra de progreso
# ---------------------------------------------------------------------------

def _make_progress_callback(total: int):
    def _cb(i: int, n: int, hit) -> None:
        icon = "✓" if hit.found else "✗"
        print(f"\r  [{i:>{len(str(n))}}/{n}] {icon} {hit.case_id:<20}", end="", flush=True)
        if i == n:
            print()
    return _cb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from backend.eval.schema import EvalDataset
    from backend.eval.retrieval.runner import EvalRunner
    from backend.eval.retrieval.aggregator import aggregate
    from backend.eval.retrieval.report import format_summary, save_json

    if not args.dataset.exists():
        print(f"[eval] ERROR: No se encuentra el dataset: {args.dataset}", file=sys.stderr)
        return 1

    dataset = EvalDataset.load(args.dataset)
    print(f"[eval] Dataset cargado: {dataset}")

    print(f"[eval] Cargando retriever '{args.retriever}'…")
    try:
        retriever = _build_retriever(args.retriever)
    except Exception as exc:
        print(f"[eval] ERROR al construir el retriever: {exc}", file=sys.stderr)
        return 1

    print(f"[eval] Evaluando {len(dataset)} casos con top_k={args.top_k}…")
    runner = EvalRunner(
        retriever=retriever,
        top_k=args.top_k,
        on_progress=_make_progress_callback(len(dataset)),
    )
    hits = runner.run(dataset)

    metrics = aggregate(hits, top_k=args.top_k)
    print(format_summary(metrics, retriever_name=args.retriever))

    if args.output:
        save_json(metrics, path=args.output, retriever_name=args.retriever,
                  extra={"dataset_path": str(args.dataset)})
        print(f"[eval] Reporte guardado → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
