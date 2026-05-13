"""
evaluate.py
===========
CLI para ejecutar la evaluación de retrieval sobre un EvalDataset existente.

Uso
---
    python -m backend.eval.retrieval.evaluate [opciones]

Opciones
--------
  --dataset    PATH   Ruta al JSON del dataset                      (requerido)
  --retriever  STR    hybrid | embedding | lsi                      (default: hybrid)
  --top-k      INT    Ventana de evaluación                         (default: 10)
  --reranker         Activa el reranker en el hybrid                (flag)
  --output     PATH   Ruta para guardar el reporte JSON             (opcional)
  --verbose           Activa logging DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa la calidad de retrieval sobre un EvalDataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",   required=True, type=Path)
    parser.add_argument("--retriever", default="hybrid",
                        choices=["hybrid", "embedding", "lsi"])
    parser.add_argument("--top-k",     default=10, type=int)
    parser.add_argument("--reranker",  action="store_true",
                        help="Activa CrossEncoderReranker en el hybrid.")
    parser.add_argument("--output",    default=None, type=Path)
    parser.add_argument("--verbose",   action="store_true")
    return parser.parse_args()


def _build_retriever(name: str, with_reranker: bool):
    from backend.retrieval.factory import (
        build_hybrid_retriever,
        build_embedding_retriever,
        build_lsi_retriever,
    )
    if name == "hybrid":
        return build_hybrid_retriever(with_reranker=with_reranker)
    if name == "embedding":
        return build_embedding_retriever()
    if name == "lsi":
        return build_lsi_retriever()
    raise ValueError(f"Retriever desconocido: {name!r}")


def _progress(total: int):
    def _cb(i, n, hit):
        icon = "✓" if hit.found else "✗"
        print(f"\r  [{i:>{len(str(n))}}/{n}] {icon} {hit.case_id:<20}", end="", flush=True)
        if i == n:
            print()
    return _cb


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
        retriever = _build_retriever(args.retriever, with_reranker=args.reranker)
    except Exception as exc:
        print(f"[eval] ERROR al construir el retriever: {exc}", file=sys.stderr)
        return 1

    print(f"[eval] Evaluando {len(dataset)} casos con top_k={args.top_k}…")
    hits    = EvalRunner(retriever=retriever, top_k=args.top_k,
                         on_progress=_progress(len(dataset))).run(dataset)
    metrics = aggregate(hits, top_k=args.top_k)
    print(format_summary(metrics, retriever_name=args.retriever))

    if args.output:
        save_json(metrics, path=args.output, retriever_name=args.retriever,
                  extra={"dataset_path": str(args.dataset)})
        print(f"[eval] Reporte guardado → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
