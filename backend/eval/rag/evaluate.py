"""
evaluate.py
===========
CLI para ejecutar la evaluación RAG end-to-end sobre un EvalDataset.

Uso
---
    python -m backend.eval.rag.evaluate [opciones]

Opciones
--------
  --dataset      PATH   Ruta al dataset JSON                        (requerido)
  --judge-model  STR    Modelo Ollama para el juez                  (default: llama3.2:3b)
  --output       PATH   Reporte JSON de métricas                    (opcional)
  --judgements   PATH   Veredictos individuales JSON                (opcional)
  --top-k        INT    top_k para el pipeline RAG                  (default: 10)
  --verbose             Activa logging DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación RAG end-to-end con LLM-as-judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True, type=Path)
    parser.add_argument("--judge-model", default="llama3.2:3b")
    parser.add_argument("--output",      default=None, type=Path)
    parser.add_argument("--judgements",  default=None, type=Path)
    parser.add_argument("--top-k",       default=10, type=int)
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def _progress(total: int):
    def _cb(i, n, j):
        err = " ⚠" if j.judge_error else ""
        print(f"\r  [{i:>{len(str(n))}}/{n}] {j.case_id:<20}{err}", end="", flush=True)
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
    from backend.eval.rag.judge import OllamaJudge
    from backend.eval.rag.runner import RAGEvalRunner
    from backend.eval.rag.aggregator import aggregate
    from backend.eval.rag.report import format_summary, save_json, save_judgements

    if not args.dataset.exists():
        print(f"[eval] ERROR: No se encuentra el dataset: {args.dataset}", file=sys.stderr)
        return 1

    dataset = EvalDataset.load(args.dataset)
    print(f"[eval] Dataset cargado: {dataset}")

    print("[eval] Cargando pipeline RAG…")
    try:
        from backend.retrieval.factory import build_hybrid_retriever
        from backend.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline(retriever=build_hybrid_retriever(with_reranker=True))
    except Exception as exc:
        print(f"[eval] ERROR al construir el pipeline RAG: {exc}", file=sys.stderr)
        return 1

    judge = OllamaJudge(model=args.judge_model, temperature=0.0)
    print(f"[eval] Juez LLM: {args.judge_model}")

    print(f"[eval] Evaluando {len(dataset)} casos…")
    judgements = RAGEvalRunner(
        pipeline=pipeline,
        judge=judge,
        on_progress=_progress(len(dataset)),
        pipeline_kwargs={"top_k": args.top_k},
    ).run(dataset)

    metrics = aggregate(judgements)
    print(format_summary(metrics, pipeline_name="RAGPipeline (hybrid + reranker)"))

    extra = {"dataset_path": str(args.dataset)}
    if args.output:
        save_json(metrics, path=args.output, pipeline_name="RAGPipeline", extra=extra)
        print(f"[eval] Reporte guardado → {args.output}")
    if args.judgements:
        save_judgements(judgements, path=args.judgements, extra=extra)
        print(f"[eval] Veredictos guardados → {args.judgements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
