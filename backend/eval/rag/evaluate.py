"""
rag/evaluate.py
================
CLI para ejecutar la evaluación RAG end-to-end con LLM-as-judge.

Carga un RAGQuerySet (generado con backend.eval.rag.generate_queries),
pasa cada consulta por el pipeline RAG y evalúa la calidad de la respuesta
con un juez LLM en dos dimensiones:

    Faithfulness     — ¿La respuesta está soportada por los documentos?
    Answer Relevance — ¿La respuesta es útil y responde la pregunta?

Uso
---
    python -m backend.eval.rag.evaluate [opciones]

Ejemplos
--------
    python -m backend.eval.rag.evaluate \\
        --queries backend/data/eval/queries_rag.json \\
        --embed-model all-MiniLM-L6-v2 \\
        --judge-model llama3.2:3b \\
        --output backend/data/eval/results_rag.json \\
        --judgements backend/data/eval/judgements_rag.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación RAG con LLM-as-judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries", required=True, type=Path,
        help="RAGQuerySet JSON generado con backend.eval.rag.generate_queries.",
    )
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--judge-model", type=str, default="llama3.2:3b")
    parser.add_argument("--top-k",       type=int, default=10)
    parser.add_argument("--output",      type=Path, default=None,
                        help="Ruta para el reporte de métricas JSON.")
    parser.add_argument("--judgements",  type=Path, default=None,
                        help="Ruta para los veredictos individuales JSON.")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def _progress(total: int):
    def _cb(i, n, j):
        err = " ⚠" if j.judge_error else ""
        print(f"\r  [{i:>{len(str(n))}}/{n}] {j.query_id:<12}{err}", end="", flush=True)
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

    from .schema import RAGQuerySet
    from .judge import OllamaJudge
    from .runner import RAGEvalRunner
    from .aggregator import aggregate
    from .report import format_summary, save_json, save_judgements

    if not args.queries.exists():
        print(f"[eval/rag] ERROR: fichero no encontrado: {args.queries}", file=sys.stderr)
        return 1

    query_set = RAGQuerySet.load(args.queries)
    print(f"[eval/rag] {query_set}")

    print(f"[eval/rag] Cargando pipeline RAG (embed_model={args.embed_model})…")
    try:
        from backend.retrieval.factory import build_hybrid_retriever
        from backend.rag.pipeline import RAGPipeline
        pipeline = RAGPipeline(
            retriever=build_hybrid_retriever(
                embed_model=args.embed_model,
                with_reranker=True,
            )
        )
    except Exception as exc:
        print(f"[eval/rag] ERROR al construir el pipeline: {exc}", file=sys.stderr)
        return 1

    judge = OllamaJudge(model=args.judge_model, temperature=0.0)
    print(f"[eval/rag] Juez LLM: {args.judge_model}")
    print(f"[eval/rag] Evaluando {len(query_set)} consultas…")

    judgements = RAGEvalRunner(
        pipeline=pipeline,
        judge=judge,
        on_progress=_progress(len(query_set)),
        pipeline_kwargs={"top_k": args.top_k},
    ).run(query_set)

    metrics = aggregate(judgements)
    print(format_summary(metrics, pipeline_name=f"RAGPipeline ({args.embed_model})"))

    extra = {"queries_path": str(args.queries), "embed_model": args.embed_model}
    if args.output:
        save_json(metrics, path=args.output, pipeline_name="RAGPipeline", extra=extra)
        print(f"[eval/rag] Reporte guardado → {args.output}")
    if args.judgements:
        save_judgements(judgements, path=args.judgements, extra=extra)
        print(f"[eval/rag] Veredictos guardados → {args.judgements}")
    return 0


if __name__ == "__main__":
    sys.exit(main())