"""
retrieval/evaluate.py
=====================
CLI para ejecutar la evaluación del HybridRetriever (sin cross-encoder).

Espera un dataset generado con ``backend.eval.retrieval.generate_dataset``
que contenga exclusivamente casos ``exact`` y/o ``semantic``. Si el dataset
contiene casos ``generated`` (propios del flujo RAG), se ignoran con aviso.

Métricas calculadas
-------------------
Hit@K  — fracción de casos donde el chunk correcto aparece en el top-K.
MRR    — Mean Reciprocal Rank.
NDCG@K — Normalized Discounted Cumulative Gain.

Desglosadas por tipo (exact / semantic) y en conjunto (overall).

Uso
---
    python -m backend.eval.retrieval.evaluate [opciones]

Ejemplos
--------
    python -m backend.eval.retrieval.evaluate \\
        --dataset backend/data/eval/dataset_retrieval.json \\
        --embed-model all-MiniLM-L6-v2 \\
        --top-k 10

    python -m backend.eval.retrieval.evaluate \\
        --dataset backend/data/eval/dataset_retrieval.json \\
        --retriever lsi --top-k 5 --output results/retrieval.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa la calidad de retrieval sobre un EvalDataset (exact + semantic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True, type=Path,
                        help="Ruta al JSON del dataset de retrieval.")
    parser.add_argument("--retriever",   default="hybrid",
                        choices=["hybrid", "embedding", "lsi"],
                        help="Retriever a evaluar.")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2",
                        help="Modelo sentence-transformers activo. Debe coincidir "
                             "con el modelo con el que se construyó el índice FAISS.")
    parser.add_argument("--top-k",       default=10, type=int,
                        help="Ventana de evaluación (cuántos resultados juzgar).")
    parser.add_argument("--output",      default=None, type=Path,
                        help="Ruta para guardar el reporte JSON (opcional).")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def _build_retriever(name: str, embed_model: str):
    from backend.retrieval.factory import (
        build_hybrid_retriever,
        build_embedding_retriever,
        build_lsi_retriever,
    )
    if name == "hybrid":
        # Sin cross-encoder: evaluamos solo el ranker RRF del hybrid
        return build_hybrid_retriever(embed_model=embed_model, with_reranker=False)
    if name == "embedding":
        return build_embedding_retriever(embed_model=embed_model)
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
        print(f"[eval/retrieval] ERROR: dataset no encontrado: {args.dataset}", file=sys.stderr)
        return 1

    full_dataset = EvalDataset.load(args.dataset)
    print(f"[eval/retrieval] Dataset cargado: {full_dataset}")

    # Filtrar solo casos exact y semantic — los generated son para eval RAG
    retrieval_cases = [
        c for c in full_dataset.cases
        if c.case_type in ("exact", "semantic")
    ]
    n_ignored = len(full_dataset) - len(retrieval_cases)
    if n_ignored:
        print(
            f"[eval/retrieval] AVISO: {n_ignored} casos 'generated' ignorados. "
            f"Para evaluarlos usa backend.eval.rag.evaluate."
        )
    if not retrieval_cases:
        print(
            "[eval/retrieval] ERROR: el dataset no contiene casos exact ni semantic.\n"
            "Genera el dataset con: python -m backend.eval.retrieval.generate_dataset",
            file=sys.stderr,
        )
        return 1

    dataset = EvalDataset(
        cases=retrieval_cases,
        db_path=full_dataset.db_path,
        generator_cfg=full_dataset.generator_cfg,
        generated_at=full_dataset.generated_at,
    )
    print(
        f"[eval/retrieval] Casos a evaluar: {len(dataset)} "
        f"(exact={dataset.n_exact}, semantic={dataset.n_semantic})"
    )

    print(f"[eval/retrieval] Cargando retriever '{args.retriever}' "
          f"(embed_model={args.embed_model})…")
    try:
        retriever = _build_retriever(args.retriever, embed_model=args.embed_model)
    except Exception as exc:
        print(f"[eval/retrieval] ERROR al construir el retriever: {exc}", file=sys.stderr)
        return 1

    print(f"[eval/retrieval] Evaluando {len(dataset)} casos con top_k={args.top_k}…")
    hits    = EvalRunner(
        retriever=retriever,
        top_k=args.top_k,
        on_progress=_progress(len(dataset)),
    ).run(dataset)
    metrics = aggregate(hits, top_k=args.top_k)
    print(format_summary(metrics, retriever_name=args.retriever))

    if args.output:
        save_json(
            metrics, path=args.output,
            retriever_name=args.retriever,
            extra={"dataset_path": str(args.dataset), "embed_model": args.embed_model},
        )
        print(f"[eval/retrieval] Reporte guardado → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())