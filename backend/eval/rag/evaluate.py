"""
evaluate.py
===========
CLI para ejecutar la evaluación RAG end-to-end sobre un EvalDataset.

Uso
---
    python -m backend.eval.rag.evaluate [opciones]

Opciones
--------
  --dataset       PATH   Ruta al dataset JSON                           (requerido)
  --judge-model   STR    Modelo Ollama usado como juez                  (default: llama3.2:3b)
  --output        PATH   Ruta para guardar el reporte JSON de métricas  (opcional)
  --judgements    PATH   Ruta para guardar todos los veredictos individuales (opcional)
  --top-k         INT    top_k para el pipeline RAG                     (default: 10)
  --max-chunks    INT    max_chunks para el contexto RAG                 (default: 5)
  --verbose              Activa logging DEBUG

Ejemplo
-------
    python -m backend.eval.rag.evaluate \\
        --dataset backend/data/eval/dataset.json \\
        --judge-model llama3.2:3b \\
        --output backend/data/eval/rag_report.json \\
        --judgements backend/data/eval/rag_judgements.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación RAG end-to-end con LLM-as-judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",     required=True, type=Path,
                        help="Ruta al dataset JSON.")
    parser.add_argument("--judge-model", default="llama3.2:3b",
                        help="Modelo Ollama para el juez.")
    parser.add_argument("--output",      default=None, type=Path,
                        help="Ruta para guardar el reporte JSON de métricas.")
    parser.add_argument("--judgements",  default=None, type=Path,
                        help="Ruta para guardar los veredictos individuales.")
    parser.add_argument("--top-k",       default=10, type=int,
                        help="top_k para el pipeline RAG.")
    parser.add_argument("--max-chunks",  default=5, type=int,
                        help="max_chunks para el contexto RAG.")
    parser.add_argument("--verbose",     action="store_true",
                        help="Activa logging DEBUG.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Barra de progreso
# ---------------------------------------------------------------------------

def _make_progress_callback(total: int):
    def _cb(i: int, n: int, j) -> None:
        err_icon = " ⚠" if j.judge_error else ""
        print(f"\r  [{i:>{len(str(n))}}/{n}] {j.case_id:<20}{err_icon}", end="", flush=True)
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
    from backend.eval.rag.judge import OllamaJudge
    from backend.eval.rag.runner import RAGEvalRunner
    from backend.eval.rag.aggregator import aggregate
    from backend.eval.rag.report import format_summary, save_json, save_judgements

    # 1. Cargar dataset
    if not args.dataset.exists():
        print(f"[eval] ERROR: No se encuentra el dataset: {args.dataset}", file=sys.stderr)
        return 1

    dataset = EvalDataset.load(args.dataset)
    print(f"[eval] Dataset cargado: {dataset}")

    # 2. Construir pipeline RAG
    print("[eval] Cargando pipeline RAG…")
    try:
        from backend.retrieval.hybrid_retriever import HybridRetriever
        from backend.retrieval.lsi_retriever import LSIRetriever, LSIRetrieverAdapter
        from backend.retrieval.embedding_retriever import EmbeddingRetriever
        from backend.retrieval.reranker import CrossEncoderReranker
        from backend.embedding.faiss.index_manager import FaissIndexManager
        from backend.embedding.pipeline import _INDEX_PATH, _ID_MAP_PATH
        from backend.embedding.embedder import DEFAULT_MODEL
        from backend.rag.pipeline import RAGPipeline
        from sentence_transformers import SentenceTransformer

        # Cargar FAISS
        dim = SentenceTransformer(DEFAULT_MODEL).get_sentence_embedding_dimension()
        faiss_mgr = FaissIndexManager(dim=dim, index_path=_INDEX_PATH, id_map_path=_ID_MAP_PATH)
        faiss_mgr.load()

        # Cargar LSI
        lsi = LSIRetriever()
        lsi.load()
        lsi_adapter = LSIRetrieverAdapter(lsi)

        pipeline = RAGPipeline(
            retriever=HybridRetriever(
                sparse=lsi_adapter,
                dense=EmbeddingRetriever(faiss_mgr=faiss_mgr),
            ),
            reranker=CrossEncoderReranker(),
        )
    except Exception as exc:
        print(f"[eval] ERROR al construir el pipeline RAG: {exc}", file=sys.stderr)
        return 1

    # 3. Construir juez
    judge = OllamaJudge(model=args.judge_model, temperature=0.0)
    print(f"[eval] Juez LLM: {args.judge_model}")

    # 4. Ejecutar evaluación
    print(f"[eval] Evaluando {len(dataset)} casos…")
    runner = RAGEvalRunner(
        pipeline=pipeline,
        judge=judge,
        on_progress=_make_progress_callback(len(dataset)),
        pipeline_kwargs={
            "top_k":      args.top_k,
            "max_chunks": args.max_chunks,
        },
    )
    judgements = runner.run(dataset)

    # 5. Agregar y mostrar
    metrics = aggregate(judgements)
    print(format_summary(metrics, pipeline_name="RAGPipeline (hybrid + reranker)"))

    # 6. Guardar reportes
    extra = {"dataset_path": str(args.dataset)}

    if args.output:
        save_json(metrics, path=args.output,
                  pipeline_name="RAGPipeline", extra=extra)
        print(f"[eval] Reporte de métricas guardado → {args.output}")

    if args.judgements:
        save_judgements(judgements, path=args.judgements, extra=extra)
        print(f"[eval] Veredictos individuales guardados → {args.judgements}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
