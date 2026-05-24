"""
retrieval/generate_dataset.py
==============================
CLI especializado para generar un EvalDataset destinado a evaluar el
HybridRetriever (sin cross-encoder).

Tipos de caso generados
-----------------------
exact    — fragmento literal extraído del chunk.
           Evalúa precisión léxica: el retriever debe encontrar el chunk
           exacto a partir de sus propias palabras.

semantic — paráfrasis del fragmento generada por LLM.
           Evalúa recuperación semántica (FAISS/dense): el retriever debe
           encontrar el chunk correcto aunque la query use palabras distintas.

Ambos tipos tienen ground-truth a nivel de chunk (expected_chunk_id), lo
que permite calcular Hit@K, MRR y NDCG con precisión exacta.

NO se generan casos "generated" (preguntas de usuario): esos corresponden
al dataset de evaluación RAG (backend/eval/rag/generate_dataset.py).

Uso
---
    python -m backend.eval.retrieval.generate_dataset [opciones]

Ejemplos
--------
    # Solo exact (rápido, sin LLM)
    python -m backend.eval.retrieval.generate_dataset --exact --sample-size 100

    # Exact + semantic (completo, requiere Ollama)
    python -m backend.eval.retrieval.generate_dataset --exact --semantic --sample-size 50

    # Solo semantic
    python -m backend.eval.retrieval.generate_dataset --semantic --model llama3.2:3b
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_DEFAULT_OUTPUT = Path("backend/data/eval/dataset_retrieval.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un EvalDataset para evaluación de retrieval (exact + semantic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="Incluir casos exact (fragmento literal del chunk). Sin LLM.",
    )
    parser.add_argument(
        "--semantic", action="store_true",
        help="Incluir casos semantic (paráfrasis del fragmento vía LLM).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Número de chunks a muestrear de la BD.",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2:3b",
        help="Modelo Ollama para paráfrasis (solo relevante con --semantic).",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Ruta de salida del dataset JSON.",
    )
    parser.add_argument(
        "--min-chars", type=int, default=200,
        help="Longitud mínima del chunk para ser elegible.",
    )
    parser.add_argument(
        "--fragment-sents", type=int, default=2,
        help="Oraciones a extraer del chunk como semilla de la query.",
    )
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    include_exact    = args.exact
    include_semantic = args.semantic

    if not include_exact and not include_semantic:
        print(
            "[eval/retrieval] No se especificó tipo. "
            "Activando --exact y --semantic por defecto."
        )
        include_exact    = True
        include_semantic = True

    from backend.eval.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(
        sample_size       = args.sample_size,
        include_exact     = include_exact,
        include_semantic  = include_semantic,
        include_generated = False,          # nunca en el dataset de retrieval
        min_chunk_chars   = args.min_chars,
        fragment_sentences= args.fragment_sents,
        paraphrase_model  = args.model,
        query_gen_model   = args.model,     # ignorado al ser include_generated=False
        seed              = args.seed,
    )

    types_str = " + ".join(
        t for t, v in [("exact", include_exact), ("semantic", include_semantic)] if v
    )
    print(
        f"[eval/retrieval] Generando dataset de retrieval: "
        f"tipos=[{types_str}]  sample_size={args.sample_size}"
    )

    dataset = gen.generate()

    if len(dataset) == 0:
        print(
            "[eval/retrieval] ERROR: dataset vacío. ¿Está la BD poblada?",
            file=sys.stderr,
        )
        return 1

    dataset.save(args.output)
    print(f"[eval/retrieval] Dataset guardado → {args.output}")
    print(f"                 {dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())