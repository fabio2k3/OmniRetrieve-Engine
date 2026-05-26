"""
generate_dataset.py
===================
CLI general para generar datasets de evaluación.

Este script sigue siendo válido para generar un dataset mixto o para
uso rápido. Para evaluaciones separadas y especializadas, usa:

    Retrieval (exact + semantic):
        python -m backend.eval.retrieval.generate_dataset

    RAG (generated):
        python -m backend.eval.rag.generate_dataset

Tipos de caso
-------------
exact     — fragmento literal del chunk como query.
            Evalúa retrieval léxico. Rápido, sin LLM.

semantic  — paráfrasis del fragmento (LLM).
            Evalúa retrieval semántico (FAISS/dense).

generated — pregunta real de usuario generada por LLM.
            Evalúa el sistema completo como lo usaría un usuario.
            Recomendado para eval RAG end-to-end.

Uso
---
    python -m backend.eval.generate_dataset [opciones]

Ejemplos
--------
    # Solo exact (sin LLM, para retrieval rápido)
    python -m backend.eval.generate_dataset --exact --sample-size 100

    # Dataset RAG (solo preguntas generadas por LLM)
    python -m backend.eval.generate_dataset --generated --sample-size 50

    # Dataset completo con los tres tipos
    python -m backend.eval.generate_dataset --exact --semantic --generated
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un EvalDataset desde la BD del sistema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-size",    type=int,  default=50)
    parser.add_argument("--exact",          action="store_true",
                        help="Incluir casos exact (fragmento literal del chunk).")
    parser.add_argument("--semantic",       action="store_true",
                        help="Incluir casos semantic (paráfrasis LLM).")
    parser.add_argument("--generated",      action="store_true",
                        help="Incluir casos generated (query real de usuario, LLM).")
    parser.add_argument("--model",          type=str,  default="llama3.2:3b",
                        help="Modelo Ollama para LLM calls (semantic y generated).")
    parser.add_argument("--output",         type=Path,
                        default=Path("backend/data/eval/dataset.json"))
    parser.add_argument("--min-chars",      type=int,  default=200)
    parser.add_argument("--fragment-sents", type=int,  default=2)
    parser.add_argument("--seed",           type=int,  default=42)
    parser.add_argument("--verbose",        action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    include_exact     = args.exact
    include_semantic  = args.semantic
    include_generated = args.generated

    if not any([include_exact, include_semantic, include_generated]):
        print("[eval] No se especificó tipo. Activando --generated por defecto.")
        include_generated = True

    from backend.eval.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(
        sample_size       = args.sample_size,
        include_exact     = include_exact,
        include_semantic  = include_semantic,
        include_generated = include_generated,
        min_chunk_chars   = args.min_chars,
        fragment_sentences= args.fragment_sents,
        paraphrase_model  = args.model,
        query_gen_model   = args.model,
        seed              = args.seed,
    )

    types_str = " + ".join(
        t for t, v in [
            ("exact",     include_exact),
            ("semantic",  include_semantic),
            ("generated", include_generated),
        ] if v
    )
    print(f"[eval] Generando dataset: tipos=[{types_str}]  sample_size={args.sample_size}")

    dataset = gen.generate()

    if len(dataset) == 0:
        print("[eval] ERROR: El dataset está vacío. ¿Está la BD poblada?", file=sys.stderr)
        return 1

    dataset.save(args.output)
    print(f"[eval] Dataset guardado → {args.output}")
    print(f"       {dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())