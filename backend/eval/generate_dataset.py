"""
generate_dataset.py
===================
CLI para generar un EvalDataset desde la BD del sistema.

Uso
---
    python -m backend.eval.generate_dataset [opciones]

Opciones
--------
  --sample-size    INT   Chunks a muestrear                         (default: 50)
  --exact                Incluir casos exact (fragmento literal)
  --semantic             Incluir casos semantic (paráfrasis LLM)
  --generated            Incluir casos generated (queries reales LLM)  ← NUEVO
  --model          STR   Modelo Ollama para LLM calls               (default: llama3.2:3b)
  --output         PATH  Ruta de salida JSON                        (default: backend/data/eval/dataset.json)
  --min-chars      INT   Tamaño mínimo del chunk                    (default: 200)
  --fragment-sents INT   Oraciones a extraer como semilla           (default: 2)
  --seed           INT   Semilla aleatoria                          (default: 42)
  --verbose              Activa logging DEBUG

Ejemplos
--------
    # Solo queries reales (recomendado para eval RAG)
    python -m backend.eval.generate_dataset --generated --sample-size 50

    # Dataset completo con los tres tipos
    python -m backend.eval.generate_dataset --exact --semantic --generated --sample-size 50

    # Rápido, sin LLM (solo exact)
    python -m backend.eval.generate_dataset --exact --sample-size 100
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

    # Si no se selecciona ningún tipo, activar generated por defecto
    include_exact     = args.exact
    include_semantic  = args.semantic
    include_generated = args.generated
    if not any([include_exact, include_semantic, include_generated]):
        print("[eval] No se especificó tipo. Activando --generated por defecto.")
        include_generated = True

    from backend.eval.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(
        sample_size=args.sample_size,
        include_exact=include_exact,
        include_semantic=include_semantic,
        include_generated=include_generated,
        min_chunk_chars=args.min_chars,
        fragment_sentences=args.fragment_sents,
        paraphrase_model=args.model,
        query_gen_model=args.model,
        seed=args.seed,
    )

    types_str = " + ".join(
        t for t, v in [("exact", include_exact),
                       ("semantic", include_semantic),
                       ("generated", include_generated)]
        if v
    )
    print(f"[eval] Generando dataset: tipos=[{types_str}] sample_size={args.sample_size}")

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
