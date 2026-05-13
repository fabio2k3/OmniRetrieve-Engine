"""
generate_dataset.py
===================
Script CLI para generar un EvalDataset desde la BD del sistema.

Uso
---
    python -m backend.eval.generate_dataset [opciones]

Opciones
--------
  --sample-size   INT    Número de chunks a muestrear         (default: 50)
  --no-semantic          Solo genera casos exact (sin LLM)
  --model         STR    Modelo Ollama para la paráfrasis      (default: llama3.2:3b)
  --output        PATH   Ruta de salida del JSON               (default: backend/data/eval/dataset.json)
  --min-chars     INT    Tamaño mínimo del chunk               (default: 200)
  --fragment-sents INT   Oraciones a extraer como fragmento    (default: 2)
  --seed          INT    Semilla aleatoria                     (default: 42)
  --verbose              Activa logging DEBUG

Ejemplos
--------
    # Solo exact (rápido, sin LLM)
    python -m backend.eval.generate_dataset --sample-size 100 --no-semantic

    # Exact + semántico con llama3.2
    python -m backend.eval.generate_dataset --sample-size 50

    # Dataset pequeño de prueba
    python -m backend.eval.generate_dataset --sample-size 10 --output /tmp/test_dataset.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un EvalDataset (exact + semantic) desde la BD del sistema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-size",    type=int,  default=50,
                        help="Número de chunks a muestrear.")
    parser.add_argument("--no-semantic",    action="store_true",
                        help="Omite la generación de casos semánticos (sin LLM).")
    parser.add_argument("--model",          type=str,  default="llama3.2:3b",
                        help="Modelo Ollama para paráfrasis.")
    parser.add_argument("--output",         type=Path,
                        default=Path("backend/data/eval/dataset.json"),
                        help="Ruta de salida del JSON.")
    parser.add_argument("--min-chars",      type=int,  default=200,
                        help="Tamaño mínimo del chunk para ser elegible.")
    parser.add_argument("--fragment-sents", type=int,  default=2,
                        help="Oraciones a extraer del chunk como semilla.")
    parser.add_argument("--seed",           type=int,  default=42,
                        help="Semilla aleatoria.")
    parser.add_argument("--verbose",        action="store_true",
                        help="Activa logging DEBUG.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Importación diferida para que el logging esté configurado primero
    from backend.eval.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(
        sample_size=args.sample_size,
        include_semantic=not args.no_semantic,
        min_chunk_chars=args.min_chars,
        fragment_sentences=args.fragment_sents,
        paraphrase_model=args.model,
        seed=args.seed,
    )

    print(f"[eval] Iniciando generación: sample_size={args.sample_size}, "
          f"semantic={'sí' if not args.no_semantic else 'no'}")

    dataset = gen.generate()

    if len(dataset) == 0:
        print("[eval] ERROR: El dataset está vacío. Verifica que la BD tiene chunks.", file=sys.stderr)
        return 1

    dataset.save(args.output)
    print(f"[eval] Dataset guardado → {args.output}")
    print(f"       {dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
