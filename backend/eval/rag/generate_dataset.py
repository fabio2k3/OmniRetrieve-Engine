"""
rag/generate_dataset.py
========================
CLI especializado para generar un EvalDataset destinado a evaluar el
pipeline RAG (retrieval + generación LLM).

Tipo de caso generado
---------------------
generated — pregunta real de usuario generada por LLM a partir del contenido
            del chunk. Simula cómo un investigador interactuaría con el sistema.

Este tipo es el más representativo para la evaluación RAG porque:
  · La query no contiene palabras del chunk (prueba comprensión semántica real).
  · La respuesta esperada no está predefinida (se evalúa con LLM-as-judge).
  · Refleja el caso de uso real del sistema.

NO se generan casos exact ni semantic: esos están diseñados para evaluar
retrieval a nivel de chunk (backend/eval/retrieval/generate_dataset.py).

La evaluación resultante mide faithfulness, answer_relevance y
context_relevance mediante un juez LLM (ver backend/eval/rag/evaluate.py).

Uso
---
    python -m backend.eval.rag.generate_dataset [opciones]

Ejemplos
--------
    # Dataset RAG con 50 preguntas generadas
    python -m backend.eval.rag.generate_dataset --sample-size 50

    # Con modelo distinto y salida personalizada
    python -m backend.eval.rag.generate_dataset --sample-size 30 \\
        --model mistral:7b --output backend/data/eval/rag_30.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_DEFAULT_OUTPUT = Path("backend/data/eval/dataset_rag.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un EvalDataset de preguntas reales para evaluación RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Número de chunks a muestrear; se genera una pregunta por chunk.",
    )
    parser.add_argument(
        "--model", type=str, default="llama3.2:3b",
        help="Modelo Ollama para generación de preguntas.",
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
        "--temperature", type=float, default=0.4,
        help="Temperatura del LLM al generar preguntas.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Intentos máximos por chunk si el LLM falla.",
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

    from backend.eval.dataset_generator import DatasetGenerator

    gen = DatasetGenerator(
        sample_size       = args.sample_size,
        include_exact     = False,          # nunca en el dataset RAG
        include_semantic  = False,          # nunca en el dataset RAG
        include_generated = True,
        min_chunk_chars   = args.min_chars,
        query_gen_model   = args.model,
        model_temperature = args.temperature,
        max_retries       = args.max_retries,
        seed              = args.seed,
    )

    print(
        f"[eval/rag] Generando dataset RAG: "
        f"sample_size={args.sample_size}  modelo={args.model}"
    )

    dataset = gen.generate()

    if len(dataset) == 0:
        print(
            "[eval/rag] ERROR: dataset vacío. "
            "¿Está la BD poblada y Ollama disponible?",
            file=sys.stderr,
        )
        return 1

    if dataset.n_generated == 0:
        print(
            "[eval/rag] AVISO: ninguna pregunta fue generada con éxito. "
            "Revisa que Ollama esté corriendo y el modelo esté disponible.",
            file=sys.stderr,
        )
        return 1

    dataset.save(args.output)
    print(f"[eval/rag] Dataset guardado → {args.output}")
    print(f"           {dataset}")
    print(f"           Preguntas generadas: {dataset.n_generated}/{args.sample_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())