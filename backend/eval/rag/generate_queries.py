"""
rag/generate_queries.py
========================
CLI para generar un RAGQuerySet — un conjunto de consultas de usuario
para evaluar el pipeline RAG.

Las consultas se generan con un LLM a partir de chunks reales de la BD.
El resultado NO incluye ningún ground truth de chunks ni documentos:
solo la pregunta que un usuario podría hacer dado ese contenido.

Uso
---
    python -m backend.eval.rag.generate_queries [opciones]

    # Desde fichero de texto (una consulta por línea, sin LLM)
    python -m backend.eval.rag.generate_queries \\
        --from-file mis_consultas.txt \\
        --output backend/data/eval/queries_rag.json

    # Generadas por LLM desde chunks de la BD
    python -m backend.eval.rag.generate_queries \\
        --sample-size 50 --model llama3.2:3b \\
        --output backend/data/eval/queries_rag.json
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from backend.database.schema import DB_PATH

_DEFAULT_OUTPUT = Path("backend/data/eval/queries_rag.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un RAGQuerySet para evaluación del pipeline RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-file", type=Path, metavar="PATH",
        help="Carga consultas desde un fichero de texto plano (una por línea). "
             "Ignora líneas vacías y las que empiezan por '#'.",
    )
    source.add_argument(
        "--sample-size", type=int, default=50,
        help="Número de chunks a muestrear para generar consultas con LLM.",
    )
    parser.add_argument("--model",       type=str,  default="llama3.2:3b")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-retries", type=int,  default=3)
    parser.add_argument("--min-chars",   type=int,  default=200)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help="Ruta de salida del RAGQuerySet JSON.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _generate_from_chunks(
    sample_size:  int,
    model:        str,
    temperature:  float,
    max_retries:  int,
    min_chars:    int,
    seed:         int,
    db_path:      Path,
) -> list[tuple[str, dict]]:
    """
    Muestrea chunks de la BD y genera una consulta por chunk con LLM.

    Devuelve lista de (query_text, metadata).
    """
    from backend.database.schema import get_connection
    from backend.eval.query_generator import QueryGenerator

    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT c.id, c.arxiv_id, c.text, d.title
        FROM   chunks c
        JOIN   documents d USING (arxiv_id)
        WHERE  length(c.text) >= ?
        ORDER  BY RANDOM()
        LIMIT  ?
        """,
        (min_chars, sample_size * 2),  # sobre-muestreo por si falla el LLM
    ).fetchall()
    conn.close()

    rng = random.Random(seed)
    rng.shuffle(rows)

    gen = QueryGenerator(model=model, temperature=temperature, max_retries=max_retries)

    results: list[tuple[str, dict]] = []
    skipped = 0

    for row in rows:
        if len(results) >= sample_size:
            break
        query = gen.generate(row["text"])
        if query is None:
            skipped += 1
            continue
        meta = {
            "source_chunk_id": row["id"],
            "source_arxiv_id": row["arxiv_id"],
            "source_title":    row["title"] or row["arxiv_id"],
        }
        results.append((query, meta))

    if skipped:
        logging.getLogger(__name__).warning(
            "[generate_queries] %d chunks descartados por fallo del LLM.", skipped
        )
    return results


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from .schema import RAGQuery, RAGQuerySet

    # ── Origen: fichero de texto ──────────────────────────────────────────────
    if args.from_file:
        if not args.from_file.exists():
            print(f"[eval/rag] ERROR: fichero no encontrado: {args.from_file}", file=sys.stderr)
            return 1
        qs = RAGQuerySet.from_text_file(args.from_file)
        print(f"[eval/rag] {len(qs)} consultas cargadas desde {args.from_file}")

    # ── Origen: LLM sobre chunks de la BD ────────────────────────────────────
    else:
        print(
            f"[eval/rag] Generando {args.sample_size} consultas "
            f"con LLM ({args.model})…"
        )
        pairs = _generate_from_chunks(
            sample_size = args.sample_size,
            model       = args.model,
            temperature = args.temperature,
            max_retries = args.max_retries,
            min_chars   = args.min_chars,
            seed        = args.seed,
            db_path     = DB_PATH,
        )
        if not pairs:
            print(
                "[eval/rag] ERROR: no se generó ninguna consulta. "
                "¿Está Ollama corriendo y la BD poblada?",
                file=sys.stderr,
            )
            return 1

        queries = [
            RAGQuery(query_id=f"q_{i:04d}", query=q, metadata=meta)
            for i, (q, meta) in enumerate(pairs)
        ]
        qs = RAGQuerySet(
            queries=queries,
            generator_cfg={
                "type":        "llm_generated",
                "model":       args.model,
                "sample_size": args.sample_size,
                "min_chars":   args.min_chars,
                "seed":        args.seed,
            },
        )
        print(f"[eval/rag] {len(qs)} consultas generadas.")

    qs.save(args.output)
    print(f"[eval/rag] RAGQuerySet guardado → {args.output}")
    print(f"           {qs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())