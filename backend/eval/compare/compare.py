"""
compare.py
==========
CLI para comparar dos reportes de evaluación (retrieval o RAG).

Uso
---
    python -m backend.eval.compare.compare [opciones]

Opciones
--------
  --baseline   PATH   Reporte JSON de referencia                    (requerido)
  --candidate  PATH   Reporte JSON candidato a comparar             (requerido)
  --output     PATH   Ruta para guardar el reporte de comparación   (opcional)
  --threshold  FLOAT  Umbral mínimo de |Δ| para salir de "neutral" (default: 0.005)

Ejemplos
--------
    # Comparar dos corridas de retrieval
    python -m backend.eval.compare.compare \\
        --baseline  backend/data/eval/report_hybrid_rrf60.json \\
        --candidate backend/data/eval/report_hybrid_rrf40.json

    # Comparar dos corridas RAG y guardar el resultado
    python -m backend.eval.compare.compare \\
        --baseline  backend/data/eval/rag_report_v1.json \\
        --candidate backend/data/eval/rag_report_v2.json \\
        --output    backend/data/eval/comparison_v1_v2.json

    # Comparar con umbral personalizado (más estricto)
    python -m backend.eval.compare.compare \\
        --baseline  report_a.json \\
        --candidate report_b.json \\
        --threshold 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara dos reportes de evaluación y muestra los deltas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline",   required=True, type=Path,
                        help="Reporte JSON de referencia.")
    parser.add_argument("--candidate",  required=True, type=Path,
                        help="Reporte JSON candidato.")
    parser.add_argument("--output",     default=None,  type=Path,
                        help="Ruta para guardar el reporte de comparación JSON.")
    parser.add_argument("--threshold",  default=0.005, type=float,
                        help="Umbral mínimo de |Δ| para 'improved'/'degraded'.")
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"[compare] ERROR: No se encuentra el fichero: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[compare] ERROR: JSON inválido en {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    args = _parse_args()

    from backend.eval.compare.differ import compare_reports
    from backend.eval.compare.report import format_summary, save_json

    baseline  = _load(args.baseline)
    candidate = _load(args.candidate)

    result = compare_reports(
        baseline=baseline,
        candidate=candidate,
        baseline_label=args.baseline.name,
        candidate_label=args.candidate.name,
        threshold=args.threshold,
    )

    print(format_summary(result))

    if args.output:
        save_json(result, path=args.output)
        print(f"\n[compare] Reporte guardado → {args.output}")

    # Código de salida no-cero si hay regresiones (útil para CI)
    return 1 if result.degraded() else 0


if __name__ == "__main__":
    sys.exit(main())
