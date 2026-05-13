"""
report.py
=========
Presentación y serialización de ComparisonResult.

Responsabilidad única
---------------------
Formatear la comparación para consola y persistirla en JSON.
Sin lógica de cálculo ni acceso a ficheros de reporte externos.

Salidas
-------
format_summary(result)       → str legible para consola.
save_json(result, path)      → JSON para análisis posterior.
load_json(path)              → dict del JSON guardado.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ._types import ComparisonResult, MetricDelta, STATUS_ICON

# Ancho de columna para el nombre de la métrica
_NAME_WIDTH = 30


# ---------------------------------------------------------------------------
# Helpers de formateo
# ---------------------------------------------------------------------------

def _fmt_delta(d: MetricDelta) -> str:
    sign   = "+" if d.delta >= 0 else ""
    pct    = f"({sign}{d.delta_pct:.1f}%)" if d.delta_pct is not None else ""
    return (
        f"  {d.icon}  {d.name:<{_NAME_WIDTH}}"
        f"  base={d.baseline:.4f}"
        f"  cand={d.candidate:.4f}"
        f"  Δ={sign}{d.delta:.4f}  {pct}"
    )


def _section(title: str, deltas: list[MetricDelta]) -> str:
    if not deltas:
        return ""
    header = f"\n  ── {title} ({'─' * max(0, 44 - len(title))})\n"
    return header + "\n".join(_fmt_delta(d) for d in deltas) + "\n"


# ---------------------------------------------------------------------------
# API pública — texto
# ---------------------------------------------------------------------------

def format_summary(result: ComparisonResult) -> str:
    """
    Devuelve un resumen de comparación legible para consola.

    Agrupa las métricas por estado (improved / degraded / neutral) y
    añade un contador de victorias/derrotas al final.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "=" * 60,
        f"  Comparison  [{result.report_type}]",
        f"  Baseline  : {result.baseline_label}",
        f"  Candidate : {result.candidate_label}",
        f"  Threshold : ±{result.threshold}   |   {now}",
        "=" * 60,
    ]

    lines.append(_section(f"{STATUS_ICON['improved']}  Improved", result.improved()))
    lines.append(_section(f"{STATUS_ICON['degraded']}  Degraded", result.degraded()))
    lines.append(_section(f"{STATUS_ICON['neutral']}  Neutral",  result.neutral()))

    n_imp  = len(result.improved())
    n_deg  = len(result.degraded())
    n_neu  = len(result.neutral())
    total  = len(result.deltas)

    lines += [
        "=" * 60,
        f"  Improved={n_imp}  Degraded={n_deg}  Neutral={n_neu}  Total={total}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API pública — JSON
# ---------------------------------------------------------------------------

def _delta_to_dict(d: MetricDelta) -> dict:
    return {
        "name":      d.name,
        "group":     d.group,
        "baseline":  d.baseline,
        "candidate": d.candidate,
        "delta":     d.delta,
        "delta_pct": d.delta_pct,
        "status":    d.status,
    }


def save_json(result: ComparisonResult, path: Path) -> None:
    """
    Guarda el ComparisonResult completo en JSON.

    Parámetros
    ----------
    result : resultado de differ.compare_reports().
    path   : ruta de destino (se crean directorios intermedios).
    """
    payload = {
        "generated_at":   result.generated_at,
        "baseline_label": result.baseline_label,
        "candidate_label":result.candidate_label,
        "report_type":    result.report_type,
        "threshold":      result.threshold,
        "summary": {
            "n_improved": len(result.improved()),
            "n_degraded": len(result.degraded()),
            "n_neutral":  len(result.neutral()),
            "n_total":    len(result.deltas),
        },
        "deltas": [_delta_to_dict(d) for d in result.deltas],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Carga un reporte de comparación guardado previamente."""
    return json.loads(path.read_text(encoding="utf-8"))
