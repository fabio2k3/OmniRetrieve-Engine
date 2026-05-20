"""
_types.py
=========
Tipos de datos del comparador de reportes de evaluación.

Contiene exclusivamente dataclasses — ninguna lógica de negocio.

Clases
------
MetricDelta      — diferencia de una métrica entre baseline y candidate.
ComparisonResult — colección de deltas con metadatos de la comparación.

Vocabulario
-----------
baseline  : el reporte de referencia (la versión "antes").
candidate : el reporte que se evalúa (la versión "después").
delta     : candidate − baseline  (positivo = mejora, negativo = regresión).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["improved", "degraded", "neutral"]

# Indicadores visuales por estado
STATUS_ICON: dict[Status, str] = {
    "improved": "✓",
    "degraded": "✗",
    "neutral":  "~",
}


@dataclass
class MetricDelta:
    """
    Diferencia de una métrica individual entre baseline y candidate.

    Campos
    ------
    name      : nombre legible de la métrica (ej. "overall.hit_at_k").
    group     : agrupación lógica (ej. "overall", "exact", "semantic").
    baseline  : valor en el reporte baseline.
    candidate : valor en el reporte candidate.
    delta     : candidate − baseline.
    delta_pct : delta relativo respecto al baseline (None si baseline == 0).
    status    : "improved" | "degraded" | "neutral" según el umbral aplicado.
    """

    name:      str
    group:     str
    baseline:  float
    candidate: float
    delta:     float
    delta_pct: float | None
    status:    Status

    @property
    def icon(self) -> str:
        return STATUS_ICON[self.status]


@dataclass
class ComparisonResult:
    """
    Resultado completo de comparar dos reportes de evaluación.

    Campos
    ------
    baseline_label  : etiqueta del reporte baseline (path o nombre).
    candidate_label : etiqueta del reporte candidate.
    report_type     : tipo de reportes comparados ("retrieval" | "rag" | "mixed").
    threshold       : umbral de delta mínimo para considerar una mejora/regresión.
    deltas          : lista de MetricDelta ordenada por grupo y nombre.
    generated_at    : timestamp ISO de la comparación.
    """

    baseline_label:  str
    candidate_label: str
    report_type:     str         # "retrieval" | "rag" | "mixed"
    threshold:       float
    deltas:          list[MetricDelta]
    generated_at:    str = field(default="")

    # ------------------------------------------------------------------
    # Helpers de filtrado
    # ------------------------------------------------------------------

    def improved(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "improved"]

    def degraded(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "degraded"]

    def neutral(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.status == "neutral"]

    def by_group(self, group: str) -> list[MetricDelta]:
        return [d for d in self.deltas if d.group == group]
