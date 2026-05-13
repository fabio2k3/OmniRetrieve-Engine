"""
_types.py
=========
Tipos de datos del submódulo de evaluación RAG end-to-end.

Contiene exclusivamente dataclasses — ninguna lógica de negocio.

Clases
------
DimensionScore      — puntuación del juez para una dimensión individual.
RAGJudgement        — veredicto completo del juez sobre un caso RAG.
DimensionStats      — estadísticas agregadas de una dimensión.
RAGAggregatedMetrics— métricas finales de una corrida de evaluación RAG.

Dimensiones evaluadas
---------------------
faithfulness       : ¿La respuesta está fundamentada en el contexto recuperado?
                     Detecta alucinaciones — afirmaciones que no están en las fuentes.
answer_relevance   : ¿La respuesta responde la pregunta planteada?
context_relevance  : ¿El contexto recuperado es relevante para la pregunta?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Dimension = Literal["faithfulness", "answer_relevance", "context_relevance"]

_SCORE_MIN = 1
_SCORE_MAX = 5


@dataclass
class DimensionScore:
    """
    Puntuación del juez LLM para una única dimensión de calidad.

    Campos
    ------
    raw_score  : puntuación entera 1–5 devuelta por el LLM.
    score      : puntuación normalizada al rango [0.0, 1.0].
    reason     : justificación breve del juez (trazabilidad).
    dimension  : nombre de la dimensión evaluada.
    """

    raw_score: int          # 1–5
    score:     float        # (raw_score - 1) / 4  → [0.0, 1.0]
    reason:    str
    dimension: Dimension

    @classmethod
    def from_raw(cls, raw: int, reason: str, dimension: Dimension) -> "DimensionScore":
        """Construye desde una puntuación entera 1-5."""
        raw_clamped = max(_SCORE_MIN, min(_SCORE_MAX, raw))
        normalized  = (raw_clamped - _SCORE_MIN) / (_SCORE_MAX - _SCORE_MIN)
        return cls(raw_score=raw_clamped, score=normalized, reason=reason, dimension=dimension)


@dataclass
class RAGJudgement:
    """
    Veredicto completo del juez LLM sobre la salida RAG de un único caso.

    Campos
    ------
    case_id             : identificador del EvalCase evaluado.
    case_type           : "exact" | "semantic".
    query               : pregunta enviada al pipeline RAG.
    answer              : respuesta generada por el pipeline.
    faithfulness        : ¿Respuesta grounded en las fuentes?  None si el juez falló.
    answer_relevance    : ¿Respuesta pertinente a la pregunta?  None si el juez falló.
    context_relevance   : ¿Contexto relevante para la pregunta? None si el juez falló.
    judge_error         : mensaje de error si alguna dimensión no pudo evaluarse.
    """

    case_id:             str
    case_type:           str
    query:               str
    answer:              str
    faithfulness:        DimensionScore | None = None
    answer_relevance:    DimensionScore | None = None
    context_relevance:   DimensionScore | None = None
    judge_error:         str | None = None

    def scores(self) -> dict[str, float | None]:
        """Devuelve las tres puntuaciones normalizadas como dict."""
        return {
            "faithfulness":      self.faithfulness.score      if self.faithfulness      else None,
            "answer_relevance":  self.answer_relevance.score  if self.answer_relevance  else None,
            "context_relevance": self.context_relevance.score if self.context_relevance else None,
        }


@dataclass
class DimensionStats:
    """
    Estadísticas descriptivas de una dimensión sobre múltiples casos.

    Campos
    ------
    n_cases : número de casos con puntuación válida para esta dimensión.
    mean    : media de las puntuaciones normalizadas [0.0, 1.0].
    minimum : puntuación mínima.
    maximum : puntuación máxima.
    """

    n_cases: int
    mean:    float
    minimum: float
    maximum: float


@dataclass
class RAGAggregatedMetrics:
    """
    Métricas RAG agregadas de una corrida de evaluación completa.

    Campos
    ------
    n_total             : total de casos evaluados.
    n_errors            : casos donde el juez no pudo puntuar al menos una dimensión.
    faithfulness        : estadísticas de faithfulness.
    answer_relevance    : estadísticas de answer_relevance.
    context_relevance   : estadísticas de context_relevance.
    by_type             : mismas métricas desglosadas por tipo ("exact" / "semantic").
    """

    n_total:           int
    n_errors:          int
    faithfulness:      DimensionStats | None
    answer_relevance:  DimensionStats | None
    context_relevance: DimensionStats | None
    by_type:           dict[str, "RAGAggregatedMetrics"] = field(default_factory=dict)
