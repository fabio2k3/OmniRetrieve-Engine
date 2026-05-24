"""
rag/_types.py
=============
Tipos de datos del submódulo de evaluación RAG.

Versión simplificada: sin ground truth de chunks ni tipos de caso.
Solo query → respuesta → veredicto del juez.

Clases
------
DimensionScore       — puntuación 1-5 del juez para una dimensión.
RAGJudgement         — veredicto completo sobre una consulta.
DimensionStats       — estadísticas agregadas de una dimensión.
RAGAggregatedMetrics — métricas finales de una corrida de evaluación.

Dimensiones evaluadas
---------------------
faithfulness     : ¿La respuesta está fundamentada en los documentos recuperados?
                   Detecta alucinaciones.
answer_relevance : ¿La respuesta responde la pregunta planteada de forma útil?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Dimension = Literal["faithfulness", "answer_relevance"]

_SCORE_MIN = 1
_SCORE_MAX = 5


@dataclass
class DimensionScore:
    """
    Puntuación del juez LLM para una única dimensión de calidad.

    Campos
    ------
    raw_score : puntuación entera 1–5 devuelta por el LLM.
    score     : puntuación normalizada al rango [0.0, 1.0].
    reason    : justificación breve del juez (trazabilidad).
    dimension : nombre de la dimensión evaluada.
    """
    raw_score: int
    score:     float
    reason:    str
    dimension: Dimension

    @classmethod
    def from_raw(cls, raw: int, reason: str, dimension: Dimension) -> "DimensionScore":
        raw_clamped = max(_SCORE_MIN, min(_SCORE_MAX, raw))
        normalized  = (raw_clamped - _SCORE_MIN) / (_SCORE_MAX - _SCORE_MIN)
        return cls(raw_score=raw_clamped, score=normalized, reason=reason, dimension=dimension)


@dataclass
class RAGJudgement:
    """
    Veredicto del juez LLM sobre la respuesta del pipeline a una consulta.

    Campos
    ------
    query_id         : ID de la RAGQuery evaluada.
    query            : consulta enviada al pipeline.
    answer           : respuesta generada por el pipeline.
    faithfulness     : ¿Respuesta grounded en las fuentes? None si el juez falló.
    answer_relevance : ¿Respuesta útil y pertinente? None si el juez falló.
    judge_error      : descripción del error si alguna dimensión no pudo evaluarse.
    """
    query_id:         str
    query:            str
    answer:           str
    faithfulness:     DimensionScore | None = None
    answer_relevance: DimensionScore | None = None
    judge_error:      str | None = None

    def scores(self) -> dict[str, float | None]:
        return {
            "faithfulness":     self.faithfulness.score     if self.faithfulness     else None,
            "answer_relevance": self.answer_relevance.score if self.answer_relevance else None,
        }


@dataclass
class DimensionStats:
    """Estadísticas descriptivas de una dimensión sobre múltiples casos."""
    n_cases: int
    mean:    float
    minimum: float
    maximum: float


@dataclass
class RAGAggregatedMetrics:
    """Métricas RAG agregadas de una corrida de evaluación completa."""
    n_total:          int
    n_errors:         int
    faithfulness:     DimensionStats | None
    answer_relevance: DimensionStats | None