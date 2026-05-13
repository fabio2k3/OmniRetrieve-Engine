"""
metrics.py
==========
Funciones matemáticas puras para métricas de información retrieval.

Reglas de diseño
----------------
· Sin imports del proyecto — solo stdlib y math.
· Sin efectos secundarios — todas las funciones son puras.
· Cada función recibe tipos primitivos (listas de floats/bools).
· Todos los casos de borde (listas vacías, k=0) devuelven 0.0.

Métricas implementadas
----------------------
hit_at_k     — fracción de casos donde el ítem relevante aparece en top-K.
mrr          — Mean Reciprocal Rank.
ndcg_at_k    — Normalized Discounted Cumulative Gain @K (relevancia binaria).
"""

from __future__ import annotations

import math


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """
    Fracción de consultas donde el ítem relevante aparece dentro del top-K.

    Con ground truth de tamaño 1, Hit@K = Precision@K = Recall@K.

    Parámetros
    ----------
    ranks : lista de posiciones 1-based donde se encontró el ítem relevante.
            None indica que no se encontró.
    k     : tamaño de la ventana de evaluación.

    Devuelve
    --------
    float en [0.0, 1.0].  0.0 si la lista está vacía o k <= 0.
    """
    if not ranks or k <= 0:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """
    Mean Reciprocal Rank.

    MRR = mean(1/rank_i) donde rank_i es la posición 1-based del ítem
    relevante para la consulta i; 0 si no se encontró.

    Parámetros
    ----------
    ranks : posiciones 1-based (None = no encontrado).

    Devuelve
    --------
    float en [0.0, 1.0].  0.0 si la lista está vacía.
    """
    if not ranks:
        return 0.0
    total = sum(1.0 / r if r is not None else 0.0 for r in ranks)
    return total / len(ranks)


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K con relevancia binaria.

    Para relevancia binaria (0/1) con exactamente un ítem relevante por
    consulta, NDCG@K se simplifica a:

        NDCG@K_i = 1 / log2(rank_i + 1)   si rank_i <= K
        NDCG@K_i = 0                        en caso contrario

    El denominador de la normalización es el DCG ideal, que es 1/log2(2) = 1.0
    (el único ítem relevante ocupa la posición 1 en el ranking ideal).

    Parámetros
    ----------
    ranks : posiciones 1-based (None = no encontrado).
    k     : tamaño de la ventana de evaluación.

    Devuelve
    --------
    float en [0.0, 1.0].  0.0 si la lista está vacía o k <= 0.
    """
    if not ranks or k <= 0:
        return 0.0

    def _dcg(rank: int | None) -> float:
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    total = sum(_dcg(r) for r in ranks)
    return total / len(ranks)
