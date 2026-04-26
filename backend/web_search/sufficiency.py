"""
sufficiency.py
==============
Decide si los documentos recuperados por el retriever son suficientes
para responder una consulta o si es necesario activar la búsqueda web.

Criterio
--------
Se considera que la información es suficiente si al menos `min_docs`
documentos superan el umbral de score `threshold`.

Los valores por defecto están calibrados para similitud coseno LSI:
    threshold = 0.15  — score mínimo aceptable por documento
    min_docs  = 1     — al menos 1 documento debe superarlo

Estos valores pueden ajustarse en la configuración del orquestador.

Uso
---
    from backend.web_search.sufficiency import SufficiencyChecker

    checker = SufficiencyChecker(threshold=0.15, min_docs=1)
    if not checker.is_sufficient(retriever_results):
        # activar búsqueda web
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class SufficiencyChecker:
    """
    Evalúa si los resultados del retriever son suficientes.

    Parámetros
    ----------
    threshold : score mínimo que debe tener un documento para considerarse relevante
    min_docs  : número mínimo de documentos que deben superar el threshold
    """

    def __init__(
        self,
        threshold: float = 0.15,
        min_docs: int = 1,
    ) -> None:
        self.threshold = threshold
        self.min_docs  = min_docs

    def is_sufficient(self, results: list[dict]) -> bool:
        """
        Devuelve True si los resultados son suficientes, False si hay
        que activar la búsqueda web.

        Parámetros
        ----------
        results : lista de dicts devuelta por LSIRetriever.retrieve()
                  cada dict debe tener al least la key 'score'
        """
        if not results:
            log.info("[Sufficiency] Sin resultados — activando búsqueda web.")
            return False

        docs_above = sum(
            1 for r in results if r.get("score", 0.0) >= self.threshold
        )

        sufficient = docs_above >= self.min_docs

        log.info(
            "[Sufficiency] %d/%d docs superan threshold=%.2f — %s",
            docs_above,
            len(results),
            self.threshold,
            "SUFICIENTE" if sufficient else "INSUFICIENTE → búsqueda web",
        )
        return sufficient

    def get_reason(self, results: list[dict]) -> str:
        """
        Devuelve una cadena explicando por qué se activó o no la búsqueda web.
        Útil para logging y depuración.
        """
        if not results:
            return "No se encontraron documentos en la base de datos local."

        docs_above = sum(
            1 for r in results if r.get("score", 0.0) >= self.threshold
        )
        best_score = max((r.get("score", 0.0) for r in results), default=0.0)

        if docs_above >= self.min_docs:
            return (
                f"{docs_above} documento(s) superan el umbral de relevancia "
                f"(threshold={self.threshold:.2f}, mejor score={best_score:.4f})."
            )
        else:
            return (
                f"Solo {docs_above}/{self.min_docs} documento(s) superan el umbral "
                f"(threshold={self.threshold:.2f}, mejor score={best_score:.4f}). "
                f"Activando búsqueda web."
            )