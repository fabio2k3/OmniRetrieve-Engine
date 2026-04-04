"""
pipeline.py
===========
Orquestador del módulo de búsqueda web.

Responsabilidad: recibir los resultados del retriever, decidir si son
suficientes y, si no lo son, buscar en la web, guardar los resultados
y devolver la lista combinada lista para el módulo RAG.

Flujo
-----
    retriever_results (list[dict] con score)
            ↓
    SufficiencyChecker → ¿suficiente?
        ├── SÍ → devuelve retriever_results sin cambios
        └── NO → WebSearcher.search(query)
                  → WebRepository.save_web_results()
                  → combina retriever_results + web_results
                  → devuelve lista combinada

Uso programático
----------------
    from backend.web_search.pipeline import WebSearchPipeline

    pipeline = WebSearchPipeline(api_key="tvly-...")
    results  = pipeline.run(
        query="fairness in machine learning",
        retriever_results=lsi_results,
    )

Uso CLI
-------
    python -m backend.web_search.pipeline --query "fairness in ML" --top 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.database.schema import DB_PATH
from backend.web_search.searcher import WebSearcher
from backend.web_search.sufficiency import SufficiencyChecker
from backend.web_search.web_repository import save_web_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class WebSearchPipeline:
    """
    Orquestador del módulo de búsqueda web.

    Parámetros
    ----------
    api_key      : clave de API de Tavily (tvly-...)
    threshold    : score mínimo para considerar un doc relevante (default: 0.15)
    min_docs     : docs mínimos que deben superar el threshold (default: 1)
    max_results  : máximo de resultados a pedir a Tavily (default: 5)
    search_depth : "basic" | "advanced" (default: "basic")
    db_path      : ruta a la BD SQLite
    """

    def __init__(
        self,
        api_key: str,
        threshold: float = 0.15,
        min_docs: int = 1,
        max_results: int = 5,
        search_depth: str = "basic",
        db_path: Path = DB_PATH,
    ) -> None:
        self.searcher   = WebSearcher(
            api_key=api_key,
            max_results=max_results,
            search_depth=search_depth,
        )
        self.checker    = SufficiencyChecker(
            threshold=threshold,
            min_docs=min_docs,
        )
        self.db_path    = db_path

    def run(
        self,
        query: str,
        retriever_results: list[dict],
    ) -> dict:
        """
        Punto de entrada principal.

        Parámetros
        ----------
        query             : consulta original del usuario
        retriever_results : resultados del LSIRetriever (lista de dicts con 'score')

        Devuelve
        --------
        dict con:
            "results"        : lista combinada de docs (retriever + web si aplica)
            "web_activated"  : True si se activó la búsqueda web
            "web_results"    : resultados web (lista vacía si no se activó)
            "reason"         : explicación de la decisión de suficiencia
            "query"          : query original
        """
        reason = self.checker.get_reason(retriever_results)

        if self.checker.is_sufficient(retriever_results):
            log.info("[WebSearch] Información suficiente — sin búsqueda web.")
            return {
                "results":       retriever_results,
                "web_activated": False,
                "web_results":   [],
                "reason":        reason,
                "query":         query,
            }

        # Información insuficiente → activar búsqueda web
        log.info("[WebSearch] Activando búsqueda web para: '%s'", query)
        web_results = self.searcher.search(query)

        # Guardar en BD para uso futuro
        if web_results:
            saved = save_web_results(
                query=query,
                results=web_results,
                db_path=self.db_path,
            )
            log.info("[WebSearch] %d documentos web guardados en BD.", saved)

        # Normalizar resultados web al mismo formato que el retriever
        web_normalized = [
            {
                "score":    r["score"],
                "arxiv_id": "",           # no tienen arxiv_id
                "title":    r["title"],
                "authors":  "Web Search",
                "abstract": r["content"][:300],
                "url":      r["url"],
                "source":   "web",
            }
            for r in web_results
        ]

        # Marcar fuente en resultados del retriever
        for r in retriever_results:
            r.setdefault("source", "local")

        # Combinar: primero los locales, luego los web
        combined = retriever_results + web_normalized

        log.info(
            "[WebSearch] Combinados: %d locales + %d web = %d total.",
            len(retriever_results), len(web_normalized), len(combined),
        )

        return {
            "results":       combined,
            "web_activated": True,
            "web_results":   web_normalized,
            "reason":        reason,
            "query":         query,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Búsqueda Web",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query",       type=str,   required=True,
                        help="Consulta de búsqueda.")
    parser.add_argument("--api-key",     type=str,   required=True,
                        help="API key de Tavily (tvly-...).")
    parser.add_argument("--threshold",   type=float, default=0.15,
                        help="Score mínimo para considerar un doc relevante.")
    parser.add_argument("--min-docs",    type=int,   default=1,
                        help="Docs mínimos que deben superar el threshold.")
    parser.add_argument("--top",         type=int,   default=5,
                        help="Máximo de resultados web a obtener.")
    parser.add_argument("--depth",       type=str,   default="basic",
                        choices=["basic", "advanced"],
                        help="Profundidad de búsqueda Tavily.")
    parser.add_argument("--db",          type=Path,  default=DB_PATH,
                        help="Ruta a la base de datos SQLite.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pipeline = WebSearchPipeline(
        api_key=args.api_key,
        threshold=args.threshold,
        min_docs=args.min_docs,
        max_results=args.top,
        search_depth=args.depth,
        db_path=args.db,
    )

    # En modo CLI simulamos que no hay resultados del retriever
    # para forzar la búsqueda web y poder probarla
    output = pipeline.run(
        query=args.query,
        retriever_results=[],
    )

    print(f"\n{'='*60}")
    print(f"Query: {output['query']}")
    print(f"Búsqueda web activada: {output['web_activated']}")
    print(f"Razón: {output['reason']}")
    print(f"Total resultados: {len(output['results'])}")
    print(f"{'='*60}")
    for i, r in enumerate(output["results"], 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    Score : {r['score']:.4f}")
        print(f"    Fuente: {r.get('source', 'local')}")
        print(f"    URL   : {r.get('url', r.get('arxiv_id', ''))}")
        print(f"    Texto : {r.get('abstract', '')[:150]}…")


if __name__ == "__main__":
    main()