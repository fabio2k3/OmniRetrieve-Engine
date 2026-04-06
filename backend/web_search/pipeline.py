"""
pipeline.py
===========
Orquestador del módulo de búsqueda web.

Responsabilidad: recibir los resultados del retriever, decidir si son
suficientes y, si no lo son, buscar en la web, guardar E INDEXAR los
resultados automáticamente, y devolver la lista combinada para RAG.

Flujo
-----
    retriever_results (list[dict] con score)
            ↓
    SufficiencyChecker → ¿suficiente?
        ├── SÍ → devuelve retriever_results sin cambios
        └── NO → WebSearcher.search(query)       ← Tavily + fallback DuckDuckGo
                  → save_web_results()            ← guarda en BD
                  → _index_web_results()          ← indexa automáticamente ← NUEVO
                  → combina local + web
                  → devuelve lista combinada

Uso programático
----------------
    from backend.web_search.pipeline import WebSearchPipeline

    pipeline = WebSearchPipeline()  # lee TAVILY_API_KEY del .env
    results  = pipeline.run(
        query="fairness in machine learning",
        retriever_results=lsi_results,
    )

Uso CLI
-------
    python -m backend.web_search.pipeline --query "fairness in ML"
    python -m backend.web_search.pipeline --query "fairness in ML" --api-key tvly-...
"""

from __future__ import annotations

import argparse
import logging
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
    api_key       : clave de API de Tavily. Si no se pasa, se lee del .env.
    threshold     : score mínimo para considerar un doc relevante (default: 0.15)
    min_docs      : docs mínimos que deben superar el threshold (default: 1)
    max_results   : máximo de resultados a pedir a Tavily (default: 5)
    search_depth  : "basic" | "advanced" (default: "basic")
    use_fallback  : usar DuckDuckGo si Tavily falla (default: True)
    auto_index    : indexar automáticamente los docs web guardados (default: True)
    db_path       : ruta a la BD SQLite
    """

    def __init__(
        self,
        api_key: str | None = None,
        threshold: float = 0.15,
        min_docs: int = 1,
        max_results: int = 5,
        search_depth: str = "basic",
        use_fallback: bool = True,
        auto_index: bool = True,
        db_path: Path = DB_PATH,
    ) -> None:
        self.searcher   = WebSearcher(
            api_key=api_key,
            max_results=max_results,
            search_depth=search_depth,
            use_fallback=use_fallback,
        )
        self.checker    = SufficiencyChecker(
            threshold=threshold,
            min_docs=min_docs,
        )
        self.auto_index = auto_index
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
            "web_results"    : resultados web normalizados
            "reason"         : explicación de la decisión de suficiencia
            "query"          : query original
            "indexed"        : número de docs web indexados (0 si auto_index=False)
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
                "indexed":       0,
            }

        # Información insuficiente → activar búsqueda web
        log.info("[WebSearch] Activando búsqueda web para: '%s'", query)
        web_results = self.searcher.search(query)

        saved   = 0
        indexed = 0

        if web_results:
            # Guardar en BD
            saved = save_web_results(
                query=query,
                results=web_results,
                db_path=self.db_path,
            )
            log.info("[WebSearch] %d documentos web guardados en BD.", saved)

            # Indexar automáticamente si hay docs nuevos
            if self.auto_index and saved > 0:
                indexed = self._index_web_results()

        # Normalizar al formato del retriever
        web_normalized = [
            {
                "score":    r["score"],
                "arxiv_id": "",
                "title":    r["title"],
                "authors":  "Web Search",
                "abstract": r["content"][:300],
                "url":      r["url"],
                "source":   r.get("source", "web"),
            }
            for r in web_results
        ]

        # Marcar fuente en resultados locales
        for r in retriever_results:
            r.setdefault("source", "local")

        # Combinar: primero locales, luego web
        combined = retriever_results + web_normalized

        log.info(
            "[WebSearch] Combinados: %d locales + %d web = %d total. "
            "Docs indexados: %d.",
            len(retriever_results), len(web_normalized), len(combined), indexed,
        )

        return {
            "results":       combined,
            "web_activated": True,
            "web_results":   web_normalized,
            "reason":        reason,
            "query":         query,
            "indexed":       indexed,
        }

    def _index_web_results(self) -> int:
        """
        Indexa los documentos web recién guardados en la BD.

        Llama al IndexingPipeline del módulo de indexación para procesar
        solo los documentos nuevos (indexación incremental).

        Devuelve el número de documentos indexados, 0 si falla.
        """
        try:
            from backend.indexing.pipeline import IndexingPipeline
            log.info("[WebSearch] Indexando documentos web nuevos…")
            pipeline = IndexingPipeline(db_path=self.db_path, field="both")
            stats    = pipeline.run(reindex=False)
            indexed  = stats.get("docs_processed", 0)
            log.info("[WebSearch] %d documentos web indexados.", indexed)
            return indexed
        except Exception as exc:
            log.warning(
                "[WebSearch] No se pudieron indexar los docs web: %s", exc
            )
            return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniRetrieve — Módulo de Búsqueda Web",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="Consulta de búsqueda.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key de Tavily. Si no se pasa, se lee del archivo .env.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Score mínimo para considerar un doc relevante.",
    )
    parser.add_argument(
        "--min-docs", type=int, default=1,
        help="Docs mínimos que deben superar el threshold.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Máximo de resultados web a obtener.",
    )
    parser.add_argument(
        "--depth", type=str, default="basic",
        choices=["basic", "advanced"],
        help="Profundidad de búsqueda Tavily.",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Desactivar fallback DuckDuckGo.",
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help="No indexar automáticamente los docs web guardados.",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Ruta a la base de datos SQLite.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pipeline = WebSearchPipeline(
        api_key=args.api_key,
        threshold=args.threshold,
        min_docs=args.min_docs,
        max_results=args.top,
        search_depth=args.depth,
        use_fallback=not args.no_fallback,
        auto_index=not args.no_index,
        db_path=args.db,
    )

    output = pipeline.run(
        query=args.query,
        retriever_results=[],
    )

    print(f"\n{'='*60}")
    print(f"Query         : {output['query']}")
    print(f"Web activada  : {output['web_activated']}")
    print(f"Razón         : {output['reason']}")
    print(f"Total results : {len(output['results'])}")
    print(f"Docs indexados: {output['indexed']}")
    print(f"{'='*60}")
    for i, r in enumerate(output["results"], 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    Score  : {r['score']:.4f}")
        print(f"    Fuente : {r.get('source', 'local')}")
        print(f"    URL    : {r.get('url', r.get('arxiv_id', ''))}")
        print(f"    Texto  : {r.get('abstract', '')[:150]}…")


if __name__ == "__main__":
    main()