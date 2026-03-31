"""
cli.py
======
Interfaz de línea de comandos interactiva del orquestador.

Contiene el bucle de input() y todas las funciones de presentación.
No tiene estado propio: recibe los callables del orquestador como
parámetros para mantener el desacoplamiento.

Comandos
--------
query <texto>   Busca usando el modelo LSI.
vquery <texto>  Busca usando similitud vectorial (embeddings).
<texto>         Atajo: texto sin prefijo se trata como query LSI.
status          Estado actual del sistema.
index           Fuerza indexación incremental ahora.
rebuild         Fuerza reconstrucción del modelo LSI ahora.
embed           Fuerza generación de embeddings ahora.
help            Muestra los comandos disponibles.
quit / exit     Detiene el sistema y sale.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def run_cli(
    shutdown:     threading.Event,
    lsi_ready:    threading.Event,
    lsi_min_docs: int,
    fn_query:     Callable[[str], list[dict]],
    fn_vquery:    Callable[[str], list[dict]],
    fn_status:    Callable[[], dict],
    fn_index:     Callable[[], dict],
    fn_rebuild:   Callable[[], dict | None],
    fn_embed:     Callable[[], dict],
    fn_stop:      Callable[[], None],
) -> None:
    """
    Bucle interactivo en el hilo principal.

    Parámetros
    ----------
    shutdown     : evento de parada — cuando se activa, el bucle termina.
    lsi_ready    : evento que indica que el modelo LSI está disponible.
    lsi_min_docs : mínimo de docs para hacer rebuild (para el mensaje de error).
    fn_query     : ejecuta una query semántica LSI y devuelve resultados.
    fn_vquery    : ejecuta una query vectorial y devuelve resultados.
    fn_status    : devuelve el dict de estado del sistema.
    fn_index     : ejecuta indexación incremental y devuelve stats.
    fn_rebuild   : reconstruye el modelo LSI y devuelve stats (o None).
    fn_embed     : ejecuta EmbeddingPipeline y devuelve stats.
    fn_stop      : señala la parada del sistema.
    """
    print_banner()
    print("  Escribe 'help' para ver los comandos disponibles.\n")

    while not shutdown.is_set():
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            fn_stop()
            break

        elif cmd == "help":
            print_help()

        elif cmd == "status":
            print_status(fn_status())

        elif cmd == "index":
            print("  → Forzando indexación incremental…")
            stats = fn_index()
            print(f"  ✔ docs={stats['docs_processed']}  "
                  f"términos={stats['terms_added']}  "
                  f"postings={stats['postings_added']}")

        elif cmd == "rebuild":
            s = fn_status()
            if s["docs_pdf_indexed"] < lsi_min_docs:
                print(f"  ✘ No hay suficientes documentos indexados "
                      f"(mínimo {lsi_min_docs}).")
            else:
                print("  → Forzando reconstrucción del modelo LSI…")
                stats = fn_rebuild()
                if stats:
                    print(f"  ✔ n_docs={stats['n_docs']}  "
                          f"n_terms={stats['n_terms']}  "
                          f"varianza={stats['var_explained']:.1%}")
                else:
                    print("  ✘ No fue posible reconstruir el modelo.")

        elif cmd == "embed":
            print("  → Forzando generación de embeddings…")
            stats = fn_embed()
            print(f"  ✔ chunks embebidos={stats['chunks_embedded']}")

        elif cmd == "vquery":
            if not arg:
                print("  Uso: vquery <texto de búsqueda>")
                continue
            _do_vquery(arg, fn_vquery)

        elif cmd == "query":
            if not arg:
                print("  Uso: query <texto de búsqueda>")
                continue
            _do_query(arg, lsi_ready, fn_query)

        else:
            # Texto sin prefijo → ambos motores
            _do_both(raw, lsi_ready, fn_query, fn_vquery)

    print("\n  Sistema detenido. Hasta luego.\n")


def _do_query(
    text:      str,
    lsi_ready: threading.Event,
    fn_query:  Callable[[str], list[dict]],
) -> None:
    """Ejecuta una query LSI y muestra los resultados."""
    if not lsi_ready.is_set():
        print("  ⏳ Modelo LSI aún no disponible. "
              "Espera a que haya suficientes documentos o usa 'rebuild'.")
        return
    results = fn_query(text)
    print_results(results, text, engine="LSI")


def _do_vquery(
    text:      str,
    fn_vquery: Callable[[str], list[dict]],
) -> None:
    """Ejecuta una query vectorial y muestra los resultados."""
    results = fn_vquery(text)
    if not results:
        print("  ⏳ Índice vectorial aún no disponible. "
              "Usa 'embed' para generar embeddings o espera al hilo automático.")
        return
    print_results(results, text, engine="Vector")


def _do_both(
    text:      str,
    lsi_ready: threading.Event,
    fn_query:  Callable[[str], list[dict]],
    fn_vquery: Callable[[str], list[dict]],
) -> None:
    """
    Consulta ambos motores y muestra sus resultados en paralelo.
    Si alguno no está listo, muestra su resultado parcial con aviso.
    """
    # LSI
    if not lsi_ready.is_set():
        print("  ⏳ [LSI] Modelo aún no disponible.")
    else:
        lsi_results = fn_query(text)
        print_results(lsi_results, text, engine="LSI")

    # Vectorial
    vec_results = fn_vquery(text)
    if not vec_results:
        print("  ⏳ [Vector] Índice aún no disponible. "
              "Usa 'embed' o espera al hilo automático.")
    else:
        print_results(vec_results, text, engine="Vector")


# ---------------------------------------------------------------------------
# Funciones de presentación
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print("\n" + "═" * 58)
    print("  OmniRetrieve-Engine — Orquestador")
    print("═" * 58)


def print_help() -> None:
    print("""
  Comandos disponibles
  ─────────────────────────────────────────────────────
  query  <texto>  Busca con el modelo LSI
  vquery <texto>  Busca con similitud vectorial (ChromaDB)
  <texto>         Atajo: consulta LSI + vectorial a la vez
  status          Estado actual del sistema
  index           Fuerza indexación incremental ahora
  rebuild         Fuerza reconstrucción del modelo LSI
  embed           Fuerza generación de embeddings ahora
  help            Muestra esta ayuda
  quit / exit     Detiene el sistema y sale
""")


def print_status(s: dict) -> None:
    lsi_ok = "✔  listo" if s["lsi_model_ready"] else "⏳ no disponible aún"
    vec_ok = "✔  listo" if s.get("vector_index_ready") else "⏳ no disponible aún"
    print(f"""
  Estado del sistema  ({s['timestamp']})
  ─────────────────────────────────────────────────────
  Documentos en BD        {s['docs_total']}
  PDFs descargados        {s['docs_pdf_indexed']}
  PDFs pendientes         {s['docs_pdf_pending']}
  Pendientes de indexar   {s['docs_not_in_index']}
  Vocabulario (terms)     {s['vocab_size']}
  Postings totales        {s['total_postings']}
  Docs en modelo LSI      {s['lsi_docs_in_model']}
  Modelo LSI              {lsi_ok}
  Chunks totales          {s.get('total_chunks', '—')}
  Chunks embebidos        {s.get('embedded_chunks', '—')}
  Chunks en índice vec.   {s.get('vector_chunks_loaded', '—')}
  Índice vectorial        {vec_ok}
""")


def print_results(results: list[dict], query: str, engine: str = "LSI") -> None:
    if not results:
        print(f"  Sin resultados para: '{query}'\n")
        return
    print(f"\n  Resultados [{engine}] para: '{query}'")
    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        print(f"  {i:2}. [{r['arxiv_id']}]  score={r['score']:.4f}")
        print(f"      {r['title']}")
        if r.get("authors"):
            print(f"      {r['authors'][:60]}")
        if r.get("abstract"):
            abstract = r["abstract"][:120].replace("\n", " ")
            print(f"      {abstract}…")
        print()