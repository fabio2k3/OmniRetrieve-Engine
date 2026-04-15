"""
cli.py
======
Interfaz de línea de comandos interactiva del orquestador.

Contiene el bucle de input() y todas las funciones de presentación.
No tiene estado propio: recibe los callables del orquestador como
parámetros para mantener el desacoplamiento.

Comandos
--------
query <texto>   Busca los 10 artículos más relevantes.
<texto>         Atajo: texto sin prefijo se trata como query.
status          Estado actual del sistema.
index           Fuerza indexación incremental ahora.
rebuild         Fuerza reconstrucción del modelo LSI ahora.
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
    shutdown:   threading.Event,
    lsi_ready:  threading.Event,
    lsi_min_docs: int,
    fn_query:   Callable[[str], list[dict]],
    fn_status:  Callable[[], dict],
    fn_index:   Callable[[], dict],
    fn_rebuild: Callable[[], dict | None],
    fn_stop:    Callable[[], None],
) -> None:
    """
    Bucle interactivo en el hilo principal.

    Parámetros
    ----------
    shutdown     : evento de parada — cuando se activa, el bucle termina.
    lsi_ready    : evento que indica que el modelo LSI está disponible.
    lsi_min_docs : mínimo de docs para hacer rebuild (para el mensaje de error).
    fn_query     : ejecuta una query semántica y devuelve resultados.
    fn_status    : devuelve el dict de estado del sistema.
    fn_index     : ejecuta indexación incremental y devuelve stats.
    fn_rebuild   : reconstruye el modelo LSI y devuelve stats (o None).
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

        elif cmd == "query":
            if not arg:
                print("  Uso: query <texto de búsqueda>")
                continue
            _do_query(arg, lsi_ready, fn_query)

        else:
            # Texto sin prefijo → query directa
            _do_query(raw, lsi_ready, fn_query)

    print("\n  Sistema detenido. Hasta luego.\n")


def _do_query(
    text:      str,
    lsi_ready: threading.Event,
    fn_query:  Callable[[str], list[dict]],
) -> None:
    """Ejecuta una query y muestra los resultados, o avisa si el modelo no está listo."""
    if not lsi_ready.is_set():
        print("  ⏳ Modelo LSI aún no disponible. "
              "Espera a que haya suficientes documentos o usa 'rebuild'.")
        return
    results = fn_query(text)
    print_results(results, text)


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
  query <texto>   Busca los 10 artículos más relevantes
  <texto>         Atajo: cualquier texto = query directa
  status          Estado actual del sistema
  index           Fuerza indexación incremental ahora
  rebuild         Fuerza reconstrucción del modelo LSI
  help            Muestra esta ayuda
  quit / exit     Detiene el sistema y sale
""")


def print_status(s: dict) -> None:
    lsi_ok = "✔  listo" if s["lsi_model_ready"] else "⏳ no disponible aún"
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
""")


def print_results(results: list[dict], query: str) -> None:
    if not results:
        print(f"  Sin resultados para: '{query}'\n")
        return
    print(f"\n  Resultados para: '{query}'")
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
