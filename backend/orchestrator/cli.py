"""
cli.py
======
Interfaz de línea de comandos interactiva del orquestador.

Contiene el bucle de input() y todas las funciones de presentación.
No tiene estado propio: recibe los callables del orquestador como
parámetros para mantener el desacoplamiento.

Comandos
--------
query <texto>    Busca los 10 artículos más relevantes (LSI local).
wsearch <texto>  Búsqueda con fallback web automático si el resultado local es pobre.
<texto>          Atajo: texto sin prefijo se trata como query directa.
status           Estado actual del sistema.
index            Fuerza indexación BM25 incremental ahora.
rebuild          Fuerza reconstrucción del modelo LSI ahora.
help             Muestra los comandos disponibles.
quit / exit      Detiene el sistema y sale.
"""

from __future__ import annotations

import threading
from typing import Callable


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def run_cli(
    shutdown:     threading.Event,
    lsi_ready:    threading.Event,
    lsi_min_docs: int,
    fn_query:     Callable[[str], list[dict]],
    fn_query_web: Callable[[str], dict],
    fn_status:    Callable[[], dict],
    fn_index:     Callable[[], dict],
    fn_rebuild:   Callable[[], dict | None],
    fn_stop:      Callable[[], None],
) -> None:
    """
    Bucle interactivo en el hilo principal.

    Parámetros
    ----------
    shutdown     : evento de parada — cuando se activa el bucle termina.
    lsi_ready    : evento que indica que el modelo LSI está disponible.
    lsi_min_docs : mínimo de docs para hacer rebuild (para el mensaje de aviso).
    fn_query     : ejecuta una query LSI local y devuelve resultados.
    fn_query_web : ejecuta query LSI + fallback web y devuelve el dict completo.
    fn_status    : devuelve el dict de estado del sistema.
    fn_index     : ejecuta indexación BM25 incremental y devuelve stats.
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
            print("  → Forzando indexación BM25 incremental…")
            stats = fn_index()
            print(
                f"  ✔ docs={stats['docs_processed']}  "
                f"términos={stats['terms_added']}  "
                f"postings={stats['postings_added']}"
            )

        elif cmd == "rebuild":
            s = fn_status()
            if s["docs_pdf_indexed"] < lsi_min_docs:
                print(
                    f"  ✘ No hay suficientes documentos indexados "
                    f"(mínimo {lsi_min_docs})."
                )
            else:
                print("  → Forzando reconstrucción del modelo LSI…")
                stats = fn_rebuild()
                if stats:
                    print(
                        f"  ✔ n_docs={stats['n_docs']}  "
                        f"n_terms={stats['n_terms']}  "
                        f"varianza={stats['var_explained']:.1%}"
                    )
                else:
                    print("  ✘ No fue posible reconstruir el modelo.")

        elif cmd == "query":
            if not arg:
                print("  Uso: query <texto de búsqueda>")
                continue
            _do_local_query(arg, lsi_ready, fn_query)

        elif cmd in ("wsearch", "web"):
            if not arg:
                print("  Uso: wsearch <texto de búsqueda>")
                continue
            _do_web_query(arg, lsi_ready, fn_query_web)

        else:
            # Texto sin prefijo → query directa local
            _do_local_query(raw, lsi_ready, fn_query)

    print("\n  Sistema detenido. Hasta luego.\n")


# ---------------------------------------------------------------------------
# Ejecución de queries
# ---------------------------------------------------------------------------

def _do_local_query(
    text:      str,
    lsi_ready: threading.Event,
    fn_query:  Callable[[str], list[dict]],
) -> None:
    """Ejecuta una query LSI local y muestra resultados."""
    if not lsi_ready.is_set():
        print(
            "  ⏳ Modelo LSI aún no disponible. "
            "Espera a que haya suficientes documentos o usa 'rebuild'."
        )
        return
    results = fn_query(text)
    print_results(results, text)


def _do_web_query(
    text:         str,
    lsi_ready:    threading.Event,
    fn_query_web: Callable[[str], dict],
) -> None:
    """Ejecuta query LSI + búsqueda web y muestra resultados combinados."""
    if not lsi_ready.is_set():
        print(
            "  ⏳ Modelo LSI aún no disponible — la búsqueda web funcionará "
            "sin resultados locales."
        )
    output = fn_query_web(text)
    print_web_results(output)


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
  query <texto>    Busca los 10 artículos más relevantes (LSI local)
  wsearch <texto>  Búsqueda con fallback web automático
  <texto>          Atajo: cualquier texto = query directa
  status           Estado actual del sistema
  index            Fuerza indexación BM25 incremental ahora
  rebuild          Fuerza reconstrucción del modelo LSI
  help             Muestra esta ayuda
  quit / exit      Detiene el sistema y sale
""")


def print_status(s: dict) -> None:
    lsi_ok   = "✔  listo"          if s["lsi_model_ready"] else "⏳ no disponible aún"
    faiss_ok = "✔  listo"          if s["faiss_ready"]     else "⏳ no disponible aún"
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
  Chunks totales          {s['total_chunks']}
  Chunks embebidos        {s['embedded_chunks']}
  Chunks pendientes       {s['pending_chunks']}
  Vectores FAISS          {s['faiss_vectors']}
  Índice FAISS            {faiss_ok}
  Modelo embedding        {s['embed_model']}
  Web fallback umbral     score≥{s['web_threshold']} (min {s['web_min_docs']} doc)
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
            print(f"      {r['abstract'][:120].replace(chr(10), ' ')}…")
        print()


def print_web_results(output: dict) -> None:
    """Muestra los resultados combinados de una búsqueda con fallback web."""
    activated = output.get("web_activated", False)
    results   = output.get("results", [])
    reason    = output.get("reason", "")

    print(f"\n  Búsqueda: '{output.get('query', '')}'")
    print(f"  Razón    : {reason}")
    if activated:
        web_n = len(output.get("web_results", []))
        idx_n = output.get("indexed", 0)
        print(f"  Web      : ✔ activada — {web_n} resultados web obtenidos "
              f"({idx_n} indexados)")
    else:
        print("  Web      : — no fue necesaria (resultados locales suficientes)")

    if not results:
        print("  Sin resultados.\n")
        return

    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        source = r.get("source", "local")
        badge  = "[WEB]  " if source in ("web", "web_fallback") else "[LOCAL]"
        label  = r.get("url") or r.get("arxiv_id", "")
        print(f"  {i:2}. {badge}  score={r['score']:.4f}  {label}")
        print(f"       {r['title']}")
        snippet = r.get("abstract", "")[:120].replace("\n", " ")
        if snippet:
            print(f"       {snippet}…")
        print()
