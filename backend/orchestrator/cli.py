"""
cli.py
======
Interfaz de línea de comandos interactiva del orquestador.

Comandos
--------
answer <pregunta>  Pipeline completo: QRF expand → Hybrid → Web → Rerank → RAG.
query <texto>      Búsqueda LSI local.
wsearch <texto>    Búsqueda LSI + fallback web.
qrf <texto>        Búsqueda QRF standalone (expand + BRF + MMR).
rag <texto>        Retrieval denso RAG sin generación (chunks).
ask <texto>        Pipeline RAG standalone: retrieval denso + respuesta LLM.
<texto>            Atajo: texto sin prefijo = pipeline unificado (answer).
status             Estado del sistema.
rebuild            Fuerza reconstrucción del modelo LSI.
help               Muestra los comandos.
quit / exit        Detiene el sistema.
"""

from __future__ import annotations

import threading
from typing import Callable


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def run_cli(
    shutdown:        threading.Event,
    lsi_ready:       threading.Event,
    qrf_ready:       threading.Event,
    rag_ready:       threading.Event,
    pipeline_ready:  threading.Event,
    lsi_min_docs:    int,
    fn_pipeline_ask: Callable[[str], dict],
    fn_query:        Callable[[str], list[dict]],
    fn_query_web:    Callable[[str], dict],
    fn_qrf_search:   Callable[[str], list[dict]],
    fn_rag_search:   Callable[[str], list[dict]],
    fn_rag_ask:      Callable[[str], dict],
    fn_status:       Callable[[], dict],
    fn_index:        Callable[[], dict],
    fn_rebuild:      Callable[[], dict | None],
    fn_stop:         Callable[[], None],
) -> None:
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

        elif cmd == "rebuild":
            s = fn_status()
            if s["lsi_docs_in_model"] < lsi_min_docs and not lsi_ready.is_set():
                print(f"  ✘ No hay suficientes documentos indexados (mínimo {lsi_min_docs}).")
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

        elif cmd == "index":
            print("  → Forzando indexación BM25 incremental…")
            stats = fn_index()
            print(
                f"  ✔ docs={stats['docs_processed']}  "
                f"términos={stats['terms_added']}  "
                f"postings={stats['postings_added']}"
            )

        elif cmd == "answer":
            if not arg:
                print("  Uso: answer <pregunta>")
                continue
            _do_pipeline_ask(arg, pipeline_ready, fn_pipeline_ask)

        elif cmd == "query":
            if not arg:
                print("  Uso: query <texto>")
                continue
            _do_local_query(arg, lsi_ready, fn_query)

        elif cmd in ("wsearch", "web"):
            if not arg:
                print("  Uso: wsearch <texto>")
                continue
            _do_web_query(arg, lsi_ready, fn_query_web)

        elif cmd == "qrf":
            if not arg:
                print("  Uso: qrf <texto>")
                continue
            _do_qrf_query(arg, qrf_ready, fn_qrf_search)

        elif cmd == "rag":
            if not arg:
                print("  Uso: rag <texto>")
                continue
            _do_rag_search(arg, rag_ready, fn_rag_search)

        elif cmd == "ask":
            if not arg:
                print("  Uso: ask <pregunta>")
                continue
            _do_rag_ask(arg, rag_ready, fn_rag_ask)

        else:
            # Texto sin prefijo → pipeline unificado
            _do_pipeline_ask(raw, pipeline_ready, fn_pipeline_ask)

    print("\n  Sistema detenido. Hasta luego.\n")


# ---------------------------------------------------------------------------
# Ejecución de cada modo
# ---------------------------------------------------------------------------

def _do_pipeline_ask(
    text:            str,
    pipeline_ready:  threading.Event,
    fn_pipeline_ask: Callable[[str], dict],
) -> None:
    if not pipeline_ready.is_set():
        print(
            "  ⏳ Pipeline unificado aún no disponible. "
            "Espera a que FAISS y LSI estén listos."
        )
        return
    print(f"  → Procesando: '{text}'…")
    result = fn_pipeline_ask(text)
    print_pipeline_result(result)


def _do_local_query(text, lsi_ready, fn_query):
    if not lsi_ready.is_set():
        print("  ⏳ Modelo LSI aún no disponible.")
        return
    print_results(fn_query(text), text)


def _do_web_query(text, lsi_ready, fn_query_web):
    if not lsi_ready.is_set():
        print("  ⏳ Modelo LSI aún no disponible.")
    print_web_results(fn_query_web(text))


def _do_qrf_query(text, qrf_ready, fn_qrf_search):
    if not qrf_ready.is_set():
        print("  ⏳ Pipeline QRF aún no disponible.")
        return
    print_qrf_results(fn_qrf_search(text), text)


def _do_rag_search(text, rag_ready, fn_rag_search):
    if not rag_ready.is_set():
        print("  ⏳ Pipeline RAG aún no disponible.")
        return
    print_rag_search_results(fn_rag_search(text), text)


def _do_rag_ask(text, rag_ready, fn_rag_ask):
    if not rag_ready.is_set():
        print("  ⏳ Pipeline RAG aún no disponible.")
        return
    print(f"  → Ejecutando RAG ask: '{text}'…")
    print_rag_ask_result(fn_rag_ask(text))


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print("\n" + "═" * 58)
    print("  OmniRetrieve-Engine — Orquestador")
    print("═" * 58)


def print_help() -> None:
    print("""
  Comandos disponibles
  ─────────────────────────────────────────────────────────
  answer <pregunta>  Pipeline completo (RECOMENDADO)
                     QRF expand → Hybrid → Web → Rerank → RAG
  <texto>            Atajo: texto sin prefijo = answer
  ─────────────────────────────────────────────────────────
  query <texto>      Búsqueda LSI local
  wsearch <texto>    Búsqueda LSI + fallback web
  qrf <texto>        Búsqueda QRF standalone (expand + BRF + MMR)
  rag <texto>        Retrieval denso RAG sin generación
  ask <texto>        Pipeline RAG standalone (retrieval + LLM)
  ─────────────────────────────────────────────────────────
  status             Estado actual del sistema
  index              Fuerza indexación BM25 incremental ahora
  rebuild            Fuerza reconstrucción del modelo LSI
  help               Muestra esta ayuda
  quit / exit        Detiene el sistema
""")


def print_status(s: dict) -> None:
    def _flag(key): return "✔  listo" if s.get(key) else "⏳ no disponible"
    print(f"""
  Estado del sistema  ({s['timestamp']})
  ─────────────────────────────────────────────────────────
  Documentos en BD        {s['docs_total']}
  PDFs descargados        {s['docs_pdf_indexed']}
  PDFs pendientes         {s['docs_pdf_pending']}
  Vocabulario (terms)     {s['vocab_size']}
  Postings totales        {s['total_postings']}
  Docs en modelo LSI      {s['lsi_docs_in_model']}
  Modelo LSI              {_flag('lsi_model_ready')}
  Chunks totales          {s['total_chunks']}
  Chunks embebidos        {s['embedded_chunks']}
  Chunks pendientes       {s['pending_chunks']}
  Vectores FAISS          {s['faiss_vectors']}
  Índice FAISS            {_flag('faiss_ready')}
  Pipeline QRF            {_flag('qrf_ready')}
  Pipeline RAG            {_flag('rag_ready')}
  Pipeline unificado      {_flag('pipeline_ready')}
  Modelo embedding        {s['embed_model']}
  Web umbral              score≥{s['web_threshold']} (min {s['web_min_docs']} doc)
""")


def print_pipeline_result(result: dict) -> None:
    """Muestra el resultado del pipeline unificado."""
    if result.get("error"):
        print(f"  [pipeline] Error: {result['error']}\n")
        return

    expanded = result.get("expanded_terms", [])
    web      = result.get("web_activated", False)

    print(f"\n  {'─' * 54}")
    if expanded:
        print(f"  Expansión LCE : +{', '.join(expanded)}")
    print(f"  Web activada  : {'✔' if web else '—'}")
    print(f"  {'─' * 54}")

    answer = result.get("answer", "").strip()
    if answer:
        for line in answer.split("\n"):
            print(f"  {line}")
    else:
        print("  (Sin respuesta generada)")

    sources = result.get("sources", [])
    if sources:
        print(f"\n  Fuentes ({len(sources)}):")
        for src in sources:
            label = src.get("title") or src.get("arxiv_id", "")
            score = src.get("score", 0.0)
            stype = src.get("score_type", "")
            print(f"    [{src.get('citation','?')}] {label}  ({stype}: {score:.4f})")
    print()


def print_results(results: list[dict], query: str) -> None:
    if not results:
        print(f"  Sin resultados para: '{query}'\n")
        return
    print(f"\n  Resultados LSI para: '{query}'")
    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        print(f"  {i:2}. [{r['arxiv_id']}]  score={r['score']:.4f}")
        print(f"       {r['title']}")
        if r.get("abstract"):
            print(f"       {r['abstract'][:120].replace(chr(10), ' ')}…")
        print()


def print_qrf_results(results: list[dict], query: str) -> None:
    if not results:
        print(f"  [QRF] Sin resultados para: '{query}'\n")
        return
    expanded = results[0].get("expanded_terms", []) if results else []
    if expanded:
        print(f"\n  [QRF] Expansión LCE: {', '.join(expanded)}")
    print(f"\n  [QRF] Resultados para: '{query}'")
    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        mmr = f"  mmr={r['mmr_score']:.4f}" if r.get("mmr_score") is not None else ""
        print(f"  {i:2}. [{r['arxiv_id']}]  score={r['score']:.4f}{mmr}")
        print(f"       {r.get('title', '')}")
        snippet = (r.get("text") or "")[:120].replace("\n", " ")
        if snippet:
            print(f"       {snippet}…")
        print()


def print_rag_search_results(results: list[dict], query: str) -> None:
    if not results:
        print(f"  [RAG] Sin resultados para: '{query}'\n")
        return
    print(f"\n  [RAG] Chunks para: '{query}'")
    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        print(
            f"  {i:2}. [{r.get('arxiv_id', '')}]  "
            f"chunk={r.get('chunk_index', '')}  "
            f"score={r.get('score', 0):.4f}  [{r.get('score_type', '')}]"
        )
        print(f"       {r.get('title', '')}")
        snippet = (r.get("text") or "")[:120].replace("\n", " ")
        if snippet:
            print(f"       {snippet}…")
        print()


def print_rag_ask_result(output: dict) -> None:
    if output.get("error"):
        print(f"  [RAG] Error: {output['error']}\n")
        return
    print(f"\n  [RAG] Respuesta para: '{output.get('query', '')}'")
    print("  " + "─" * 54)
    answer = output.get("answer", "").strip()
    if answer:
        for line in answer.split("\n"):
            print(f"  {line}")
    else:
        print("  (Sin respuesta generada)")
    sources = output.get("sources", [])
    if sources:
        print(f"\n  Fuentes ({len(sources)}):")
        for src in sources:
            print(f"    • {src.get('title') or src.get('arxiv_id', '')}")
    print()


def print_web_results(output: dict) -> None:
    activated = output.get("web_activated", False)
    results   = output.get("results", [])
    reason    = output.get("reason", "")
    print(f"\n  Búsqueda: '{output.get('query', '')}'")
    print(f"  Razón    : {reason}")
    print(f"  Web      : {'✔ activada' if activated else '— no necesaria'}")
    if not results:
        print("  Sin resultados.\n")
        return
    print("  " + "─" * 54)
    for i, r in enumerate(results, 1):
        badge = "[WEB]  " if r.get("source") in ("web", "web_fallback") else "[LOCAL]"
        label = r.get("url") or r.get("arxiv_id", "")
        print(f"  {i:2}. {badge}  score={r['score']:.4f}  {label}")
        print(f"       {r['title']}")
        snippet = r.get("abstract", "")[:120].replace("\n", " ")
        if snippet:
            print(f"       {snippet}…")
        print()
