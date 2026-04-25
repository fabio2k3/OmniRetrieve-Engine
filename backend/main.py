"""
main.py
=======
Punto de entrada principal de OmniRetrieve-Engine.

Responsabilidades
-----------------
1. Inicializar la base de datos (idempotente — seguro llamar siempre).
2. Proveer un `setup()` reutilizable que pueden llamar tanto la CLI del
   orquestador como la interfaz Streamlit antes de arrancar.
3. Exponer un bloque `__main__` para ejecutar el orquestador completo:

       python -m backend.main                   # arranca con config por defecto
       python -m backend.main --help            # muestra opciones del orquestador
"""

from __future__ import annotations

import logging
import sys

from backend.database.schema import DB_PATH, init_db

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def setup(db_path=DB_PATH, *, verbose: bool = True) -> None:
    """
    Inicializa la base de datos y registra las rutas activas.

    Es idempotente: si las tablas ya existen no hace nada dañino.
    Llámalo al inicio de cualquier módulo que necesite acceso a la BD
    sin pasar por el orquestador completo (p. ej. la interfaz Streamlit).

    Parámetros
    ----------
    db_path : ruta al archivo SQLite. Por defecto usa DB_PATH de schema.py.
    verbose : si True, imprime la ruta al inicializar.
    """
    try:
        init_db(db_path)
        if verbose:
            print(f"[OmniRetrieve] ✓ BD lista  →  {db_path}")
    except Exception as exc:
        # No lanzamos: si la BD falla la app lo manejará por su cuenta.
        log.warning("[OmniRetrieve] No se pudo inicializar la BD: %s", exc)


# ---------------------------------------------------------------------------
# Bloque __main__ — delega en el orquestador
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuración mínima de logging para el arranque
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Inicializar BD antes de pasar el control al orquestador
    setup()

    # Delegar en el orquestador principal pasando todos los argumentos
    try:
        from backend.orchestrator.main import main as orchestrator_main
        sys.exit(orchestrator_main())
    except ImportError as exc:
        log.error(
            "No se pudo importar el orquestador: %s\n"
            "Asegúrate de ejecutar desde la raíz del proyecto:\n"
            "    python -m backend.main",
            exc,
        )
        sys.exit(1)