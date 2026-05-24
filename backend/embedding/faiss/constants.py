"""
faiss/constants.py
==================
Constantes internas del subpaquete FAISS.

Solo contiene valores que son verdaderamente fijos e internos al algoritmo
FAISS (heurísticas de entrenamiento).

Los parámetros configurables por el usuario (nlist, m, nbits, nprobe,
rebuild_every) se han centralizado en OrchestratorConfig como embed_nlist,
embed_m, embed_nbits, embed_nprobe y embed_rebuild_every.
FaissIndexManager los recibe como parámetros del constructor desde _faiss.py.
"""

# Número mínimo de vectores por celda de Voronoi que FAISS necesita para
# entrenar de forma estable (heurística interna de K-means).
# El umbral real es: nlist * MIN_TRAIN_FACTOR
MIN_TRAIN_FACTOR = 39