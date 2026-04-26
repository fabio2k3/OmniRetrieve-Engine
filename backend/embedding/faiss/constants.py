"""
faiss/constants.py
==================
Constantes del subpaquete FAISS.

Centraliza todos los valores por defecto del índice para que ningún otro
módulo los duplique o hardcodee.
"""

# Número mínimo de vectores por celda de Voronoi que FAISS necesita para
# entrenar de forma estable (heurística interna de K-means).
# El umbral real es: nlist * _MIN_TRAIN_FACTOR
MIN_TRAIN_FACTOR = 39

# Parámetros por defecto del índice IndexIVFPQ
DEFAULT_NLIST  = 100   # celdas de Voronoi
DEFAULT_M      = 8     # subvectores PQ  (384 / 8 = 48 → válido para dim=384)
DEFAULT_NBITS  = 8     # bits por código (256 centroides/subvector)
DEFAULT_NPROBE = 10    # celdas inspeccionadas en cada búsqueda

# Cada cuántos chunks añadidos se dispara una reconstrucción completa
DEFAULT_REBUILD_EVERY = 10_000
