"""
eval
====
Módulo de evaluación de calidad del sistema RAG.

Pipeline de evaluación completo
---------------------------------
1. Generar dataset  → DatasetGenerator  (eval.dataset_generator)
2. Evaluar retrieval → EvalRunner       (eval.retrieval)
3. Evaluar RAG      → RAGEvalRunner     (eval.rag)
4. Comparar corridas → compare_reports  (eval.compare)

Subpaquetes
-----------
eval.retrieval  — métricas de retrieval: Hit@K, MRR, NDCG.
eval.rag        — evaluación end-to-end con LLM-as-judge (faithfulness,
                  answer relevance, context relevance).
eval.compare    — comparador de reportes entre corridas (detección de
                  regresiones).

Módulos raíz
------------
schema            — EvalCase, EvalDataset.
paraphraser       — paráfrasis semánticas con Ollama.
dataset_generator — generación automática de datasets.
generate_dataset  — CLI de generación.
"""

from .schema import EvalCase, EvalDataset
from .paraphraser import Paraphraser
from .dataset_generator import DatasetGenerator

# Subpaquetes — importación explícita para que IDEs los descubran
from . import retrieval  # noqa: F401
from . import rag        # noqa: F401
from . import compare    # noqa: F401

__all__ = [
    # Tipos de datos
    "EvalCase",
    "EvalDataset",
    # Generación de dataset
    "Paraphraser",
    "DatasetGenerator",
    # Subpaquetes
    "retrieval",
    "rag",
    "compare",
]
