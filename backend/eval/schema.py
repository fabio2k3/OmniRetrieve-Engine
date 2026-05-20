"""
schema.py
=========
Tipos de datos centrales del módulo de evaluación.

Clases
------
EvalCase     — un caso de test individual (query + respuesta esperada).
EvalDataset  — colección de casos con metadatos de generación.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


CaseType = Literal["exact", "semantic", "generated"]
"""
Tipos de caso de evaluación:

exact     — query es un fragmento literal del chunk.
            Prueba precisión del retrieval léxico.

semantic  — query es una paráfrasis del fragmento (LLM).
            Prueba el retrieval semántico (FAISS/dense).

generated — query es una pregunta real generada por LLM a partir del chunk.
            Prueba el sistema completo tal como lo usaría un usuario real.
            Es el tipo más representativo para evaluar el pipeline RAG.
"""


@dataclass
class EvalCase:
    """
    Un caso de evaluación individual.

    Campos
    ------
    case_id             : identificador único (ej. "generated_0042").
    case_type           : "exact" | "semantic" | "generated".
    query               : texto que se enviará al retriever / RAG pipeline.
    expected_chunk_id   : id (PK) del chunk que debe aparecer en los resultados.
    expected_arxiv_id   : arxiv_id del documento fuente.
    expected_chunk_index: índice del chunk dentro del documento.
    source_text         : texto íntegro del chunk de referencia.
    fragment_used       : semilla usada para generar la query.
                          exact    → el propio fragmento.
                          semantic → fragmento que se parafraseó.
                          generated → extracto del chunk enviado al LLM.
    paraphrase_model    : modelo Ollama usado (None para exact).
    metadata            : campo libre (título del doc, etc.).
    """

    case_id:              str
    case_type:            CaseType
    query:                str
    expected_chunk_id:    int
    expected_arxiv_id:    str
    expected_chunk_index: int
    source_text:          str
    fragment_used:        str
    paraphrase_model:     str | None = None
    metadata:             dict       = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(**d)


@dataclass
class EvalDataset:
    """
    Colección de EvalCase con metadatos de generación.
    """

    cases:         list[EvalCase]
    generated_at:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    db_path:       str = ""
    n_exact:       int = 0
    n_semantic:    int = 0
    n_generated:   int = 0
    generator_cfg: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_exact     = sum(1 for c in self.cases if c.case_type == "exact")
        self.n_semantic  = sum(1 for c in self.cases if c.case_type == "semantic")
        self.n_generated = sum(1 for c in self.cases if c.case_type == "generated")

    def exact_cases(self)     -> list[EvalCase]:
        return [c for c in self.cases if c.case_type == "exact"]

    def semantic_cases(self)  -> list[EvalCase]:
        return [c for c in self.cases if c.case_type == "semantic"]

    def generated_cases(self) -> list[EvalCase]:
        return [c for c in self.cases if c.case_type == "generated"]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at":  self.generated_at,
            "db_path":       self.db_path,
            "n_exact":       self.n_exact,
            "n_semantic":    self.n_semantic,
            "n_generated":   self.n_generated,
            "generator_cfg": self.generator_cfg,
            "cases":         [c.to_dict() for c in self.cases],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = [EvalCase.from_dict(c) for c in payload.pop("cases")]
        # compatibilidad con datasets anteriores sin n_generated
        payload.setdefault("n_generated", 0)
        return cls(cases=cases, **payload)

    def __len__(self) -> int:
        return len(self.cases)

    def __repr__(self) -> str:
        return (
            f"EvalDataset(total={len(self)}, "
            f"exact={self.n_exact}, semantic={self.n_semantic}, "
            f"generated={self.n_generated}, "
            f"generated_at={self.generated_at!r})"
        )
