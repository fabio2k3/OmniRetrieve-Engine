"""
rag/schema.py
=============
Tipos de datos del dataset de evaluación RAG.

Diseño deliberadamente simple: solo queries, sin ground truth de chunks.
El sistema evalúa si la RESPUESTA es útil para el usuario, no si recuperó
el chunk correcto — eso es responsabilidad del módulo de retrieval.

Clases
------
RAGQuery    — una consulta individual con su ID y metadatos opcionales.
RAGQuerySet — colección de consultas con I/O JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RAGQuery:
    """
    Una consulta individual para evaluar el pipeline RAG.

    Campos
    ------
    query_id : identificador único (ej. "q_0001").
    query    : texto de la consulta tal como la escribiría un usuario.
    metadata : campo libre — puede incluir el chunk origen, topic, etc.
               No se usa en la evaluación, solo para trazabilidad.
    """
    query_id: str
    query:    str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RAGQuery":
        return cls(**d)


@dataclass
class RAGQuerySet:
    """
    Colección de RAGQuery con metadatos de generación.

    Campos
    ------
    queries       : lista de consultas.
    generated_at  : timestamp ISO de creación.
    generator_cfg : parámetros usados para generar las consultas.
    """
    queries:       list[RAGQuery]
    generated_at:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    generator_cfg: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.queries)

    def __repr__(self) -> str:
        return (
            f"RAGQuerySet(n={len(self)}, "
            f"generated_at={self.generated_at!r})"
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at":  self.generated_at,
            "generator_cfg": self.generator_cfg,
            "n_queries":     len(self.queries),
            "queries":       [q.to_dict() for q in self.queries],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "RAGQuerySet":
        payload  = json.loads(Path(path).read_text(encoding="utf-8"))
        queries  = [RAGQuery.from_dict(q) for q in payload.pop("queries")]
        payload.pop("n_queries", None)
        return cls(queries=queries, **payload)

    @classmethod
    def from_text_file(cls, path: Path) -> "RAGQuerySet":
        """
        Carga consultas desde un fichero de texto plano (una por línea).

        Útil para datasets creados manualmente o importados desde otra fuente.
        Las líneas vacías y las que empiezan por '#' se ignoran.
        """
        lines = [
            l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        queries = [
            RAGQuery(query_id=f"q_{i:04d}", query=line)
            for i, line in enumerate(lines)
        ]
        return cls(
            queries=queries,
            generator_cfg={"source": str(path), "type": "manual"},
        )