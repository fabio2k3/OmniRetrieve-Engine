"""
chunker.py
==========
Algoritmo centralizado de fragmentación (chunking) de texto.

Extraído de pdf_extractor.py para que sea reutilizable por cualquier
cliente sin duplicar lógica.  pdf_extractor.py importa desde aquí,
por lo que el comportamiento existente no cambia.

API pública
-----------
make_chunks(text, chunk_size, overlap_sentences) -> List[str]
    Punto de entrada único.  Aplica limpieza + fragmentación.

clean_text(text) -> str
    Limpieza de texto crudo (normalización de espacios y saltos).
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Constantes (mismos valores que en pdf_extractor.py)
# ---------------------------------------------------------------------------
MIN_CHUNK_CHARS = 100   # chunks más cortos se descartan
MIN_SENT_CHARS  = 20    # oraciones muy cortas se fusionan con la siguiente

# Separador de oraciones respetando texto científico:
#   · Punto seguido de mayúscula o dígito (evita "Fig. 3", "et al.")
#   · !? siempre separan
#   · ; como frontera blanda
_SENT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z\"\'\(0-9])'
    r'|(?<=;)\s+'
)


# ---------------------------------------------------------------------------
# Limpieza de texto
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normaliza el texto crudo eliminando ruido tipográfico.

    · Colapsa más de 2 saltos de línea consecutivos.
    · Elimina líneas que solo contienen números (números de página).
    · Colapsa espacios/tabulaciones múltiples a uno solo.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# División en oraciones
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Divide un bloque de texto en oraciones usando fronteras lingüísticas.

    Fusiona oraciones muy cortas (< MIN_SENT_CHARS) con la siguiente
    para evitar chunks de una sola palabra o abreviatura suelta.
    """
    raw = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    merged: List[str] = []
    buf = ""
    for sent in raw:
        buf = (buf + " " + sent).strip() if buf else sent
        if len(buf) >= MIN_SENT_CHARS:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = (merged[-1] + " " + buf).strip()
        else:
            merged.append(buf)
    return merged


# ---------------------------------------------------------------------------
# Fragmentación en chunks
# ---------------------------------------------------------------------------

def _split_into_chunks(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Divide el texto en chunks con solapamiento semántico a nivel de oración.

    Algoritmo
    ---------
    1. Divide por párrafos (\\n\\n) — fronteras duras entre chunks.
    2. Dentro de cada párrafo extrae oraciones con _split_sentences().
    3. Acumula oraciones hasta alcanzar max_chars.
    4. Al emitir un chunk, las últimas `overlap_sentences` oraciones se
       reutilizan como prefijo del siguiente, dando contexto de transición.

    Parámetros
    ----------
    text               : texto completo del documento.
    max_chars          : tamaño máximo de cada chunk en caracteres.
    overlap_sentences  : oraciones compartidas entre chunks consecutivos
                         dentro del mismo párrafo. 0 = sin solapamiento.

    Ejemplo con overlap_sentences=2
    --------------------------------
    Oraciones: [A B C D E F G H]
    Chunk 1 -> A B C D
    Chunk 2 -> C D E F     <- C y D repetidas como contexto
    Chunk 3 -> E F G H     <- E y F repetidas como contexto
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        sentences = _split_sentences(para)
        if not sentences:
            continue

        window: List[str] = []
        window_len: int   = 0

        for sent in sentences:
            sent_len = len(sent) + 1  # +1 por el espacio de unión

            if window and window_len + sent_len > max_chars:
                # Emitir chunk actual
                candidate = " ".join(window)
                if len(candidate) >= MIN_CHUNK_CHARS:
                    chunks.append(candidate)

                # Solapamiento: conservar las últimas N oraciones
                if overlap_sentences > 0 and len(window) > overlap_sentences:
                    window     = window[-overlap_sentences:]
                    window_len = sum(len(s) + 1 for s in window)
                else:
                    window     = []
                    window_len = 0

            window.append(sent)
            window_len += sent_len

        # Emitir el último chunk del párrafo
        if window:
            candidate = " ".join(window)
            if len(candidate) >= MIN_CHUNK_CHARS:
                chunks.append(candidate)

    return chunks


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def make_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Punto de entrada único del módulo.

    Aplica limpieza al texto y lo divide en chunks usando el algoritmo
    de solapamiento semántico a nivel de oración.

    Parámetros
    ----------
    text               : texto crudo del documento.
    chunk_size         : tamaño máximo de cada chunk en caracteres.
    overlap_sentences  : oraciones de contexto compartidas entre chunks
                         consecutivos del mismo párrafo.

    Returns
    -------
    List[str]
        Lista de fragmentos de texto listos para embedding.
    """
    cleaned = clean_text(text)
    return _split_into_chunks(
        cleaned,
        max_chars=chunk_size,
        overlap_sentences=overlap_sentences,
    )
