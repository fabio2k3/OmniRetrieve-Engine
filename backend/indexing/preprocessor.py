"""
preprocessor.py
===============
Pipeline de limpieza y tokenización de texto para el módulo de indexación.

Recibir texto crudo y devolver una lista de tokens
normalizados, listos para ser contados por el indexador TF-IDF.
"""

from __future__ import annotations

import logging
import re
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

log = logging.getLogger(__name__)

class TextPreprocessor:
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation + "–—\u201c\u201d\u2018\u2019")
    _URL_RE      = re.compile(r"https?://\S+|www\.\S+")
    _LATEX_RE    = re.compile(r"\$[^$]*\$|\\\w+\{[^}]*\}")
    _DIGIT_RE    = re.compile(r"\b\d+\b")

    def __init__(self, use_stemming: bool = False, min_token_len: int = 3, language: str = "english",) -> None:
        self.use_stemming  = use_stemming and _HAS_NLTK
        self.min_token_len = min_token_len

        if _HAS_NLTK:
            _ensure_nltk_data()
            self._stopwords = set(stopwords.words(language))
            self._stemmer   = SnowballStemmer(language) if self.use_stemming else None
        else:
            log.warning("NLTK no encontrado — usando stopwords básicas de fallback.")
            self._stopwords = _BASIC_STOPWORDS
            self._stemmer   = None


    def process(self, text: str) -> list[str]:
        """
        Procesa y normaliza un texto crudo y devuelve la lista de tokens resultante.

        Pasos principales:
            1. Si el texto es vacío/nulo, devuelve lista vacía.
            2. Pasa el texto a minúsculas.
            3. Elimina URLs, expresiones LaTeX y números aislados.
            4. Elimina puntuación usando una tabla de traducción (_PUNCT_TABLE).
            5. Tokeniza por espacios (text.split()).
            6. Filtra tokens por longitud mínima (self.min_token_len).
            7. Filtra stopwords (self._stopwords).
            8. Conserva solo tokens alfabéticos (isalpha()).
            9. Aplica stemming si está configurado y disponible.

        Parámetros:
            text (str): texto crudo a preprocesar.

        Retorna:
            list[str]: lista de tokens normalizados listos para indexación.
        """
        if not text:
            return []

        text = text.lower()
        text = self._URL_RE.sub(" ", text)
        text = self._LATEX_RE.sub(" ", text)
        text = self._DIGIT_RE.sub(" ", text)
        text = text.translate(self._PUNCT_TABLE)

        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= self.min_token_len]
        tokens = [t for t in tokens if t not in self._stopwords]
        tokens = [t for t in tokens if t.isalpha()]

        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens


def _ensure_nltk_data() -> None:
    """
    Verifica y descarga (si es necesario) recursos básicos de NLTK requeridos.

    Recursos verificados:
        - 'stopwords'  -> path 'corpora/stopwords'
        - 'punkt_tab'  -> path 'tokenizers/punkt_tab' (nombre usado en este proyecto)

    Comportamiento:
        - Para cada recurso intenta localizarlo con nltk.data.find().
        - Si falta, registra un mensaje informativo y descarga el recurso con nltk.download(..., quiet=True).
    """
    for resource, path in [
        ("stopwords",  "corpora/stopwords"),
        ("punkt_tab",  "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            log.info("Descargando recurso NLTK: %s …", resource)
            nltk.download(resource, quiet=True)

_BASIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "between", "both", "but", "by", "can", "cannot", "could",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "get", "got", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "if", "in",
    "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "out", "over", "own", "paper", "result", "results", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "use", "used", "using", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would", "you", "your",
}