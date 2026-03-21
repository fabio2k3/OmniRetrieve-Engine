"""
preprocessor.py
===============
Pipeline de limpieza y tokenización de texto para el módulo de indexación.

Recibir texto crudo y devolver una lista de tokens
normalizados, listos para ser contados por el indexador BM25.

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

    def __init__(
        self,
        use_stemming: bool = False,
        min_token_len: int = 3,
        language: str = "english",
    ) -> None:
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
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt_tab", "tokenizers/punkt_tab"),
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
