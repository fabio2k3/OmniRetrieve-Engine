from unittest.mock import MagicMock, patch
from backend.web_search.fallback_searcher import DuckDuckGoSearcher


# ── Tests de búsqueda DuckDuckGo ──────────────────────────────────────────────

def test_ddg_search_success():
    """DuckDuckGo devuelve resultados normalizados correctamente."""
    fake_ddg_response = [
        {
            "title": "DDG Result",
            "href":  "http://ddg.com/result",
            "body":  "Some content from DuckDuckGo",
        }
    ]

    searcher = DuckDuckGoSearcher(max_results=5)

    with patch("backend.web_search.fallback_searcher.DDGS") as mock_ddgs:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__  = MagicMock(return_value=False)
        mock_ctx.text.return_value = fake_ddg_response
        mock_ddgs.return_value = mock_ctx

        results = searcher.search("AI ethics")

    assert len(results)             == 1
    assert results[0]["title"]      == "DDG Result"
    assert results[0]["url"]        == "http://ddg.com/result"
    assert results[0]["content"]    == "Some content from DuckDuckGo"
    assert results[0]["source"]     == "web_fallback"
    assert isinstance(results[0]["score"], float)


def test_ddg_search_failure_returns_empty():
    """Si DuckDuckGo falla, devuelve lista vacía sin romper."""
    searcher = DuckDuckGoSearcher()

    with patch("backend.web_search.fallback_searcher.DDGS") as mock_ddgs:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__  = MagicMock(return_value=False)
        mock_ctx.text.side_effect = Exception("DDG error")
        mock_ddgs.return_value = mock_ctx

        results = searcher.search("test")

    assert results == []


def test_ddg_source_is_web_fallback():
    """Todos los resultados tienen source='web_fallback'."""
    fake_response = [
        {"title": "A", "href": "url1", "body": "content1"},
        {"title": "B", "href": "url2", "body": "content2"},
    ]

    searcher = DuckDuckGoSearcher()

    with patch("backend.web_search.fallback_searcher.DDGS") as mock_ddgs:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__  = MagicMock(return_value=False)
        mock_ctx.text.return_value = fake_response
        mock_ddgs.return_value = mock_ctx

        results = searcher.search("query")

    assert all(r["source"] == "web_fallback" for r in results)


def test_ddg_score_is_neutral():
    """DuckDuckGo no tiene score propio — debe ser 0.5 (valor neutro)."""
    fake_response = [{"title": "T", "href": "url", "body": "content"}]

    searcher = DuckDuckGoSearcher()

    with patch("backend.web_search.fallback_searcher.DDGS") as mock_ddgs:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__  = MagicMock(return_value=False)
        mock_ctx.text.return_value = fake_response
        mock_ddgs.return_value = mock_ctx

        results = searcher.search("query")

    assert results[0]["score"] == 0.5


def test_ddg_missing_fields():
    """Campos ausentes en resultados DDG se manejan con valores por defecto."""
    searcher = DuckDuckGoSearcher()

    with patch("backend.web_search.fallback_searcher.DDGS") as mock_ddgs:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__  = MagicMock(return_value=False)
        mock_ctx.text.return_value = [{}]   # dict vacío
        mock_ddgs.return_value = mock_ctx

        results = searcher.search("query")

    assert results[0]["title"]   == "Sin título"
    assert results[0]["url"]     == ""
    assert results[0]["content"] == ""


def test_ddg_import_error_returns_empty():
    """Si duckduckgo-search no está instalado, devuelve lista vacía."""
    searcher = DuckDuckGoSearcher()

    with patch.dict("sys.modules", {"duckduckgo_search": None}):
        with patch(
            "backend.web_search.fallback_searcher.DDGS",
            side_effect=ImportError("No module"),
        ):
            results = searcher.search("query")

    assert results == []