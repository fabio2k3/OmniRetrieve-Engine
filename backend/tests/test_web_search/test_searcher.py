from unittest.mock import MagicMock, patch
from backend.web_search.searcher import WebSearcher


# ── Tests de normalización ────────────────────────────────────────────────────

def test_normalization():
    searcher = WebSearcher(api_key="fake")

    raw = [
        {
            "title":   "Test",
            "url":     "http://example.com",
            "content": "text",
            "score":   0.9,
        }
    ]

    normalized = searcher._normalize(raw)

    assert normalized[0]["title"]  == "Test"
    assert normalized[0]["source"] == "web"
    assert isinstance(normalized[0]["score"], float)


def test_normalization_missing_fields():
    """Campos ausentes deben usar valores por defecto."""
    searcher = WebSearcher(api_key="fake")

    normalized = searcher._normalize([{}])

    assert normalized[0]["title"]   == "Sin título"
    assert normalized[0]["url"]     == ""
    assert normalized[0]["content"] == ""
    assert normalized[0]["score"]   == 0.0
    assert normalized[0]["source"]  == "web"


# ── Tests de búsqueda Tavily ──────────────────────────────────────────────────

@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_search_success(mock_client):
    """Tavily responde correctamente."""
    mock = MagicMock()
    mock.search.return_value = {
        "results": [
            {
                "title":   "A",
                "url":     "url",
                "content": "content",
                "score":   0.8,
            }
        ]
    }
    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake")
    results  = searcher.search("test")

    assert len(results) == 1
    assert results[0]["source"] == "web"


@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_search_failure_with_fallback(mock_client, fake_ddg_results):
    """Tavily falla → activa DuckDuckGo automáticamente."""
    mock = MagicMock()
    mock.search.side_effect = Exception("API error")
    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake", use_fallback=True)

    with patch.object(searcher._fallback, "search", return_value=fake_ddg_results):
        results = searcher.search("test")

    assert len(results) == 1
    assert results[0]["source"] == "web_fallback"


@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_search_failure_no_fallback(mock_client):
    """Tavily falla y fallback desactivado → lista vacía."""
    mock = MagicMock()
    mock.search.side_effect = Exception("API error")
    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake", use_fallback=False)
    results  = searcher.search("test")

    assert results == []


@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_fallback_not_called_when_tavily_ok(mock_client, fake_tavily_results):
    """DuckDuckGo NO se llama si Tavily funciona correctamente."""
    mock = MagicMock()
    mock.search.return_value = {"results": [
        {"title": "T", "url": "u", "content": "c", "score": 0.8}
    ]}
    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake", use_fallback=True)

    with patch.object(searcher._fallback, "search") as mock_ddg:
        results = searcher.search("test")
        mock_ddg.assert_not_called()

    assert results[0]["source"] == "web"


# ── Tests de use_fallback flag ────────────────────────────────────────────────

def test_use_fallback_default_is_true():
    """use_fallback debe ser True por defecto."""
    searcher = WebSearcher(api_key="fake")
    assert searcher.use_fallback is True


def test_use_fallback_can_be_disabled():
    """use_fallback puede desactivarse."""
    searcher = WebSearcher(api_key="fake", use_fallback=False)
    assert searcher.use_fallback is False