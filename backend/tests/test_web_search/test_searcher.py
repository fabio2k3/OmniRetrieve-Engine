from unittest.mock import MagicMock, patch
from backend.web_search.searcher import WebSearcher


def test_normalization():
    searcher = WebSearcher(api_key="fake")

    raw = [
        {
            "title": "Test",
            "url": "http://example.com",
            "content": "text",
            "score": 0.9,
        }
    ]

    normalized = searcher._normalize(raw)

    assert normalized[0]["title"] == "Test"
    assert normalized[0]["source"] == "web"
    assert isinstance(normalized[0]["score"], float)


@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_search_success(mock_client):
    mock = MagicMock()
    mock.search.return_value = {
        "results": [
            {
                "title": "A",
                "url": "url",
                "content": "content",
                "score": 0.8,
            }
        ]
    }

    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake")
    results = searcher.search("test")

    assert len(results) == 1
    assert results[0]["source"] == "web"


@patch("backend.web_search.searcher.WebSearcher._get_client")
def test_search_failure(mock_client):
    mock = MagicMock()
    mock.search.side_effect = Exception("API error")

    mock_client.return_value = mock

    searcher = WebSearcher(api_key="fake")
    results = searcher.search("test")

    assert results == []