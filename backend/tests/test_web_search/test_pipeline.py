from unittest.mock import patch, MagicMock
from backend.web_search.pipeline import WebSearchPipeline


def test_pipeline_no_web_when_sufficient(sample_retriever_results):
    pipeline = WebSearchPipeline(api_key="fake")

    with patch.object(pipeline.searcher, "search") as mock_search:
        result = pipeline.run("query", sample_retriever_results)

        assert result["web_activated"] is False
        mock_search.assert_not_called()


@patch("backend.web_search.pipeline.save_web_results")
def test_pipeline_triggers_web(mock_save):
    pipeline = WebSearchPipeline(api_key="fake")

    low_results = [{"score": 0.01}]

    fake_web = [
        {
            "title": "Web Doc",
            "url": "http://web.com",
            "content": "info",
            "score": 0.9,
        }
    ]

    with patch.object(pipeline.searcher, "search", return_value=fake_web):
        result = pipeline.run("query", low_results)

        assert result["web_activated"] is True
        assert len(result["web_results"]) == 1
        mock_save.assert_called_once()


def test_pipeline_combines_results():
    pipeline = WebSearchPipeline(api_key="fake")

    local = [{"score": 0.01, "title": "Local"}]

    web = [
        {
            "title": "Web",
            "url": "url",
            "content": "content",
            "score": 0.9,
        }
    ]

    with patch.object(pipeline.searcher, "search", return_value=web):
        with patch("backend.web_search.pipeline.save_web_results"):
            result = pipeline.run("query", local)

    combined = result["results"]

    assert len(combined) == 2
    assert combined[0]["source"] == "local"
    assert combined[1]["source"] == "web"


def test_pipeline_web_returns_empty():
    pipeline = WebSearchPipeline(api_key="fake")

    with patch.object(pipeline.searcher, "search", return_value=[]):
        with patch("backend.web_search.pipeline.save_web_results"):
            result = pipeline.run("query", [{"score": 0.01}])

    assert result["web_activated"] is True
    assert result["web_results"] == []