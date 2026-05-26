from unittest.mock import patch, MagicMock
from backend.web_search.pipeline import WebSearchPipeline


# ── Suficiencia ───────────────────────────────────────────────────────────────

def test_pipeline_no_web_when_sufficient(sample_retriever_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search") as mock_search:
        result = pipeline.run("query", sample_retriever_results)
    assert result["web_activated"] is False
    mock_search.assert_not_called()


def test_pipeline_sufficient_returns_local_results(sample_retriever_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search"):
        result = pipeline.run("query", sample_retriever_results)
    assert result["results"] == sample_retriever_results
    assert result["web_results"] == []


# ── Activación web ────────────────────────────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_triggers_web(mock_save, fake_tavily_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [{"score": 0.01}])
    assert result["web_activated"] is True
    assert len(result["web_results"]) == 2
    mock_save.assert_called_once()


@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_triggers_web_on_empty(mock_save, fake_tavily_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [])
    assert result["web_activated"] is True


# ── Combinación de resultados ─────────────────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_combines_results(mock_save):
    pipeline = WebSearchPipeline(api_key="fake")
    local = [{"score": 0.01, "title": "Local"}]
    web   = [{"title": "Web", "url": "url", "content": "content", "score": 0.9}]
    with patch.object(pipeline.searcher, "search", return_value=web):
        result = pipeline.run("query", local)
    combined = result["results"]
    assert len(combined) == 2
    assert combined[0]["source"] == "local"
    assert combined[1]["source"] == "web"


@patch("backend.web_search.pipeline.save_web_results", return_value=0)
def test_pipeline_web_returns_empty(mock_save):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search", return_value=[]):
        result = pipeline.run("query", [{"score": 0.01}])
    assert result["web_activated"] is True
    assert result["web_results"] == []


@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_web_results_preserve_source_fallback(mock_save, fake_ddg_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search", return_value=fake_ddg_results):
        result = pipeline.run("query", [{"score": 0.01}])
    assert result["web_results"][0]["source"] == "web_fallback"


# ── Claves del dict de retorno ────────────────────────────────────────────────

def test_pipeline_return_keys_when_sufficient(sample_retriever_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search"):
        result = pipeline.run("query", sample_retriever_results)
    for key in ("results", "web_activated", "web_results", "reason", "query"):
        assert key in result


@patch("backend.web_search.pipeline.save_web_results", return_value=0)
def test_pipeline_return_keys_when_web_activated(mock_save, fake_tavily_results):
    pipeline = WebSearchPipeline(api_key="fake")
    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [{"score": 0.01}])
    for key in ("results", "web_activated", "web_results", "reason", "query"):
        assert key in result