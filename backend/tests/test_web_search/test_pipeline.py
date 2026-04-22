from unittest.mock import patch, MagicMock
from backend.web_search.pipeline import WebSearchPipeline


# ── Tests de suficiencia ──────────────────────────────────────────────────────

def test_pipeline_no_web_when_sufficient(sample_retriever_results):
    """No activa búsqueda web si los docs locales son suficientes."""
    pipeline = WebSearchPipeline(api_key="fake")

    with patch.object(pipeline.searcher, "search") as mock_search:
        result = pipeline.run("query", sample_retriever_results)

        assert result["web_activated"] is False
        assert result["indexed"]       == 0
        mock_search.assert_not_called()


def test_pipeline_sufficient_returns_local_results(sample_retriever_results):
    """Cuando es suficiente, devuelve exactamente los resultados locales."""
    pipeline = WebSearchPipeline(api_key="fake")

    with patch.object(pipeline.searcher, "search"):
        result = pipeline.run("query", sample_retriever_results)

    assert result["results"] == sample_retriever_results
    assert result["web_results"] == []


# ── Tests de activación de búsqueda web ──────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_triggers_web(mock_save, fake_tavily_results):
    """Activa búsqueda web cuando los scores son insuficientes."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [{"score": 0.01}])

    assert result["web_activated"]    is True
    assert len(result["web_results"]) == 2
    mock_save.assert_called_once()


@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_triggers_web_on_empty(mock_save, fake_tavily_results):
    """Activa búsqueda web cuando no hay resultados locales."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [])

    assert result["web_activated"] is True


# ── Tests de combinación de resultados ───────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_combines_results(mock_save):
    """Combina correctamente resultados locales y web."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    local = [{"score": 0.01, "title": "Local"}]
    web   = [{"title": "Web", "url": "url", "content": "content", "score": 0.9}]

    with patch.object(pipeline.searcher, "search", return_value=web):
        result = pipeline.run("query", local)

    combined = result["results"]

    assert len(combined)              == 2
    assert combined[0]["source"]      == "local"
    assert combined[1]["source"]      == "web"


@patch("backend.web_search.pipeline.save_web_results", return_value=0)
def test_pipeline_web_returns_empty(mock_save):
    """Maneja correctamente que la búsqueda web devuelva vacío."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=[]):
        result = pipeline.run("query", [{"score": 0.01}])

    assert result["web_activated"] is True
    assert result["web_results"]   == []
    assert result["indexed"]       == 0


# ── Tests de fuente en resultados web ────────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_web_results_preserve_source_fallback(mock_save, fake_ddg_results):
    """Los resultados de DuckDuckGo mantienen source='web_fallback'."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=fake_ddg_results):
        result = pipeline.run("query", [{"score": 0.01}])

    assert result["web_results"][0]["source"] == "web_fallback"


# ── Tests de indexación automática ───────────────────────────────────────────

@patch("backend.web_search.pipeline.save_web_results", return_value=2)
def test_pipeline_auto_index_called_when_docs_saved(mock_save, fake_tavily_results):
    """_index_web_results se llama cuando se guardan docs nuevos."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=True)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        with patch.object(pipeline, "_index_web_results", return_value=2) as mock_index:
            result = pipeline.run("query", [{"score": 0.01}])

    mock_index.assert_called_once()
    assert result["indexed"] == 2


@patch("backend.web_search.pipeline.save_web_results", return_value=0)
def test_pipeline_auto_index_not_called_when_no_new_docs(mock_save, fake_tavily_results):
    """_index_web_results NO se llama si no se guardaron docs nuevos."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=True)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        with patch.object(pipeline, "_index_web_results") as mock_index:
            result = pipeline.run("query", [{"score": 0.01}])

    mock_index.assert_not_called()
    assert result["indexed"] == 0


@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_auto_index_disabled(mock_save, fake_tavily_results):
    """auto_index=False no indexa aunque haya docs nuevos."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        with patch.object(pipeline, "_index_web_results") as mock_index:
            result = pipeline.run("query", [{"score": 0.01}])

    mock_index.assert_not_called()
    assert result["indexed"] == 0


@patch("backend.web_search.pipeline.save_web_results", return_value=1)
def test_pipeline_auto_index_failure_does_not_crash(mock_save, fake_tavily_results):
    """Si el indexador falla, el pipeline continúa sin romperse."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=True)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        with patch.object(pipeline, "_index_web_results", return_value=0):
            result = pipeline.run("query", [{"score": 0.01}])

    assert result["web_activated"] is True
    assert result["indexed"]       == 0


# ── Tests del dict de retorno ─────────────────────────────────────────────────

def test_pipeline_return_keys_when_sufficient(sample_retriever_results):
    """El dict de retorno tiene todas las keys esperadas."""
    pipeline = WebSearchPipeline(api_key="fake")

    with patch.object(pipeline.searcher, "search"):
        result = pipeline.run("query", sample_retriever_results)

    assert "results"        in result
    assert "web_activated"  in result
    assert "web_results"    in result
    assert "reason"         in result
    assert "query"          in result
    assert "indexed"        in result


@patch("backend.web_search.pipeline.save_web_results", return_value=0)
def test_pipeline_return_keys_when_web_activated(mock_save, fake_tavily_results):
    """El dict de retorno tiene todas las keys también cuando se activa la web."""
    pipeline = WebSearchPipeline(api_key="fake", auto_index=False)

    with patch.object(pipeline.searcher, "search", return_value=fake_tavily_results):
        result = pipeline.run("query", [{"score": 0.01}])

    assert "results"        in result
    assert "web_activated"  in result
    assert "web_results"    in result
    assert "reason"         in result
    assert "query"          in result
    assert "indexed"        in result