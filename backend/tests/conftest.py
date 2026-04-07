from pathlib import Path
import pytest


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def network() -> bool:
    return False


# ── Fixtures compartidas para web_search ─────────────────────────────────────

@pytest.fixture
def sample_retriever_results():
    """Resultados del retriever con scores suficientes."""
    return [
        {"score": 0.8, "arxiv_id": "2301.00001", "title": "Fairness in ML"},
        {"score": 0.6, "arxiv_id": "2301.00002", "title": "Bias Detection"},
    ]


@pytest.fixture
def low_score_results():
    """Resultados del retriever con scores insuficientes."""
    return [
        {"score": 0.05, "arxiv_id": "2301.00003", "title": "Some Paper"},
    ]


@pytest.fixture
def empty_results():
    """Lista vacía — sin resultados del retriever."""
    return []


@pytest.fixture
def fake_tavily_results():
    """Resultados simulados de Tavily."""
    return [
        {
            "title":   "Web Result 1",
            "url":     "http://example.com/1",
            "content": "Some web content about AI ethics",
            "score":   0.9,
            "source":  "web",
        },
        {
            "title":   "Web Result 2",
            "url":     "http://example.com/2",
            "content": "More content about fairness",
            "score":   0.7,
            "source":  "web",
        },
    ]


@pytest.fixture
def fake_ddg_results():
    """Resultados simulados de DuckDuckGo."""
    return [
        {
            "title":   "DDG Result 1",
            "url":     "http://ddg.com/1",
            "content": "DuckDuckGo content about AI",
            "score":   0.5,
            "source":  "web_fallback",
        },
    ]