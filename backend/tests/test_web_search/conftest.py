import pytest


@pytest.fixture
def sample_retriever_results():
    return [
        {"score": 0.2, "title": "Doc 1"},
        {"score": 0.1, "title": "Doc 2"},
    ]


@pytest.fixture
def low_score_results():
    return [
        {"score": 0.05},
        {"score": 0.1},
    ]


@pytest.fixture
def empty_results():
    return []