from backend.web_search.sufficiency import SufficiencyChecker


def test_empty_results_triggers_web(empty_results):
    checker = SufficiencyChecker()
    assert checker.is_sufficient(empty_results) is False


def test_low_scores_not_sufficient(low_score_results):
    checker = SufficiencyChecker(threshold=0.15, min_docs=1)
    assert checker.is_sufficient(low_score_results) is False


def test_enough_high_scores(sample_retriever_results):
    checker = SufficiencyChecker(threshold=0.15, min_docs=1)
    assert checker.is_sufficient(sample_retriever_results) is True


def test_multiple_docs_required():
    checker = SufficiencyChecker(threshold=0.15, min_docs=2)
    results = [
        {"score": 0.2},
        {"score": 0.1},
    ]
    assert checker.is_sufficient(results) is False


def test_missing_score_field():
    checker = SufficiencyChecker()
    results = [{}]
    assert checker.is_sufficient(results) is False