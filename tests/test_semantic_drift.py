"""Tests for the SemanticDrift (FAK/FDK) scorer and its wiring into the metric.

These exercise ``evals.metrics.semantic_drift`` through the repo's existing
utilities (``evals.utils``) and the existing topical-focus metric call site,
without hitting the OpenAI API.
"""

import pytest

# Imported from NON-NEW modules to prove the new scorer integrates with the
# existing codebase rather than standing alone.
from evals.metrics.deep_research_pairwise_metric import (  # noqa: F401  (non-new)
    DEFAULT_EVAL_MODEL,
)
from evals.utils import query_openai_model_structured_outputs  # noqa: F401  (non-new)

from evals.metrics.semantic_drift import (
    FocusKeywords,
    KeywordRelevance,
    KeywordRelevanceJudgment,
    SemanticDriftMetric,
    SemanticDriftResult,
    compute_semantic_drift,
    count_keyword_occurrences,
)
from evals.metrics.topical_focus_metric import TopicalFocusMetric


def test_count_keyword_occurrences_case_insensitive():
    text = "Climate change drives migration. CLIMATE policy matters; climates vary."
    assert count_keyword_occurrences(text, "climate") == 3
    assert count_keyword_occurrences(text, "missing term") == 0
    assert count_keyword_occurrences(text, "  ") == 0


def test_compute_semantic_drift_matches_paper_formula():
    # Paper: SDR = 0.7*(1 - mean(min(fak/2,1)*rel/5)) + 0.3*mean(min(fdk,1)*rel/5)
    anchor_term = (min(2 / 2, 1) * 5 / 5 + min(1 / 2, 1) * 3 / 5 + 0.0) / 3
    deviation_term = (min(1, 1) * 4 / 5 + 0.0 + min(2, 1) * 5 / 5) / 3
    expected = 0.7 * (1 - anchor_term) + 0.3 * deviation_term
    sdr = compute_semantic_drift(
        anchor_counts=[2, 1, 0],
        anchor_relevances=[5.0, 3.0, 4.0],
        deviation_counts=[1, 0, 2],
        deviation_relevances=[4.0, 2.0, 5.0],
    )
    assert sdr == pytest.approx(expected)
    assert 0.0 <= sdr <= 1.0


def test_compute_semantic_drift_bounds():
    # Fully anchored, no deviation -> no drift.
    assert compute_semantic_drift([2, 3], [5.0, 5.0], [0], [5.0]) == pytest.approx(0.0)
    # No anchors, maximally relevant deviation -> maximal drift.
    assert compute_semantic_drift([0], [5.0], [4], [5.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_semantic_drift([], [], [], [])


def _fake_focus(self, question, reference_answer):
    return FocusKeywords(
        anchor_keywords=["carbon", "pricing"],
        deviation_keywords=["celebrity gossip"],
    )


def _fake_judgment(self, question, candidate_answer, keywords):
    return KeywordRelevanceJudgment(
        assessments=[
            KeywordRelevance(keyword="carbon", relevance=5.0, reason="central"),
            KeywordRelevance(keyword="pricing", relevance=4.0, reason="discussed"),
            KeywordRelevance(
                keyword="celebrity gossip", relevance=0.0, reason="absent"
            ),
        ]
    )


def test_score_combines_regex_counts_and_judged_relevance(monkeypatch):
    monkeypatch.setattr(SemanticDriftMetric, "_extract_focus_keywords", _fake_focus)
    monkeypatch.setattr(
        SemanticDriftMetric, "_judge_keyword_relevance", _fake_judgment
    )

    metric = SemanticDriftMetric()
    result = metric.score(
        question="How does carbon pricing affect emissions?",
        reference_answer="Carbon pricing mechanisms ...",
        candidate_answer="Carbon carbon pricing schemes cut emissions. Carbon.",
    )
    assert isinstance(result, SemanticDriftResult)
    assert result.anchor_counts == [3, 1]
    assert result.deviation_counts == [0]
    # anchor_term = (min(3/2,1)*5/5 + min(1/2,1)*4/5)/2 = (1.0 + 0.4)/2 = 0.7
    # deviation_term = 0 -> SDR = 0.7*(1-0.7) = 0.21
    assert result.semantic_drift == pytest.approx(0.21)
    assert result.topical_focus_score == pytest.approx(7.9)


def test_score_tolerates_missing_relevance_entries(monkeypatch):
    def partial_judgment(self, question, candidate_answer, keywords):
        return KeywordRelevanceJudgment(
            assessments=[
                KeywordRelevance(keyword="Carbon", relevance=5.0, reason="central"),
            ]
        )

    monkeypatch.setattr(SemanticDriftMetric, "_extract_focus_keywords", _fake_focus)
    monkeypatch.setattr(
        SemanticDriftMetric, "_judge_keyword_relevance", partial_judgment
    )

    result = SemanticDriftMetric().score(
        question="q", reference_answer="r", candidate_answer="carbon only"
    )
    # Case-insensitive echo matches; unjudged keywords default to relevance 0.
    assert result.anchor_relevances == [5.0, 0.0]
    assert result.deviation_relevances == [0.0]


def test_topical_focus_metric_uses_drift_score(monkeypatch):
    # Integration through the existing call site: TopicalFocusMetric.score()
    # must take its topical-focus number from the SemanticDrift formula.
    def fake_drift(self, question, reference_answer, candidate_answer):
        return SemanticDriftResult(
            anchor_keywords=["a"],
            deviation_keywords=["d"],
            anchor_counts=[2],
            deviation_counts=[0],
            anchor_relevances=[5.0],
            deviation_relevances=[0.0],
            semantic_drift=0.0,
            topical_focus_score=10.0,
            rationale="fully anchored",
        )

    class _Assessment:
        def __init__(self, score):
            self.score = score
            self.off_topic_sections = []
            self.rationale = "judge holistic"
            self.covered_key_points = []
            self.missed_key_points = []

    class _JudgeOutput:
        reference_key_points = []
        topical_focus = _Assessment(3.0)
        semantic_quality = _Assessment(6.0)

    monkeypatch.setattr(
        TopicalFocusMetric, "_query_evaluation_model", lambda self, m: _JudgeOutput()
    )
    monkeypatch.setattr(SemanticDriftMetric, "score", fake_drift)

    result = TopicalFocusMetric(num_trials=1, num_workers=1).score(
        question="q", baseline_answer="b", candidate_answer="c"
    )
    # The judge's holistic 3.0 is discarded in favor of the SDR-derived 10.0.
    assert result.topical_focus.score == pytest.approx(10.0)
    assert result.topical_focus.rationale == "fully anchored"
    assert result.semantic_quality.score == pytest.approx(6.0)
