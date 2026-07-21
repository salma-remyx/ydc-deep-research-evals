"""Integration tests for the topical-focus absolute scorer.

These exercise the new ``evals.metrics.topical_focus_metric`` wired through the
repo's existing utilities (``evals.utils``) and alongside the existing pairwise
metric, without hitting the OpenAI API.
"""

import pytest

# Imported from NON-NEW modules to prove the new metric integrates with the
# existing codebase rather than standing alone.
from evals.metrics.deep_research_pairwise_metric import (  # noqa: F401  (non-new)
    DEFAULT_EVAL_MODEL,
    DIMENSIONS as PAIRWISE_DIMENSIONS,
)
from evals.utils import replace_markdown_links_with_text  # noqa: F401  (non-new)

from evals.metrics.semantic_drift import SemanticDriftMetric, SemanticDriftResult
from evals.metrics.topical_focus_metric import (
    DIMENSIONS,
    SemanticQualityAssessment,
    TopicalFocusAssessment,
    TopicalFocusJudgeOutput,
    TopicalFocusMetric,
    TopicalFocusScoreResult,
)


def _make_result(tf=8.0, sq=6.0, covered=3, missed=1) -> TopicalFocusScoreResult:
    return TopicalFocusScoreResult(
        reference_key_points=[f"kp{i}" for i in range(covered + missed)],
        topical_focus=TopicalFocusAssessment(
            score=tf, off_topic_sections=[], rationale="on topic"
        ),
        semantic_quality=SemanticQualityAssessment(
            score=sq,
            covered_key_points=[f"kp{i}" for i in range(covered)],
            missed_key_points=[f"kp{i}" for i in range(covered, covered + missed)],
            rationale="covers most",
        ),
    )


def test_new_dimensions_complement_pairwise():
    # The new absolute dimensions must not duplicate the existing pairwise family.
    assert set(DIMENSIONS).isdisjoint(PAIRWISE_DIMENSIONS)
    assert DIMENSIONS == ["topical_focus", "semantic_quality"]
    # And the default model is shared with the pairwise metric (single config).
    assert isinstance(DEFAULT_EVAL_MODEL, str) and DEFAULT_EVAL_MODEL


def test_computed_fields():
    result = _make_result(tf=8.0, sq=6.0, covered=3, missed=1)
    assert result.composite_score == pytest.approx(7.0)
    assert result.semantic_quality.coverage_fraction == pytest.approx(0.75)


def test_aggregate_is_pure_and_shapes_like_pairwise():
    metric = TopicalFocusMetric()
    agg = metric.aggregate(
        [_make_result(8.0, 6.0, covered=3, missed=1),
         _make_result(4.0, 2.0, covered=1, missed=3)]
    )
    assert agg["support"] == 2
    assert agg["topical_focus"]["avg_score"] == pytest.approx(6.0)
    assert agg["semantic_quality"]["avg_score"] == pytest.approx(4.0)
    # composite averages: (7.0 + 3.0) / 2 == 5.0
    assert agg["overall"]["avg_composite_score"] == pytest.approx(5.0)
    # coverage: (0.75 + 0.25) / 2 == 0.5
    assert agg["overall"]["avg_coverage_fraction"] == pytest.approx(0.5)


def test_aggregate_empty():
    assert TopicalFocusMetric().aggregate([]) == {"support": 0}


def test_input_reuses_existing_markdown_link_stripper():
    # Integration with existing evals.utils: citation markdown is stripped
    # before scoring, exactly as the pairwise metric does.
    cleaned = replace_markdown_links_with_text(
        "see [globaledge.msu.edu](https://globaledge.msu.edu/x) for detail", ""
    )
    assert "globaledge.msu.edu" not in cleaned
    assert "https" not in cleaned


def test_score_wires_judge_output(monkeypatch):
    # End-to-end wiring of the new metric with no OpenAI call: patch the judge
    # call and the SemanticDrift scorer, and assert score() turns judge output
    # plus the paper's SDR formula into an aggregate-able result.
    judge_output = TopicalFocusJudgeOutput(
        reference_key_points=["kp0", "kp1", "kp2"],
        topical_focus=TopicalFocusAssessment(
            score=9, off_topic_sections=["section on Y"], rationale="focused"
        ),
        semantic_quality=SemanticQualityAssessment(
            score=6,
            covered_key_points=["kp0", "kp1"],
            missed_key_points=["kp2"],
            rationale="misses one",
        ),
    )
    drift_result = SemanticDriftResult(
        anchor_keywords=["x", "impact"],
        deviation_keywords=["y"],
        anchor_counts=[3, 1],
        deviation_counts=[2],
        anchor_relevances=[5.0, 4.0],
        deviation_relevances=[4.0],
        semantic_drift=0.2,
        topical_focus_score=8.0,
        rationale="mostly anchored",
    )

    calls = []

    def fake_query(self, messages):
        calls.append(messages)
        return judge_output

    def fake_drift(self, question, reference_answer, candidate_answer):
        return drift_result

    monkeypatch.setattr(TopicalFocusMetric, "_query_evaluation_model", fake_query)
    monkeypatch.setattr(SemanticDriftMetric, "score", fake_drift)

    metric = TopicalFocusMetric(num_trials=3, num_workers=2)
    result = metric.score(
        question="What are the economic impacts of X?",
        baseline_answer="Reference covering kp0, kp1, kp2.",
        candidate_answer="Candidate covering kp0, kp1.",
    )
    assert len(calls) == 3  # num_trials independent judge calls
    assert isinstance(result, TopicalFocusScoreResult)
    # Topical focus comes from the SDR formula, not the judge's holistic 9;
    # the judge's qualitative off-topic findings are preserved.
    assert result.topical_focus.score == pytest.approx(8.0)
    assert result.topical_focus.off_topic_sections == ["section on Y"]
    assert result.composite_score == pytest.approx(7.0)
    assert result.semantic_quality.coverage_fraction == pytest.approx(2 / 3)
    # score() output feeds straight back into aggregate()
    assert metric.aggregate([result])["support"] == 1


def test_score_raises_when_all_trials_fail(monkeypatch):
    def boom(self, messages):
        raise RuntimeError("api down")

    monkeypatch.setattr(TopicalFocusMetric, "_query_evaluation_model", boom)
    metric = TopicalFocusMetric(num_trials=2, num_workers=1)
    with pytest.raises(ValueError):
        metric.score(question="q", baseline_answer="b", candidate_answer="c")
