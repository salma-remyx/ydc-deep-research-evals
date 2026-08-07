"""Integration tests for the Bayesian rubric-reliability hook in ``aggregate``.

These tests import the *existing* metric module (the call site) and exercise
the wired-in reliability posterior end to end, proving the integration rather
than only unit-testing the new module in isolation. ``conftest.py`` supplies
the dummy OpenAI credentials that ``evals.utils`` needs at import time.
"""

import pytest

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.metrics.rubric_reliability import (
    beta_credible_interval,
    beta_posterior_mean,
    compute_reliability_from_votes,
)


def _dimension(preferred, score):
    """Build a DimensionResult with the majority-vote grade the metric expects."""
    n_b = preferred.count("b")
    n_a = preferred.count("a")
    if n_b > n_a:
        grade = "win"
    elif n_b < n_a:
        grade = "lose"
    else:
        grade = "tie"
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=score,
        preferred=list(preferred),
        raw_preferences={"original": [], "flipped": []},
    )


def _score_result():
    """One row where every dimension has a distinct agreement profile.

    Scores are chosen so the reliability-weighted average differs from the
    flat average (high-reliability dimensions carry the higher scores).
    """
    return DeepResearchScoreResult(
        instruction_following=_dimension(["b"] * 6, score=9.0),  # unanimous
        comprehensiveness=_dimension(["a", "a", "a", "b", "b", "b"], score=1.0),  # 3-3 tie
        completeness=_dimension(["b", "b", "b", "b", "b", "a"], score=5.0),  # 5-1
        writing_quality=_dimension(["b"] * 6, score=9.0),  # unanimous
    )


@pytest.fixture
def metric(monkeypatch):
    """A metric whose explanation-summary call is stubbed out (no API hit)."""
    monkeypatch.setattr(
        DeepResearchPairwiseMetric,
        "_generate_explanation_summary_from_raw",
        lambda self, raw: "stubbed summary",
    )
    return DeepResearchPairwiseMetric()


def test_aggregate_surfaces_reliability_fields(metric):
    aggregated = metric.aggregate([_score_result()])

    # Existing metrics survive (no regression at the call site).
    for dimension in DIMENSIONS:
        dim = aggregated[dimension]
        for key in ("win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"):
            assert key in dim
        # New reliability fields are surfaced per dimension.
        assert "reliability" in dim
        assert "reliability_ci" in dim
        assert "measurable" in dim
        assert dim["num_votes"] == 6  # one row of 6 pooled votes
        lo, hi = dim["reliability_ci"]
        assert 0.0 <= lo <= dim["reliability"] <= hi <= 1.0

    # Overall reliability summary.
    overall = aggregated["overall"]
    for key in ("mean_reliability", "num_measurable_dimensions", "reliability_weighted_avg_score"):
        assert key in overall
    assert 0.0 <= overall["mean_reliability"] <= 1.0


def test_aggregate_delegates_to_rubric_reliability(metric):
    """aggregate() must compute reliability via the new module (the wiring)."""
    rows = [_score_result(), _score_result(), _score_result()]
    aggregated = metric.aggregate(rows)

    direct = compute_reliability_from_votes(
        [r.instruction_following.preferred for r in rows]
    )
    assert aggregated["instruction_following"]["reliability"] == pytest.approx(
        direct.reliability
    )
    assert aggregated["instruction_following"]["measurable"] is direct.measurable


def test_measurability_filter_distinguishes_dimensions(metric):
    aggregated = metric.aggregate(
        [_score_result(), _score_result(), _score_result()]
    )

    # Unanimous dimensions: 3 rows x 6 agreeing votes -> lower credible bound
    # (~0.82) clears the 0.75 threshold -> measurable.
    assert aggregated["instruction_following"]["measurable"] is True
    assert aggregated["writing_quality"]["measurable"] is True

    # Tie dimension: judges split 3-3 on every row -> ~0.5 reliability, lower
    # bound (~0.29) far below threshold -> not measurable.
    assert aggregated["comprehensiveness"]["measurable"] is False

    # Reliability ordering reflects the underlying agreement.
    rel = {d: aggregated[d]["reliability"] for d in DIMENSIONS}
    assert rel["comprehensiveness"] < rel["completeness"] < rel["instruction_following"]

    # Two of the four dimensions clear the measurability bar.
    assert aggregated["overall"]["num_measurable_dimensions"] == 2


def test_reliability_weighted_average_differs_from_flat(metric):
    aggregated = metric.aggregate(
        [_score_result(), _score_result(), _score_result()]
    )
    flat = aggregated["overall"]["avg_score"]
    weighted = aggregated["overall"]["reliability_weighted_avg_score"]
    assert isinstance(weighted, float)
    assert flat == pytest.approx(6.0)  # (9 + 1 + 5 + 9) / 4
    # High-reliability dimensions carry the higher scores, so weighting up.
    assert weighted == pytest.approx(6.75, abs=0.01)
    assert weighted > flat


# --- pure-math sanity checks on the new module's Beta machinery ----------


def test_beta_mean():
    assert beta_posterior_mean(1.0, 1.0) == pytest.approx(0.5)
    assert beta_posterior_mean(3.0, 1.0) == pytest.approx(0.75)


def test_uniform_credible_interval():
    lo, hi = beta_credible_interval(1.0, 1.0)
    # Uniform Beta(1, 1) -> central 95% interval is [0.025, 0.975].
    assert lo == pytest.approx(0.025, abs=1e-3)
    assert hi == pytest.approx(0.975, abs=1e-3)
    assert lo <= hi


def test_empty_votes_fall_back_to_uninformative_prior():
    result = compute_reliability_from_votes([])
    assert result.num_votes == 0
    assert result.reliability == pytest.approx(0.5)  # Beta(1, 1) prior
    assert result.measurable is False  # wide prior -> lower bound below threshold
