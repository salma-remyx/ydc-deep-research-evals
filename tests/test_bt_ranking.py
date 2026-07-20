"""Integration tests for the Bradley-Terry ranking wired into aggregate().

These import the existing call-site module
(``evals.metrics.deep_research_pairwise_metric``) and assert the integrated
behavior end-to-end, plus direct sanity checks on the BT estimator.
"""

import math
import os

# The call-site module imports evals.utils, which constructs an OpenAI client
# at import time from these env vars. Set placeholders before importing so the
# test is self-contained; no API call is made (the summary method is patched).
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from evals.metrics.bt_ranking import estimate_bt_ranking
from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
    DimensionResult,
)


def _make_score_result(preferred_per_dim):
    """Build a DeepResearchScoreResult with identical per-trial preferences
    across all four dimensions."""
    dims = {}
    grade = "win" if preferred_per_dim.count("b") > preferred_per_dim.count(
        "a"
    ) else "lose"
    for d in DIMENSIONS:
        dims[d] = DimensionResult(
            grade=grade,
            is_win=grade == "win",
            is_tie=False,
            is_lose=grade == "lose",
            score=6.0,
            preferred=list(preferred_per_dim),
            raw_preferences={},
        )
    return DeepResearchScoreResult(**dims)


def test_aggregate_includes_bt_ranking(monkeypatch):
    """aggregate() must now emit a bt_ranking entry per dimension and overall."""
    metric = DeepResearchPairwiseMetric()
    # Keep the test offline: the explanation summary normally calls the model.
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda raw, n=20: "stub"
    )

    # 5 candidate wins / 1 baseline win per dimension per row, across 3 rows.
    prefs = ["b", "b", "b", "b", "b", "a"]
    scores = [_make_score_result(prefs) for _ in range(3)]
    aggregated = metric.aggregate(scores)

    bt = aggregated["instruction_following"]["bt_ranking"]
    assert bt["num_wins"] == 15  # 5 * 3 rows
    assert bt["num_losses"] == 3
    assert bt["ability"] > 0.0
    assert bt["win_probability"] > 0.5
    # Win-probability CI honours the [0, 1] boundary and brackets the estimate.
    assert 0.0 <= bt["win_probability_ci_low"]
    assert bt["win_probability_ci_low"] <= bt["win_probability"]
    assert bt["win_probability"] <= bt["win_probability_ci_high"]
    assert bt["win_probability_ci_high"] <= 1.0
    # A 15:3 split should be a statistically significant candidate edge.
    assert bt["significant"] is True
    # Existing output contract is preserved alongside the new key. Note
    # net_winrate is computed at the row-consensus level (3/3 winning rows =>
    # 1.0, saturated) whereas bt_ranking operates on the trial-level split
    # (15:3) and so carries the uncertainty the row-level rate discards.
    assert "net_winrate" in aggregated["instruction_following"]
    assert aggregated["instruction_following"]["net_winrate"] == 1.0
    # Overall pooled ranking is present too.
    assert "bt_ranking" in aggregated["overall"]
    assert aggregated["overall"]["bt_ranking"]["num_wins"] == 15 * len(DIMENSIONS)


def test_aggregate_bt_ranking_balanced_is_not_significant(monkeypatch):
    """A 50/50 split must produce a non-significant ranking centered at 0."""
    metric = DeepResearchPairwiseMetric()
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda raw, n=20: "stub"
    )

    prefs = ["a", "b", "a", "b"]
    aggregated = metric.aggregate([_make_score_result(prefs)])
    bt = aggregated["comprehensiveness"]["bt_ranking"]
    assert bt["num_wins"] == 2
    assert bt["num_losses"] == 2
    assert abs(bt["ability"]) < 1e-9
    assert abs(bt["win_probability"] - 0.5) < 1e-9
    assert bt["significant"] is False


def test_estimator_handles_degenerate_all_wins():
    """All-candidate-wins must stay finite (regularised) and significant."""
    ranking = estimate_bt_ranking(["b"] * 20)
    assert math.isfinite(ranking.ability)
    assert ranking.ability > 0.0
    assert ranking.significant is True
    assert ranking.win_probability > 0.5


def test_estimator_bootstrap_ci_brackets_point_estimate():
    ranking = estimate_bt_ranking(
        ["b"] * 14 + ["a"] * 6, bootstrap_iterations=500, seed=0
    )
    assert ranking.bootstrap_ci_low is not None
    assert ranking.bootstrap_ci_high is not None
    assert ranking.bootstrap_ci_low <= ranking.win_probability
    assert ranking.win_probability <= ranking.bootstrap_ci_high
