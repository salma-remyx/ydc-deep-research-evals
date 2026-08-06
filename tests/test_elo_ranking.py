"""Tests for the Elo ranking integration into the pairwise metric.

The integration test imports the *existing* metric module
(``evals.metrics.deep_research_pairwise_metric``) and exercises the
``aggregate()`` wiring, proving the new ``elo_ranking`` capability is actually
reached by the repo's aggregation contract rather than only self-tested.
"""

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DimensionResult,
    DeepResearchScoreResult,
)
from evals.metrics.elo_ranking import (
    DEFAULT_BASE_RATING,
    compute_elo_ratings,
    expected_score,
    rank_with_elo,
    resolve_outcome,
)


def _dim(preferred, grade=None, score=5.0) -> DimensionResult:
    """Build a DimensionResult, deriving the grade from the votes if omitted."""
    if grade is None:
        b = preferred.count("b")
        a = preferred.count("a")
        grade = "win" if b > a else "lose" if a > b else "tie"
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=score,
        preferred=list(preferred),
        raw_preferences={},
    )


def _row(preferred) -> DeepResearchScoreResult:
    """Build a full score result with every dimension sharing the same votes."""
    dim = _dim(preferred)
    return DeepResearchScoreResult(**{dimension: dim for dimension in DIMENSIONS})


def test_aggregate_emits_elo_ranking_with_candidate_ahead(monkeypatch):
    """aggregate() must reach the Elo wiring and rank a winning candidate higher."""
    # 3 rows where the candidate wins on a 3-1 split, 1 row where it loses.
    scores = [
        _row(["b", "b", "b", "a"]),
        _row(["b", "b", "b", "a"]),
        _row(["b", "b", "b", "a"]),
        _row(["a", "a", "a", "b"]),
    ]

    metric = DeepResearchPairwiseMetric()
    # aggregate() calls the OpenAI-backed summary generator; stub it so the
    # test never touches the network.
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda _: "stub"
    )

    aggregated = metric.aggregate(scores)

    assert "elo_ranking" in aggregated
    elo = aggregated["elo_ranking"]
    overall = elo["overall"]
    assert overall["candidate_rating"] > DEFAULT_BASE_RATING
    assert overall["candidate_rating"] > overall["baseline_rating"]
    # Every row x dimension match is decisive at the majority threshold.
    assert overall["decided_matches"] == len(scores) * len(DIMENSIONS)
    assert overall["total_matches"] == len(scores) * len(DIMENSIONS)
    # Per-dimension summaries are present for every dimension.
    assert set(elo["dimensions"]) == set(DIMENSIONS)


def test_aggregate_agreement_threshold_trades_coverage_for_confidence(monkeypatch):
    """Raising the threshold to unanimity turns 3-1 splits into ties (coverage drops)."""
    scores = [_row(["b", "b", "b", "a"]), _row(["a", "a", "b", "b"])]

    majority = DeepResearchPairwiseMetric(agreement_threshold=0.5)
    unanimous = DeepResearchPairwiseMetric(agreement_threshold=1.0)
    monkeypatch.setattr(
        majority, "_generate_explanation_summary_from_raw", lambda _: "stub"
    )
    monkeypatch.setattr(
        unanimous, "_generate_explanation_summary_from_raw", lambda _: "stub"
    )

    majority_coverage = majority.aggregate(scores)["elo_ranking"]["overall"]["coverage"]
    unanimous_coverage = (
        unanimous.aggregate(scores)["elo_ranking"]["overall"]["coverage"]
    )

    # At unanimity every match is a tie -> zero coverage, the paper's
    # confidence-vs-coverage tradeoff.
    assert 0.0 < majority_coverage
    assert unanimous_coverage == 0.0


def test_resolve_outcome_threshold_semantics():
    assert resolve_outcome(["b", "b", "b", "a"], 0.5) == "win"
    # Same votes at unanimity -> not decisive -> tie.
    assert resolve_outcome(["b", "b", "b", "a"], 1.0) == "tie"
    assert resolve_outcome(["a", "a", "a", "a"], 1.0) == "loss"
    # An even split is always a tie.
    assert resolve_outcome(["a", "b", "a", "b"], 0.5) == "tie"
    assert resolve_outcome([]) == "tie"


def test_expected_score_is_symmetric():
    assert expected_score(1000.0, 1000.0) == 0.5
    assert abs(expected_score(1100.0, 1000.0) - (1 - expected_score(1000.0, 1100.0))) < 1e-12


def test_compute_elo_ratings_zero_sum_and_convergence():
    # A single decisive match moves both players by equal and opposite amounts.
    ratings = compute_elo_ratings([("a", "b", "win")])
    assert ratings["a"] > DEFAULT_BASE_RATING
    assert ratings["b"] < DEFAULT_BASE_RATING
    assert abs(ratings["a"] + ratings["b"] - 2 * DEFAULT_BASE_RATING) < 1e-9

    # Repeated clean sweeps converge to a stable gap regardless of order.
    sweeps = [("a", "b", "win")] * 50
    converged = compute_elo_ratings(sweeps)
    shuffled = compute_elo_ratings(list(reversed(sweeps)))
    assert abs(converged["a"] - shuffled["a"]) < 1e-9
    assert converged["a"] > converged["b"]


def test_rank_with_elo_overall_pools_all_dimensions():
    scores = [_row(["b", "b", "b", "a"]), _row(["a", "a", "a", "b"])]
    elo = rank_with_elo(scores, DIMENSIONS, agreement_threshold=0.5)
    # 2 rows x 4 dimensions = 8 pooled matches, all decisive at majority.
    assert elo["overall"]["total_matches"] == 8
    assert elo["overall"]["decided_matches"] == 8
    assert elo["agreement_threshold"] == 0.5
