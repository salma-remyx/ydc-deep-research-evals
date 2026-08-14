"""Integration tests for the pivotal-vote (affected-set) wiring.

These import from the EXISTING metric module
``evals.metrics.deep_research_pairwise_metric`` and assert that the
``vote_margin`` / ``is_pivotal`` computed fields land on ``DimensionResult``
and that ``DeepResearchPairwiseMetric.aggregate`` emits the
``pivotal_vote`` breakdown -- proving the new
``evals.metrics.pivotal_vote_metric`` capability is actually reached from
the existing call site.

The OpenAI client in ``evals.utils`` is constructed at import time, so the
dummy credentials below must be set before importing anything under
``evals``.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-dummy-org")

from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.metrics.pivotal_vote_metric import (  # noqa: E402
    ballot_margin,
    is_pivotal_ballot,
    pivotal_breakdown,
    pivotal_breakdown_across_dimensions,
)


def _dimension(grade: str, preferred: list) -> DimensionResult:
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=5.0,
        preferred=preferred,
        raw_preferences={},
    )


def _score(preferred_by_dim: dict) -> DeepResearchScoreResult:
    return DeepResearchScoreResult(
        **{
            dim: _dimension(("win" if prefs.count("b") > prefs.count("a")
                             else "lose" if prefs.count("b") < prefs.count("a")
                             else "tie"), prefs)
            for dim, prefs in preferred_by_dim.items()
        }
    )


# Ballot lists (6 = 2 * num_trials flipped-order votes each). Margins are
# even: 0 (tie), 2 (narrowest decision), 6 (wide decision).
_TIE = ["a", "a", "b", "b", "a", "b"]              # 3-3, margin 0
_NARROW_WIN = ["b", "b", "b", "b", "a", "a"]       # 4-2, margin 2
_NARROW_LOSE = ["b", "b", "a", "a", "a", "a"]      # 2-4, margin 2
_WIDE_WIN = ["b", "b", "b", "b", "b", "b"]         # 6-0, margin 6
_WIDE_LOSE = ["a", "a", "a", "a", "a", "a"]        # 0-6, margin 6

SCORES = [
    _score(
        {
            "instruction_following": _TIE,
            "comprehensiveness": _NARROW_WIN,
            "completeness": _WIDE_WIN,
            "writing_quality": _WIDE_LOSE,
        }
    ),
    _score(
        {
            "instruction_following": _NARROW_LOSE,
            "comprehensiveness": _TIE,
            "completeness": _NARROW_WIN,
            "writing_quality": _WIDE_WIN,
        }
    ),
]


def test_dimension_result_exposes_pivotal_computed_fields():
    """The wiring adds vote_margin/is_pivotal derived from preferred."""
    assert _dimension("tie", _TIE).vote_margin == 0
    assert _dimension("tie", _TIE).is_pivotal is True

    assert _dimension("win", _NARROW_WIN).vote_margin == 2
    assert _dimension("win", _NARROW_WIN).is_pivotal is True  # flippable -> tie

    assert _dimension("win", _WIDE_WIN).vote_margin == 6
    assert _dimension("win", _WIDE_WIN).is_pivotal is False  # stable


def test_pure_helpers():
    assert ballot_margin(_TIE) == 0
    assert ballot_margin(_NARROW_WIN) == 2
    assert ballot_margin(_WIDE_LOSE) == 6
    assert is_pivotal_ballot(_TIE) is True
    assert is_pivotal_ballot(_WIDE_WIN) is False


def test_aggregate_emits_pivotal_breakdown():
    """aggregate() (existing call site) reaches the new capability."""
    metric = DeepResearchPairwiseMetric()
    # Avoid the OpenAI explanation-summary call during the unit test.
    metric._generate_explanation_summary_from_raw = lambda raw: "stub"

    aggregated = metric.aggregate(SCORES)

    assert "pivotal_vote" in aggregated
    pv = aggregated["pivotal_vote"]

    # Every dimension is reported, plus a pooled overall entry.
    assert set(pv) == set(DIMENSIONS) | {"overall"}

    # instruction_following: both decisions pivotal (margins 0 and 2).
    assert pv["instruction_following"]["num_pivotal"] == 2
    assert pv["instruction_following"]["pivotal_rate"] == 1.0

    # completeness: one narrow (margin 2) and one wide (margin 6) -> 1/2.
    assert pv["completeness"]["num_pivotal"] == 1
    assert pv["completeness"]["pivotal_rate"] == 0.5

    # writing_quality: two wide decisions -> none pivotal.
    assert pv["writing_quality"]["num_pivotal"] == 0
    assert pv["writing_quality"]["pivotal_rate"] == 0.0

    # Overall: 5 of 8 (query, dimension) decisions are pivotal.
    assert pv["overall"]["support"] == 8
    assert pv["overall"]["num_pivotal"] == 5
    assert pv["overall"]["pivotal_rate"] == 0.625
    assert pv["overall"]["num_ties"] == 2
    assert pv["overall"]["decided_rate"] == 0.75


def test_breakdown_directly_on_score_results():
    breakdown = pivotal_breakdown_across_dimensions(SCORES, DIMENSIONS)
    assert breakdown["overall"]["pivotal_rate"] == 0.625
    # Empty group degrades gracefully.
    assert pivotal_breakdown([])["support"] == 0
    assert pivotal_breakdown([])["pivotal_rate"] == 0.0
