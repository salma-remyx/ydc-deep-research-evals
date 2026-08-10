"""Integration tests for the judge-stability wiring.

These tests import from the *existing* module
``evals.metrics.deep_research_pairwise_metric`` and exercise the wiring
edits (``DimensionResult.stability`` field + ``aggregate()`` stability
blocks) adapted from arXiv:2608.06202's "measure consistency across
runs" thesis. They never call the OpenAI API: ``aggregate()``'s only
network touch is the explanation summary, which is monkeypatched out.
"""

# evals.utils constructs an OpenAI client at import time, reading these env
# vars. Dummy values are fine -- no request is made at import.
import os

os.environ.setdefault("OPENAI_API_KEY", "test-dummy")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-dummy")

import math  # noqa: E402

from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.metrics.judge_stability import compute_judge_stability  # noqa: E402


def _dim(stability) -> DimensionResult:
    """Build a DimensionResult carrying an explicit stability block."""
    return DimensionResult(
        grade="win",
        is_win=True,
        is_tie=False,
        is_lose=False,
        score=6.0,
        preferred=["b", "b"],
        raw_preferences={},
        stability=stability,
    )


def _score_result(stability) -> DeepResearchScoreResult:
    return DeepResearchScoreResult(**{d: _dim(stability) for d in DIMENSIONS})


def test_aggregate_emits_judge_stability_per_dimension_and_overall(monkeypatch):
    metric = DeepResearchPairwiseMetric()
    # aggregate() otherwise calls the OpenAI explanation summary.
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda raw: "stub"
    )

    unanimous = compute_judge_stability(["b", "b", "b", "b"], [3, 3, 3, 3])
    split = compute_judge_stability(["a", "b", "a", "b"], [3, -3, 3, -3])
    assert unanimous.is_unanimous is True
    assert split.is_unanimous is False

    agg = metric.aggregate([_score_result(unanimous), _score_result(split)])

    # Each dimension carries a stability block averaged across the two rows.
    for dimension in DIMENSIONS:
        stability = agg[dimension]["stability"]
        assert set(stability) == {
            "agreement_rate",
            "preference_entropy",
            "score_std",
            "inconsistent_rate",
        }
        assert math.isclose(stability["agreement_rate"], 0.75)  # mean(1.0, 0.5)
        assert math.isclose(stability["preference_entropy"], 0.5)  # mean(0.0, 1.0)
        assert math.isclose(stability["inconsistent_rate"], 0.5)  # mean(0, 1)

    # Overall is the cross-dimension mean (all four dimensions identical here).
    overall = agg["overall"]["stability"]
    assert math.isclose(overall["agreement_rate"], 0.75)
    assert math.isclose(overall["inconsistent_rate"], 0.5)


def test_aggregate_skips_legacy_rows_without_stability(monkeypatch):
    """Rows re-validated from pre-stability serialized output lack the field.

    ``stability`` defaults to None, so such rows must not break aggregation --
    they are simply excluded from the stability average.
    """
    metric = DeepResearchPairwiseMetric()
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda raw: "stub"
    )

    legacy_row = _score_result(None)  # stability unset
    stable_row = _score_result(compute_judge_stability(["b", "b"], [2, 2]))

    agg = metric.aggregate([legacy_row, stable_row])

    # Only the stable row contributed -> agreement 1.0, no inconsistency.
    for dimension in DIMENSIONS:
        stability = agg[dimension]["stability"]
        assert math.isclose(stability["agreement_rate"], 1.0)
        assert math.isclose(stability["inconsistent_rate"], 0.0)
