"""Tests for the partial-evaluation decision layer and its wiring into the
pairwise metric's aggregate step."""

import os

# evals.utils reads the OpenAI config at import time; these tests stub the
# network call, so set safe defaults if the real values are absent.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.metrics.partial_eval_decision import (
    ABSTAIN,
    BETTER,
    NEEDS_EVIDENCE,
    NOT_BETTER,
    ComparisonPolicy,
    decide,
    summarize_run,
)


def _dimension(grade: str) -> DimensionResult:
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=1.0,
        preferred=[],
        raw_preferences={},
    )


def _score(grade: str) -> DeepResearchScoreResult:
    """A question whose every dimension resolves to the same grade."""
    return DeepResearchScoreResult(**{dim: _dimension(grade) for dim in DIMENSIONS})


WIN = (True, False, False)
LOSE = (False, True, False)
TIE = (False, False, True)


def test_decide_clean_wins_and_losses():
    assert decide([WIN] * 30) == BETTER
    assert decide([LOSE] * 30) == NOT_BETTER
    # too few decided outcomes to commit
    assert decide([WIN, WIN]) == NEEDS_EVIDENCE


def test_decide_balanced_abstains_when_pinned_to_tie():
    outcomes = [WIN] * 100 + [LOSE] * 100
    assert decide(outcomes, ComparisonPolicy(margin=0.1)) == ABSTAIN


def test_summarize_run_reports_early_stopping_for_clean_sweep():
    summary = summarize_run([WIN] * 50)
    assert summary["verdict"] == BETTER
    assert summary["decided"] == 50
    assert summary["questions_to_decision"] is not None
    # a clean sweep is decidable well before the full run
    assert summary["fraction_needed"] < 0.5


def test_summarize_run_never_resolves_reports_full_run():
    # near-50/50 with too few outcomes to pin the indifference band
    summary = summarize_run([WIN, LOSE, WIN, LOSE, WIN, LOSE])
    assert summary["verdict"] == NEEDS_EVIDENCE
    assert summary["questions_to_decision"] is None
    assert summary["fraction_needed"] == 1.0


def test_aggregate_emits_partial_eval_decision(monkeypatch):
    metric = DeepResearchPairwiseMetric()
    # aggregate() also calls the LLM for an explanation summary; stub it so the
    # test exercises the wiring without network access.
    monkeypatch.setattr(
        metric, "_generate_explanation_summary_from_raw", lambda raw, n=20: "stub"
    )
    scores = [_score("win") for _ in range(30)]

    aggregated = metric.aggregate(scores)

    decision = aggregated["partial_eval_decision"]
    assert set(decision) == {"policy", "overall", "by_dimension"}
    assert decision["policy"] == ComparisonPolicy().to_dict()
    assert decision["overall"]["verdict"] == BETTER
    assert decision["overall"]["fraction_needed"] < 1.0
    assert set(decision["by_dimension"]) == set(DIMENSIONS)
    for dim in DIMENSIONS:
        assert decision["by_dimension"][dim]["verdict"] == BETTER
