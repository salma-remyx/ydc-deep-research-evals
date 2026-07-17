"""Integration tests for the absolute (pointwise) scoring protocol.

These tests import from the *existing* (non-new) modules -- the shared
``DimensionResult`` / ``DeepResearchScoreResult`` contracts and the
``DeepResearchEvaluator`` base class -- and exercise the new wiring:
``AbsoluteScoringEvaluator`` subclassing ``DeepResearchEvaluator`` and
backed by ``DeepResearchAbsoluteMetric``. The structured-output judge call
is mocked, so no network or API key is required.
"""

from unittest.mock import patch

from evals.absolute_scoring_evals import AbsoluteScoringEvaluator
from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_absolute_metric import (
    DeepResearchAbsoluteMetric,
    DeepResearchAbsoluteScoreOutput,
)
from evals.metrics.deep_research_pairwise_metric import (
    DeepResearchScoreResult,
    DimensionResult,
)

JUDGE_PATH = "evals.metrics.deep_research_absolute_metric.query_openai_model_structured_outputs"


def _make_output(if_score, comp, compl, wq):
    """Build a judge response with per-dimension absolute scores."""
    return DeepResearchAbsoluteScoreOutput(
        instruction_following={"explanation": "ok", "score": if_score},
        comprehensiveness={"explanation": "ok", "score": comp},
        completeness={"explanation": "ok", "score": compl},
        writing_quality={"explanation": "ok", "score": wq},
    )


def test_metric_reuses_shared_result_contract():
    """The new metric returns the shared DeepResearchScoreResult contract."""
    metric = DeepResearchAbsoluteMetric(num_trials=1, num_workers=1)
    # Single worker -> candidate task is submitted/run before the baseline
    # task, so completion order (and side_effect order) is deterministic.
    candidate = _make_output(9, 9, 9, 9)
    baseline = _make_output(3, 3, 3, 3)

    with patch(JUDGE_PATH, side_effect=[candidate, baseline]):
        result = metric.score(
            question="What are the economic impacts of climate change?",
            baseline_answer="baseline report",
            candidate_answer="candidate report",
        )

    assert isinstance(result, DeepResearchScoreResult)
    assert isinstance(result.writing_quality, DimensionResult)
    # Candidate outranks baseline on every dimension -> win, score ~9/10.
    assert result.comprehensiveness.grade == "win"
    assert result.comprehensiveness.is_win is True
    assert result.writing_quality.score == 9.0


def test_tie_band_prevents_verdict_flip_on_small_difference():
    """Scores within TIE_BAND resolve as a tie instead of flipping the verdict."""
    metric = DeepResearchAbsoluteMetric(num_trials=1, num_workers=1)
    candidate = _make_output(8, 8, 8, 8)
    baseline = _make_output(8, 8, 8, 8)  # identical -> tie

    with patch(JUDGE_PATH, side_effect=[candidate, baseline]):
        result = metric.score(
            question="q", baseline_answer="b", candidate_answer="c"
        )

    assert result.instruction_following.grade == "tie"
    assert result.instruction_following.is_tie is True


def test_absolute_evaluator_subclasses_and_wires_metric():
    """AbsoluteScoringEvaluator is a DeepResearchEvaluator wired to the absolute metric,
    and evaluate_single flows through that metric via the inherited base-class path."""
    evaluator = AbsoluteScoringEvaluator(
        model="o3-mini-2025-01-31", metric_num_trials=1, metric_num_workers=1
    )

    # Wiring: the inherited DeepResearchEvaluator delegates to the injected
    # metric, which must be the absolute (pointwise) protocol.
    assert isinstance(evaluator, DeepResearchEvaluator)
    assert isinstance(evaluator.pairwise_metric, DeepResearchAbsoluteMetric)

    candidate = _make_output(9, 9, 9, 9)
    baseline = _make_output(2, 2, 2, 2)
    with patch(JUDGE_PATH, side_effect=[candidate, baseline]):
        out = evaluator.evaluate_single("q", "baseline", "candidate")

    assert out["success"] is True
    assert out["comprehensiveness_grade"] == "win"
    assert out["writing_quality_score"] == 9.0


def test_aggregate_is_comparable_to_pairwise_shape():
    """Aggregate exposes the same per-dimension metrics as the pairwise protocol."""
    metric = DeepResearchAbsoluteMetric(num_trials=1, num_workers=1)
    candidate = _make_output(9, 9, 9, 9)
    baseline = _make_output(2, 2, 2, 2)
    with patch(JUDGE_PATH, side_effect=[candidate, baseline]):
        result = metric.score(question="q", baseline_answer="b", candidate_answer="c")

    aggregate = metric.aggregate([result])
    assert aggregate["support"] == 1
    for dimension in ("instruction_following", "comprehensiveness", "completeness", "writing_quality"):
        for key in ("win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"):
            assert key in aggregate[dimension]
    assert aggregate["overall"]["win_rate"] == 1.0
