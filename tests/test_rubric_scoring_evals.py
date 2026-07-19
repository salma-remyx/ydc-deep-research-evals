"""Tests for the rubric-based scoring integration.

These cover two things:
  1. The wiring edit in the existing call site (``evals.deep_research_pairwise_evals``)
     — the ``--rubric-scoring`` flag delegates to ``evals.rubric_scoring_evals.run``.
  2. The paper's core diagnostic — the aggregate weight-tier pass-rate breakdown
     that reveals critical-criteria underperformance.

Neither test makes a real OpenAI call: the wiring test monkeypatches ``run``,
and the aggregation test exercises the pure (no-model) math directly.
"""

import sys


def test_rubric_scoring_flag_delegates_to_run(monkeypatch):
    """The existing CLI's --rubric-scoring flag must delegate to the new path."""
    from evals import deep_research_pairwise_evals, rubric_scoring_evals

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"delegated": True}

    monkeypatch.setattr(rubric_scoring_evals, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--rubric-scoring",
            "--input-data",
            "in.csv",
            "--output-dir",
            "/tmp/rubric-out",
            "--model",
            "o3-mini-2025-01-31",
            "--num-workers",
            "2",
        ],
    )

    result = deep_research_pairwise_evals.main()

    assert result == {"delegated": True}
    assert captured["input_data"] == "in.csv"
    assert captured["output_dir"] == "/tmp/rubric-out"
    assert captured["model"] == "o3-mini-2025-01-31"
    assert captured["num_workers"] == 2


def test_rubric_scoring_flag_default_is_off(monkeypatch):
    """Without --rubric-scoring the flag is False (pairwise path unchanged)."""
    from evals import deep_research_pairwise_evals

    monkeypatch.setattr(sys, "argv", ["prog", "--output-dir", "/tmp/out"])
    args = deep_research_pairwise_evals.parse_args()
    assert args.rubric_scoring is False


def test_aggregate_reports_priority_inversion_by_weight():
    """Weight-tier breakdown must separate weight-1 from weight-5 pass rates.

    Mirrors the paper's central finding: low-weight criteria pass at high rates
    while critical (weight-5) criteria pass at low rates, which an aggregate
    score alone would hide.
    """
    from evals.metrics.deep_research_pairwise_metric import DIMENSIONS
    from evals.rubric_scoring_evals import (
        MAX_WEIGHT,
        MIN_WEIGHT,
        RubricCriterionJudgment,
        RubricScoreResult,
        RubricScoringMetric,
    )

    def judgment(dimension, weight, met):
        return RubricCriterionJudgment(
            dimension=dimension,
            weight=weight,
            criterion="c",
            met=met,
            explanation="e",
        )

    # Four weight-1 criteria all met; four weight-5 criteria all missed.
    criteria = [
        judgment("comprehensiveness", MIN_WEIGHT, True),
        judgment("instruction_following", MIN_WEIGHT, True),
        judgment("writing_quality", MIN_WEIGHT, True),
        judgment("completeness", MIN_WEIGHT, True),
        judgment("comprehensiveness", MAX_WEIGHT, False),
        judgment("completeness", MAX_WEIGHT, False),
        judgment("comprehensiveness", MAX_WEIGHT, False),
        judgment("completeness", MAX_WEIGHT, False),
    ]
    result = RubricScoreResult(question="q", criteria=criteria)

    aggregate = RubricScoringMetric().aggregate([result])

    assert aggregate["support"] == 1
    assert aggregate["total_criteria"] == 8
    # The headline inversion: weight-1 fully met, weight-5 fully missed.
    assert aggregate["pass_rate_by_weight"][MIN_WEIGHT] == 1.0
    assert aggregate["pass_rate_by_weight"][MAX_WEIGHT] == 0.0
    # Mean hides the inversion (50%), as in the paper.
    assert aggregate["mean_pass_rate"] == 0.5
    assert aggregate["critical_unsatisfied_rate"] == 1.0
    # Dimension keys stay within the repo's shared dimension vocabulary.
    assert set(aggregate["pass_rate_by_dimension"]).issubset(set(DIMENSIONS))


def test_aggregate_handles_empty_input():
    """An empty corpus must not raise; it reports zero support."""
    from evals.rubric_scoring_evals import RubricScoringMetric

    aggregate = RubricScoringMetric().aggregate([])
    assert aggregate["support"] == 0
    assert aggregate["total_criteria"] == 0


def test_weighted_pass_rate_weights_critical_criteria_higher():
    """weighted_pass_rate must fall when a critical criterion is missed."""
    from evals.rubric_scoring_evals import (
        RubricCriterionJudgment,
        RubricScoreResult,
    )

    def make(met_critical):
        return RubricScoreResult(
            question="q",
            criteria=[
                RubricCriterionJudgment(
                    dimension="completeness",
                    weight=1,
                    criterion="c",
                    met=True,
                    explanation="e",
                ),
                RubricCriterionJudgment(
                    dimension="comprehensiveness",
                    weight=5,
                    criterion="c",
                    met=met_critical,
                    explanation="e",
                ),
            ],
        )

    # Passed weight: 1 + 5 = 6 of 6 -> 1.0
    assert make(True).weighted_pass_rate == 1.0
    # Passed weight: 1 of 6 -> ~0.1667
    assert round(make(False).weighted_pass_rate, 4) == round(1 / 6, 4)
