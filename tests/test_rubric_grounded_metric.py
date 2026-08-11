"""Integration tests for the rubric-grounded metric.

These import from the *existing* modules (``evals.deep_research_pairwise_evals``
and ``evals.metrics.deep_research_pairwise_metric``) and exercise the
``--rubric-grounded-scoring`` wiring plus the pure structure-weighting logic.
The LLM-backed synthesis/scoring methods are stubbed so no network is used.
"""

import pytest

# Imported from a NON-NEW module: the existing evaluator entry point.
from evals.deep_research_pairwise_evals import DeepResearchEvaluator
# Imported from a NON-NEW module: the existing result contract.
from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
)
# Imported from the NEW module under test.
from evals.metrics.rubric_grounded_metric import (
    CheckpointVerdict,
    ResearchDAG,
    ResearchStep,
    RubricGroundedMetric,
)

QUESTION = "Compare the economic and environmental impacts of solar vs wind energy."


def _build_dag(_question: str) -> ResearchDAG:
    """A small DAG: a root step (s1) with two descendants, each with checkpoints."""
    return ResearchDAG(
        question=QUESTION,
        steps=[
            ResearchStep(
                id="s1",
                description="Identify the economic factors for each technology.",
                depends_on=[],
                dimension="comprehensiveness",
                checkpoints=["capital cost per watt", "operating cost", "available subsidies"],
            ),
            ResearchStep(
                id="s2",
                description="Quantify the environmental tradeoffs.",
                depends_on=["s1"],
                dimension="completeness",
                checkpoints=["land use footprint", "lifecycle emissions", "wildlife impact"],
            ),
            ResearchStep(
                id="s3",
                description="Synthesize a recommendation with caveats.",
                depends_on=["s1", "s2"],
                dimension="completeness",
                checkpoints=["explicit recommendation", "stated caveats"],
            ),
        ],
    )


def _verdicts_for(dag: ResearchDAG, coverage: str) -> list:
    """Return one verdict per checkpoint, all at the given coverage level."""
    return [
        CheckpointVerdict(
            step_id=sid,
            dimension=dim,
            checkpoint=cp,
            coverage=coverage,
            evidence=f"stub {coverage}",
        )
        for sid, dim, cp in dag.checkpoints()
    ]


# --------------------------------------------------------------------------- #
# Wiring: the --rubric-grounded-scoring flag routes the evaluator to the new metric.
# --------------------------------------------------------------------------- #


def test_default_evaluator_uses_pairwise_metric():
    evaluator = DeepResearchEvaluator()
    assert isinstance(evaluator.pairwise_metric, DeepResearchPairwiseMetric)


def test_rubric_flag_swaps_in_rubric_metric():
    evaluator = DeepResearchEvaluator(use_rubric_grounded_metric=True)
    assert isinstance(evaluator.pairwise_metric, RubricGroundedMetric)


def test_evaluate_single_routes_through_rubric_metric(monkeypatch):
    """The edited call site (evaluate_single) must invoke the rubric metric."""
    evaluator = DeepResearchEvaluator(
        use_rubric_grounded_metric=True, metric_num_workers=1
    )
    metric = evaluator.pairwise_metric
    assert isinstance(metric, RubricGroundedMetric)

    # Stub the LLM-backed methods: baseline misses every checkpoint, candidate
    # covers them all, so the candidate should win the grounded dimensions.
    dag = _build_dag(QUESTION)
    monkeypatch.setattr(metric, "build_research_dag", lambda question: dag)
    monkeypatch.setattr(
        metric,
        "score_checkpoints",
        lambda d, answer, label: (
            _verdicts_for(d, "covered")
            if label == "report_b"
            else _verdicts_for(d, "missing")
        ),
    )

    result = evaluator.evaluate_single(
        question=QUESTION,
        baseline_answer="A sparse answer that skips most points.",
        candidate_answer="A thorough answer covering every checkpoint in depth.",
    )

    assert result["success"] is True
    # The result object must satisfy the existing (non-new) result contract.
    score_result = DeepResearchScoreResult.model_validate(result["score_result"])
    assert score_result.comprehensiveness.is_win
    assert score_result.completeness.is_win
    # Candidate covered everything -> per-dimension score at the ceiling.
    assert score_result.comprehensiveness.score == pytest.approx(10.0)
    assert score_result.completeness.score == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Pure logic: structure weighting + pointwise aggregation (no network).
# --------------------------------------------------------------------------- #


def test_structure_weights_favor_ancestor_steps_and_normalize():
    dag = _build_dag(QUESTION)
    weights = RubricGroundedMetric.structure_weights(dag)
    # s1 unlocks s2 and s3, s2 unlocks s3, s3 is a leaf.
    assert weights["s1"] > weights["s2"] > weights["s3"]
    # Weights form a probability distribution over steps.
    assert sum(weights.values()) == pytest.approx(1.0)


def test_aggregate_verdicts_reports_win_when_candidate_covers_more():
    dag = _build_dag(QUESTION)
    weights = RubricGroundedMetric.structure_weights(dag)
    result = RubricGroundedMetric.aggregate_verdicts_to_dimensions(
        dag,
        baseline_verdicts=_verdicts_for(dag, "partial"),
        candidate_verdicts=_verdicts_for(dag, "covered"),
        weights=weights,
    )
    assert isinstance(result, DeepResearchScoreResult)
    # Every grounded dimension should show a candidate win.
    for dim in ("comprehensiveness", "completeness"):
        assert getattr(result, dim).is_win
        assert getattr(result, dim).grade == "win"
    # Raw preferences carry the checkpoint-level transparency payload.
    assert result.completeness.raw_preferences["num_checkpoints"] == 5


def test_aggregate_method_emits_standard_shape_plus_rubric_extras():
    dag = _build_dag(QUESTION)
    weights = RubricGroundedMetric.structure_weights(dag)
    score = RubricGroundedMetric.aggregate_verdicts_to_dimensions(
        dag,
        baseline_verdicts=_verdicts_for(dag, "missing"),
        candidate_verdicts=_verdicts_for(dag, "covered"),
        weights=weights,
    )
    aggregated = RubricGroundedMetric().aggregate([score])

    assert aggregated["support"] == 1
    assert set(DIMENSIONS).issubset(aggregated.keys())
    # Same overall keys the pairwise metric exposes, plus rubric extras.
    for key in ("win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"):
        assert key in aggregated["overall"]
    assert aggregated["overall"]["avg_candidate_checkpoint_coverage"] > 0.0
    assert aggregated["overall"]["avg_baseline_checkpoint_coverage"] == pytest.approx(0.0)
    assert aggregated["overall"]["total_checkpoints"] == 8


def test_score_rejects_empty_answers():
    metric = RubricGroundedMetric()
    with pytest.raises(ValueError):
        metric.score(question=QUESTION, baseline_answer="   ", candidate_answer="ok")
