"""Integration tests for the CM-LRS bankability scoring wiring.

These exercise the call-site edit in ``evals.deep_research_pairwise_evals`` -- the
existing, non-new module -- so they prove the new ``BankabilityMetric`` is actually
invoked by the repo's evaluator. The OpenAI structured-output helper is monkeypatched
so the tests are deterministic and network-free.
"""

import evals.metrics.bankability_metric as bankability_module
import evals.metrics.deep_research_pairwise_metric as pairwise_module
from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.bankability_metric import (
    BANKABILITY_DIMENSIONS,
    BankabilityDimensionScore,
    BankabilityMetric,
    BankabilityOutput,
    BankabilityScoreResult,
    _resolve_weights,
    weighted_aggregate,
)
from evals.metrics.deep_research_pairwise_metric import (
    DeepResearchPairwisePreferenceOutput,
    Preference,
)


def _canned_bankability_output() -> BankabilityOutput:
    def dim(score: int, explanation: str) -> BankabilityDimensionScore:
        return BankabilityDimensionScore(score=score, explanation=explanation)

    return BankabilityOutput(
        factual_accuracy=dim(4, "Figures match the cited prospectus."),
        evidence_traceability=dim(3, "Most claims cite a source."),
        numerical_consistency=dim(5, "Totals reconcile with components."),
        workflow_completeness=dim(4, "All requested sections present."),
        source_discipline=dim(2, "Two sources are under-specified."),
        decision_usefulness=dim(5, "Clear recommendation with trade-offs."),
        reviewability_auditability=dim(3, "Reasoning is mostly traceable."),
    )


def _canned_pairwise_output() -> DeepResearchPairwisePreferenceOutput:
    pref = Preference(explanation="B is more complete.", preferred="b", gap_score=2)
    return DeepResearchPairwisePreferenceOutput(
        instruction_following=pref,
        comprehensiveness=pref,
        completeness=pref,
        writing_quality=pref,
    )


def _install_fakes(monkeypatch) -> None:
    def fake_structured(messages, output_class, **kwargs):
        if output_class is BankabilityOutput:
            return _canned_bankability_output()
        if output_class is DeepResearchPairwisePreferenceOutput:
            return _canned_pairwise_output()
        raise AssertionError(f"Unexpected output_class: {output_class}")

    monkeypatch.setattr(
        bankability_module,
        "query_openai_model_structured_outputs",
        fake_structured,
    )
    monkeypatch.setattr(
        pairwise_module,
        "query_openai_model_structured_outputs",
        fake_structured,
    )
    # aggregate_results -> pairwise aggregate -> explanation summary hits the text API.
    monkeypatch.setattr(
        pairwise_module, "query_openai_model", lambda **kwargs: {"content": "summary"}
    )


# --- pure aggregation logic (no network, no monkeypatch) ---


def test_weighted_aggregate_equal_weights_defaults_to_mean():
    scores = {dim: 5.0 for dim in BANKABILITY_DIMENSIONS}
    assert weighted_aggregate(scores) == 5.0
    scores = {dim: 0.0 for dim in BANKABILITY_DIMENSIONS}
    assert weighted_aggregate(scores) == 0.0


def test_weighted_aggregate_honors_tunable_weights():
    scores = {
        "factual_accuracy": 0.0,
        "evidence_traceability": 0.0,
        "numerical_consistency": 0.0,
        "workflow_completeness": 0.0,
        "source_discipline": 0.0,
        "decision_usefulness": 5.0,
        "reviewability_auditability": 0.0,
    }
    # Put all the weight on decision_usefulness (the paper's most discriminative dim).
    weights = {"decision_usefulness": 1.0}
    assert weighted_aggregate(scores, weights) == 5.0


def test_resolve_weights_normalizes_partial_map():
    resolved = _resolve_weights({"decision_usefulness": 3.0, "factual_accuracy": 1.0})
    assert resolved["decision_usefulness"] == 0.75
    assert resolved["factual_accuracy"] == 0.25
    # Omitted dimensions carry zero weight: the caller chose to emphasize only these.
    for dim in BANKABILITY_DIMENSIONS:
        if dim not in ("decision_usefulness", "factual_accuracy"):
            assert resolved[dim] == 0.0
    assert abs(sum(resolved.values()) - 1.0) < 1e-9


def test_resolve_weights_empty_map_falls_back_to_equal():
    resolved = _resolve_weights({})
    assert abs(sum(resolved.values()) - 1.0) < 1e-9
    first = resolved[BANKABILITY_DIMENSIONS[0]]
    assert all(abs(resolved[dim] - first) < 1e-9 for dim in BANKABILITY_DIMENSIONS)


# --- call-site integration (exercises the wiring in the existing module) ---


def test_evaluator_constructs_bankability_metric_when_enabled():
    evaluator = DeepResearchEvaluator(bankability_scoring=True)
    assert isinstance(evaluator.bankability_metric, BankabilityMetric)


def test_evaluator_omits_bankability_metric_by_default():
    evaluator = DeepResearchEvaluator()
    assert evaluator.bankability_metric is None


def test_evaluate_single_emits_bankability_scores(monkeypatch):
    _install_fakes(monkeypatch)
    evaluator = DeepResearchEvaluator(
        bankability_scoring=True, metric_num_trials=1, metric_num_workers=1
    )

    result = evaluator.evaluate_single(
        question="Summarize the debt issuance and use of proceeds.",
        baseline_answer="Baseline report text.",
        candidate_answer="Candidate report text with citations.",
    )

    assert result["bankability_success"] is True
    assert "bankability_score_result" in result
    # The aggregate is the paper's tunable weighted mean over the seven dimensions.
    assert 0.0 <= result["bankability_aggregate_score"] <= 5.0
    for dim in BANKABILITY_DIMENSIONS:
        assert 0.0 <= result[f"bankability_{dim}_score"] <= 5.0
    # Round-trips through the pydantic result model.
    BankabilityScoreResult.model_validate(result["bankability_score_result"])


def test_aggregate_results_includes_bankability_block(monkeypatch):
    _install_fakes(monkeypatch)
    evaluator = DeepResearchEvaluator(
        bankability_scoring=True, metric_num_trials=1, metric_num_workers=1
    )

    results = [
        evaluator.evaluate_single(
            question="Q",
            baseline_answer="baseline",
            candidate_answer="candidate",
        )
    ]
    aggregated = evaluator.aggregate_results(results)

    assert "bankability" in aggregated
    bankability = aggregated["bankability"]
    assert bankability["support"] == 1
    assert 0.0 <= bankability["overall"]["avg_score"] <= 5.0
    for dim in BANKABILITY_DIMENSIONS:
        assert 0.0 <= bankability[dim]["avg_score"] <= 5.0
