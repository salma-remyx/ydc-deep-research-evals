"""Integration tests for the multi-agent debate judge architecture.

These tests exercise the debate metric's wiring against the repo's *existing*
pairwise infrastructure (``evals.metrics.deep_research_pairwise_metric``): the
debate metric subclasses ``DeepResearchPairwiseMetric``, overrides the
single-pass call site ``_query_evaluation_model``, and its verdicts must flow
through the inherited flipped-trial consensus and aggregation unchanged.

No real API calls are made: ``query_openai_model_structured_outputs`` is
monkeypatched on the debate module's namespace. Dummy OpenAI credentials are set
before import because ``evals.utils`` constructs its client at import time.
"""

import os

# evals.utils reads these at import time to build the OpenAI client; the values
# are never used because the model call is monkeypatched below.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

import pytest  # noqa: E402

from evals.deep_research_pairwise_evals import (  # noqa: E402
    DeepResearchEvaluator,
)
from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchPairwisePreferenceOutput,
    DeepResearchScoreResult,
    Preference,
)
from evals.multi_agent_debate_evals import (  # noqa: E402
    MultiAgentDebateEvaluator,
)
from evals.multi_agent_debate_metric import (  # noqa: E402
    DEFAULT_DEBATER_PERSPECTIVES,
    MultiAgentDebateMetric,
)


def _full_verdict() -> DeepResearchPairwisePreferenceOutput:
    """A valid verdict with every dimension populated (candidate 'b' preferred)."""
    return DeepResearchPairwisePreferenceOutput(
        **{
            dim: Preference(
                explanation=f"candidate b is stronger on {dim}",
                preferred="b",
                gap_score=2,
            )
            for dim in DIMENSIONS
        }
    )


@pytest.fixture
def fake_model(monkeypatch):
    """Monkeypatch the debate's model seam; record the system prompts it sees."""
    seen_systems = []

    def _fake_structured_call(messages, output_class, **kwargs):
        seen_systems.append(messages[0]["content"])
        return _full_verdict()

    monkeypatch.setattr(
        "evals.multi_agent_debate_metric.query_openai_model_structured_outputs",
        _fake_structured_call,
    )
    return seen_systems


def test_debate_metric_is_subclass_overriding_call_site():
    # The debate metric plugs into the existing single-pass metric by overriding
    # the named call site _query_evaluation_model.
    assert issubclass(MultiAgentDebateMetric, DeepResearchPairwiseMetric)
    assert (
        MultiAgentDebateMetric._query_evaluation_model
        is not DeepResearchPairwiseMetric._query_evaluation_model
    )


def test_score_flows_through_inherited_consensus_and_schema(fake_model):
    metric = MultiAgentDebateMetric(
        num_trials=1, num_workers=1, n_debaters=2, n_rounds=1
    )
    result = metric.score(
        question="What are the economic impacts of climate change?",
        baseline_answer=(
            "Climate change affects agriculture, infrastructure, and healthcare."
        ),
        candidate_answer=(
            "Climate change drives agricultural disruption, coastal damage, "
            "rising healthcare costs, and energy demand shifts."
        ),
    )

    # Inherited schema: a full DeepResearchScoreResult with all four dimensions.
    assert isinstance(result, DeepResearchScoreResult)
    for dim in DIMENSIONS:
        dim_result = getattr(result, dim)
        assert dim_result.grade in {"win", "lose", "tie"}
        assert 0.0 <= dim_result.score <= 10.0

    # Inherited aggregation runs unchanged against debate verdicts.
    aggregate = metric.aggregate([result])
    assert aggregate["support"] == 1
    assert set(aggregate["overall"]) == {
        "win_rate",
        "tie_rate",
        "lose_rate",
        "avg_score",
        "net_winrate",
    }


def test_debate_uses_more_calls_than_single_pass(fake_model):
    # The paper's cost finding: debate issues many more model calls than the
    # single-pass judge for the same rows. Per trial-half the debate runs
    # n_debaters opening + n_debaters*n_rounds revisions + 1 chair.
    n_debaters, n_rounds, num_trials = 2, 1, 1
    metric = MultiAgentDebateMetric(
        num_trials=num_trials, num_workers=1, n_debaters=n_debaters, n_rounds=n_rounds
    )
    metric.score(
        question="Summarize the trade-offs of remote work.",
        baseline_answer="Remote work offers flexibility.",
        candidate_answer="Remote work offers flexibility but blurs boundaries.",
    )

    expected = 2 * num_trials * (n_debaters * (1 + n_rounds) + 1)
    assert metric.model_call_count == expected
    # Single-pass would issue exactly one call per trial-half.
    assert metric.model_call_count > 2 * num_trials

    # And the recorded system prompts prove the debate structure ran: debater
    # perspectives plus the chair synthesis prompt.
    joined = "\n".join(fake_model)
    assert "Debater" in joined
    assert "chair of a panel" in joined


def test_evaluator_swaps_in_debate_metric():
    evaluator = MultiAgentDebateEvaluator(
        debaters=3, debate_rounds=1, metric_num_trials=1
    )
    assert isinstance(evaluator, DeepResearchEvaluator)
    assert isinstance(evaluator.pairwise_metric, MultiAgentDebateMetric)
    assert evaluator.pairwise_metric.debater_perspectives == (
        DEFAULT_DEBATER_PERSPECTIVES
    )
