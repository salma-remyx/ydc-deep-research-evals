"""Integration tests for the investment-logic (P->E->R->D->O) process-trace eval.

These import the NON-NEW call-site module ``evals.deep_research_pairwise_evals``
(the file whose ``--investment-logic-scoring`` flag dispatches into the new
capability) and exercise the wiring plus the metric's scoring/aggregation path
with a mocked judge, so no OpenAI API key or network call is required.
"""

# evals.utils constructs the OpenAI client at import time, so the API env vars
# must be present before any evals module is imported.
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from unittest import mock  # noqa: E402

# Import from the NON-NEW call-site module to prove integration.
import evals.deep_research_pairwise_evals as pairwise_cli  # noqa: E402
from evals.investment_logic_evals import InvestmentLogicEvaluator  # noqa: E402
from evals.metrics.investment_logic_metric import (  # noqa: E402
    INVESTMENT_LOGIC_DIMENSIONS,
    InvestmentLogicMetric,
    InvestmentLogicPreferenceOutput,
    InvestmentLogicScoreResult,
)
from evals.metrics.deep_research_pairwise_metric import Preference  # noqa: E402

VALID_GRADES = {"win", "tie", "lose"}


def _synthetic_preference_output() -> InvestmentLogicPreferenceOutput:
    """A fixed judge output: every dimension prefers report_b, gap 4."""
    return InvestmentLogicPreferenceOutput(
        logical_plausibility=Preference(
            explanation="b reasons more coherently.", preferred="b", gap_score=4
        ),
        event_grounding=Preference(
            explanation="b cites specific market evidence.", preferred="b", gap_score=4
        ),
        process_completeness=Preference(
            explanation="b traces evidence to a decision.", preferred="b", gap_score=4
        ),
    )


def _patched_metric(num_trials=1, num_workers=1) -> InvestmentLogicMetric:
    metric = InvestmentLogicMetric(
        eval_model="o3-mini-2025-01-31",
        num_trials=num_trials,
        num_workers=num_workers,
    )
    # The same fixed preference is returned for both original and flipped
    # trials; flipping it back yields a tie at score 5.0 -- deterministic.
    metric._query_evaluation_model = mock.Mock(
        side_effect=lambda messages: _synthetic_preference_output()
    )
    return metric


QUESTION = (
    "Analyze the investment implications of TikTok bans for ad-dependent "
    "companies and recommend how they can strategically navigate the risk."
)
BASELINE = (
    "TikTok bans would hurt some advertisers. Companies should diversify "
    "and consider alternative platforms to manage the risk."
)
CANDIDATE = (
    "TikTok reaches ~170M US users; a ban would shift ~$10-20B of ad spend. "
    "Ad-dependent names (e.g., Meta, Snap) see a mix shift toward Reels/"
    "Shorts. Recommendation: overweight Meta ( diversified scale, Reels "
    "monetization improving), underweight pure-play short-video names with "
    "concentrated TikTok exposure, 12-month horizon."
)


def test_call_site_flag_is_recognized():
    """The wiring edit: --investment-logic-scoring is parsed by the base CLI."""
    with mock.patch(
        "sys.argv",
        [
            "deep_research_pairwise_evals.py",
            "--input-data",
            "x.csv",
            "--output-dir",
            "out",
            "--investment-logic-scoring",
        ],
    ):
        args = pairwise_cli.parse_args()
    assert args.investment_logic_scoring is True


def test_call_site_dispatch_invokes_sibling_cli():
    """When the flag is set, main() delegates to the investment-logic CLI."""
    sibling_main = mock.Mock(return_value=None)
    with mock.patch(
        "sys.argv",
        [
            "deep_research_pairwise_evals.py",
            "--output-dir",
            "out",
            "--investment-logic-scoring",
        ],
    ), mock.patch(
        "evals.investment_logic_evals.main", side_effect=sibling_main
    ) as patched:
        pairwise_cli.main()
    patched.assert_called_once()


def test_metric_score_is_well_formed():
    metric = _patched_metric()
    result = metric.score(
        question=QUESTION, baseline_answer=BASELINE, candidate_answer=CANDIDATE
    )
    assert isinstance(result, InvestmentLogicScoreResult)
    # All three P->E->R->D->O dimensions present, no more, no less.
    assert set(result.model_dump().keys()) == set(INVESTMENT_LOGIC_DIMENSIONS)
    for dimension in INVESTMENT_LOGIC_DIMENSIONS:
        dim = getattr(result, dimension)
        assert dim.grade in VALID_GRADES
        assert 0.0 <= dim.score <= 10.0
        assert (dim.is_win, dim.is_tie, dim.is_lose).count(True) == 1
    # Fixed-mock tie case: flipped preference cancels out -> tie at score 5.0.
    assert result.logical_plausibility.is_tie
    assert result.logical_plausibility.score == 5.0


def test_aggregate_emits_grounding_diagnostic():
    metric = _patched_metric()
    scored = [
        metric.score(
            question=QUESTION, baseline_answer=BASELINE, candidate_answer=CANDIDATE
        )
        for _ in range(2)
    ]
    agg = metric.aggregate(scored)
    assert agg["support"] == 2
    assert set(agg["overall"]) == {
        "win_rate",
        "tie_rate",
        "lose_rate",
        "avg_score",
        "net_winrate",
    }
    # Plausibility and grounding both average to 5.0 here -> no gap.
    assert agg["process_grounding_gap"] == 0.0
    assert agg["grounding_warning"] == "plausibility_and_grounding_aligned"


def test_aggregate_flags_polished_but_weakly_grounded():
    metric = _patched_metric()
    scored = [
        metric.score(
            question=QUESTION, baseline_answer=BASELINE, candidate_answer=CANDIDATE
        )
        for _ in range(2)
    ]
    # Simulate InvestLogicBench's headline: high plausibility, low grounding.
    for s in scored:
        s.event_grounding = s.event_grounding.model_copy(
            update={"score": 2.0, "is_win": False, "is_tie": True, "is_lose": False}
        )
        s.logical_plausibility = s.logical_plausibility.model_copy(
            update={"score": 8.0}
        )
    agg = metric.aggregate(scored)
    assert agg["process_grounding_gap"] > 1.0
    assert agg["grounding_warning"] == "polished_but_weakly_grounded"


def test_evaluator_integration_path():
    """Evaluator -> metric -> score integration with a mocked judge."""
    evaluator = InvestmentLogicEvaluator(
        model="o3-mini-2025-01-31",
        metric_num_workers=1,
        metric_num_trials=1,
    )
    evaluator.pairwise_metric._query_evaluation_model = mock.Mock(
        side_effect=lambda messages: _synthetic_preference_output()
    )
    result = evaluator.evaluate_single(
        question=QUESTION,
        baseline_answer=BASELINE,
        candidate_answer=CANDIDATE,
    )
    assert result["success"] is True
    # The score_result round-trips through the process-trace schema.
    parsed = InvestmentLogicScoreResult.model_validate(result["score_result"])
    assert set(parsed.model_dump().keys()) == set(INVESTMENT_LOGIC_DIMENSIONS)
    # Per-dimension columns the base evaluate_single writes are present.
    for dimension in INVESTMENT_LOGIC_DIMENSIONS:
        assert f"{dimension}_score" in result
        assert f"{dimension}_grade" in result
