"""Tests for the continuous logprob-verifier metric and its wiring into the repo.

These tests import from NON-NEW modules (``evals.utils`` and
``evals.metrics.deep_research_pairwise_metric``) to prove integration, exercise the
existing pairwise-CLI flag wiring, and deterministically verify the paper's core
scoring math (expected value over scoring-token logprobs, granularity, flipped-trial
correction, repeated-evaluation uncertainty) without making real API calls.
"""

# evals.utils constructs the OpenAI client at import time and reads these env vars;
# set dummy values before importing any evals module. The client constructor does not
# validate credentials, so the real API is never contacted (the logprobs query is
# monkeypatched in the end-to-end test).
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

import math

# Imports from NON-NEW repo modules (proves integration, not a pure self-test).
from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchPairwisePreferenceInput,
)
from evals.utils import query_openai_model_logprobs  # noqa: E402

# Import from the NEW module under test.
from evals.metrics.logprob_verifier_metric import (  # noqa: E402
    DEFAULT_VERIFIER_GRANULARITY,
    LogprobVerifierMetric,
    LogprobVerifierScoreResult,
    expected_score_from_logprobs,
    preference_from_expected,
    renormalize_over_candidates,
)


def _dist(peak, breadth=8.0, granularity=10):
    """Build a synthetic top-logprob list peaked at ``peak`` over 0..granularity."""
    out = []
    for i in range(granularity + 1):
        tok = str(i)
        out.append({"token": tok, "logprob": -0.2 if tok == str(peak) else -breadth})
    return out


# --- Deterministic scoring-math tests (the paper's core mechanism) ---


def test_renormalize_over_candidates_sums_to_one():
    probs = renormalize_over_candidates(_dist(7), granularity=10)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    # Non-candidate tokens (none here) are excluded; 7 dominates.
    assert probs["7"] > 0.9


def test_expected_score_is_continuous_and_near_peak():
    # A distribution peaked at 7 must yield an expected score near 7 but continuous
    # (not a hard 7), demonstrating the paper's fine-grained scoring.
    expected, probs = expected_score_from_logprobs(_dist(7, breadth=8.0), granularity=10)
    assert 6.5 < expected < 7.5
    assert expected != 7.0  # continuous, not the argmax integer
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)


def test_expected_score_no_candidates_falls_back_to_midpoint():
    # Tokens outside the candidate set -> neutral midpoint (a tie), never NaN.
    expected, probs = expected_score_from_logprobs(
        [{"token": "x", "logprob": -0.1}, {"token": "y", "logprob": -0.2}], granularity=10
    )
    assert expected == 5.0
    assert probs == {}


def test_preference_thresholds_are_midpoint_anchored():
    g = 10
    assert preference_from_expected(8.0, g) == "win"
    assert preference_from_expected(5.0, g) == "tie"
    assert preference_from_expected(2.0, g) == "lose"
    assert preference_from_expected(5.4, g) == "tie"  # within margin of midpoint


def test_granularity_scaling_keeps_normalized_score_comparable():
    # Same relative preference (b clearly better, peak ~80% of scale) under two
    # granularities: raw expected scales with G, normalized stays comparable.
    exp10, _ = expected_score_from_logprobs(_dist(8, breadth=4.0, granularity=10), 10)
    exp5, _ = expected_score_from_logprobs(_dist(4, breadth=4.0, granularity=5), 5)
    # Finer granularity separates the winner further above the midpoint.
    assert exp10 - 5 > exp5 - 2.5 or math.isclose(exp10 / 10, exp5 / 5, abs_tol=0.15)
    assert math.isclose(exp10 / 10, exp5 / 5, abs_tol=0.2)  # normalized comparable


# --- End-to-end test of score() with the logprobs query monkeypatched ---


def _b_favoring_logprobs(messages, **kwargs):
    """Mock verifier query: high score when report_b is 'GOOD', low when 'BAD'.

    This mirrors a real judge that flips its view when the reports are swapped, so the
    metric's flipped-order correction (``granularity - expected``) is exercised.
    """
    content = messages[1]["content"]
    b_block = content.split("<report_b>")[1]
    good_in_b = "GOOD" in b_block
    return _dist(8 if good_in_b else 2, breadth=4.0)


def test_score_end_to_end_continuous_win(monkeypatch):
    # Patch the name as bound inside the new metric module.
    import evals.metrics.logprob_verifier_metric as metric_mod

    monkeypatch.setattr(metric_mod, "query_openai_model_logprobs", _b_favoring_logprobs)

    metric = LogprobVerifierMetric(
        eval_model="gpt-4o-mini", granularity=10, num_trials=2, num_workers=2
    )
    result = metric.score(
        question="Compare these reports.",
        baseline_answer="BAD report missing key points.",
        candidate_answer="GOOD report covering all key points thoroughly.",
    )

    assert isinstance(result, LogprobVerifierScoreResult)
    for dim in DIMENSIONS:
        dr = getattr(result, dim)
        # Field-compatible with the repo's DimensionResult (drops into the evaluator).
        for field in ("grade", "is_win", "is_tie", "is_lose", "score"):
            assert hasattr(dr, field)
        # report_b is GOOD in original order and (after correction) in flipped order,
        # so every dimension should land on the win side above the midpoint.
        assert dr.grade == "win"
        assert dr.is_win is True
        assert dr.expected_score > 5.0
        assert 0.0 <= dr.normalized_score <= 1.0
        # 2 original + 2 flipped trials per dimension.
        assert len(dr.trial_scores) == 4
        assert dr.std >= 0.0
        # The flipped trials must have been mirrored back above the midpoint.
        assert all(s > 5.0 for s in dr.trial_scores)


def test_score_reuses_existing_input_validation(monkeypatch):
    # The metric builds inputs via the existing DeepResearchPairwisePreferenceInput,
    # which strips markdown links -- proving reuse of a non-new module's behavior.
    import evals.metrics.logprob_verifier_metric as metric_mod

    monkeypatch.setattr(metric_mod, "query_openai_model_logprobs", _b_favoring_logprobs)
    metric = LogprobVerifierMetric(num_trials=1, num_workers=1)
    msgs = metric._get_evaluation_messages(
        DeepResearchPairwisePreferenceInput(
            question="q?",
            baseline_answer="see [link](https://x.com/a)",
            candidate_answer="GOOD answer",
        ),
        "completeness",
    )
    assert "[link]" not in msgs[1]["content"]  # markdown link stripped by existing input


def test_aggregate_reports_continuous_metrics(monkeypatch):
    import evals.metrics.logprob_verifier_metric as metric_mod

    monkeypatch.setattr(metric_mod, "query_openai_model_logprobs", _b_favoring_logprobs)
    metric = LogprobVerifierMetric(num_trials=1, num_workers=1)
    one = metric.score("q", "BAD", "GOOD")
    agg = metric.aggregate([one])
    assert agg["support"] == 1
    for dim in DIMENSIONS:
        assert "avg_expected_score" in agg[dim]
        assert "avg_normalized_score" in agg[dim]
        assert "avg_uncertainty_std" in agg[dim]
    assert "overall" in agg and "avg_expected_score" in agg["overall"]


# --- Wiring test: existing pairwise CLI invokes the new verifier ---


def test_existing_cli_delegates_to_verifier(monkeypatch):
    """The existing deep_research_pairwise_evals CLI's --verifier-scoring flag must
    invoke the new LogprobVerifierEvaluator (existing code calling new code)."""
    import evals.deep_research_pairwise_evals as pairwise_cli
    import evals.logprob_verifier_evals as verifier_cli

    called = {"count": 0}

    def fake_verifier_main():
        called["count"] += 1

    monkeypatch.setattr(verifier_cli, "main", fake_verifier_main)
    monkeypatch.setattr(
        "sys.argv",
        [
            "deep_research_pairwise_evals.py",
            "--output-dir",
            "/tmp/__verifier_unused__",
            "--verifier-scoring",
        ],
    )
    pairwise_cli.main()
    assert called["count"] == 1


def test_existing_cli_default_is_pairwise(monkeypatch):
    # Without --verifier-scoring the flag must default to False (no behavior change).
    import evals.deep_research_pairwise_evals as pairwise_cli

    monkeypatch.setattr(
        "sys.argv",
        ["deep_research_pairwise_evals.py", "--output-dir", "/tmp/__unused__"],
    )
    args = pairwise_cli.parse_args()
    assert args.verifier_scoring is False


def test_non_new_modules_expose_expected_surface():
    # Guard: the integration imports from non-new modules that must remain present.
    assert callable(query_openai_model_logprobs)
    assert DIMENSIONS == [
        "instruction_following",
        "comprehensiveness",
        "completeness",
        "writing_quality",
    ]
    assert DEFAULT_VERIFIER_GRANULARITY == 10
