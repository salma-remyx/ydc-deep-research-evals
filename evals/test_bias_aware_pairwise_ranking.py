"""Integration tests for the bias-aware pairwise ranker.

These import from the NON-NEW ``evals.metrics.deep_research_pairwise_metric``
module (its real ``DeepResearchScoreResult`` / ``DimensionResult`` types) and
exercise the wiring through the new capability module and the runner, proving
the bias-aware aggregation consumes the repo's actual eval contract and
corrects the position bias that naive majority voting introduces.

The metric module builds an OpenAI client at import time, so dummy credentials
are set before importing it.
"""

import math
import os

# evals.utils constructs an OpenAI client at import using these env vars.
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "dummy")

import random  # noqa: E402

from evals.bias_aware_pairwise_evals import bias_aware_aggregate  # noqa: E402
from evals.bias_aware_pairwise_ranking import (  # noqa: E402
    Comparison,
    acquire_next,
    acquire_round_robin,
    comparisons_from_score_results,
    debiased_pairwise_summary,
    fit,
    rank_topk,
)
from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_dimension(original_prefs, flipped_prefs) -> DimensionResult:
    """Build a real DimensionResult carrying position-labeled raw preferences."""
    # preferred is the canonical-frame vote list (original + remapped flipped).
    canonical = list(original_prefs) + [
        "a" if p == "b" else "b" for p in flipped_prefs
    ]
    num_wins = sum(1 for p in canonical if p == "b")
    num_losses = sum(1 for p in canonical if p == "a")
    if num_wins > num_losses:
        grade = "win"
    elif num_wins < num_losses:
        grade = "lose"
    else:
        grade = "tie"

    def to_pref_dump(p: str):
        return {
            "explanation": "synthetic",
            "preferred": p,
            "gap_score": 1,
            "score_b": 1 if p == "b" else -1,
        }

    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=5.0,
        preferred=canonical,
        raw_preferences={
            "original": [to_pref_dump(p) for p in original_prefs],
            "flipped": [to_pref_dump(p) for p in flipped_prefs],
        },
    )


def _biased_score_result(theta: float, beta: float, n_trials: int, rng: random.Random):
    """One DeepResearchScoreResult where candidate truth=theta, judge bias=beta.

    Position convention matches the repo: original order puts the candidate in
    slot b (position_b=1); flipped order puts it in slot a (position_b=0).
    """
    dims = {}
    for dim in DIMENSIONS:
        original, flipped = [], []
        for _ in range(n_trials):
            # original frame: a=baseline, b=candidate; candidate in slot b.
            p_orig = _sigmoid(theta + beta)
            original.append("b" if rng.random() < p_orig else "a")
            # flipped frame: a=candidate, b=baseline; candidate in slot a.
            p_flip = _sigmoid(theta)  # bias term is 0 for the candidate here
            flipped.append("a" if rng.random() < p_flip else "b")
        dims[dim] = _make_dimension(original, flipped)
    return DeepResearchScoreResult(**dims)


def test_adapter_reads_non_new_score_result_types():
    """comparisons_from_score_results must consume real DeepResearchScoreResult."""
    rng = random.Random(7)
    sr = _biased_score_result(theta=0.0, beta=2.0, n_trials=5, rng=rng)
    per_dim = comparisons_from_score_results([sr])
    # 4 dimensions, 2 * n_trials comparisons each (original + flipped).
    for dim in DIMENSIONS:
        assert len(per_dim[dim]) == 10, dim
        positions = {c.covariates["position_b"] for c in per_dim[dim]}
        assert positions == {0.0, 1.0}, dim
        # item identities follow the repo's candidate/baseline contract.
        assert all(c.item_a == "candidate" and c.item_b == "baseline" for c in per_dim[dim])


def test_debiasing_beats_naive_vote_under_position_bias():
    """With a strong position bias and true quality ~0, naive vote is wrong;
    the bias-aware fit recovers ~0 and flags the bias as real."""
    rng = random.Random(1)
    theta_true, beta_true = 0.0, 2.0
    score_results = [
        _biased_score_result(theta_true, beta_true, n_trials=5, rng=rng)
        for _ in range(30)
    ]
    summary = debiased_pairwise_summary(score_results)

    for dim in DIMENSIONS:
        m = summary[dim]
        # Naive vote is inflated well above the true 0.5 win rate.
        assert m["naive_winrate"] > 0.6, (dim, m["naive_winrate"])
        # Debiased advantage recovers the truth (0.0) within ~2 posterior std.
        assert abs(m["candidate_advantage_mean"]) < 2 * m["candidate_advantage_std"], dim
        # The position bias is detected as real (not shrunk to zero) and positive.
        assert not m["position_bias_shrunken"], dim
        assert m["position_bias_mean"] > 0.5, dim


def test_unbiased_judge_yields_shrunken_bias():
    """With no position bias and limited data, the shrinkage prior pulls the
    bias estimate toward zero -- contrasting the ~2.4 recovered under a real
    bias in test_debiasing_beats_naive_vote_under_position_bias. The paper's
    shrinkage matters most in the limited-data regime where a spurious bias
    could otherwise be over-estimated."""
    rng = random.Random(2)
    score_results = [
        _biased_score_result(theta=0.5, beta=0.0, n_trials=3, rng=rng)
        for _ in range(6)
    ]
    # Strong shrinkage (small scale) so the prior actually bites on few trials.
    summary = debiased_pairwise_summary(score_results, prior_bias_scale=0.25)
    for dim in DIMENSIONS:
        m = summary[dim]
        # Estimate stays near zero (contrast: ~2.4 when the bias is real).
        assert abs(m["position_bias_mean"]) < 0.5, (dim, m)
        assert m["position_bias_shrunken"], (dim, m)


def test_runner_aggregate_consumes_jsonl_row_dicts():
    """bias_aware_aggregate validates rows through the non-new metric model."""
    rng = random.Random(3)
    rows = []
    for _ in range(20):
        sr = _biased_score_result(theta=0.0, beta=2.0, n_trials=5, rng=rng)
        rows.append({"success": True, "score_result": sr.model_dump()})
    rows.append({"success": False, "error": "boom"})  # filtered out

    aggregate = bias_aware_aggregate(rows)
    assert aggregate["support"] == 20
    assert "instruction_following" in aggregate
    m = aggregate["instruction_following"]
    assert m["naive_winrate"] > 0.6
    assert abs(m["candidate_advantage_mean"]) < 2 * m["candidate_advantage_std"]


def test_general_ranking_orders_items_by_quality():
    """The general N-item model ranks items by latent quality."""
    rng = random.Random(4)
    qualities = {"a": 3.0, "b": 1.0, "c": -1.0, "d": -3.0}
    pairs = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"), ("b", "d"), ("a", "d")]
    comps = []
    for _ in range(40):
        hi, lo = pairs[rng.randrange(len(pairs))]
        p = _sigmoid(qualities[hi] - qualities[lo])
        comps.append(Comparison(hi, lo, rng.random() < p, {}))
    res = fit(comps)
    order = rank_topk(res, k=2)["order"]
    assert order[:2] == ["a", "b"], order


def test_acquisition_rules_contract():
    rng = random.Random(5)
    comps = [
        Comparison("a", "b", rng.random() < 0.7, {"position_b": 1.0})
        for _ in range(30)
    ] + [
        Comparison("a", "b", rng.random() < 0.7, {"position_b": 0.0})
        for _ in range(30)
    ]
    res = fit(comps)
    cands = [
        Comparison("a", "b", True, {"position_b": 1.0}),
        Comparison("a", "b", True, {"position_b": 0.0}),
    ]
    # topk / d_optimal return one of the candidates.
    assert acquire_next(res, cands, k=1, rule="topk") in cands
    assert acquire_next(res, cands, k=1, rule="d_optimal") in cands
    # round_robin rotates deterministically.
    assert acquire_round_robin(cands, spent=0).covariates["position_b"] == 1.0
    assert acquire_round_robin(cands, spent=1).covariates["position_b"] == 0.0
    # round_robin is rejected by the stateless acquire_next.
    try:
        acquire_next(res, cands, rule="round_robin")
        assert False, "expected ValueError"
    except ValueError:
        pass
