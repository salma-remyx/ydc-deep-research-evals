"""Judge self-consistency metrics for the pairwise evaluator.

Adapted from: "What Current AI Benchmarks Leave Unmeasured: Modality,
Search, Citations, and Implications (for Safety Evaluations)"
(arXiv:2608.06202).

Mode 3 (inspired experiment). The paper audits LLM benchmarks and argues
that single-run accuracy obscures behavioral variation -- it found
repeated runs of the same prompt produced inconsistent responses in up
to 21% of prompts, and that this "response consistency" across runs is a
dimension benchmarks routinely leave unmeasured. We do NOT port the
paper's modality / web-search audit (this repo uses a single access
modality and access path); we apply the paper's "measure consistency
across runs" insight to *this* repo's pairwise judge.

The mapping is direct: ``DeepResearchPairwiseMetric`` already runs
``num_trials`` original trials plus ``num_trials`` flipped trials and
collects a per-trial preference (``all_preferred``) and signed gap score
(``all_scores``) per dimension, but consumes them only via a single
majority vote -- discarding the within-prompt stability signal. The
functions below turn that already-collected trial data into a
self-consistency read: how often does the judge agree with itself across
repeated trials? That is the judge-side analog of the paper's multi-run
response-consistency dimension.
"""

import math
from typing import Dict, List

from pydantic import BaseModel

# Metric keys surfaced by ``aggregate_judge_stability``. Kept as a tuple so the
# per-dimension and overall aggregation in ``DeepResearchPairwiseMetric`` stay
# in lockstep without restating the key list.
STABILITY_AGGREGATE_KEYS = (
    "agreement_rate",
    "preference_entropy",
    "score_std",
    "inconsistent_rate",
)


class JudgeStability(BaseModel):
    """Self-consistency of the pairwise judge across repeated trials.

    Attributes:
        agreement_rate: share of trials whose preference matches the majority
            verdict. Range ``[0.5, 1.0]``; ``1.0`` means the judge was
            unanimous across every trial.
        preference_entropy: normalized Shannon entropy of the per-trial
            preference distribution (binary outcome, so range ``[0, 1]``).
            ``0`` = unanimous, ``1`` = an even a/b split.
        score_std: standard deviation of the per-trial signed gap scores.
            Lower means the judge's strength-of-preference was stable across
            trials; ``0`` means identical magnitudes.
        is_unanimous: ``True`` when every trial agreed on the same preference
            (``agreement_rate == 1.0``). Convenience flag for the aggregate
            "inconsistent rate" analog of the paper's headline statistic.
        num_trials: number of trials the metrics were computed over.
    """

    agreement_rate: float
    preference_entropy: float
    score_std: float
    is_unanimous: bool
    num_trials: int


def compute_judge_stability(
    all_preferred: List[str],
    all_scores: List[float],
) -> JudgeStability:
    """Compute judge stability from per-trial preference and score data.

    Args:
        all_preferred: per-trial preference labels (``"a"`` or ``"b"``)
            collected across original and flipped trials, already
            re-oriented so ``"b"`` always means "candidate preferred".
        all_scores: per-trial signed gap scores on the same orientation as
            ``all_preferred`` (positive = candidate preferred).

    Returns:
        A :class:`JudgeStability`. Empty trial input returns zeroed metrics
        with ``num_trials == 0`` so callers can aggregate without special
        handling.
    """
    num_trials = len(all_preferred)
    if num_trials == 0:
        return JudgeStability(
            agreement_rate=0.0,
            preference_entropy=0.0,
            score_std=0.0,
            is_unanimous=False,
            num_trials=0,
        )

    num_b = sum(1 for p in all_preferred if p == "b")
    num_a = sum(1 for p in all_preferred if p == "a")

    # Agreement rate: share of trials matching the majority preference.
    majority = max(num_a, num_b)
    agreement_rate = majority / num_trials

    # Normalized Shannon entropy over {a, b} preferences. For a binary
    # outcome the maximum is log2(2) == 1, so entropy is already in [0, 1].
    probs = [count / num_trials for count in (num_a, num_b) if count > 0]
    preference_entropy = -sum(p * math.log2(p) for p in probs)

    # Standard deviation of the per-trial signed gap scores.
    if all_scores:
        mean_score = sum(all_scores) / len(all_scores)
        variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
        score_std = math.sqrt(variance)
    else:
        score_std = 0.0

    return JudgeStability(
        agreement_rate=agreement_rate,
        preference_entropy=preference_entropy,
        score_std=score_std,
        is_unanimous=agreement_rate == 1.0,
        num_trials=num_trials,
    )


def aggregate_judge_stability(stabilities: List[JudgeStability]) -> Dict[str, float]:
    """Average per-row :class:`JudgeStability` into an aggregate block.

    Args:
        stabilities: per-row stability objects for one dimension (rows whose
            ``stability`` was ``None`` -- e.g. re-validated from pre-stability
            serialized output -- should be filtered out by the caller).

    Returns:
        Dict with :data:`STABILITY_AGGREGATE_KEYS`. ``inconsistent_rate`` is the
        share of rows where the judge was *not* unanimous -- the closest
        native analog to the paper's "inconsistent in up to 21% of prompts"
        statistic. An empty input returns zeroed metrics.
    """
    if not stabilities:
        return {key: 0.0 for key in STABILITY_AGGREGATE_KEYS}

    n = len(stabilities)
    return {
        "agreement_rate": sum(s.agreement_rate for s in stabilities) / n,
        "preference_entropy": sum(s.preference_entropy for s in stabilities) / n,
        "score_std": sum(s.score_std for s in stabilities) / n,
        "inconsistent_rate": sum(0 if s.is_unanimous else 1 for s in stabilities) / n,
    }
