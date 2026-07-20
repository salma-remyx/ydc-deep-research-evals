"""Bradley-Terry statistical ranking for pairwise-evaluation outcomes.

Adapted from "A Statistical Framework for Ranking LLM-Based Chatbots"
(arXiv:2412.18407), which advocates estimating a latent quality ("ability")
score from pairwise comparisons via the Bradley-Terry model together with
maximum-likelihood uncertainty quantification, rather than reporting raw
win-rate fractions with no measure of statistical reliability.

The repo's ``aggregate()`` step already stores, per dimension, the
orientation-corrected per-trial flipped preferences ("a"/"b") produced by
the eval pipeline. This module consumes exactly those outcomes and returns,
per dimension:

  * a Bradley-Terry *ability* score on a logit scale -- the paper's advocated
    latent-quality metric. It is additive and comparable across dimensions,
    unlike a raw fraction. The baseline ability is anchored at 0, so a
    positive ability means the candidate is preferred over the baseline.
  * a Fisher-information standard error and confidence interval on that
    ability -- the core statistical-inference contribution (the asymptotic
    MLE variance of the two-player BT estimator).
  * the implied win probability with its CI (the ability CI mapped through
    the logistic CDF, which respects the bounded [0, 1] scale -- unlike a
    naive normal interval placed directly on a raw fraction).
  * an optional nonparametric *bootstrap* CI on the win probability -- the
    resampling-based uncertainty route the paper leads with; robust in the
    small-sample / degenerate-count regime where the normal approximation is
    loose.
  * a ``significant`` flag: whether the candidate's edge over the baseline is
    distinguishable from a tie at the available trial count (the ability CI
    excludes 0, equivalently the win-probability CI excludes 0.5).

Mode-2 scoping (substituted / omitted components):
  * The paper's per-trial outcomes in THIS repo are binary ("a" or "b"), so
    the Davidson tie-mass extension is unidentified here and is omitted.
    Ties are a consensus-level artifact in this pipeline, not
    per-comparison outcomes.
  * The paper's multi-model global ranking (BT over a graph of >2 models)
    plus its position/verbosity-bias corrections require cross-model
    comparison graphs that this single-pair (baseline vs candidate) pipeline
    does not store; they are left to a downstream integration. The two-player
    BT MLE is the exact special case of the paper's model that fits here.
"""

import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

# Per-trial preference labels as stored on DimensionResult.preferred.
# "b" == the candidate (report_b) is preferred; "a" == the baseline wins.
CANDIDATE = "b"
BASELINE = "a"

# Jeffreys/Laplace pseudo-count added to each side's win count. It keeps the
# MLE finite when one side has zero wins (log(W_b / 0) would be infinite) and
# stabilises the Fisher standard error in the small-sample regime. This is the
# standard regularised Bradley-Terry prior.
DEFAULT_PRIOR = 0.5

# Normal critical values for the confidence intervals we support.
_Z_TABLE = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def _z_for_confidence(confidence: float) -> float:
    """Normal critical value ``z`` for a two-sided central interval."""
    if confidence in _Z_TABLE:
        return _Z_TABLE[confidence]
    return _Z_TABLE[min(_Z_TABLE, key=lambda k: abs(k - confidence))]


def _sigmoid(x: float) -> float:
    """Numerically stable logistic CDF."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class BTRanking:
    """Bradley-Terry ranking summary for a single comparison dimension."""

    ability: float
    ability_se: float
    ability_ci_low: float
    ability_ci_high: float
    win_probability: float
    win_probability_ci_low: float
    win_probability_ci_high: float
    significant: bool
    num_wins: int  # candidate ("b") preferences
    num_losses: int  # baseline ("a") preferences
    num_trials: int
    bootstrap_ci_low: Optional[float] = None
    bootstrap_ci_high: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["method"] = "bradley-terry-mle-fisher"
        return d


def _win_probability_bootstrap_ci(
    outcomes: Sequence[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Nonparametric bootstrap CI on the candidate win probability.

    Resamples the empirical per-trial outcomes with replacement -- the
    resampling-based uncertainty route the paper leads with. Makes no
    asymptotic / Fisher-information assumption, so it stays well-behaved when
    one side has zero or very few wins.
    """
    n = len(outcomes)
    if n == 0:
        return 0.0, 1.0
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(outcomes, k=n)
        estimates.append(sample.count(CANDIDATE) / n)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = min(int(alpha * n_bootstrap), n_bootstrap - 1)
    hi_idx = min(int((1.0 - alpha) * n_bootstrap), n_bootstrap - 1)
    return estimates[lo_idx], estimates[hi_idx]


def estimate_bt_ranking(
    outcomes: Sequence[str],
    prior: float = DEFAULT_PRIOR,
    confidence: float = 0.95,
    bootstrap_iterations: int = 0,
    seed: int = 0,
) -> BTRanking:
    """Estimate a Bradley-Terry ranking from per-trial preference labels.

    Args:
        outcomes: Sequence of per-trial preferences ("a" = baseline wins,
            "b" = candidate wins). Anything else is ignored.
        prior: Pseudo-count added to each side (regularises the MLE; see
            ``DEFAULT_PRIOR``).
        confidence: Two-sided central confidence level for the intervals
            (0.80 / 0.90 / 0.95 / 0.99 supported).
        bootstrap_iterations: If > 0, also compute a nonparametric bootstrap
            CI on the win probability using this many resamples.
        seed: Seed for the bootstrap RNG (deterministic for reproducibility).

    Returns:
        A :class:`BTRanking` with the latent-ability estimate, its
        Fisher-information uncertainty, and the implied win probability.
    """
    valid = [o for o in outcomes if o == CANDIDATE or o == BASELINE]
    num_wins = valid.count(CANDIDATE)
    num_losses = valid.count(BASELINE)
    num_trials = num_wins + num_losses

    # Regularised BT MLE of the candidate's latent ability (baseline anchored
    # at 0): theta_hat = log((W_b + prior) / (W_a + prior)).
    wins = num_wins + prior
    losses = num_losses + prior
    ability = math.log(wins / losses)

    # Fisher-information (asymptotic MLE) variance for the two-player BT
    # estimator: Var(theta_hat) = 1/W_a + 1/W_b.
    ability_se = math.sqrt(1.0 / losses + 1.0 / wins)

    z = _z_for_confidence(confidence)
    ability_ci_low = ability - z * ability_se
    ability_ci_high = ability + z * ability_se

    win_probability = _sigmoid(ability)
    # Map the ability CI through the logistic CDF so the win-probability CI
    # honours the [0, 1] boundary rather than a naive normal interval.
    win_probability_ci_low = _sigmoid(ability_ci_low)
    win_probability_ci_high = _sigmoid(ability_ci_high)

    # The candidate's edge is "significant" once the tie threshold (ability 0
    # / win-probability 0.5) falls outside the CI.
    significant = ability_ci_low > 0.0 or ability_ci_high < 0.0

    bootstrap_ci_low: Optional[float] = None
    bootstrap_ci_high: Optional[float] = None
    if bootstrap_iterations > 0 and num_trials > 0:
        bootstrap_ci_low, bootstrap_ci_high = _win_probability_bootstrap_ci(
            valid, bootstrap_iterations, confidence, seed
        )

    return BTRanking(
        ability=ability,
        ability_se=ability_se,
        ability_ci_low=ability_ci_low,
        ability_ci_high=ability_ci_high,
        win_probability=win_probability,
        win_probability_ci_low=win_probability_ci_low,
        win_probability_ci_high=win_probability_ci_high,
        significant=significant,
        num_wins=num_wins,
        num_losses=num_losses,
        num_trials=num_trials,
        bootstrap_ci_low=bootstrap_ci_low,
        bootstrap_ci_high=bootstrap_ci_high,
    )
