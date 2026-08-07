"""Bayesian reliability scoring for pairwise evaluation dimensions.

Adapted from CalibratedRubric (arXiv:2607.29252), specifically its Bayesian
*rubric-measurability* filter. A dimension (rubric) is "measurable" when
repeated judge votes agree often enough that we are confident the rubric can
actually tell two responses apart. We estimate that with a Beta-Bernoulli
agreement posterior over the pooled trial votes and flag a dimension as
measurable only when the *lower* credible bound of that posterior clears a
threshold -- so low-redundancy rubrics are not over-claimed.

Ported at full fidelity (the paper's core mechanism):
  * The per-rubric Beta-Bernoulli agreement posterior.
  * An uncertainty-aware measurability filter keyed on the lower credible
    bound, which is exactly the property the paper highlights: calibration
    gains depend on sufficient judge redundancy.

Intentionally substituted / scoped out (Mode 2 adapted port):
  * The paper's learned, type-specific scoring components are replaced by a
    parameter-free Beta(1, 1) prior -- a no-fit proxy for the same agreement
    signal.
  * The paper's IRT-based, submodular rubric-bank assembly is out of scope:
    it needs a held-out calibration set and a separate IRT fitting path this
    repo does not host. Bank assembly / benchmark evaluation belongs in a
    downstream PR.

This module is dependency-free (standard-library ``math`` only): the Beta
credible interval is computed via the regularized incomplete beta function
(Lentz continued fraction) and inverted by bisection.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Uninformative Beta(1, 1) prior -- a parameter-free stand-in for the paper's
# learned/type-specific agreement estimator (Mode 2 substitution).
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0
DEFAULT_CONFIDENCE = 0.95
# A dimension is flagged "measurable" when the lower bound of its agreement
# posterior is at least this high. Conservative by design: with few judge
# votes the posterior is wide and the lower bound stays below threshold, so
# under-redundant dimensions are not over-claimed (the paper's main caveat).
DEFAULT_MEASURABLE_THRESHOLD = 0.75


@dataclass(frozen=True)
class DimensionReliability:
    """Bayesian reliability summary for a single evaluation dimension."""

    reliability: float  # posterior mean agreement (Beta mean)
    ci_low: float  # lower bound of the credible interval
    ci_high: float  # upper bound of the credible interval
    measurable: bool  # does ci_low clear the measurability threshold?
    num_votes: int  # total judge votes pooled across rows
    posterior_alpha: float  # Beta posterior shape (successes + prior)
    posterior_beta: float  # Beta posterior shape (failures + prior)
    threshold: float


# --- regularized incomplete beta (Beta CDF) and its inverse -------------
# Lentz continued-fraction implementation following Numerical Recipes.
_EPS = 1e-14
_TINY = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` -- the Beta(a, b) CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(math.log(x) * a + math.log1p(-x) * b + lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _inverse_reg_incomplete_beta(p: float, a: float, b: float) -> float:
    """Return ``x`` such that ``I_x(a, b) = p`` via bisection (CDF is monotonic)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _reg_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def beta_posterior_mean(alpha: float, beta: float) -> float:
    """Posterior mean of a Beta(alpha, beta) distribution."""
    return alpha / (alpha + beta)


def beta_credible_interval(
    alpha: float,
    beta: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Central ``confidence`` credible interval for Beta(alpha, beta)."""
    tail = (1.0 - confidence) / 2.0
    low = _inverse_reg_incomplete_beta(tail, alpha, beta)
    high = _inverse_reg_incomplete_beta(1.0 - tail, alpha, beta)
    return low, high


def compute_reliability_from_votes(
    vote_lists: Iterable[Sequence[str]],
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
    confidence: float = DEFAULT_CONFIDENCE,
    measurable_threshold: float = DEFAULT_MEASURABLE_THRESHOLD,
) -> DimensionReliability:
    """Estimate a dimension's reliability from per-row judge vote lists.

    Each element of ``vote_lists`` is the sequence of per-trial preferences
    ("a"/"b") for one evaluated row (original and flipped trials combined).
    For every row the votes agreeing with that row's majority side count as
    agreement "successes" and the rest as "failures"; pooling across rows
    yields a Beta-Bernoulli posterior over the dimension's agreement rate.

    A tie row (equal "a"/"b" counts) therefore contributes 50% agreement,
    matching the majority-vote grade the metric already assigns it.
    """
    successes = 0
    failures = 0
    num_votes = 0
    for votes in vote_lists:
        n = len(votes)
        if n == 0:
            continue
        num_votes += n
        count_a = sum(1 for v in votes if v == "a")
        count_b = n - count_a
        successes += max(count_a, count_b)
        failures += min(count_a, count_b)

    alpha = prior_alpha + successes
    beta = prior_beta + failures
    ci_low, ci_high = beta_credible_interval(alpha, beta, confidence)
    return DimensionReliability(
        reliability=beta_posterior_mean(alpha, beta),
        ci_low=ci_low,
        ci_high=ci_high,
        measurable=ci_low >= measurable_threshold,
        num_votes=num_votes,
        posterior_alpha=alpha,
        posterior_beta=beta,
        threshold=measurable_threshold,
    )


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MEASURABLE_THRESHOLD",
    "DEFAULT_PRIOR_ALPHA",
    "DEFAULT_PRIOR_BETA",
    "DimensionReliability",
    "beta_credible_interval",
    "beta_posterior_mean",
    "compute_reliability_from_votes",
]
