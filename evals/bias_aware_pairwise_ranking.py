"""Bias-aware Bayesian pairwise ranking for LLM-judge comparisons.

Adapted from "Ask the Right Comparison: Bias-Aware Bayesian Active Top-k
Ranking with LLM Judges" (arXiv:2607.02104). Attribution lives here in the
docstring and in README.md, never in the file/module name.

The repo's pairwise metric already mitigates *input-level* position bias by
running flipped-order trials and then averaging the votes (a naive majority
aggregation). This module moves the bias correction to the *aggregation*
layer, as the paper proposes: it casts the flipped-trial outcomes as Bayesian
inference over latent item quality with an explicit, judge-specific position
covariate, regularized by a shrinkage prior so the data decide whether a given
judge actually exhibits the bias. It emits a debiased ranking with posterior
uncertainty and a top-k-aware acquisition rule that picks the next comparison
to most reduce top-k *membership* uncertainty (rather than round-robin or a
global-uncertainty / D-optimal rule).

Implementation mode: **Mode 2 (adapted port)**. The paper's core mechanism is
kept at full fidelity; the following auxiliary components are substituted with
target-native equivalents:

* Inference  -> Laplace approximation (Newton MAP + Hessian inverse) in the
  Python standard library, replacing the paper's MCMC/HMC. No numpy / PyMC /
  Stan dependency (the repo pins none).
* Shrinkage  -> a Laplace (L1) prior on bias covariates (smoothed so a Hessian
  exists), giving the same sparse, "data-decided" bias selection as the
  paper's horseshoe prior.
* Benchmark  -> cut; the model consumes this repo's own flipped-trial outcomes
  via :func:`comparisons_from_score_results`. Standalone evaluation against the
  paper's controlled ground-truth benchmark belongs in a downstream PR.
* Covariates -> position (the repo's documented bias concern). Verbosity is
  structurally supported (pass any covariate dict) but out of scope here, since
  the saved eval outputs carry no length signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Mirrors evals.metrics.deep_research_pairwise_metric.DIMENSIONS so this module
# stays importable without pulling in openai/pydantic (see adapter below).
DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "instruction_following",
    "comprehensiveness",
    "completeness",
    "writing_quality",
)

# Smoothing for the L1 prior so the objective is twice-differentiable and the
# Hessian (hence the Laplace covariance) is well defined.
_L1_EPS = 1e-4


@dataclass
class Comparison:
    """A single pairwise comparison with optional bias covariates.

    ``item_a`` is modelled against ``item_b``; ``a_wins`` is the judge's vote
    (``True`` if ``item_a`` won). ``covariates`` are judge/position features
    (e.g. ``{"position_b": 1.0}`` when ``item_a`` sat in slot b).
    """

    item_a: str
    item_b: str
    a_wins: bool
    covariates: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiasEstimate:
    name: str
    mean: float
    std: float
    shrunken: bool  # True when the data did not support a nonzero bias

    @property
    def significant(self) -> bool:
        return (not self.shrunken) and abs(self.mean) > self.std


@dataclass
class FitResult:
    items: List[str]
    covariates: List[str]
    theta_mean: Dict[str, float]
    theta_std: Dict[str, float]
    bias: Dict[str, BiasEstimate]
    covariance: List[List[float]]
    n_comparisons: int
    converged: bool
    n_iter: int

    def advantage(self, item_a: str, item_b: str) -> Tuple[float, float]:
        """Posterior (mean, std) of theta_a - theta_b."""
        ia = self.items.index(item_a)
        ib = self.items.index(item_b)
        cov = self.covariance
        mean = self.theta_mean[item_a] - self.theta_mean[item_b]
        var = cov[ia][ia] + cov[ib][ib] - 2.0 * cov[ia][ib]
        return mean, math.sqrt(max(var, 0.0))


# --------------------------------------------------------------------------- #
# linear algebra helpers (pure stdlib)                                        #
# --------------------------------------------------------------------------- #
def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _mat_inv(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Invert a small symmetric PD matrix via Gauss-Jordan with pivoting."""
    n = len(matrix)
    aug = [list(matrix[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] += 1e-9  # jitter a singular pivot
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col]:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


# --------------------------------------------------------------------------- #
# core: bias-aware Bayesian pairwise fit (Laplace approximation)             #
# --------------------------------------------------------------------------- #
def fit(
    comparisons: Sequence[Comparison],
    prior_bias_scale: float = 1.0,
    prior_quality_scale: float = 10.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> FitResult:
    """Fit latent item qualities + bias covariates by MAP, return posteriors.

    Model: ``logit P(a beats b) = theta_a - theta_b + sum_d beta_d * x_d``.
    Bias coefficients ``beta`` carry a Laplace (L1) shrinkage prior with scale
    ``prior_bias_scale``; item qualities carry a weak ridge prior with scale
    ``prior_quality_scale`` (this also fixes the additive constant). Posteriors
    come from a Laplace approximation: the inverse Hessian of the negative log
    posterior at the MAP.

    Args:
        comparisons: trial-level pairwise outcomes with covariates.
        prior_bias_scale: Laplace scale ``b`` for bias priors; smaller shrinks
            bias harder toward zero (the "data decides" knob).
        prior_quality_scale: ridge scale for item quality; large = weak.
        max_iter / tol: Newton convergence controls.

    Returns:
        A :class:`FitResult` with debiased quality means/stds and bias
        estimates (flagged ``shrunken`` when the data did not support them).
    """
    item_names = sorted({c.item_a for c in comparisons} | {c.item_b for c in comparisons})
    cov_names = sorted({k for c in comparisons for k in c.covariates})
    item_idx = {name: i for i, name in enumerate(item_names)}
    n_items, n_cov = len(item_names), len(cov_names)
    n_params = n_items + n_cov

    if not comparisons:
        raise ValueError("Cannot fit a model with zero comparisons.")

    # Pre-materialize per-comparison design vectors.
    rows: List[Tuple[int, int, float, List[float]]] = []
    for c in comparisons:
        xvec = [float(c.covariates.get(name, 0.0)) for name in cov_names]
        rows.append((item_idx[c.item_a], item_idx[c.item_b], 1.0 if c.a_wins else 0.0, xvec))

    lam = 1.0 / prior_bias_scale  # Laplace(b) penalty weight = 1/b
    sq_q = prior_quality_scale ** 2
    inv_sq_q = 1.0 / sq_q

    theta = [0.0] * n_items
    beta = [0.0] * n_cov
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        grad = [0.0] * n_params
        hess = [[0.0] * n_params for _ in range(n_params)]

        for w, l, y, xvec in rows:
            eta = theta[w] - theta[l]
            for d in range(n_cov):
                eta += beta[d] * xvec[d]
            p = _sigmoid(eta)
            r = p - y  # d(NLL)/d eta
            weight = p * (1.0 - p)

            grad[w] += r
            grad[l] -= r
            for d in range(n_cov):
                grad[n_items + d] += r * xvec[d]

            hess[w][w] += weight
            hess[l][l] += weight
            hess[w][l] -= weight
            hess[l][w] -= weight
            for d in range(n_cov):
                bd = n_items + d
                hess[w][bd] += weight * xvec[d]
                hess[bd][w] += weight * xvec[d]
                hess[l][bd] -= weight * xvec[d]
                hess[bd][l] -= weight * xvec[d]
                for e in range(n_cov):
                    be = n_items + e
                    hess[bd][be] += weight * xvec[d] * xvec[e]

        # Priors: ridge on theta, smoothed-L1 (Laplace) on beta.
        for i in range(n_items):
            grad[i] += theta[i] * inv_sq_q
            hess[i][i] += inv_sq_q
        for d in range(n_cov):
            bd = n_items + d
            b = beta[d]
            denom = math.sqrt(b * b + _L1_EPS)
            grad[bd] += lam * b / denom
            hess[bd][bd] += lam * _L1_EPS / (denom ** 3)

        cov_inv = _mat_inv(hess)
        # Newton step on the negative log posterior: params -= H^{-1} grad.
        delta = [0.0] * n_params
        for i in range(n_params):
            delta[i] = sum(cov_inv[i][j] * grad[j] for j in range(n_params))
        step_norm = math.sqrt(sum(d * d for d in delta))

        for i in range(n_items):
            theta[i] -= delta[i]
        for d in range(n_cov):
            beta[d] -= delta[n_items + d]

        if step_norm < tol:
            converged = True
            break

    theta_std = {
        item_names[i]: math.sqrt(max(cov_inv[i][i], 0.0)) for i in range(n_items)
    }
    bias: Dict[str, BiasEstimate] = {}
    for d, name in enumerate(cov_names):
        bd = n_items + d
        std = math.sqrt(max(cov_inv[bd][bd], 0.0))
        mean = beta[d]
        shrunken = abs(mean) <= std  # not distinguishable from zero
        bias[name] = BiasEstimate(name=name, mean=mean, std=std, shrunken=shrunken)

    return FitResult(
        items=item_names,
        covariates=cov_names,
        theta_mean={item_names[i]: theta[i] for i in range(n_items)},
        theta_std=theta_std,
        bias=bias,
        covariance=cov_inv,
        n_comparisons=len(comparisons),
        converged=converged,
        n_iter=n_iter,
    )


# --------------------------------------------------------------------------- #
# top-k ranking + membership uncertainty                                      #
# --------------------------------------------------------------------------- #
def rank_topk(result: FitResult, k: int = 1) -> Dict[str, Any]:
    """Return the top-k items by posterior mean quality, with membership doubt.

    Membership doubt is the posterior uncertainty on each item's quality
    relative to the rank-k boundary; the acquisition rule targets exactly this
    quantity rather than full-ranking variance.
    """
    order = sorted(result.items, key=lambda n: -result.theta_mean[n])
    k = max(1, min(k, len(order)))
    topk = order[:k]
    boundary = result.theta_mean[order[k - 1]]
    membership: Dict[str, float] = {}
    for name in order:
        mean = result.theta_mean[name]
        sd = result.theta_std[name] or 1e-9
        # Probability the item clears the current boundary (Laplace approx).
        membership[name] = _sigmoid((mean - boundary) / sd)
    return {"topk": topk, "order": order, "membership_probability": membership}


def acquire_next(
    result: FitResult,
    candidates: Sequence[Comparison],
    k: int = 1,
    rule: str = "topk",
) -> Comparison:
    """Pick the next comparison to run under a fixed budget.

    Rules mirror the paper's contrast:

    * ``"topk"``      -- reduce top-k *membership* uncertainty: score by outcome
      uncertainty times the membership sensitivity of both items (peaked at the
      rank-k boundary).
    * ``"d_optimal"`` -- global Fisher-information / uncertainty rule that
      ignores the boundary (score by outcome uncertainty alone).

    The round-robin baseline (the repo's current equal original/flipped trial
    allocation) is stateful, so use :func:`acquire_round_robin` for it.
    """
    if not candidates:
        raise ValueError("No candidate comparisons to acquire from.")
    if rule == "round_robin":
        raise ValueError(
            "rule='round_robin' is stateful; use acquire_round_robin(candidates, spent)."
        )
    if rule not in ("topk", "d_optimal"):
        raise ValueError(f"Unknown acquisition rule: {rule!r}")

    rk = rank_topk(result, k=k)
    membership = rk["membership_probability"]
    # Sensitivity is highest for items whose membership is most in doubt (~0.5).
    sens = {name: 1.0 - abs(2.0 * p - 1.0) for name, p in membership.items()}

    best, best_score = candidates[0], -1.0
    for c in candidates:
        a_w = result.theta_mean.get(c.item_a, 0.0)
        b_w = result.theta_mean.get(c.item_b, 0.0)
        beta_term = sum(
            result.bias[name].mean * c.covariates.get(name, 0.0)
            for name in result.covariates
        )
        p = _sigmoid(a_w - b_w + beta_term)
        outcome_uncertainty = p * (1.0 - p)
        if rule == "d_optimal":
            score = outcome_uncertainty
        else:  # "topk"
            s = sens.get(c.item_a, 0.0) + sens.get(c.item_b, 0.0)
            score = outcome_uncertainty * s
        if score > best_score:
            best_score, best = score, c
    return best


def acquire_round_robin(candidates: Sequence[Comparison], spent: int) -> Comparison:
    """Round-robin acquisition baseline (uniform rotation over candidates)."""
    if not candidates:
        raise ValueError("No candidate comparisons to acquire from.")
    return candidates[spent % len(candidates)]


# --------------------------------------------------------------------------- #
# adapter: consume the repo's DeepResearchScoreResult contract               #
# --------------------------------------------------------------------------- #
def _dig(obj: Any, key: str) -> Any:
    return getattr(obj, key) if hasattr(obj, key) else obj[key]


def comparisons_from_score_results(
    score_results: Iterable[Any],
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
) -> Dict[str, List[Comparison]]:
    """Convert repo eval outputs into position-labeled comparisons.

    Accepts either ``DeepResearchScoreResult`` pydantic objects (from the
    non-new ``evals.metrics.deep_research_pairwise_metric`` module) or their
    ``model_dump()`` dicts, e.g. rows saved by the eval runner's JSONL output.
    For each dimension it recovers the per-trial, per-position preferences from
    ``raw_preferences.original`` / ``raw_preferences.flipped`` -- *not* the
    already-majority-voted ``preferred`` list -- so the position covariate is
    attributed correctly.

    Position convention matches the repo's metric: in the original order the
    candidate sits in slot ``b`` (``position_b=1``); in the flipped order it
    sits in slot ``a`` (``position_b=0``).
    """
    out: Dict[str, List[Comparison]] = {dim: [] for dim in dimensions}
    for score_result in score_results:
        for dim in dimensions:
            dim_obj = _dig(score_result, dim)
            raw = _dig(dim_obj, "raw_preferences")
            original = _dig(raw, "original")
            flipped = _dig(raw, "flipped")
            for entry in original:
                pref = _dig(entry, "preferred")
                # original frame: a=baseline, b=candidate -> candidate in slot b
                out[dim].append(
                    Comparison(
                        item_a=candidate_name,
                        item_b=baseline_name,
                        a_wins=(pref == "b"),
                        covariates={"position_b": 1.0},
                    )
                )
            for entry in flipped:
                pref = _dig(entry, "preferred")
                # flipped frame: a=candidate, b=baseline -> candidate in slot a
                out[dim].append(
                    Comparison(
                        item_a=candidate_name,
                        item_b=baseline_name,
                        a_wins=(pref == "a"),
                        covariates={"position_b": 0.0},
                    )
                )
    return out


def debiased_pairwise_summary(
    score_results: Iterable[Any],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    prior_bias_scale: float = 1.0,
) -> Dict[str, Any]:
    """Drop-in bias-aware replacement for the metric's naive per-dimension vote.

    For each dimension, fits the 2-item (candidate vs baseline) model and
    reports the debiased candidate advantage, the shrunk position-bias
    estimate, the naive win rate (for contrast), the posterior probability that
    the candidate is truly better, and the next trial order the top-k-aware
    acquisition rule recommends spending the budget on.
    """
    per_dim = comparisons_from_score_results(score_results, dimensions=dimensions)
    summary: Dict[str, Any] = {}
    for dim, comps in per_dim.items():
        if not comps:
            continue
        fit_res = fit(comps, prior_bias_scale=prior_bias_scale)
        adv_mean, adv_std = fit_res.advantage("candidate", "baseline")
        bias = fit_res.bias.get("position_b")
        naive_winrate = sum(1 for c in comps if c.a_wins) / len(comps)
        p_candidate_better = _sigmoid(adv_mean / (adv_std or 1e-9))

        # Acquisition: should the next trial be original (position_b=1) or
        # flipped (position_b=0) to most reduce membership (win/lose) doubt?
        cand_orig = Comparison("candidate", "baseline", True, {"position_b": 1.0})
        cand_flip = Comparison("candidate", "baseline", True, {"position_b": 0.0})
        next_trial = acquire_next(fit_res, [cand_orig, cand_flip], k=1, rule="topk")
        summary[dim] = {
            "n_comparisons": len(comps),
            "candidate_advantage_mean": adv_mean,
            "candidate_advantage_std": adv_std,
            "p_candidate_better": p_candidate_better,
            "position_bias_mean": bias.mean if bias else 0.0,
            "position_bias_std": bias.std if bias else 0.0,
            "position_bias_shrunken": bias.shrunken if bias else True,
            "naive_winrate": naive_winrate,
            "recommended_next_position_b": next_trial.covariates["position_b"],
            "converged": fit_res.converged,
        }
    return summary
