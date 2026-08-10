"""Judge-sensitivity and paired-significance audit for pairwise evaluations.

Adapted from: "Evaluating medical AI under missing information: same-provider
judges and human raters change apparent safety" (arXiv:2607.18828v1).

The paper's two robust *evaluator-facing* findings are that (1) the choice of
LLM judge materially changes apparent model rankings -- inter-judge agreement is
only moderate (Fleiss' kappa ~0.65) and a same-provider self-preference survives
adjustment for per-judge leniency (exact permutation p ~0.04) -- and (2) LLM
judges diverge from a consensus reference (judge-vs-consensus kappa ~0.20-0.43).

This module ports those evaluator-facing statistics onto the repo's existing
pairwise contract. It consumes the per-judge ``DimensionResult.preferred`` vote
vectors and per-row ``grade`` values that
:class:`evals.metrics.deep_research_pairwise_metric.DeepResearchPairwiseMetric`
already emits, and reports:

* Fleiss' kappa across judges on the per-row consensus grade (inter-judge
  agreement).
* Per-judge candidate-favor rates (leniency) and a residualized same-provider
  permutation test (self-preference).
* Wilson and bootstrap confidence intervals on the win/tie/lose consensus.
* Per-judge Cohen's kappa against a leave-one-out majority consensus.

Mode 2 (adapted port). The paper's core inferential engine -- leniency
adjustment followed by an exact permutation test for the same-provider effect --
is preserved. Two auxiliary components are substituted with target-native
equivalents: (a) the medical / HealthBench domain becomes the repo's DeepConsult
pairwise vote vectors, and (b) the paper's vote-level *logistic* regression for
leniency adjustment is implemented as judge-mean residualization (a parameter-
free proxy of the same signal that is identifiable in both single-candidate and
multi-candidate-panel settings and adds no dependencies). The clinician-anchored
reference has no target-native equivalent, so finding (2) is delivered as
judge-vs-leave-one-out-majority-consensus agreement instead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchScoreResult,
)

GRADES = ("win", "tie", "lose")

# Paper's four-provider panel, mapped from model / judge names.
_PROVIDER_RULES: List[Tuple[str, str]] = [
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("openai", "openai"),
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("xai", "xai"),
]

_Z_SCORES = {0.10: 1.6448536269514722, 0.05: 1.959963984540054, 0.01: 2.5758293035489004}


def infer_provider(name: str) -> str:
    """Map a model / judge name to a provider family (paper's 4-provider panel)."""
    key = name.lower()
    for token, provider in _PROVIDER_RULES:
        if token in key:
            return provider
    return "unknown"


@dataclass
class JudgeRun:
    """One judge's scored output over the shared, aligned set of rows."""

    name: str
    provider: str
    scores: List[DeepResearchScoreResult]


# --------------------------------------------------------------------------- #
# Core statistics (pure-Python; no numpy / scipy dependency).
# --------------------------------------------------------------------------- #


def _fleiss_kappa(matrix: List[List[int]]) -> Optional[float]:
    """Fleiss' kappa for ``n_items`` x ``n_cats`` counts, each row summing to K raters."""
    n_items = len(matrix)
    if n_items == 0:
        return None
    n_raters = sum(matrix[0])
    if n_raters < 2:
        return None
    n_cats = len(matrix[0])
    col_tot = [sum(matrix[i][j] for i in range(n_items)) for j in range(n_cats)]
    total = n_items * n_raters
    p_j = [c / total for c in col_tot]
    p_e = sum(p * p for p in p_j)
    p_i = []
    for i in range(n_items):
        s = sum(matrix[i][j] * matrix[i][j] for j in range(n_cats))
        p_i.append((s - n_raters) / (n_raters * (n_raters - 1)))
    p_bar = sum(p_i) / n_items
    if (1 - p_e) == 0:
        # Every rating landed in one category: perfect but degenerate agreement.
        return 1.0 if p_bar >= 0.9999 else None
    return (p_bar - p_e) / (1 - p_e)


def _cohen_kappa(r1: Sequence[str], r2: Sequence[str]) -> Optional[float]:
    """Cohen's kappa between two equally-long rater vectors over arbitrary labels."""
    n = len(r1)
    if n == 0:
        return None
    cats = sorted(set(r1) | set(r2))
    idx = {c: k for k, c in enumerate(cats)}
    m = [[0] * len(cats) for _ in range(len(cats))]
    for a, b in zip(r1, r2):
        m[idx[a]][idx[b]] += 1
    p_o = sum(m[k][k] for k in range(len(cats))) / n
    row = [sum(m[k]) for k in range(len(cats))]
    col = [sum(m[k][j] for k in range(len(cats))) for j in range(len(cats))]
    p_e = sum((row[k] / n) * (col[k] / n) for k in range(len(cats)))
    if (1 - p_e) == 0:
        return 1.0 if p_o >= 0.9999 else None
    return (p_o - p_e) / (1 - p_e)


def _wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return None, None
    z = _Z_SCORES.get(alpha, _Z_SCORES[0.05])
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of a pre-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def _majority_grade(grades: Sequence[str]) -> str:
    """Plurality consensus grade across judges for one row (deterministic)."""
    counts = {g: 0 for g in GRADES}
    for g in grades:
        if g in counts:
            counts[g] += 1
    top = max(counts.values())
    winners = [g for g in GRADES if counts[g] == top]
    if len(winners) == 1:
        return winners[0]
    if "win" in winners and "lose" in winners:
        return "tie"  # a head-to-head split is genuinely undecided
    return "tie" if "tie" in winners else winners[0]


# --------------------------------------------------------------------------- #
# Audit.
# --------------------------------------------------------------------------- #


class JudgeSensitivityAudit:
    """Compute judge-sensitivity + paired-significance diagnostics over judge runs."""

    def __init__(
        self,
        runs: List[JudgeRun],
        candidate_provider: Optional[str] = None,
        candidate_providers: Optional[List[str]] = None,
        alpha: float = 0.05,
        n_perm: int = 9999,
        n_bootstrap: int = 2000,
        seed: int = 0,
    ):
        if not runs:
            raise ValueError("at least one JudgeRun is required")
        lengths = {len(r.scores) for r in runs}
        if len(lengths) != 1:
            raise ValueError(
                "all judge runs must cover the same rows in the same order; "
                f"got distinct row counts {sorted(lengths)}"
            )
        if candidate_providers is not None and len(candidate_providers) != len(runs[0].scores):
            raise ValueError(
                "candidate_providers must have one provider per row "
                f"({len(candidate_providers)} vs {len(runs[0].scores)} rows)"
            )
        self.runs = runs
        self.candidate_provider = candidate_provider
        self.candidate_providers = candidate_providers
        self.alpha = alpha
        self.n_perm = n_perm
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    @property
    def n_rows(self) -> int:
        return len(self.runs[0].scores) if self.runs else 0

    def _sameprov(self, judge_idx: int, row_idx: int) -> int:
        provider = self.runs[judge_idx].provider
        if self.candidate_providers is not None:
            return 1 if provider == self.candidate_providers[row_idx] else 0
        if self.candidate_provider is None:
            return 0
        return 1 if provider == self.candidate_provider else 0

    def _grades_by_row(self, dimension: str) -> List[List[str]]:
        return [
            [getattr(run.scores[ri], dimension).grade for run in self.runs]
            for ri in range(self.n_rows)
        ]

    def fleiss_kappa(self, dimension: str) -> Optional[float]:
        """Inter-judge agreement (Fleiss' kappa) on the per-row consensus grade."""
        if len(self.runs) < 2:
            return None
        matrix = []
        for row in self._grades_by_row(dimension):
            matrix.append([row.count(g) for g in GRADES])
        return _fleiss_kappa(matrix)

    def leniency(self, dimension: str) -> Dict[str, Dict[str, Any]]:
        """Per-judge candidate-favor rate (fraction of votes for ``b``)."""
        out: Dict[str, Dict[str, Any]] = {}
        for run in self.runs:
            favored = 0
            votes = 0
            for ri in range(self.n_rows):
                for v in getattr(run.scores[ri], dimension).preferred:
                    votes += 1
                    if v == "b":
                        favored += 1
            out[run.name] = {
                "provider": run.provider,
                "favored_b": favored,
                "votes": votes,
                "favor_rate": favored / votes if votes else None,
            }
        return out

    def _vote_records(self, dimensions: Sequence[str]) -> List[Tuple[int, int, int]]:
        recs: List[Tuple[int, int, int]] = []
        for ji, run in enumerate(self.runs):
            for ri in range(self.n_rows):
                sp = self._sameprov(ji, ri)
                for dim in dimensions:
                    for v in getattr(run.scores[ri], dim).preferred:
                        recs.append((1 if v == "b" else 0, ji, sp))
        return recs

    def same_provider_effect(
        self, dimension: Optional[str] = None
    ) -> Dict[str, Any]:
        """Leniency-adjusted same-provider self-preference via exact permutation test.

        ``dimension=None`` pools votes across all dimensions. Returns
        ``identifiable=False`` when the same-provider label is constant within
        every judge (confounded with judge identity) -- which is the single-
        candidate case; supply a multi-candidate panel (``candidate_providers``)
        to estimate the effect.
        """
        dims = DIMENSIONS if dimension is None else [dimension]
        recs = self._vote_records(dims)
        base: Dict[str, Any] = {
            "n_votes": len(recs),
            "judge_leniency": {d: self.leniency(d) for d in dims},
        }
        if not recs:
            return {**base, "identifiable": False, "note": "no votes available"}

        # Confounding check: same-provider must vary *within* at least one judge.
        pure = True
        for ji in range(len(self.runs)):
            labels = {sp for (_, j, sp) in recs if j == ji}
            if len(labels) != 1:
                pure = False
                break
        if pure:
            return {
                **base,
                "identifiable": False,
                "note": (
                    "same-provider label is constant within each judge (confounded "
                    "with judge identity); pass per-row candidate_providers (a "
                    "multi-candidate panel) to estimate the same-provider effect."
                ),
            }

        # Leniency adjustment: subtract each judge's mean favor rate (proxy for the
        # paper's vote-level logistic-regression adjustment for judge leniency).
        judge_mean: Dict[int, float] = {}
        for ji in range(len(self.runs)):
            ys = [y for (y, j, _sp) in recs if j == ji]
            judge_mean[ji] = sum(ys) / len(ys)
        resid = [(y - judge_mean[ji], sp) for (y, ji, sp) in recs]

        def stat(pairs: Sequence[Tuple[float, int]]) -> Optional[float]:
            sp1 = [r for (r, sp) in pairs if sp == 1]
            sp0 = [r for (r, sp) in pairs if sp == 0]
            if not sp1 or not sp0:
                return None
            return sum(sp1) / len(sp1) - sum(sp0) / len(sp0)

        s_obs = stat(resid)
        if s_obs is None:
            return {**base, "identifiable": False, "note": "only one provider present"}

        rng = random.Random(self.seed)
        residuals = [r for (r, _) in resid]
        labels = [sp for (_, sp) in resid]
        ge = 0
        for _ in range(self.n_perm):
            perm = labels[:]
            rng.shuffle(perm)
            s = stat(list(zip(residuals, perm)))
            if s is not None and abs(s) >= abs(s_obs):
                ge += 1
        # +1 smoothing: standard Monte-Carlo exact-permutation p-value estimator.
        p_value = (1 + ge) / (1 + self.n_perm)
        return {
            **base,
            "identifiable": True,
            "same_provider_advantage": s_obs,
            "p_value": p_value,
            "n_perm": self.n_perm,
        }

    def consensus_with_cis(self, dimension: str) -> Dict[str, Any]:
        """Win/tie/lose consensus rates with Wilson and bootstrap confidence intervals."""
        consensus = [_majority_grade(row) for row in self._grades_by_row(dimension)]
        n = len(consensus)
        if n == 0:
            return {"n_rows": 0, "rates": {}, "wilson": {}, "bootstrap": {}}
        rates = {g: consensus.count(g) / n for g in GRADES}
        wilson = {
            g: _wilson_interval(consensus.count(g), n, self.alpha) for g in GRADES
        }
        rng = random.Random(self.seed)
        reps = {g: [] for g in GRADES}
        for _ in range(self.n_bootstrap):
            sample = [consensus[rng.randrange(n)] for _ in range(n)]
            for g in GRADES:
                reps[g].append(sample.count(g) / n)
        half = self.alpha / 2
        bootstrap = {
            g: (
                _percentile(sorted(reps[g]), half),
                _percentile(sorted(reps[g]), 1 - half),
            )
            for g in GRADES
        }
        return {"n_rows": n, "rates": rates, "wilson": wilson, "bootstrap": bootstrap}

    def judge_vs_consensus_agreement(self) -> Dict[str, Any]:
        """Per-judge Cohen's kappa vs leave-one-out majority consensus, per dimension."""
        out: Dict[str, Any] = {"per_dimension": {}, "mean": None}
        if len(self.runs) < 2:
            return out
        means = []
        for dim in DIMENSIONS:
            grades = self._grades_by_row(dim)
            per_judge: Dict[str, Optional[float]] = {}
            kappas = []
            for ji, run in enumerate(self.runs):
                loo = [
                    _majority_grade([grades[ri][j] for j in range(len(self.runs)) if j != ji])
                    for ri in range(self.n_rows)
                ]
                kap = _cohen_kappa([grades[ri][ji] for ri in range(self.n_rows)], loo)
                per_judge[run.name] = kap
                if kap is not None:
                    kappas.append(kap)
            mean = sum(kappas) / len(kappas) if kappas else None
            if mean is not None:
                means.append(mean)
            out["per_dimension"][dim] = {"per_judge": per_judge, "mean": mean}
        out["mean"] = sum(means) / len(means) if means else None
        return out

    def audit(self) -> Dict[str, Any]:
        """Full judge-sensitivity + paired-significance report."""
        per_dimension: Dict[str, Any] = {}
        for dim in DIMENSIONS:
            per_dimension[dim] = {
                "fleiss_kappa": self.fleiss_kappa(dim),
                "judge_leniency": self.leniency(dim),
                "consensus_with_cis": self.consensus_with_cis(dim),
                "same_provider_effect": self.same_provider_effect(dim),
            }
        fleiss_vals = [
            v for v in (per_dimension[d]["fleiss_kappa"] for d in DIMENSIONS) if v is not None
        ]
        return {
            "config": {
                "n_judges": len(self.runs),
                "n_rows": self.n_rows,
                "candidate_provider": self.candidate_provider,
                "candidate_providers": self.candidate_providers,
                "alpha": self.alpha,
                "n_perm": self.n_perm,
                "n_bootstrap": self.n_bootstrap,
                "seed": self.seed,
            },
            "judges": [
                {"name": r.name, "provider": r.provider} for r in self.runs
            ],
            "per_dimension": per_dimension,
            "overall": {
                "mean_fleiss_kappa": sum(fleiss_vals) / len(fleiss_vals) if fleiss_vals else None,
                "same_provider_effect_pooled": self.same_provider_effect(None),
            },
            "judge_vs_consensus_agreement": self.judge_vs_consensus_agreement(),
        }


# --------------------------------------------------------------------------- #
# CLI: consumes existing deep_research_pairwise_evals.py JSONL outputs.
# --------------------------------------------------------------------------- #


def load_run(path: str, name: str) -> JudgeRun:
    """Load one judge's JSONL output into aligned DeepResearchScoreResult objects."""
    scores: List[DeepResearchScoreResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sr = obj.get("score_result", obj)
            if not sr:
                continue
            scores.append(DeepResearchScoreResult.model_validate(sr))
    if not scores:
        raise ValueError(f"no scored rows with a 'score_result' field found in {path}")
    return JudgeRun(name=name, provider=infer_provider(name), scores=scores)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Judge-sensitivity & paired-significance audit over multi-judge "
            "deep_research_pairwise_evals.py JSONL outputs."
        )
    )
    parser.add_argument(
        "--judge-results",
        nargs="+",
        required=True,
        help="One JSONL path per judge run (output of deep_research_pairwise_evals.py), "
        "all over the SAME rows in the SAME order.",
    )
    parser.add_argument(
        "--judge-names",
        nargs="+",
        help="Judge / model names, one per --judge-results file (drives provider inference). "
        "Defaults to judge_0, judge_1, ...",
    )
    parser.add_argument(
        "--candidate-provider",
        default=None,
        help="Provider family of the candidate (b) answers: openai/anthropic/google/xai. "
        "Use --candidate-providers-file for a multi-candidate panel.",
    )
    parser.add_argument(
        "--candidate-providers-file",
        default=None,
        help="Optional path: one candidate provider per row (enables the same-provider test).",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to write the audit report.")
    parser.add_argument("--n-perm", type=int, default=9999, help="Permutations for the same-provider test.")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap replicates for consensus CIs.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for confidence intervals.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for permutation / bootstrap.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = _parse_args(argv)
    names = args.judge_names or [f"judge_{i}" for i in range(len(args.judge_results))]
    if len(names) != len(args.judge_results):
        raise SystemExit("--judge-names must match --judge-results count")
    runs = [load_run(p, n) for p, n in zip(args.judge_results, names)]

    candidate_providers = None
    if args.candidate_providers_file:
        candidate_providers = [
            ln.strip() for ln in Path(args.candidate_providers_file).read_text().splitlines() if ln.strip()
        ]

    audit = JudgeSensitivityAudit(
        runs,
        candidate_provider=args.candidate_provider,
        candidate_providers=candidate_providers,
        alpha=args.alpha,
        n_perm=args.n_perm,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    report = audit.audit()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "judge_sensitivity_audit.json"
    dest.write_text(json.dumps(report, indent=2))

    overall = report["overall"]
    jvc = report["judge_vs_consensus_agreement"]
    print(f"Judges: {report['config']['n_judges']}  Rows: {report['config']['n_rows']}")
    print(f"Report written to {dest}")
    print("Per-dimension Fleiss' kappa (inter-judge agreement):")
    for dim in DIMENSIONS:
        k = report["per_dimension"][dim]["fleiss_kappa"]
        print(f"  {dim}: {k:.3f}" if k is not None else f"  {dim}: n/a")
    sp = overall["same_provider_effect_pooled"]
    if sp.get("identifiable"):
        print(
            "Same-provider effect (pooled): "
            f"advantage={sp['same_provider_advantage']:.3f}, p={sp['p_value']:.4f}"
        )
    else:
        print(f"Same-provider effect (pooled): not identifiable ({sp.get('note', '')})")
    if jvc.get("mean") is not None:
        print(f"Judge-vs-consensus mean Cohen's kappa: {jvc['mean']:.3f}")
    return report


if __name__ == "__main__":
    main()
