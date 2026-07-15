"""Bias-aware aggregation of pairwise deep-research eval results.

Sibling entry point to ``deep_research_pairwise_evals.py``. The base pipeline
runs flipped-order trials and aggregates them by naive majority vote; this
script re-aggregates *already-computed* eval results with the bias-aware
Bayesian ranker from :mod:`evals.bias_aware_pairwise_ranking`, separating
latent candidate quality from the judge's position bias and reporting
posterior uncertainty plus a top-k-aware recommendation for the next trial
order. It consumes the exact JSONL contract that
``deep_research_pairwise_evals.py`` writes, so existing runs can be
re-analyzed without spending more API budget.

Adapted from "Ask the Right Comparison: Bias-Aware Bayesian Active Top-k
Ranking with LLM Judges" (arXiv:2607.02104); see the capability module
docstring for the Mode-2 substitutions (Laplace-approx inference, L1 bias
shrinkage, repo-native inputs).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evals.bias_aware_pairwise_ranking import debiased_pairwise_summary
from evals.metrics.deep_research_pairwise_metric import DeepResearchScoreResult


def bias_aware_aggregate(
    results: List[Dict[str, Any]],
    prior_bias_scale: float = 1.0,
) -> Dict[str, Any]:
    """Aggregate eval-result rows with the bias-aware ranker.

    Args:
        results: rows as produced by ``DeepResearchEvaluator.evaluate_batch``
            (each carries ``success`` and a ``score_result`` dict).
        prior_bias_scale: Laplace scale for the position-bias prior; smaller
            shrinks bias harder toward zero.

    Returns:
        Per-dimension debiased candidate advantage, shrunk position-bias
        estimate, posterior ``p_candidate_better``, the naive win rate for
        contrast, and the next-trial order the acquisition rule recommends.
    """
    score_results = []
    for row in results:
        if not row.get("success", False):
            continue
        raw = row.get("score_result")
        if not raw:
            continue
        # Validate through the non-new metric's data model so the aggregation
        # is anchored on the repo's real contract, not a parallel schema.
        score_results.append(DeepResearchScoreResult.model_validate(raw))

    if not score_results:
        return {"support": 0, "error": "No successful evaluations found"}

    summary = debiased_pairwise_summary(
        score_results, prior_bias_scale=prior_bias_scale
    )
    summary["support"] = len(score_results)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-aggregate Deep Research pairwise eval results with the "
        "bias-aware Bayesian ranker (debiased for judge position bias)."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="JSONL of eval results as written by deep_research_pairwise_evals.py "
        "(rows with 'success' and 'score_result').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the bias-aware aggregate JSON.",
    )
    parser.add_argument(
        "--prior-bias-scale",
        type=float,
        default=1.0,
        help="Laplace prior scale for the position-bias coefficient; smaller "
        "shrinks the bias harder toward zero (the 'data decides' knob).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval results from {args.input_data}")
    results_df = pd.read_json(args.input_data, lines=True)
    results = results_df.to_dict(orient="records")
    print(f"Loaded {len(results)} rows")

    aggregate = bias_aware_aggregate(results, prior_bias_scale=args.prior_bias_scale)

    aggregate_path = output_dir / "bias_aware_aggregate.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Bias-aware aggregate saved to {aggregate_path}")

    if "error" in aggregate:
        print(aggregate["error"])
        return

    print(f"\nSupport: {aggregate.get('support', 0)}")
    for dim, m in aggregate.items():
        if dim == "support" or not isinstance(m, dict):
            continue
        flag = " (shrunken->0)" if m["position_bias_shrunken"] else ""
        print(
            f"  {dim}: debiased advantage {m['candidate_advantage_mean']:+.3f}"
            f" +/- {m['candidate_advantage_std']:.3f} | "
            f"p(candidate better)={m['p_candidate_better']:.3f} | "
            f"naive winrate={m['naive_winrate']:.3f} | "
            f"position bias={m['position_bias_mean']:+.3f}{flag} | "
            f"next position_b={m['recommended_next_position_b']:.0f}"
        )


if __name__ == "__main__":
    main()
