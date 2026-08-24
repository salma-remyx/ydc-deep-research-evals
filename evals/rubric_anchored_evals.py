"""Rubric-anchored deep research evaluation CLI.

Adapted from "Grading Needs a Rubric, Not Intelligence"
(arXiv:2608.17938). The paper's claim is that grading reliability does not
require an intelligent judge once grading is anchored in an explicit rubric
containing the official answer. This entry point runs the existing pairwise
evaluation pipeline with the judge anchored that way, and can run the
paper's two ablation arms alongside it so a single invocation shows how
much agreement the official answer is carrying.

Usage::

    python -m evals.rubric_anchored_evals \\
        --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \\
        --output-dir out/rubric_anchored \\
        --compare-default

Omit ``--compare-default`` to run the anchored arm alone.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from evals.deep_research_pairwise_evals import (
    DEFAULT_EVAL_MODEL,
    DeepResearchEvaluator,
)
from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchScoreResult,
)
from evals.metrics.reference_rubric import AnchorMode
from evals.metrics.rubric_anchored_pairwise_metric import (
    RubricAnchoredPairwiseMetric,
)

ANCHOR_MODES = [AnchorMode.FULL, AnchorMode.ANSWER_ONLY, AnchorMode.NONE]


class RubricAnchoredEvaluator(DeepResearchEvaluator):
    """Deep research evaluator that grades against an explicit rubric.

    Drops in where :class:`DeepResearchEvaluator` does - same CLI surface,
    same JSONL output shape - but scores with
    :class:`RubricAnchoredPairwiseMetric` and reports the agreement analysis
    the paper is about, alongside the standard aggregates.
    """

    def __init__(
        self,
        *args: Any,
        anchor_mode: AnchorMode = AnchorMode.FULL,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_mode = anchor_mode
        self.pairwise_metric = RubricAnchoredPairwiseMetric(
            eval_model=self.model,
            num_trials=self.metric_num_trials,
            num_workers=self.pairwise_metric.num_workers,
            anchor_mode=anchor_mode,
        )

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Standard aggregates plus the agreement/variance analysis."""
        aggregated = super().aggregate_results(results)

        score_results = self._score_results(results)
        if score_results:
            aggregated["rubric_agreement"] = self.pairwise_metric.analyze_agreement(
                score_results
            )
        return aggregated

    def _score_results(
        self, results: List[Dict[str, Any]]
    ) -> List[DeepResearchScoreResult]:
        """Rehydrate the stored per-row score results."""
        score_results = []
        for result in results:
            if not result.get("success", False):
                continue
            try:
                score_results.append(
                    DeepResearchScoreResult.model_validate(result["score_result"])
                )
            except Exception as e:
                print(f"Error parsing score result: {e}")
        return score_results


def run_arm(
    evaluator: RubricAnchoredEvaluator,
    data: pd.DataFrame,
    output_path: Path,
) -> Dict[str, Any]:
    """Run one anchoring arm end to end and persist its rows and aggregate."""
    results = evaluator.evaluate_batch(data)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    aggregate = evaluator.aggregate_results(results)
    return aggregate


def _print_arm(mode: AnchorMode, aggregate: Dict[str, Any]) -> None:
    print(f"\n=== Anchor mode: {mode.value} ===")
    print(f"Support: {aggregate.get('support', 0)}")

    if "overall" in aggregate:
        print("Overall:")
        for metric, value in aggregate["overall"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    agreement = aggregate.get("rubric_agreement")
    if not agreement:
        return

    print("Agreement (judge reliability under this anchor):")
    overall = agreement.get("overall", {})
    icc = overall.get("icc")
    icc_str = f"{icc:.4f}" if isinstance(icc, float) else "n/a"
    print(f"  icc: {icc_str}")
    print(f"  preference_agreement: {overall.get('preference_agreement', 0.0):.4f}")
    print(
        f"  mean_trial_score_spread: "
        f"{overall.get('mean_trial_score_spread', 0.0):.4f}"
    )

    print("  per dimension:")
    for dimension in DIMENSIONS:
        row = agreement["per_dimension"].get(dimension, {})
        dim_icc = row.get("icc")
        dim_icc_str = f"{dim_icc:.4f}" if isinstance(dim_icc, float) else "n/a"
        print(
            f"    {dimension}: icc={dim_icc_str} "
            f"pref_agreement={row.get('preference_agreement', 0.0):.4f} "
            f"spread={row.get('mean_trial_score_spread', 0.0):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run deep research pairwise evaluations with the judge anchored in an "
            "explicit rubric built from the reference answer, optionally alongside "
            "the paper's ablation arms."
        )
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV with 'question', 'baseline_answer', 'candidate_answer'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for evaluation",
    )
    parser.add_argument(
        "--metric-num-workers",
        type=int,
        default=1,
        help="Number of worker threads used in the pairwise metric per row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation (each runs original + flipped).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=[m.value for m in ANCHOR_MODES],
        default=AnchorMode.FULL.value,
        help=(
            "Which parts of the rubric anchor the judge: 'full' (criteria + official "
            "answer), 'answer_only' (official answer, criteria removed), or 'none' "
            "(the stock flat prompt)."
        ),
    )
    parser.add_argument(
        "--compare-default",
        action="store_true",
        help=(
            "Additionally run the 'answer_only' and 'none' ablation arms so one "
            "invocation shows how much agreement the official answer carries."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    modes = [AnchorMode(args.anchor)]
    if args.compare_default:
        modes = list(ANCHOR_MODES)

    aggregate_by_mode: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        suffix = mode.value if mode is not AnchorMode.FULL else "anchored"
        rows_path = output_dir / f"deep_research_results_{args.model}_{suffix}.jsonl"
        aggregate_path = output_dir / f"deep_research_aggregate_{args.model}_{suffix}.json"

        print(
            f"\nRunning anchor mode '{mode.value}' with model {args.model} "
            f"({args.num_workers} workers, {args.metric_num_workers} metric workers, "
            f"{args.metric_num_trials} trials)..."
        )

        evaluator = RubricAnchoredEvaluator(
            model=args.model,
            output_path=rows_path,
            num_workers=args.num_workers,
            metric_num_workers=args.metric_num_workers,
            metric_num_trials=args.metric_num_trials,
            anchor_mode=mode,
        )

        aggregate = run_arm(evaluator, df, rows_path)

        with open(aggregate_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Results saved to {rows_path}")
        print(f"Aggregate metrics saved to {aggregate_path}")

        aggregate_by_mode[mode.value] = aggregate
        _print_arm(mode, aggregate)

    if len(aggregate_by_mode) > 1:
        comparison_path = output_dir / f"anchor_comparison_{args.model}.json"
        comparison = {
            mode: agg.get("rubric_agreement", {}) for mode, agg in aggregate_by_mode.items()
        }
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nAnchor-mode comparison saved to {comparison_path}")
        print(
            "Compare 'icc' and 'preference_agreement' across modes: the gap between "
            "'answer_only' and 'none' is how much agreement the official answer "
            "carries, per arXiv:2608.17938."
        )


if __name__ == "__main__":
    main()
