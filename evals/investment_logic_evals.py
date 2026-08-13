"""Sibling CLI for the investment-logic process-trace evaluation.

Runs InvestLogicBench's P->E->R->D->O process-trace audit over a
DeepConsult-style CSV (``question`` / ``baseline_answer`` /
``candidate_answer``) using the repo's existing pairwise judge infra.
Reached from the main pairwise CLI via ``--investment-logic-scoring``, or
directly as ``python -m evals.investment_logic_evals``.

Adapted from InvestLogicBench (arXiv:2608.06108); see
``evals/metrics/investment_logic_metric.py`` for the methodology and the
full scope notes (what is ported at fidelity vs. intentionally omitted).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.investment_logic_metric import (
    DEFAULT_EVAL_MODEL,
    InvestmentLogicMetric,
    InvestmentLogicScoreResult,
)


class InvestmentLogicEvaluator(DeepResearchEvaluator):
    """DeepResearchEvaluator wired to the investment-logic process-trace metric.

    Inherits ``evaluate_single`` / ``evaluate_batch`` unchanged (they are
    metric-agnostic) and only swaps the underlying metric for
    ``InvestmentLogicMetric`` and re-points aggregation at the process-trace
    score-result type.
    """

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
    ):
        self.model = model
        self.output_path = output_path
        self.num_workers = num_workers
        self.metric_num_trials = metric_num_trials
        self.pairwise_metric = InvestmentLogicMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,
        )

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate evaluation results on the process-trace dimensions."""
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"support": 0, "error": "No successful evaluations found"}

        score_results = []
        for result in successful_results:
            try:
                score_results.append(
                    InvestmentLogicScoreResult.model_validate(
                        result["score_result"]
                    )
                )
            except Exception as e:
                print(f"Error parsing score result: {e}")

        return self.pairwise_metric.aggregate(score_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run investment-logic (P->E->R->D->O) process-trace evaluations"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV with 'question', 'baseline_answer', and 'candidate_answer'.",
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
        help="Number of worker threads used in the metric computation on each row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation. Each trial runs the evaluation twice (original and flipped).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output path
    output_path = output_dir / f"investment_logic_results_{args.model}.jsonl"

    # Load input data
    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    # Initialize evaluator
    evaluator = InvestmentLogicEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    # Run evaluation
    print(
        f"Starting investment-logic evaluation with model {args.model} "
        f"using {args.num_workers} workers and {args.metric_num_workers} "
        f"metric workers..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    # Compute and save aggregate metrics
    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        aggregate_path = output_dir / f"investment_logic_aggregate_{args.model}.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)

        print(f"Aggregate metrics saved to {aggregate_path}")

        # Display key metrics
        print("\nKey Metrics:")
        print(f"Total examples: {aggregate_metrics.get('support', 0)}")

        if "overall" in aggregate_metrics:
            print("\nOverall Metrics:")
            for metric, value in aggregate_metrics["overall"].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

        if "process_grounding_gap" in aggregate_metrics:
            print("\nProcess-vs-Grounding Diagnostic:")
            print(
                "plausibility_minus_grounding: "
                f"{aggregate_metrics['process_grounding_gap']:.4f}"
            )
            print(f"verdict: {aggregate_metrics['grounding_warning']}")


if __name__ == "__main__":
    main()
