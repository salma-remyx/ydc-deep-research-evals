"""Absolute (pointwise) scoring evaluation entry point.

Sibling CLI to ``deep_research_pairwise_evals`` that scores deep research
reports with the bias-robust pointwise protocol from
``deep_research_absolute_metric`` instead of the pairwise protocol. It
reuses :class:`DeepResearchEvaluator`'s batching, parallelism and
aggregation by swapping the injected metric, so the output schema and
downstream tooling are unchanged.

Run with::

    python evals/absolute_scoring_evals.py \\
      --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \\
      --output-dir path/to/output \\
      --model o3-mini-2025-01-31

Adapted from "Pairwise or Pointwise? Evaluating Feedback Protocols for
Bias in LLM-Based Evaluation" (arXiv:2504.14716).
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_absolute_metric import DeepResearchAbsoluteMetric
from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL


class AbsoluteScoringEvaluator(DeepResearchEvaluator):
    """Deep research evaluator backed by the absolute (pointwise) metric.

    Inherits :class:`DeepResearchEvaluator`'s ``evaluate_single`` /
    ``evaluate_batch`` / ``aggregate_results`` (which all delegate to the
    injected metric) and swaps that metric for the pointwise protocol. The
    attribute intentionally keeps the base class's ``pairwise_metric`` name
    so the inherited methods bind without editing the base class.
    """

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
    ):
        super().__init__(
            model=model,
            output_path=output_path,
            num_workers=num_workers,
            metric_num_workers=metric_num_workers,
            metric_num_trials=metric_num_trials,
        )
        # Swap the injected metric for the bias-robust pointwise protocol.
        self.pairwise_metric = DeepResearchAbsoluteMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Research absolute (pointwise) evaluations"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV file. The CSV should have columns 'question', 'baseline_answer', and 'candidate_answer'.",
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
        help="Number of worker threads used in the absolute metric computation on each row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation. Each trial scores the candidate and the baseline independently (there is no presentation order to bias).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output path
    output_path = output_dir / f"deep_research_absolute_results_{args.model}.jsonl"

    # Load input data
    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    # Initialize evaluator
    evaluator = AbsoluteScoringEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    # Run evaluation
    print(
        f"Starting absolute-scoring evaluation with model {args.model} using "
        f"{args.num_workers} workers and {args.metric_num_workers} metric workers..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    # Compute aggregate metrics
    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        # Save aggregate metrics
        aggregate_path = output_dir / f"deep_research_absolute_aggregate_{args.model}.json"
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


if __name__ == "__main__":
    main()
