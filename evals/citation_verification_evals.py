"""Standalone sibling CLI for citation-quality evaluation.

Mirrors ``deep_research_pairwise_evals`` (the established sibling-CLI pattern on
this fork) but swaps in :class:`CitationVerificationMetric` so the existing
per-row / batch / parallel wiring from :class:`DeepResearchEvaluator` is reused
unchanged via the shared ``score`` / ``aggregate`` contract. Citation quality is
judged along the two rubric dimensions from the paper (source relevance +
factual support); see :mod:`evals.metrics.citation_verification_metric`.

Run with::

    python -m evals.citation_verification_evals \\
        --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \\
        --output-dir path/to/output --model o3-mini-2025-01-31
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.citation_verification_metric import (
    DEFAULT_EVAL_MODEL,
    CitationVerificationMetric,
    CitationVerificationScoreResult,
)


class CitationVerificationEvaluator(DeepResearchEvaluator):
    """Report-level pairwise evaluator that scores citation quality.

    Subclasses the existing pairwise evaluator and swaps in the
    citation-verification metric. ``evaluate_single`` / ``evaluate_batch`` are
    generic over the metric contract and are reused as-is; only
    ``aggregate_results`` is overridden because the result model differs.
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
        # Swap the report-quality metric for the citation-quality metric.
        self.pairwise_metric = CitationVerificationMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,
        )

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate citation-quality results using the citation metric."""
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"support": 0, "error": "No successful evaluations found"}

        score_results = []
        for result in successful_results:
            try:
                score_result = CitationVerificationScoreResult.model_validate(
                    result["score_result"]
                )
                score_results.append(score_result)
            except Exception as e:
                print(f"Error parsing score result: {e}")

        return self.pairwise_metric.aggregate(score_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run citation-quality (source relevance + factual support) "
        "pairwise evaluations"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV file. The CSV should have columns 'question', "
        "'baseline_answer', and 'candidate_answer'. Citation markdown links "
        "are preserved (not stripped) so they can be judged.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
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
        help="Number of worker threads used in the pairwise metric on each row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation. Each trial runs the "
        "evaluation twice (original and flipped inputs) to mitigate position "
        "bias.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"citation_verification_results_{args.model}.jsonl"

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    evaluator = CitationVerificationEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    print(
        f"Starting citation-quality evaluation with model {args.model} using "
        f"{args.num_workers} workers and {args.metric_num_workers} metric workers..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        aggregate_path = output_dir / f"citation_verification_aggregate_{args.model}.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)
        print(f"Aggregate metrics saved to {aggregate_path}")

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
