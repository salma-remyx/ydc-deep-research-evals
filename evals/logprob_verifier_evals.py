"""Sibling CLI for continuous logprob-verifier scoring of deep-research reports.

Mirrors ``evals.deep_research_pairwise_evals`` (same CSV -> JSONL contract, same CLI
flags) but swaps the discrete structured-output pairwise judge for the
``LogprobVerifierMetric``, which scores each dimension by the expected value over
scoring-token logits (adapted from LLM-as-a-Verifier, arXiv:2607.05391). See
``evals.metrics.logprob_verifier_metric`` for the mechanism and Mode-2 scoping notes.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.metrics.logprob_verifier_metric import (
    DEFAULT_VERIFIER_GRANULARITY,
    LogprobVerifierMetric,
    LogprobVerifierScoreResult,
)


class LogprobVerifierEvaluator(DeepResearchEvaluator):
    """Evaluator that scores reports with the continuous logprob-verifier metric.

    Reuses the base evaluator's batch/parallel plumbing and per-row extraction (the
    verifier result is field-compatible with the repo's ``DimensionResult``), only
    swapping the metric and validating aggregate inputs with the verifier result class.
    """

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
        granularity: int = DEFAULT_VERIFIER_GRANULARITY,
    ):
        # Skip the base __init__'s pairwise-metric wiring and install the verifier.
        self.model = model
        self.output_path = output_path
        self.num_workers = num_workers
        self.metric_num_trials = metric_num_trials
        self.granularity = granularity
        self.pairwise_metric = LogprobVerifierMetric(
            eval_model=model,
            granularity=granularity,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,
        )

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate verifier results using the verifier result class + metric."""
        successful_results = [r for r in results if r.get("success", False)]
        if not successful_results:
            return {"support": 0, "error": "No successful evaluations found"}

        score_results = []
        for result in successful_results:
            try:
                score_results.append(
                    LogprobVerifierScoreResult.model_validate(result["score_result"])
                )
            except Exception as e:
                print(f"Error parsing score result: {e}")
        return self.pairwise_metric.aggregate(score_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run continuous logprob-verifier evaluations of deep research reports"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV with columns 'question', 'baseline_answer', 'candidate_answer'.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EVAL_MODEL, help="Model to use for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of worker threads for evaluation"
    )
    parser.add_argument(
        "--metric-num-workers",
        type=int,
        default=1,
        help="Number of worker threads used in the verifier metric per row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per dimension (each trial runs original + flipped order).",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=DEFAULT_VERIFIER_GRANULARITY,
        help="Candidate score-token scale 0..granularity (finer = more separation).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"deep_research_verifier_results_{args.model}.jsonl"

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    evaluator = LogprobVerifierEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
        granularity=args.granularity,
    )

    print(
        f"Starting logprob-verifier evaluation with model {args.model} "
        f"(granularity 0..{args.granularity}, {args.metric_num_trials} trials)..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)
        aggregate_path = output_dir / f"deep_research_verifier_aggregate_{args.model}.json"
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
