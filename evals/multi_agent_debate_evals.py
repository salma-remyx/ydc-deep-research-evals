"""Sibling CLI for deep-research pairwise evaluation via multi-agent debate.

Mirrors ``evals.deep_research_pairwise_evals`` but swaps the single-pass
``DeepResearchPairwiseMetric`` for :class:`MultiAgentDebateMetric`, so a
multi-agent debate judge architecture can be run on the same DeepConsult inputs
as the single-pass judge. Results are written to ``debate_``-prefixed files so
the two architectures' outputs coexist for a head-to-head comparison.

Run as a module (matching the repo's existing entry points)::

    python -m evals.multi_agent_debate_evals \\
        --input-data datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv \\
        --output-dir path/to/output \\
        --debaters 3 --debate-rounds 1

Adapted (Mode 2) from "Does Multi-Agent Debate Improve AI Feedback on Research
Papers?", arxiv: https://arxiv.org/abs/2607.14713v1 -- see the metric module's
docstring for the substitution/scoping notes.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.multi_agent_debate_metric import MultiAgentDebateMetric


class MultiAgentDebateEvaluator(DeepResearchEvaluator):
    """Evaluator that scores report pairs with the multi-agent debate metric.

    Subclasses :class:`DeepResearchEvaluator` and replaces its ``pairwise_metric``
    with :class:`MultiAgentDebateMetric`, reusing the inherited
    ``evaluate_single`` / ``evaluate_batch`` / ``aggregate_results`` pipeline so
    the two judge architectures are scored on identical footing.
    """

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
        debaters: int = 3,
        debate_rounds: int = 1,
    ):
        """
        Args:
            model: The model to use for every debater and chair call.
            output_path: Path to save evaluation results.
            num_workers: Workers for parallel processing of evaluation tasks.
            metric_num_workers: Workers for the underlying metric trial fan-out.
            metric_num_trials: Number of (original + flipped) trials per row.
            debaters: Number of debating agents on the panel.
            debate_rounds: Number of debate revision rounds after opening verdicts.
        """
        super().__init__(
            model=model,
            output_path=output_path,
            num_workers=num_workers,
            metric_num_workers=metric_num_workers,
            metric_num_trials=metric_num_trials,
        )
        # Swap the inherited single-pass metric for the debate architecture.
        self.pairwise_metric = MultiAgentDebateMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,
            n_debaters=debaters,
            n_rounds=debate_rounds,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Research pairwise evaluations via multi-agent debate"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV file with 'question', 'baseline_answer', and 'candidate_answer' columns.",
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
        help="Model to use for every debater and chair call",
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
        help="Number of worker threads used in the metric trial fan-out",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation (each runs original + flipped).",
    )
    parser.add_argument(
        "--debaters",
        type=int,
        default=3,
        help="Number of debating agents on the panel",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=1,
        help="Number of debate revision rounds after the opening verdicts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # debate_-prefixed so single-pass and debate outputs coexist for comparison.
    output_path = output_dir / f"debate_deep_research_results_{args.model}.jsonl"

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    evaluator = MultiAgentDebateEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
        debaters=args.debaters,
        debate_rounds=args.debate_rounds,
    )

    print(
        f"Starting debate evaluation with model {args.model} "
        f"({args.debaters} debaters x {args.debate_rounds} rounds + chair, "
        f"{args.metric_num_trials} trials)..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    results_df = pd.read_json(output_path, lines=True)
    print("\nResults summary:")
    print(f"Model: {args.model}")
    print(f"Total evaluations: {len(results_df)}")
    print(
        f"Successful evaluations: {len(results_df[results_df.get('success', False)])}"
    )
    print(f"Failed evaluations: {len(results_df[~results_df.get('success', False)])}")

    # Surface the token-cost trade-off the paper highlights: debate issues many
    # more model calls than the single-pass judge for the same rows.
    print(
        f"Total structured-output model calls: {evaluator.pairwise_metric.model_call_count}"
    )

    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        aggregate_path = (
            output_dir / f"debate_deep_research_aggregate_{args.model}.json"
        )
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
