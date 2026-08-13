"""Investment-logic process-trace metric for deep research reports.

Adapted from InvestLogicBench (arXiv:2608.06108), which evaluates financial
LLMs against a P->E->R->D->O process trace -- investor Profile, observable
market Events, investment Reasoning, executable Decision, delayed Outcome.
The paper's headline finding is that *logical plausibility* and *event
grounding* disagree: reports can read as polished and internally coherent
while being weakly tethered to observable evidence -- a failure mode that
output-only evaluation hides.

This metric ports that insight onto the repo's pairwise judge. An o3-mini
judge decomposes each report into a P->E->R->D->O trace and compares a
candidate against a baseline on three process-focused dimensions --
``logical_plausibility``, ``event_grounding``, and ``process_completeness``
-- using flipped-order trials to mitigate position bias, exactly as the
base ``DeepResearchPairwiseMetric`` does. The aggregate emits an explicit
plausibility-vs-grounding gap so "polished but weakly grounded" reasoning
is surfaced rather than hidden.

Scope (Mode 2 -- adapted port): the paper's dataset (201k decisions from
151 real investors), its profile-conditioned generation task, point-in-time
event binding, and end-to-end replay ledger are intentionally NOT ported --
they require data-system infrastructure this repo does not host. What is
kept at full fidelity is the *methodology*: the P->E->R->D->O process trace
and the plausibility-vs-grounding grounding audit, contract-anchored to the
existing judge infra.
"""

import concurrent.futures
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DimensionResult,
    DeepResearchPairwisePreferenceInput,
    Preference,
)
from evals.utils import query_openai_model_structured_outputs

INVESTMENT_LOGIC_DIMENSIONS = [
    "logical_plausibility",
    "event_grounding",
    "process_completeness",
]

# A pairwise investment-logic input has the same shape as the base pairwise
# input (a question plus two reports). Aliasing reuses its answer
# normalization -- the non-empty check and markdown-link stripping -- so the
# process-trace judge sees the same cleaned text the output-quality judge does.
InvestmentLogicPreferenceInput = DeepResearchPairwisePreferenceInput


class InvestmentLogicPreferenceOutput(BaseModel):
    """Pairwise investment-logic preference output across the P->E->R->D->O axes."""

    logical_plausibility: Preference = Field(
        description=(
            "Which report's investment reasoning (the R in P->E->R->D->O) is more "
            "internally coherent and logically sound."
        )
    )
    event_grounding: Preference = Field(
        description=(
            "Which report's reasoning is better tethered to specific, observable "
            "market evidence (the E) rather than generic or unsupported assertions. "
            "A plausible-sounding report that rests on vague or unspecified claims "
            "scores LOW here."
        )
    )
    process_completeness: Preference = Field(
        description=(
            "Which report better traces a complete Profile -> Events -> Reasoning -> "
            "Decision chain, connecting concrete evidence to an actionable "
            "conclusion instead of jumping straight to a recommendation."
        )
    )


class InvestmentLogicScoreResult(BaseModel):
    """Process-trace evaluation results across the investment-logic dimensions."""

    logical_plausibility: DimensionResult
    event_grounding: DimensionResult
    process_completeness: DimensionResult


INVESTMENT_LOGIC_PROMPT = """
You are an expert evaluator for investment and market-analysis research reports. You will compare two reports responding to the same research question: report_a and report_b.

First, mentally decompose EACH report into a P->E->R->D->O process trace:
- P (Profile): the implied investor goals, horizon, and risk context the report addresses.
- E (Events): the observable market evidence, data points, and specific factual claims the reasoning relies on.
- R (Reasoning): the investment logic that connects events to conclusions.
- D (Decision): the actionable recommendation or conclusion the report reaches.
- O (Outcome): any projected outcome, horizon, or risk scenario the report states.

Then compare the two reports on these dimensions:
1. logical_plausibility: which report's investment reasoning (R) is more internally coherent and logically sound?
2. event_grounding: which report's reasoning is better tethered to specific, observable market evidence (E) rather than generic, hand-wavy, or unsupported assertions? A report that sounds polished but rests on vague or unspecified claims scores LOW on grounding.
3. process_completeness: which report better traces a complete Profile -> Events -> Reasoning -> Decision chain, connecting concrete evidence to an actionable conclusion, rather than jumping straight to a recommendation?

CRITICAL: logical_plausibility and event_grounding are DIFFERENT axes. A report can be fluent and internally coherent (high plausibility) yet weakly tethered to real evidence (low grounding). Judge each axis independently on its own merits -- do not let a confident writing style inflate the grounding score, and do not let sparse-but-cited evidence depress the plausibility score.

For each dimension, indicate which report you prefer (either "a" or "b") and provide a concise explanation citing specific evidence from the reports.
Also provide a gap score that measures the difference in quality between the two reports for that dimension. The gap score is a number from 0 to 5, where 0 indicates both reports have similar quality and 5 is the maximum difference in quality.

Be fair and objective. Do not be biased towards either report A or B. The length of a report is not necessarily an indicator of quality -- focus on the substance of the reasoning and how well it is grounded.
"""


class InvestmentLogicMetric:
    """Pairwise metric that audits the investment-logic *process* behind reports.

    Ports InvestLogicBench's P->E->R->D->O process-trace evaluation: rather
    than judging only final output quality, the judge decomposes each report
    into a process trace and scores whether the reasoning is plausible AND
    grounded -- surfacing "polished but weakly grounded" reasoning that
    output-only metrics miss.
    """

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
    ):
        """
        Initialize the metric.

        Args:
            eval_model: The model to use for evaluation
            num_trials: Number of times to run the evaluation model
            num_workers: Number of parallel workers to use for processing trials
        """
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers

    def _get_evaluation_messages(
        self, metric_input: InvestmentLogicPreferenceInput
    ) -> List[ChatCompletionMessageParam]:
        """Generate the messages for the evaluation model."""
        return [
            {"role": "system", "content": INVESTMENT_LOGIC_PROMPT},
            {
                "role": "user",
                "content": f"""
<prompt>
{metric_input.question}
</prompt>

<report_a>
{metric_input.baseline_answer}
</report_a>

<report_b>
{metric_input.candidate_answer}
</report_b>
""",
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query_evaluation_model(
        self, messages: List[ChatCompletionMessageParam]
    ) -> InvestmentLogicPreferenceOutput:
        """Query the evaluation model with retry logic."""
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=InvestmentLogicPreferenceOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return InvestmentLogicPreferenceOutput.model_validate(output)

    def _get_pairwise_preference(
        self, metric_input: InvestmentLogicPreferenceInput
    ) -> dict:
        """Get pairwise preference between a baseline and candidate report."""
        # Create flipped input (baseline=B, candidate=A) to mitigate position bias
        input_flipped = InvestmentLogicPreferenceInput(
            question=metric_input.question,
            baseline_answer=metric_input.candidate_answer,
            candidate_answer=metric_input.baseline_answer,
        )

        messages = self._get_evaluation_messages(metric_input)
        messages_flipped = self._get_evaluation_messages(input_flipped)

        all_outputs: List[InvestmentLogicPreferenceOutput] = []
        all_outputs_flipped: List[InvestmentLogicPreferenceOutput] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            original_futures = [
                executor.submit(self._query_evaluation_model, messages)
                for _ in range(self.num_trials)
            ]
            flipped_futures = [
                executor.submit(self._query_evaluation_model, messages_flipped)
                for _ in range(self.num_trials)
            ]

            for future in concurrent.futures.as_completed(original_futures):
                try:
                    output = future.result()
                    if output is not None:
                        all_outputs.append(output)
                except Exception as exc:
                    print(f"Original trial generated an exception: {exc}")

            for future in concurrent.futures.as_completed(flipped_futures):
                try:
                    output = future.result()
                    if output is not None:
                        all_outputs_flipped.append(output)
                except Exception as exc:
                    print(f"Flipped trial generated an exception: {exc}")

        if not all_outputs or not all_outputs_flipped:
            raise ValueError("Failed to get enough outputs from evaluation model")

        results = {}
        for dimension in INVESTMENT_LOGIC_DIMENSIONS:
            all_preferred: List[str] = []

            preferences = [getattr(output, dimension) for output in all_outputs]
            preferences_flipped = [
                getattr(output, dimension) for output in all_outputs_flipped
            ]

            for pref in preferences:
                all_preferred.append(pref.preferred)
            for pref in preferences_flipped:
                # For flipped evaluations, flip the preference back to the
                # original frame (a/b refer to the swapped order there).
                flipped_pref = "a" if pref.preferred == "b" else "b"
                all_preferred.append(flipped_pref)

            num_wins = sum(1 for p in all_preferred if p == "b")
            num_losses = sum(1 for p in all_preferred if p == "a")

            if num_wins > num_losses:
                grade = "win"
            elif num_wins < num_losses:
                grade = "lose"
            else:
                grade = "tie"

            original_scores = [p.score_b for p in preferences]
            flipped_scores = [-p.score_b for p in preferences_flipped]
            all_scores = original_scores + flipped_scores
            consensus_score = sum(all_scores) / len(all_scores) + 5
            results[dimension] = {
                "raw_preferences": {
                    "original": [p.model_dump() for p in preferences],
                    "flipped": [p.model_dump() for p in preferences_flipped],
                },
                "all_preferred": all_preferred,
                "all_scores": all_scores,
                "consensus_grade": grade,
                "consensus_score": consensus_score,
            }

        return results

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> InvestmentLogicScoreResult:
        """
        Score a single question-answer pair on the process trace.

        Args:
            question: The research question
            baseline_answer: The baseline answer (report_a)
            candidate_answer: The candidate answer (report_b)

        Returns:
            Object containing process-trace dimension scores and grades
        """
        metric_input = InvestmentLogicPreferenceInput(
            question=question,
            baseline_answer=baseline_answer,
            candidate_answer=candidate_answer,
        )

        output_dict = self._get_pairwise_preference(metric_input)

        dimension_results = {}
        for dimension in INVESTMENT_LOGIC_DIMENSIONS:
            dimension_output_dict = output_dict[dimension]
            dimension_results[dimension] = DimensionResult(
                grade=dimension_output_dict["consensus_grade"],
                is_win=dimension_output_dict["consensus_grade"] == "win",
                is_tie=dimension_output_dict["consensus_grade"] == "tie",
                is_lose=dimension_output_dict["consensus_grade"] == "lose",
                score=dimension_output_dict["consensus_score"],
                preferred=dimension_output_dict["all_preferred"],
                raw_preferences=dimension_output_dict["raw_preferences"],
            )

        return InvestmentLogicScoreResult(**dimension_results)

    def aggregate(
        self, scores_list: List[InvestmentLogicScoreResult]
    ) -> Dict[str, Any]:
        """
        Aggregate process-trace metrics from multiple scored rows.

        Args:
            scores_list: List of score result objects from multiple rows

        Returns:
            Dictionary containing aggregated metrics, including a
            plausibility-vs-grounding diagnostic that mirrors InvestLogicBench's
            headline result (polished reasoning can hide weak grounding).
        """
        aggregated_metrics: Dict[str, Any] = {"support": len(scores_list)}

        for dimension in INVESTMENT_LOGIC_DIMENSIONS:
            dimension_results = [
                getattr(score_result, dimension) for score_result in scores_list
            ]

            win_rate = sum(r.is_win for r in dimension_results) / len(
                dimension_results
            )
            tie_rate = sum(r.is_tie for r in dimension_results) / len(
                dimension_results
            )
            lose_rate = sum(r.is_lose for r in dimension_results) / len(
                dimension_results
            )
            avg_score = sum(r.score for r in dimension_results) / len(
                dimension_results
            )

            num_wins = sum(r.is_win for r in dimension_results)
            num_losses = sum(r.is_lose for r in dimension_results)
            net_winrate = (
                num_wins / (num_wins + num_losses)
                if (num_wins + num_losses) > 0
                else 0.0
            )

            aggregated_metrics[dimension] = {
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "lose_rate": lose_rate,
                "avg_score": avg_score,
                "net_winrate": net_winrate,
            }

        aggregated_metrics["overall"] = {}
        metrics = ["win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"]
        for metric in metrics:
            overall_avg = sum(
                aggregated_metrics[dimension][metric]
                for dimension in INVESTMENT_LOGIC_DIMENSIONS
            ) / len(INVESTMENT_LOGIC_DIMENSIONS)
            aggregated_metrics["overall"][metric] = overall_avg

        # Process-vs-grounding diagnostic (InvestLogicBench's headline): when
        # average logical_plausibility materially exceeds event_grounding, the
        # reports read as polished but are weakly tethered to evidence -- the
        # exact failure output-only evaluation hides.
        plausibility = aggregated_metrics["logical_plausibility"]["avg_score"]
        grounding = aggregated_metrics["event_grounding"]["avg_score"]
        aggregated_metrics["process_grounding_gap"] = plausibility - grounding
        aggregated_metrics["grounding_warning"] = (
            "polished_but_weakly_grounded"
            if plausibility - grounding > 1.0
            else "plausibility_and_grounding_aligned"
        )

        return aggregated_metrics
