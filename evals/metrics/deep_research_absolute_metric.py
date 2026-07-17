"""Absolute (pointwise) scoring metric for deep research reports.

A bias-robust sibling to the pairwise metric in
``deep_research_pairwise_metric``. Instead of asking the judge to choose
between two reports -- which is sensitive to presentation order and to
distractor features such as report length -- the judge scores a *single*
report on an absolute 0-10 scale for each of the four established
dimensions. Both the candidate and the baseline are scored independently,
and the relative verdict (win/tie/lose) is *derived* from the absolute
scores rather than elicited directly. Verdicts within a small score band
(``TIE_BAND``) are treated as ties, so minor score differences no longer
flip the result.

Adapted from "Pairwise or Pointwise? Evaluating Feedback Protocols for
Bias in LLM-Based Evaluation" (arXiv:2504.14716), which finds absolute
(pointwise) scoring markedly more robust to distractor-feature bias than
pairwise preference elicitation.

Mode 2 (adapted port): the paper's core mechanism -- absolute instead of
relative feedback -- is implemented at full fidelity. Auxiliary pieces are
target-native: the paper's own bias benchmark suite is intentionally out of
scope (evaluation belongs downstream), and we reuse the repo's existing
o3-mini structured-output judge path, the four dimensions, and the
``DimensionResult`` / ``DeepResearchScoreResult`` output contracts so the
metric drops into the existing aggregation pipeline unchanged. The pairwise
metric's auxiliary LLM "explanation summary" call is cut from aggregation
here (a static descriptor is returned instead) to keep aggregation
network-free.
"""

import concurrent.futures
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.utils import query_openai_model_structured_outputs

# Absolute scores within this band of each other are treated as a tie when
# deriving the relative verdict. This is the crux of the protocol's
# robustness: small absolute-score differences do not flip the verdict,
# whereas a pairwise judge can flip its preference on a distractor feature.
TIE_BAND = 0.5


class AbsoluteScore(BaseModel):
    """A single absolute quality score for one dimension."""

    explanation: str
    score: int = Field(
        description="Absolute quality score from 0 (extremely poor) to 10 (excellent)."
    )


class DeepResearchAbsoluteScoreOutput(BaseModel):
    """Structured output for the absolute scoring judge."""

    instruction_following: AbsoluteScore = Field(
        description="Evaluates response's fidelity to user specified instructions and constraints."
    )
    comprehensiveness: AbsoluteScore = Field(
        description="Measures breadth and range of information covered in response, addressing the scope of user request."
    )
    completeness: AbsoluteScore = Field(
        description="Measures the depth and thoroughness of information for topics addressed in the report."
    )
    writing_quality: AbsoluteScore = Field(
        description="Evaluates clarity, conciseness, logical organization and overall readability of the report."
    )


DEEP_RESEARCH_ABSOLUTE_PROMPT = """
You are an expert evaluator for reports answering a research question. You will score a single report on an absolute quality scale.

Evaluate the report on these dimensions:
1. Instruction following: Evaluates response's fidelity to user specified instructions and constraints.
2. Comprehensiveness: Measures breadth and range of information covered in response, addressing the scope of user request.
3. Completeness: Measures the depth and thoroughness of information for topics addressed in the report.
4. Writing quality: Evaluates clarity, conciseness, logical organization and overall readability of the report.

For each dimension, assign an absolute quality score from 0 to 10 (0 = extremely poor, 10 = excellent) and provide a concise explanation that cites specific examples from the report to justify the score.

Score this report strictly on its own merits. Do not compare it to any other report.
The length of a report is not necessarily an indicator of quality - focus on the substance and how well it meets the user's needs.
"""


class DeepResearchAbsoluteMetric:
    """Pointwise/absolute scoring metric for deep research reports.

    Mirrors the interface of :class:`DeepResearchPairwiseMetric`
    (``score`` -> :class:`DeepResearchScoreResult`, ``aggregate`` -> dict)
    so it can be swapped into :class:`DeepResearchEvaluator` unchanged.
    """

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
    ):
        """
        Args:
            eval_model: The model to use for evaluation
            num_trials: Number of independent scorings per report
            num_workers: Number of parallel workers to use across trials
        """
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers

    def _get_scoring_messages(
        self, question: str, report: str
    ) -> List[ChatCompletionMessageParam]:
        """Generate the messages for scoring a single report."""
        return [
            {"role": "system", "content": DEEP_RESEARCH_ABSOLUTE_PROMPT},
            {
                "role": "user",
                "content": f"""
<prompt>
{question}
</prompt>

<report>
{report}
</report>
""",
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query_scoring_model(
        self, messages: List[ChatCompletionMessageParam]
    ) -> DeepResearchAbsoluteScoreOutput:
        """Query the scoring model with retry logic."""
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=DeepResearchAbsoluteScoreOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return DeepResearchAbsoluteScoreOutput.model_validate(output)

    def _get_absolute_scores(
        self, question: str, candidate_answer: str, baseline_answer: str
    ) -> Dict[str, Any]:
        """Score the candidate and baseline reports independently across trials."""
        candidate_messages = self._get_scoring_messages(question, candidate_answer)
        baseline_messages = self._get_scoring_messages(question, baseline_answer)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            candidate_futures = [
                executor.submit(self._query_scoring_model, candidate_messages)
                for _ in range(self.num_trials)
            ]
            baseline_futures = [
                executor.submit(self._query_scoring_model, baseline_messages)
                for _ in range(self.num_trials)
            ]

            candidate_outputs: List[DeepResearchAbsoluteScoreOutput] = []
            for future in concurrent.futures.as_completed(candidate_futures):
                try:
                    candidate_outputs.append(future.result())
                except Exception as exc:
                    print(f"Candidate trial generated an exception: {exc}")

            baseline_outputs: List[DeepResearchAbsoluteScoreOutput] = []
            for future in concurrent.futures.as_completed(baseline_futures):
                try:
                    baseline_outputs.append(future.result())
                except Exception as exc:
                    print(f"Baseline trial generated an exception: {exc}")

        if not candidate_outputs or not baseline_outputs:
            raise ValueError("Failed to get enough outputs from evaluation model")

        results: Dict[str, Any] = {}
        for dimension in DIMENSIONS:
            candidate_scores = [getattr(o, dimension).score for o in candidate_outputs]
            baseline_scores = [getattr(o, dimension).score for o in baseline_outputs]

            # Derive a per-trial preference from the absolute scores. Trials
            # inside the tie band contribute to neither side, so a near-tie
            # resolves as "tie" rather than flipping on noise.
            all_preferred: List[str] = []
            for c_score, b_score in zip(candidate_scores, baseline_scores):
                if c_score - b_score > TIE_BAND:
                    all_preferred.append("b")
                elif b_score - c_score > TIE_BAND:
                    all_preferred.append("a")
                else:
                    all_preferred.append("tie")

            results[dimension] = {
                "candidate_scores": candidate_scores,
                "baseline_scores": baseline_scores,
                "all_preferred": all_preferred,
                "raw_outputs": {
                    "candidate": [o.model_dump() for o in candidate_outputs],
                    "baseline": [o.model_dump() for o in baseline_outputs],
                },
            }
        return results

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> DeepResearchScoreResult:
        """
        Score a single question-answer pair using absolute (pointwise) scoring.

        Args:
            question: The research question
            baseline_answer: The reference answer (scored independently)
            candidate_answer: The candidate answer to evaluate

        Returns:
            Object containing dimension-level scores and grades
        """
        metric_output = self._get_absolute_scores(
            question=question,
            candidate_answer=candidate_answer,
            baseline_answer=baseline_answer,
        )

        dimension_results: Dict[str, DimensionResult] = {}
        for dimension in DIMENSIONS:
            dim = metric_output[dimension]
            num_wins = sum(1 for p in dim["all_preferred"] if p == "b")
            num_losses = sum(1 for p in dim["all_preferred"] if p == "a")
            if num_wins > num_losses:
                grade = "win"
            elif num_wins < num_losses:
                grade = "lose"
            else:
                grade = "tie"

            dimension_results[dimension] = DimensionResult(
                grade=grade,
                is_win=grade == "win",
                is_tie=grade == "tie",
                is_lose=grade == "lose",
                # Candidate's mean absolute quality score on a 0-10 scale.
                # (Under the pairwise protocol this field is a gap-normalized
                # score centered at 5; the contract is shared, the semantics
                # differ by protocol.)
                score=sum(dim["candidate_scores"]) / len(dim["candidate_scores"]),
                preferred=dim["all_preferred"],
                raw_preferences=dim["raw_outputs"],
            )

        return DeepResearchScoreResult(**dimension_results)

    def aggregate(
        self, scores_list: List[DeepResearchScoreResult]
    ) -> Dict[str, Any]:
        """
        Aggregate absolute-scoring metrics from multiple scored rows.

        Produces the same per-dimension structure (win_rate, tie_rate,
        lose_rate, avg_score, net_winrate) as the pairwise metric so the two
        protocols' aggregates are directly comparable.
        """
        aggregated_metrics: Dict[str, Any] = {"support": len(scores_list)}

        for dimension in DIMENSIONS:
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
        for metric in [
            "win_rate",
            "tie_rate",
            "lose_rate",
            "avg_score",
            "net_winrate",
        ]:
            aggregated_metrics["overall"][metric] = sum(
                aggregated_metrics[dimension][metric] for dimension in DIMENSIONS
            ) / len(DIMENSIONS)

        # Static descriptor instead of an extra LLM call: keeps aggregation
        # network-free and documents how to read avg_score under this protocol.
        aggregated_metrics["explanation_summary"] = (
            "Pointwise/absolute protocol: per-dimension avg_score is the "
            "candidate's mean absolute quality score (0-10); win/tie/lose is "
            "derived from absolute scores within a tie band of +/- "
            f"{TIE_BAND}."
        )
        return aggregated_metrics
