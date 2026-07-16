"""Citation-quality pairwise metric for deep-research reports.

Adapted from "Do You Need a Frontier Model as a Citation Verifier? Benchmarking
Rubric LLMs for Deep-Research Source Attribution" (arXiv:2607.08700v1).

Mode 2 (adapted port). The paper judges citation quality as a structured rubric
along two LLM-graded dimensions — *source relevance* (is the cited source
pertinent to the claim?) and *factual support* (does it substantiate the
claim?) — and scores per attribution-citation pair against gold labels
(F1 / kappa / FPR / FNR). Those rubric dimensions are kept verbatim; the
aggregation frame is substituted with this repo's native report-level pairwise
comparison + flipped-order consensus (the same position-bias mitigation the
existing metric uses), since the repo has no gold-labeled citation dataset or
classification machinery. The paper's multi-judge benchmark harness is out of
scope; ``eval_model`` is exposed so its suggested experiment (a frontier judge
vs. a cheaper one) runs by changing one arg.

The paper's other headline — scalar metrics obscure the judge's *directional*
bias — is surfaced as ``flip_disagreement`` (did reversing the report order flip
the verdict?), aggregated as ``flip_disagreement_rate``.

Unlike ``DeepResearchPairwisePreferenceInput``, this metric deliberately
*preserves* markdown citation links; citations cannot be judged once stripped.
"""

import concurrent.futures
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, field_validator
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    Preference,
)
from evals.utils import query_openai_model_structured_outputs

# The two citation-quality rubric dimensions from the paper.
CITATION_DIMENSIONS = ["source_relevance", "factual_support"]


class CitationVerificationPreferenceOutput(BaseModel):
    """Structured rubric output for one (baseline, candidate) pair."""

    source_relevance: Preference = Field(
        description=(
            "Evaluates whether the cited sources are actually pertinent to the "
            "claims they are attached to. Penalize off-topic or decorative "
            "citations and substantive claims left uncited."
        )
    )
    factual_support: Preference = Field(
        description=(
            "Evaluates whether the cited sources genuinely substantiate the "
            "claims they back. Penalize citations that contradict or omit the "
            "evidence the report attributes to them."
        )
    )


CITATION_VERIFICATION_PROMPT = """You are an expert evaluator for the citation quality of deep-research reports.
You will compare two responses to a research question — report_a and report_b — each of which may cite its sources using markdown links of the form [label](url). A high-quality report supports its claims with citations that are both relevant and factually accurate.

Evaluate both reports on these two citation-quality dimensions:
1. Source relevance: Are the cited sources actually pertinent to the claims they are attached to? A report scores poorly when it cites sources that are off-topic, tangential, or merely decorative relative to the claim, or when it makes substantive claims with no supporting citation at all.
2. Factual support: Do the cited sources genuinely substantiate the claims they back? A report scores poorly when citations do not actually contain the evidence the report attributes to them (the source contradicts or does not contain the claim), or when important factual claims are left uncited.

For each dimension, indicate which report you prefer (either "a" or "b") and provide a concise explanation citing specific claims and their supporting citations. Point out what could be improved in the other report.
Also provide a gap score that measures the difference in citation quality between the two reports for that dimension. The gap score should be a number from 0 to 5, where 0 indicates both reports have similar citation quality and 5 is the maximum difference.

Be fair and objective in your evaluation. Do not be biased towards either report A or B. A larger number of citations is not by itself a sign of quality — weigh whether the citations are relevant and whether they actually support the claims.
"""


class CitationVerificationInput(BaseModel):
    """Input for a single citation-verification comparison.

    Deliberately preserves markdown citation links (does NOT call
    ``replace_markdown_links_with_text``), since citation quality cannot be
    judged once the links are stripped.
    """

    question: str
    baseline_answer: str
    candidate_answer: str

    @field_validator("candidate_answer", "baseline_answer")
    @classmethod
    def validate_non_empty_answer(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Answer cannot be empty")
        return v


class CitationDimensionResult(BaseModel):
    """Evaluation result for a single citation-quality dimension."""

    grade: str
    is_win: bool
    is_tie: bool
    is_lose: bool
    score: float
    preferred: List[str]
    raw_preferences: dict
    # Paper's directional-bias signal: did flipping the report order flip the
    # verdict for this dimension? Aggregated as ``flip_disagreement_rate``.
    flip_disagreement: bool


class CitationVerificationScoreResult(BaseModel):
    """Citation-quality evaluation results across both rubric dimensions."""

    source_relevance: CitationDimensionResult
    factual_support: CitationDimensionResult


class CitationVerificationMetric:
    """Pairwise metric judging citation quality (source relevance + factual support).

    Mirrors the ``DeepResearchPairwiseMetric`` contract (``score`` / ``aggregate``)
    so it plugs into the same evaluation pipeline.
    """

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
    ):
        """
        Initialize the evaluator.

        Args:
            eval_model: The model to use for evaluation
            num_trials: Number of times to run the evaluation model
            num_workers: Number of parallel workers to use for processing trials
        """
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers

    def _get_evaluation_messages(
        self, metric_input: CitationVerificationInput
    ) -> List[ChatCompletionMessageParam]:
        """Generate the messages for the evaluation model."""
        return [
            {"role": "system", "content": CITATION_VERIFICATION_PROMPT},
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
    ) -> CitationVerificationPreferenceOutput:
        """Query the evaluation model with retry logic."""
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=CitationVerificationPreferenceOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return CitationVerificationPreferenceOutput.model_validate(output)

    @staticmethod
    def _verdict(preferences: List[str]) -> str:
        """Map a list of canonical preferences ('a'/'b') to a win/lose/tie grade."""
        num_wins = sum(1 for p in preferences if p == "b")
        num_losses = sum(1 for p in preferences if p == "a")
        if num_wins > num_losses:
            return "win"
        if num_wins < num_losses:
            return "lose"
        return "tie"

    def _consensus_for_dimension(
        self,
        preferences: List[Preference],
        preferences_flipped: List[Preference],
    ) -> Dict[str, Any]:
        """Reduce original + flipped trials for one dimension to a consensus.

        Pure (no API): combines flipped-order trials with the original ones
        (position-bias mitigation, matching the existing metric) and computes a
        consensus grade/score plus ``flip_disagreement``.
        """
        # Canonicalize so 'b' always means the candidate answer wins.
        canonical_original = [p.preferred for p in preferences]
        canonical_flipped = [
            "a" if p.preferred == "b" else "b" for p in preferences_flipped
        ]
        all_preferred = canonical_original + canonical_flipped

        grade = self._verdict(all_preferred)
        flip_disagreement = self._verdict(canonical_original) != self._verdict(
            canonical_flipped
        )

        original_scores = [p.score_b for p in preferences]
        flipped_scores = [-p.score_b for p in preferences_flipped]
        all_scores = original_scores + flipped_scores
        consensus_score = sum(all_scores) / len(all_scores) + 5

        return {
            "raw_preferences": {
                "original": [p.model_dump() for p in preferences],
                "flipped": [p.model_dump() for p in preferences_flipped],
            },
            "all_preferred": all_preferred,
            "all_scores": all_scores,
            "consensus_grade": grade,
            "consensus_score": consensus_score,
            "flip_disagreement": flip_disagreement,
        }

    def _build_dimension_results(
        self,
        all_outputs: List[CitationVerificationPreferenceOutput],
        all_outputs_flipped: List[CitationVerificationPreferenceOutput],
    ) -> Dict[str, Dict[str, Any]]:
        """Build per-dimension consensus dicts from original + flipped outputs."""
        results: Dict[str, Dict[str, Any]] = {}
        for dimension in CITATION_DIMENSIONS:
            preferences = [getattr(output, dimension) for output in all_outputs]
            preferences_flipped = [
                getattr(output, dimension) for output in all_outputs_flipped
            ]
            results[dimension] = self._consensus_for_dimension(
                preferences, preferences_flipped
            )
        return results

    def _get_pairwise_preference(
        self, metric_input: CitationVerificationInput
    ) -> Dict[str, Dict[str, Any]]:
        """Get pairwise citation preference between a baseline and candidate answer."""
        input_flipped = CitationVerificationInput(
            question=metric_input.question,
            baseline_answer=metric_input.candidate_answer,
            candidate_answer=metric_input.baseline_answer,
        )

        messages = self._get_evaluation_messages(metric_input)
        messages_flipped = self._get_evaluation_messages(input_flipped)

        all_outputs: List[CitationVerificationPreferenceOutput] = []
        all_outputs_flipped: List[CitationVerificationPreferenceOutput] = []

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

        return self._build_dimension_results(all_outputs, all_outputs_flipped)

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> CitationVerificationScoreResult:
        """
        Score a single question-answer pair on citation quality.

        Args:
            question: The research question
            baseline_answer: The baseline answer (report_a)
            candidate_answer: The candidate answer (report_b)

        Returns:
            Object containing per-dimension citation scores and grades
        """
        metric_input = CitationVerificationInput(
            question=question,
            baseline_answer=baseline_answer,
            candidate_answer=candidate_answer,
        )

        output_dict = self._get_pairwise_preference(metric_input)

        dimension_results: Dict[str, CitationDimensionResult] = {}
        for dimension in CITATION_DIMENSIONS:
            dimension_output_dict = output_dict[dimension]
            dimension_results[dimension] = CitationDimensionResult(
                grade=dimension_output_dict["consensus_grade"],
                is_win=dimension_output_dict["consensus_grade"] == "win",
                is_tie=dimension_output_dict["consensus_grade"] == "tie",
                is_lose=dimension_output_dict["consensus_grade"] == "lose",
                score=dimension_output_dict["consensus_score"],
                preferred=dimension_output_dict["all_preferred"],
                raw_preferences=dimension_output_dict["raw_preferences"],
                flip_disagreement=dimension_output_dict["flip_disagreement"],
            )

        return CitationVerificationScoreResult(**dimension_results)

    def aggregate(
        self, scores_list: List[CitationVerificationScoreResult]
    ) -> Dict[str, Any]:
        """
        Aggregate citation-quality metrics from multiple scored rows.

        Args:
            scores_list: List of score result objects from multiple rows

        Returns:
            Dictionary containing aggregated metrics values, including a
            ``flip_disagreement_rate`` per dimension (the paper's directional
            bias signal: how often flipping report order flipped the verdict).
        """
        aggregated_metrics: Dict[str, Any] = {
            "support": len(scores_list),
        }

        for dimension in CITATION_DIMENSIONS:
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

            # Directional-bias diagnostic from the paper.
            flip_disagreement_rate = sum(
                r.flip_disagreement for r in dimension_results
            ) / len(dimension_results)

            aggregated_metrics[dimension] = {
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "lose_rate": lose_rate,
                "avg_score": avg_score,
                "net_winrate": net_winrate,
                "flip_disagreement_rate": flip_disagreement_rate,
            }

        aggregated_metrics["overall"] = {}
        metrics = [
            "win_rate",
            "tie_rate",
            "lose_rate",
            "avg_score",
            "net_winrate",
            "flip_disagreement_rate",
        ]
        for metric in metrics:
            overall_avg = sum(
                aggregated_metrics[dimension][metric]
                for dimension in CITATION_DIMENSIONS
            ) / len(CITATION_DIMENSIONS)
            aggregated_metrics["overall"][metric] = overall_avg

        return aggregated_metrics
