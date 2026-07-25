"""Absolute (pointwise) topical-focus + semantic-quality metric for deep-research reports.

Adapted from "A Rigorous Benchmark with Multidimensional Evaluation for Deep
Research Agents: From Answers to Reports" (Dr. Bench, arXiv:2510.02190). The
paper introduces two evaluation dimensions that complement this repo's existing
pairwise dimensions (instruction_following, comprehensiveness, completeness,
writing_quality):

  * **Topical focus** -- does the report stay on the research question's core
    topics, or does it drift into irrelevant material?
  * **Semantic quality** -- how completely does the report cover the reference
    answer's key information points (the "reference bundle")?

Adaptations from the paper (Mode 2 -- adapted port):

  * The paper constructs reference *bundles* by retrieving many reference
    reports. We substitute the repo's existing single ``baseline_answer``
    column as the reference bundle (parameter-free, target-native).
  * The paper's learned semantic-matching estimator is replaced by an
    LLM-as-judge that extracts the reference's key points and scores coverage,
    reusing this repo's existing OpenAI structured-output plumbing
    (``evals.utils.query_openai_model_structured_outputs``).
  * The topical-focus *score* is not a holistic judge rating: it is computed
    with the paper's SemanticDrift (SDR) formula over focus-anchor/deviation
    keyword (FAK/FDK) regex counts and judged keyword relevance -- see
    ``evals.metrics.semantic_drift``. The judge's topical-focus assessment is
    kept only for its qualitative ``off_topic_sections``.
  * The paper's separate benchmark / leaderboard framework is intentionally out
    of scope; this module plugs into the existing ``evals/`` CLI family instead.

This is *absolute* (single-report) scoring, not pairwise, so there is no
position to flip; instead we average ``num_trials`` independent judge calls to
reduce variance, mirroring the pairwise metric's multi-trial robustness pattern.
"""

import concurrent.futures
from typing import Any, Callable, Dict, List

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, computed_field, field_validator
from retry import retry

from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.metrics.semantic_drift import SemanticDriftMetric
from evals.utils import (
    query_openai_model_structured_outputs,
    replace_markdown_links_with_text,
)

# These intentionally do not overlap with the pairwise DIMENSIONS.
DIMENSIONS = ["topical_focus", "semantic_quality"]


class TopicalFocusAssessment(BaseModel):
    """Topical-focus dimension: on-topicness of the report vs. the question."""

    score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "How focused the report is on the question's core topics, 0-10. "
            "10 = entirely on-topic; 0 = mostly off-topic."
        ),
    )
    off_topic_sections: List[str] = Field(
        default_factory=list,
        description="Specific sections or claims that drift off the question's core topics.",
    )
    rationale: str = Field(description="One or two sentences justifying the score.")


class SemanticQualityAssessment(BaseModel):
    """Semantic-quality dimension: coverage of the reference bundle's key points."""

    score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "How completely the report covers the reference's key points, 0-10. "
            "10 = every key point covered with correct substance."
        ),
    )
    covered_key_points: List[str] = Field(
        default_factory=list,
        description="Reference key points the report substantively addresses.",
    )
    missed_key_points: List[str] = Field(
        default_factory=list,
        description="Reference key points the report omits or gets wrong.",
    )
    rationale: str = Field(description="One or two sentences justifying the score.")

    @computed_field
    def coverage_fraction(self) -> float:
        """Fraction of the reference key points substantively covered."""
        total = len(self.covered_key_points) + len(self.missed_key_points)
        if total == 0:
            return 0.0
        return len(self.covered_key_points) / total


class TopicalFocusMetricInput(BaseModel):
    question: str
    baseline_answer: str
    candidate_answer: str

    @field_validator("candidate_answer", "baseline_answer")
    @classmethod
    def validate_non_empty_answer(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Answer cannot be empty")
        # Reuse the existing pairwise metric's markdown-link stripping so the
        # judge never sees raw citation markdown (consistent with sibling metric).
        return replace_markdown_links_with_text(v, "")


class TopicalFocusJudgeOutput(BaseModel):
    """Raw structured output of a single LLM-as-judge call."""

    reference_key_points: List[str] = Field(
        default_factory=list,
        description=(
            "The key information points extracted from the reference answer "
            "(the reference bundle) that a complete report should cover."
        ),
    )
    topical_focus: TopicalFocusAssessment
    semantic_quality: SemanticQualityAssessment


class TopicalFocusScoreResult(BaseModel):
    """Complete absolute-scoring result across both dimensions."""

    reference_key_points: List[str] = Field(default_factory=list)
    topical_focus: TopicalFocusAssessment
    semantic_quality: SemanticQualityAssessment

    @computed_field
    def composite_score(self) -> float:
        """Mean of the two dimension scores (0-10)."""
        return (self.topical_focus.score + self.semantic_quality.score) / 2.0


TOPICAL_FOCUS_PROMPT = """You are an expert evaluator for deep-research reports.

You will receive a research <question>, a <reference_answer> (a high-quality reference report), and a <candidate_answer> (the report to evaluate).

Do two things:

1. Extract the key information points from the <reference_answer> into `reference_key_points`. These are the substantive claims, data points, or analyses a complete report on this question should cover.

2. Evaluate the <candidate_answer> on two dimensions, each scored 0-10:

   - **topical_focus**: Does the candidate stay on the question's core topics, or does it drift into irrelevant material? 10 = entirely on-topic and relevant; 0 = mostly off-topic. List any drifting material in `off_topic_sections`.

   - **semantic_quality**: How completely and correctly does the candidate cover the `reference_key_points`? Assign each key point to `covered_key_points` (substantively addressed, correct) or `missed_key_points` (omitted or wrong). 10 = every key point covered with correct substance; 0 = none covered.

Be strict and objective. Length is not quality: a concise report that covers the key points on-topic should score well; a long report that drifts or omits key points should not.
"""


class TopicalFocusMetric:
    """Absolute metric scoring topical focus + semantic quality against a reference."""

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
            num_trials: Number of independent judge calls per row; scores are
                averaged to reduce variance (absolute scoring has no position to
                flip, unlike the pairwise metric).
            num_workers: Number of parallel workers to use for processing trials
        """
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers
        # Topical-focus scores come from the paper's SemanticDrift formula
        # (FAK/FDK counts + judged relevance), not a holistic judge rating.
        self.semantic_drift = SemanticDriftMetric(eval_model=eval_model)

    def _get_evaluation_messages(
        self, metric_input: TopicalFocusMetricInput
    ) -> List[ChatCompletionMessageParam]:
        """Generate the messages for the evaluation model."""
        return [
            {"role": "system", "content": TOPICAL_FOCUS_PROMPT},
            {
                "role": "user",
                "content": f"""
<question>
{metric_input.question}
</question>

<reference_answer>
{metric_input.baseline_answer}
</reference_answer>

<candidate_answer>
{metric_input.candidate_answer}
</candidate_answer>
""",
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query_evaluation_model(
        self, messages: List[ChatCompletionMessageParam]
    ) -> TopicalFocusJudgeOutput:
        """Query the evaluation model with retry logic."""
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=TopicalFocusJudgeOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return TopicalFocusJudgeOutput.model_validate(output)

    def _merge_trial_outputs(
        self, outputs: List[TopicalFocusJudgeOutput]
    ) -> TopicalFocusScoreResult:
        """Average per-dimension scores across trials.

        Absolute scoring has no presentation position to flip, so we reduce
        variance by averaging scores; qualitative fields (rationale, key-point
        assignments) are taken from the trial closest to the averaged score.
        """
        if not outputs:
            raise ValueError("No judge outputs to aggregate")

        # Reference bundle = union of key points across trials (dedup, keep order).
        seen = set()
        reference_key_points: List[str] = []
        for out in outputs:
            for key_point in out.reference_key_points:
                if key_point not in seen:
                    seen.add(key_point)
                    reference_key_points.append(key_point)

        def _representative(score_of: Callable[[TopicalFocusJudgeOutput], float]):
            avg = sum(score_of(o) for o in outputs) / len(outputs)
            return min(outputs, key=lambda o: abs(score_of(o) - avg))

        tf_avg = sum(o.topical_focus.score for o in outputs) / len(outputs)
        sq_avg = sum(o.semantic_quality.score for o in outputs) / len(outputs)
        tf_rep = _representative(lambda o: o.topical_focus.score).topical_focus
        sq_rep = _representative(lambda o: o.semantic_quality.score).semantic_quality

        return TopicalFocusScoreResult(
            reference_key_points=reference_key_points,
            topical_focus=TopicalFocusAssessment(
                score=tf_avg,
                off_topic_sections=tf_rep.off_topic_sections,
                rationale=tf_rep.rationale,
            ),
            semantic_quality=SemanticQualityAssessment(
                score=sq_avg,
                covered_key_points=sq_rep.covered_key_points,
                missed_key_points=sq_rep.missed_key_points,
                rationale=sq_rep.rationale,
            ),
        )

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> TopicalFocusScoreResult:
        """
        Score a single question-answer pair absolutely against the reference.

        Args:
            question: The research question
            baseline_answer: The reference answer (reference bundle source)
            candidate_answer: The candidate answer to evaluate

        Returns:
            Object containing per-dimension scores, the extracted reference
            bundle, and a composite score.
        """
        metric_input = TopicalFocusMetricInput(
            question=question,
            baseline_answer=baseline_answer,
            candidate_answer=candidate_answer,
        )
        messages = self._get_evaluation_messages(metric_input)

        outputs: List[TopicalFocusJudgeOutput] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self._query_evaluation_model, messages)
                for _ in range(self.num_trials)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    outputs.append(future.result())
                except Exception as exc:
                    print(f"Topical-focus trial generated an exception: {exc}")

        if not outputs:
            raise ValueError("Failed to get any outputs from evaluation model")

        result = self._merge_trial_outputs(outputs)

        # Replace the judge's holistic topical-focus rating with the paper's
        # formulaic SemanticDrift score; keep the judge's off-topic findings.
        drift = self.semantic_drift.score(
            question=metric_input.question,
            reference_answer=metric_input.baseline_answer,
            candidate_answer=metric_input.candidate_answer,
        )
        result.topical_focus = TopicalFocusAssessment(
            score=drift.topical_focus_score,
            off_topic_sections=result.topical_focus.off_topic_sections,
            rationale=drift.rationale,
        )
        return result

    def aggregate(
        self, scores_list: List[TopicalFocusScoreResult]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple scored rows.

        Args:
            scores_list: List of score result objects from multiple rows

        Returns:
            Dictionary containing aggregated metrics values
        """
        aggregated_metrics: Dict[str, Any] = {"support": len(scores_list)}
        if not scores_list:
            return aggregated_metrics

        for dimension in DIMENSIONS:
            dim_scores = [
                getattr(score_result, dimension).score
                for score_result in scores_list
            ]
            aggregated_metrics[dimension] = {
                "avg_score": sum(dim_scores) / len(dim_scores),
            }

        composite_scores = [s.composite_score for s in scores_list]
        coverage_fractions = [s.semantic_quality.coverage_fraction for s in scores_list]
        aggregated_metrics["overall"] = {
            "avg_composite_score": sum(composite_scores) / len(composite_scores),
            "avg_coverage_fraction": sum(coverage_fractions)
            / len(coverage_fractions),
        }
        return aggregated_metrics
