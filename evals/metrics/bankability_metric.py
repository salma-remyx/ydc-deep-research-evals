"""Pointwise bankability scoring for deep-research reports.

Adapted from "Capital Markets LLM Reliability Score (CM-LRS): From Plausible to
Bankable" (arXiv:2607.21340v1). The paper's core mechanism is kept at full
fidelity: a seven-dimension rubric scored 0-5 by an LLM judge, with an aggregate
that is tunable to the workflow.

Adaptations (Mode 2 port) of the paper's auxiliary components:

  * The paper's capital-markets workflow benchmark (DCM / ECM transaction-terms
    extraction, precedent retrieval, issuer profiling, M&A comps) and its SEC
    EDGAR / UK-takeover datasets are not ported. The rubric transfers directly to
    the DeepConsult business / consulting reports this repo already evaluates, so
    scoring is wired into the existing DeepResearchEvaluator pipeline instead of a
    bespoke harness.
  * The paper's four-judge inter-model agreement analysis (mean r) is out of
    scope here -- that is a downstream reporting step, not part of the scoring
    mechanism.

The aggregate is the paper's "tunable to the workflow" weighted mean over the
seven dimensions, defaulting to equal weights so it is parameter-free out of the
box and overridable per dimension.
"""

import concurrent.futures
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, field_validator
from retry import retry

from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.utils import query_openai_model_structured_outputs

# The seven CM-LRS bankability dimensions, in the paper's order.
BANKABILITY_DIMENSIONS = [
    "factual_accuracy",
    "evidence_traceability",
    "numerical_consistency",
    "workflow_completeness",
    "source_discipline",
    "decision_usefulness",
    "reviewability_auditability",
]

# Signal anchors for each dimension, lifted from the paper's rubric framing.
# These anchor the judge's 0-5 score on what a reviewer in a regulated setting
# actually checks, rather than on surface fluency.
DIMENSION_ANCHORS = {
    "factual_accuracy": (
        "Material claims and figures are correct and verifiable; the report does "
        "not assert facts or data points that are wrong or invented."
    ),
    "evidence_traceability": (
        "Every material claim is tied to a specific, followable source or citation "
        "so a reader can trace it back to the underlying document."
    ),
    "numerical_consistency": (
        "Figures are internally consistent: totals match their components, units are "
        "correct, percentages reconcile, and no numbers contradict each other."
    ),
    "workflow_completeness": (
        "The output addresses the requested task end to end, with no missing required "
        "sections, fields, or sub-questions."
    ),
    "source_discipline": (
        "Sources are relevant and authoritative, neither over-cited nor under-cited, "
        "with no fabricated or mismatched citations."
    ),
    "decision_usefulness": (
        "The output supports a concrete decision or action -- a clear takeaway, the "
        "relevant trade-offs, and where appropriate a recommendation -- rather than "
        "just restating information."
    ),
    "reviewability_auditability": (
        "A reviewer or auditor can reconstruct how the output was derived: the "
        "reasoning chain and evidence trail are transparent and checkable."
    ),
}

BANKABILITY_PROMPT = f"""\
You are a strict reviewer evaluating a research report for bankability: whether it
is defensible in front of a counter-party or a regulator, with the documents in hand.
Plausibility is cheap; bankability is the bar. Score the report on each of the seven
dimensions below from 0 (absent or critically flawed) to 5 (excellent, fully
defensible), where 2 means notable gaps, 3 means adequate, and 4 means strong with
only minor gaps.

For each dimension give an integer score in [0, 5] and a one-sentence explanation
that cites the specific signal in the report that drove the score.

The dimensions and what each measures:
1. factual_accuracy: {DIMENSION_ANCHORS["factual_accuracy"]}
2. evidence_traceability: {DIMENSION_ANCHORS["evidence_traceability"]}
3. numerical_consistency: {DIMENSION_ANCHORS["numerical_consistency"]}
4. workflow_completeness: {DIMENSION_ANCHORS["workflow_completeness"]}
5. source_discipline: {DIMENSION_ANCHORS["source_discipline"]}
6. decision_usefulness: {DIMENSION_ANCHORS["decision_usefulness"]}
7. reviewability_auditability: {DIMENSION_ANCHORS["reviewability_auditability"]}

Be demanding and objective. Length and fluency are not bankability; a polished but
unsupported or untraceable report scores low.
"""


class BankabilityDimensionScore(BaseModel):
    """A judge's 0-5 score for a single bankability dimension."""

    score: int = Field(description="Integer score from 0 to 5.")
    explanation: str = Field(description="One-sentence justification citing a signal.")

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, v: int) -> int:
        # Judges occasionally return 6 or -1; clamp to the valid 0-5 band so a
        # single noisy grade never fails the whole evaluation.
        return max(0, min(5, int(v)))


class BankabilityOutput(BaseModel):
    """Structured judge output: one 0-5 score per bankability dimension."""

    factual_accuracy: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["factual_accuracy"]
    )
    evidence_traceability: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["evidence_traceability"]
    )
    numerical_consistency: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["numerical_consistency"]
    )
    workflow_completeness: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["workflow_completeness"]
    )
    source_discipline: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["source_discipline"]
    )
    decision_usefulness: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["decision_usefulness"]
    )
    reviewability_auditability: BankabilityDimensionScore = Field(
        description=DIMENSION_ANCHORS["reviewability_auditability"]
    )


class BankabilityDimensionResult(BaseModel):
    """Aggregated score for one dimension across the trial judges."""

    score: float
    explanations: List[str]


class BankabilityScoreResult(BaseModel):
    """Per-report bankability result: mean score per dimension plus the aggregate."""

    factual_accuracy: BankabilityDimensionResult
    evidence_traceability: BankabilityDimensionResult
    numerical_consistency: BankabilityDimensionResult
    workflow_completeness: BankabilityDimensionResult
    source_discipline: BankabilityDimensionResult
    decision_usefulness: BankabilityDimensionResult
    reviewability_auditability: BankabilityDimensionResult
    aggregate: float


def _resolve_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Resolve caller weights into a normalized distribution over all dimensions.

    A weights map is interpreted as relative emphasis: omitted dimensions get zero
    weight and the supplied values are renormalized to sum to 1. ``None`` (and an
    empty or all-zero map, which would otherwise divide by zero) yields equal
    weighting -- the paper's neutral default, since "the aggregate is tunable to the
    workflow".
    """
    equal = 1.0 / len(BANKABILITY_DIMENSIONS)
    if weights is None:
        return {dim: equal for dim in BANKABILITY_DIMENSIONS}

    filled = {dim: float(weights.get(dim, 0.0)) for dim in BANKABILITY_DIMENSIONS}
    total = sum(filled.values())
    if total <= 0:
        # Empty / all-zero weights map: fall back to the neutral equal weighting
        # instead of dividing by zero.
        return {dim: equal for dim in BANKABILITY_DIMENSIONS}
    return {dim: w / total for dim, w in filled.items()}


def weighted_aggregate(
    dimension_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted mean of per-dimension scores -- the CM-LRS tunable aggregate.

    Pure and network-free so it can be unit-tested and reused for cross-row
    aggregation.
    """
    resolved = _resolve_weights(weights)
    return sum(dimension_scores[dim] * resolved[dim] for dim in BANKABILITY_DIMENSIONS)


class BankabilityMetric:
    """Pointwise 0-5 bankability scorer for a single research report.

    Unlike the pairwise metric this scores one report absolutely (no baseline, no
    position bias), matching the CM-LRS rubric. Multi-trial scoring averages out
    judge noise.
    """

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers
        self.weights = _resolve_weights(weights)

    def _get_messages(
        self, question: str, answer: str
    ) -> List[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": BANKABILITY_PROMPT},
            {
                "role": "user",
                "content": f"""
<task>
{question}
</task>

<report>
{answer}
</report>
""",
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query(self, messages: List[ChatCompletionMessageParam]) -> BankabilityOutput:
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=BankabilityOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return BankabilityOutput.model_validate(output)

    def _run_trials(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[BankabilityOutput]:
        outputs: List[BankabilityOutput] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(self._query, messages) for _ in range(self.num_trials)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    output = future.result()
                    if output is not None:
                        outputs.append(output)
                except Exception as exc:
                    print(f"Bankability trial generated an exception: {exc}")
        return outputs

    def score(self, question: str, answer: str) -> BankabilityScoreResult:
        """Score a single report on the seven-dimension bankability rubric."""
        messages = self._get_messages(question, answer)
        outputs = self._run_trials(messages)
        if not outputs:
            raise ValueError(
                "Failed to get any bankability outputs from evaluation model"
            )

        dimension_results: Dict[str, BankabilityDimensionResult] = {}
        dimension_scores: Dict[str, float] = {}
        for dimension in BANKABILITY_DIMENSIONS:
            per_trial = [getattr(output, dimension) for output in outputs]
            mean_score = sum(p.score for p in per_trial) / len(per_trial)
            dimension_scores[dimension] = mean_score
            dimension_results[dimension] = BankabilityDimensionResult(
                score=mean_score,
                explanations=[p.explanation for p in per_trial],
            )

        return BankabilityScoreResult(
            **dimension_results,
            aggregate=weighted_aggregate(dimension_scores, self.weights),
        )

    def aggregate(self, scores_list: List[BankabilityScoreResult]) -> Dict[str, Any]:
        """Aggregate bankability scores across multiple reports.

        Mirrors the shape of DeepResearchPairwiseMetric.aggregate so it slots into
        the existing reporting tooling.
        """
        aggregated: Dict[str, Any] = {"support": len(scores_list)}
        if not scores_list:
            return aggregated

        per_dimension_mean: Dict[str, float] = {}
        for dimension in BANKABILITY_DIMENSIONS:
            scores = [getattr(r, dimension).score for r in scores_list]
            per_dimension_mean[dimension] = sum(scores) / len(scores)
            aggregated[dimension] = {"avg_score": per_dimension_mean[dimension]}

        aggregated["overall"] = {
            "avg_score": weighted_aggregate(per_dimension_mean, self.weights)
        }
        return aggregated


if __name__ == "__main__":
    # Smoke test with a short example report.
    metric = BankabilityMetric(num_trials=1, num_workers=1)
    question = (
        "Summarize Acme Corp's most recent debt issuance and its use of proceeds."
    )
    answer = (
        "Acme Corp issued $500M of 5-year senior unsecured notes at a 5.25% coupon "
        "(see prospectus, p. 12). Proceeds refinance $300M of maturing 2024 notes "
        "and fund $200M of share repurchases. The issue was rated BBB-/Baa3."
    )
    result = metric.score(question=question, answer=answer)
    print(result.model_dump_json(indent=2))
    print("\nAggregated over one report:")
    import json

    print(json.dumps(metric.aggregate([result]), indent=2))
