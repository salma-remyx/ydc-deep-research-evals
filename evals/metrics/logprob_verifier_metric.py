"""Logprob-verifier metric: continuous verification scoring from token logits.

Adapted (Mode 2) from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). Core mechanism retained at full fidelity: instead of reading a
single discrete score a judge emits, take the *expectation over the distribution of
scoring-token logits* at the scoring position to produce a continuous, calibrated
score. The chat-completions API exposes ``top_logprobs`` for generated tokens, so this
needs no hidden states and no extra training.

The paper's three verification-scaling axes map onto this metric:
  1. Score granularity -- ``granularity`` sets the candidate score-token set (0..G);
     finer = better separation. Scores are also reported normalized to [0, 1].
  2. Repeated evaluation -- ``num_trials`` in original + flipped order (the repo's
     position-bias mitigation); cross-trial std is reported as an uncertainty signal.
  3. Criteria decomposition -- the repo's four dimensions are each scored continuously.

Mode 2 substitutions (AUXILIARY paper components intentionally not ported): the
best-of-N ranking algorithm, the Claude-Code progress monitor, and the RL dense-reward
use (SAC/GRPO) are out of scope; the agentic benchmarks are replaced by the repo's
DeepConsult CSV->JSONL contract and four dimensions; and the paper's correctness-
verification target is reframed to relative report-quality judgment.
"""

import concurrent.futures
import math
from typing import Any, Dict, List, Tuple

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchPairwisePreferenceInput,
)
from evals.utils import query_openai_model_logprobs

# Candidate score tokens are 0..granularity (inclusive). The paper shows finer
# granularity improves positive/negative separation; 10 mirrors the repo's 0-10
# consensus_score scale while staying a single decimal digit (one token).
DEFAULT_VERIFIER_GRANULARITY = 10

# A score within +/- VERIFIER_TIE_MARGIN of the midpoint counts as a tie.
VERIFIER_TIE_MARGIN = 0.5


LOGPROB_VERIFIER_PROMPT = """
You are an expert evaluator comparing two reports answering a research question: report_a and report_b.

Focus ONLY on this dimension:
{dimension_description}

Rate how much better report_b is than report_a on this dimension, using a single integer from 0 to {granularity}:
- 0 means report_a is far better than report_b on this dimension.
- {midpoint} means report_b and report_a are roughly tied on this dimension.
- {granularity} means report_b is far better than report_a on this dimension.

Be fair and objective; do not be biased toward either report. The length of a report is not necessarily an indicator of quality.

Respond with ONLY a single integer between 0 and {granularity} and nothing else.
"""

DIMENSION_DESCRIPTIONS = {
    "instruction_following": (
        "Instruction following: response's fidelity to user-specified instructions "
        "and constraints."
    ),
    "comprehensiveness": (
        "Comprehensiveness: breadth and range of information covered, addressing the "
        "scope of the user's request."
    ),
    "completeness": (
        "Completeness: depth and thoroughness of information for the topics addressed."
    ),
    "writing_quality": (
        "Writing quality: clarity, conciseness, logical organization, and overall "
        "readability."
    ),
}


def candidate_score_tokens(granularity: int) -> List[str]:
    """The set of decimal-integer tokens the judge may emit (0..granularity)."""
    return [str(i) for i in range(granularity + 1)]


def renormalize_over_candidates(
    token_logprobs: List[Dict[str, Any]], granularity: int
) -> Dict[str, float]:
    """Restrict a token-logprob distribution to the candidate score tokens.

    The chat API only returns the top-k token logprobs, so candidate tokens outside
    that top-k are treated as negligible mass. Surviving candidate probabilities are
    exponentiated and renormalized to sum to 1 over the observed candidate subset.
    Returns ``{token_str: probability}``.
    """
    logprob_of = {entry["token"]: entry["logprob"] for entry in token_logprobs}
    probs = {
        tok: math.exp(logprob_of[tok])
        for tok in candidate_score_tokens(granularity)
        if tok in logprob_of
    }
    total = sum(probs.values())
    if total <= 0.0:
        return {}
    return {tok: p / total for tok, p in probs.items()}


def expected_score_from_logprobs(
    token_logprobs: List[Dict[str, Any]], granularity: int
) -> Tuple[float, Dict[str, float]]:
    """Expected score over the candidate-token distribution (the paper's core quantity).

    Returns ``(expected_score, probs)`` where ``expected_score = sum(value(token) * p)``
    on the 0..granularity scale. If no candidate tokens were observed, falls back to the
    midpoint (a neutral tie).
    """
    probs = renormalize_over_candidates(token_logprobs, granularity)
    if not probs:
        return float(granularity) / 2.0, probs
    expected = sum(int(tok) * p for tok, p in probs.items())
    return expected, probs


def preference_from_expected(
    expected: float, granularity: int, margin: float = VERIFIER_TIE_MARGIN
) -> str:
    """Map a continuous expected score to a win/tie/lose grade (midpoint-anchored)."""
    midpoint = granularity / 2.0
    if expected > midpoint + margin:
        return "win"
    if expected < midpoint - margin:
        return "lose"
    return "tie"


class VerifierDimensionResult(BaseModel):
    """Per-dimension continuous verifier result.

    Field-compatible with the repo's ``DimensionResult`` (``grade``/``is_win``/
    ``is_tie``/``is_lose``/``score``) so it drops into the existing evaluator's
    per-row extraction, plus the verifier-specific continuous signals.
    """

    grade: str
    is_win: bool
    is_tie: bool
    is_lose: bool
    score: float
    expected_score: float
    normalized_score: float
    std: float
    trial_scores: List[float]
    raw_token_logprobs: dict


class LogprobVerifierScoreResult(BaseModel):
    """Continuous verifier results across the repo's four evaluation dimensions."""

    instruction_following: VerifierDimensionResult = Field(
        description=DIMENSION_DESCRIPTIONS["instruction_following"]
    )
    comprehensiveness: VerifierDimensionResult = Field(
        description=DIMENSION_DESCRIPTIONS["comprehensiveness"]
    )
    completeness: VerifierDimensionResult = Field(
        description=DIMENSION_DESCRIPTIONS["completeness"]
    )
    writing_quality: VerifierDimensionResult = Field(
        description=DIMENSION_DESCRIPTIONS["writing_quality"]
    )


class LogprobVerifierMetric:
    """Continuous verification metric: expected score over scoring-token logprobs.

    Mirrors the surface of ``DeepResearchPairwiseMetric`` (``score`` / ``aggregate``)
    and reuses its input validation (``DeepResearchPairwisePreferenceInput``), so it is
    a drop-in alternative that swaps discrete structured-output scoring for the paper's
    probabilistic continuous scoring.
    """

    def __init__(
        self,
        eval_model: str = "o3-mini-2025-01-31",
        granularity: int = DEFAULT_VERIFIER_GRANULARITY,
        num_trials: int = 3,
        num_workers: int = 3,
    ):
        self.eval_model = eval_model
        self.granularity = granularity
        self.num_trials = num_trials
        self.num_workers = num_workers

    def _get_evaluation_messages(
        self, metric_input: DeepResearchPairwisePreferenceInput, dimension: str
    ) -> List[ChatCompletionMessageParam]:
        midpoint = self.granularity / 2.0
        prompt = LOGPROB_VERIFIER_PROMPT.format(
            dimension_description=DIMENSION_DESCRIPTIONS[dimension],
            granularity=self.granularity,
            midpoint=midpoint,
        )
        return [
            {"role": "system", "content": prompt},
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
    def _query_verifier(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[Dict[str, Any]]:
        """Query the judge and return the first generated token's top-logprob entries."""
        token_logprobs = query_openai_model_logprobs(
            messages=messages,
            model=self.eval_model,
            max_completion_tokens=5,
            top_logprobs=20,
            temperature=0,
        )
        if not token_logprobs:
            raise ValueError("Judge returned no logprobs for the scoring token")
        return token_logprobs

    def _score_dimension(
        self, metric_input: DeepResearchPairwisePreferenceInput, dimension: str
    ) -> VerifierDimensionResult:
        """Score one dimension via expected value over original + flipped trials."""
        input_flipped = DeepResearchPairwisePreferenceInput(
            question=metric_input.question,
            baseline_answer=metric_input.candidate_answer,
            candidate_answer=metric_input.baseline_answer,
        )
        original_messages = self._get_evaluation_messages(metric_input, dimension)
        flipped_messages = self._get_evaluation_messages(input_flipped, dimension)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            original_futures = [
                executor.submit(self._query_verifier, original_messages)
                for _ in range(self.num_trials)
            ]
            flipped_futures = [
                executor.submit(self._query_verifier, flipped_messages)
                for _ in range(self.num_trials)
            ]
            original_raw = [f.result() for f in original_futures]
            flipped_raw = [f.result() for f in flipped_futures]

        trial_scores: List[float] = []
        raw = {"original": [], "flipped": []}
        for token_logprobs in original_raw:
            expected, probs = expected_score_from_logprobs(token_logprobs, self.granularity)
            trial_scores.append(expected)
            raw["original"].append({"expected": expected, "probs": probs})
        for token_logprobs in flipped_raw:
            # Flipped order asks "how much better is original_a"; mirror about the
            # midpoint to express it as "how much better is original_b (report_b)".
            expected, probs = expected_score_from_logprobs(token_logprobs, self.granularity)
            corrected = self.granularity - expected
            trial_scores.append(corrected)
            raw["flipped"].append({"expected": expected, "corrected": corrected, "probs": probs})

        mean_expected = sum(trial_scores) / len(trial_scores)
        std = (
            (sum((s - mean_expected) ** 2 for s in trial_scores) / len(trial_scores)) ** 0.5
            if len(trial_scores) > 1
            else 0.0
        )
        grade = preference_from_expected(mean_expected, self.granularity)
        return VerifierDimensionResult(
            grade=grade,
            is_win=grade == "win",
            is_tie=grade == "tie",
            is_lose=grade == "lose",
            score=mean_expected,
            expected_score=mean_expected,
            normalized_score=mean_expected / self.granularity,
            std=std,
            trial_scores=trial_scores,
            raw_token_logprobs=raw,
        )

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> LogprobVerifierScoreResult:
        """Score a single question-answer pair continuously across all dimensions."""
        metric_input = DeepResearchPairwisePreferenceInput(
            question=question,
            baseline_answer=baseline_answer,
            candidate_answer=candidate_answer,
        )
        return LogprobVerifierScoreResult(
            **{dim: self._score_dimension(metric_input, dim) for dim in DIMENSIONS}
        )

    def aggregate(
        self, scores_list: List[LogprobVerifierScoreResult]
    ) -> Dict[str, Any]:
        """Aggregate continuous verifier scores across rows.

        Reports per-dimension and overall mean expected score, mean normalized score,
        win/tie/lose rates, net winrate, and the mean cross-trial uncertainty (std) as a
        calibration diagnostic.
        """
        aggregated: Dict[str, Any] = {"support": len(scores_list)}
        if not scores_list:
            return aggregated

        for dimension in DIMENSIONS:
            dim_results = [getattr(s, dimension) for s in scores_list]
            n = len(dim_results)
            win_rate = sum(r.is_win for r in dim_results) / n
            tie_rate = sum(r.is_tie for r in dim_results) / n
            lose_rate = sum(r.is_lose for r in dim_results) / n
            num_wins = sum(r.is_win for r in dim_results)
            num_losses = sum(r.is_lose for r in dim_results)
            net_winrate = (
                num_wins / (num_wins + num_losses) if (num_wins + num_losses) > 0 else 0.0
            )
            aggregated[dimension] = {
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "lose_rate": lose_rate,
                "net_winrate": net_winrate,
                "avg_expected_score": sum(r.expected_score for r in dim_results) / n,
                "avg_normalized_score": sum(r.normalized_score for r in dim_results) / n,
                "avg_uncertainty_std": sum(r.std for r in dim_results) / n,
            }

        aggregated["overall"] = {}
        for metric in [
            "win_rate",
            "tie_rate",
            "lose_rate",
            "net_winrate",
            "avg_expected_score",
            "avg_normalized_score",
            "avg_uncertainty_std",
        ]:
            aggregated["overall"][metric] = sum(
                aggregated[d][metric] for d in DIMENSIONS
            ) / len(DIMENSIONS)
        return aggregated
