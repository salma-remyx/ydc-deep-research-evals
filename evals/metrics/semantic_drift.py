"""Semantic-drift scoring via focus anchor/deviation keywords (FAK/FDK).

Adapted from the SemanticDrift (SDR) component of "A Rigorous Benchmark with
Multidimensional Evaluation for Deep Research Agents: From Answers to Reports"
(Dr. Bench, arXiv:2510.02190). The paper measures topical focus *formulaically*
rather than with a single holistic judge score:

  * Each query carries 5 focus-anchor keywords (FAKs) -- terms a focused report
    must engage -- and 5 focus-deviation keywords (FDKs) -- terms signalling
    drift into adjacent but off-query topics.
  * Keyword occurrences in the candidate report are counted by regex.
  * A judge rates each keyword's relevance to the report on a 0-5 scale.
  * SemanticDrift combines them (paper Section 3, SDR definition)::

        SDR = 0.7 * (1 - mean(min(fak_count / 2, 1) * fak_relevance / 5))
            + 0.3 * mean(min(fdk_count, 1) * fdk_relevance / 5)

    so anchor presence pulls drift toward 0 and deviation presence pushes it
    toward 1. The 0-10 topical-focus score is ``(1 - SDR) * 10``.

Adaptations from the paper (Mode 2 -- adapted port):

  * DrBench ships per-query FAK/FDK lists in its reference bundles; this repo's
    CSV convention (``question`` / ``baseline_answer`` / ``candidate_answer``)
    has no such column, so a judge extracts the anchor/deviation keywords from
    the question and reference answer (target-native substitute for the
    dataset's curated keyword lists).
  * The paper's gpt-4o judge with regex-parsed ``[score] reason`` output is
    replaced by this repo's existing structured-output plumbing
    (``evals.utils.query_openai_model_structured_outputs``), with the model
    defaulting to the repo-wide ``DEFAULT_EVAL_MODEL``.
  * The paper's QUA rubric scoring, TrustworthyBoost URL matching, and the
    integrated leaderboard are intentionally out of scope; this module delivers
    only the topical-focus (SDR) mechanism.
"""

import re
from typing import Dict, List

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry

from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.utils import query_openai_model_structured_outputs

# The paper fixes 5 anchor + 5 deviation keywords per query.
NUM_FOCUS_KEYWORDS = 5


class FocusKeywords(BaseModel):
    """FAK/FDK keyword lists for one research question."""

    anchor_keywords: List[str] = Field(
        description=(
            "Focus-anchor keywords (FAKs): short terms or phrases central to "
            "the research question that a focused report must substantively "
            "engage with."
        ),
    )
    deviation_keywords: List[str] = Field(
        description=(
            "Focus-deviation keywords (FDKs): short terms or phrases marking "
            "adjacent but off-question topics whose presence signals drift."
        ),
    )


class KeywordRelevance(BaseModel):
    """Judge-assessed relevance of one keyword to the candidate report."""

    keyword: str = Field(description="The keyword being assessed, echoed verbatim.")
    relevance: float = Field(
        ge=0.0,
        le=5.0,
        description=(
            "How relevant the keyword is to the candidate report's actual "
            "content, 0-5. 5 = the report engages it centrally and correctly; "
            "0 = absent or only mentioned in passing."
        ),
    )
    reason: str = Field(description="One sentence justifying the relevance rating.")


class KeywordRelevanceJudgment(BaseModel):
    """Relevance assessments for every keyword in a single judge call."""

    assessments: List[KeywordRelevance]


class SemanticDriftResult(BaseModel):
    """Complete SDR computation for one candidate report."""

    anchor_keywords: List[str]
    deviation_keywords: List[str]
    anchor_counts: List[int]
    deviation_counts: List[int]
    anchor_relevances: List[float]
    deviation_relevances: List[float]
    semantic_drift: float = Field(
        description="SDR in [0, 1]; higher means more topical drift."
    )
    topical_focus_score: float = Field(
        description="Topical focus on a 0-10 scale: (1 - SDR) * 10."
    )
    rationale: str


def count_keyword_occurrences(text: str, keyword: str) -> int:
    """Count case-insensitive occurrences of ``keyword`` in ``text`` by regex.

    Mirrors the paper's occurrence counting: plain substring matching, no
    IDF re-weighting or other heuristic scaling.
    """
    if not keyword.strip():
        return 0
    return len(re.findall(re.escape(keyword.strip()), text, flags=re.IGNORECASE))


def compute_semantic_drift(
    anchor_counts: List[int],
    anchor_relevances: List[float],
    deviation_counts: List[int],
    deviation_relevances: List[float],
) -> float:
    """The paper's SemanticDrift (SDR) formula, in [0, 1].

    SDR = 0.7 * (1 - mean(min(fak_count / 2, 1) * fak_relevance / 5))
        + 0.3 * mean(min(fdk_count, 1) * fdk_relevance / 5)

    An anchor keyword contributes fully once it appears at least twice with
    maximal relevance; any occurring, relevant deviation keyword adds drift.
    """
    if not anchor_counts and not deviation_counts:
        raise ValueError("Cannot compute semantic drift without keywords")

    anchor_term = 1.0
    if anchor_counts:
        anchor_term = sum(
            min(count / 2, 1.0) * (relevance / 5.0)
            for count, relevance in zip(anchor_counts, anchor_relevances)
        ) / len(anchor_counts)

    deviation_term = 0.0
    if deviation_counts:
        deviation_term = sum(
            min(count, 1.0) * (relevance / 5.0)
            for count, relevance in zip(deviation_counts, deviation_relevances)
        ) / len(deviation_counts)

    return 0.7 * (1.0 - anchor_term) + 0.3 * deviation_term


FOCUS_KEYWORDS_PROMPT = """You are an expert evaluator for deep-research reports.

Given a research <question> and a high-quality <reference_answer>, define the topical-focus keywords used to detect drift in candidate reports:

1. `anchor_keywords`: exactly {n} focus-anchor keywords (FAKs) -- short terms or phrases (1-4 words) central to the question. A focused report MUST engage these substantively. Prefer terms that literally appear in the question or reference so they can be counted by string matching.

2. `deviation_keywords`: exactly {n} focus-deviation keywords (FDKs) -- short terms or phrases (1-4 words) marking adjacent but OFF-question topics. A report drifting into these topics is losing topical focus. Pick plausible drift directions for this question, not random unrelated words.

Return keywords as they would literally appear in a report (singular/plural as most natural)."""

KEYWORD_RELEVANCE_PROMPT = """You are an expert evaluator for deep-research reports.

Given a research <question>, a <candidate_answer>, and a list of <keywords>, rate how relevant each keyword is to what the candidate report actually discusses, on a 0-5 scale:

- 5: the report engages the keyword's topic centrally, with correct substance.
- 3: the report discusses the topic meaningfully but not centrally.
- 1: the keyword appears only in passing.
- 0: the keyword is absent or its mentions are vacuous.

Echo each keyword verbatim in `keyword`. Assess every keyword provided, no more, no fewer."""


class SemanticDriftMetric:
    """Scores a candidate report's topical drift with the paper's SDR formula."""

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_keywords: int = NUM_FOCUS_KEYWORDS,
    ):
        """
        Initialize the metric.

        Args:
            eval_model: The model to use for keyword extraction and relevance
                judging (defaults to the repo-wide evaluation model)
            num_keywords: Number of anchor and deviation keywords per query
                (the paper fixes this at 5)
        """
        self.eval_model = eval_model
        self.num_keywords = num_keywords

    @retry(tries=3, delay=1, backoff=2)
    def _extract_focus_keywords(
        self, question: str, reference_answer: str
    ) -> FocusKeywords:
        """Judge call: derive FAK/FDK lists (substitute for DrBench's bundles)."""
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": FOCUS_KEYWORDS_PROMPT.format(n=self.num_keywords),
            },
            {
                "role": "user",
                "content": f"""
<question>
{question}
</question>

<reference_answer>
{reference_answer}
</reference_answer>
""",
            },
        ]
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=FocusKeywords,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=2000,
        )
        if output is None:
            raise ValueError("Failed to extract focus keywords")
        return FocusKeywords.model_validate(output)

    @retry(tries=3, delay=1, backoff=2)
    def _judge_keyword_relevance(
        self, question: str, candidate_answer: str, keywords: List[str]
    ) -> KeywordRelevanceJudgment:
        """Judge call: 0-5 relevance of every keyword to the candidate report."""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": KEYWORD_RELEVANCE_PROMPT},
            {
                "role": "user",
                "content": f"""
<question>
{question}
</question>

<keywords>
{chr(10).join(f"- {kw}" for kw in keywords)}
</keywords>

<candidate_answer>
{candidate_answer}
</candidate_answer>
""",
            },
        ]
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=KeywordRelevanceJudgment,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=5000,
        )
        if output is None:
            raise ValueError("Failed to judge keyword relevance")
        return KeywordRelevanceJudgment.model_validate(output)

    @staticmethod
    def _relevances_for(
        keywords: List[str], judgment: KeywordRelevanceJudgment
    ) -> List[float]:
        """Align judged relevances to the keyword order; missing -> 0.0."""
        by_keyword: Dict[str, float] = {
            a.keyword.strip().lower(): a.relevance for a in judgment.assessments
        }
        return [
            by_keyword.get(keyword.strip().lower(), 0.0) for keyword in keywords
        ]

    def score(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> SemanticDriftResult:
        """
        Compute SemanticDrift for a candidate report.

        Args:
            question: The research question
            reference_answer: The reference answer (FAK/FDK derivation source)
            candidate_answer: The candidate report to evaluate

        Returns:
            Object with the keyword lists, occurrence counts, judged
            relevances, the SDR value, and the 0-10 topical-focus score.
        """
        focus = self._extract_focus_keywords(question, reference_answer)
        anchor_keywords = focus.anchor_keywords[: self.num_keywords]
        deviation_keywords = focus.deviation_keywords[: self.num_keywords]

        anchor_counts = [
            count_keyword_occurrences(candidate_answer, kw) for kw in anchor_keywords
        ]
        deviation_counts = [
            count_keyword_occurrences(candidate_answer, kw)
            for kw in deviation_keywords
        ]

        judgment = self._judge_keyword_relevance(
            question, candidate_answer, anchor_keywords + deviation_keywords
        )
        anchor_relevances = self._relevances_for(anchor_keywords, judgment)
        deviation_relevances = self._relevances_for(deviation_keywords, judgment)

        sdr = compute_semantic_drift(
            anchor_counts, anchor_relevances, deviation_counts, deviation_relevances
        )
        topical_focus_score = (1.0 - sdr) * 10.0

        rationale = (
            f"SemanticDrift={sdr:.3f} from {sum(anchor_counts)} anchor "
            f"occurrences ({', '.join(anchor_keywords)}) and "
            f"{sum(deviation_counts)} deviation occurrences "
            f"({', '.join(deviation_keywords)})."
        )

        return SemanticDriftResult(
            anchor_keywords=anchor_keywords,
            deviation_keywords=deviation_keywords,
            anchor_counts=anchor_counts,
            deviation_counts=deviation_counts,
            anchor_relevances=anchor_relevances,
            deviation_relevances=deviation_relevances,
            semantic_drift=sdr,
            topical_focus_score=topical_focus_score,
            rationale=rationale,
        )
