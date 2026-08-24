"""Rubric-anchored pairwise metric for deep research evaluation.

Adapted from "Grading Needs a Rubric, Not Intelligence"
(arXiv:2608.17938). The paper shows grading reliability is decoupled from
judge intelligence once the judge is anchored in an explicit rubric, and
that within the rubric the official answer does nearly all the work.

The stock :class:`DeepResearchPairwiseMetric` grades against four flat
dimension descriptions and no reference standard, which is exactly the
paper's unanchored condition. This subclass keeps the entire existing
contract - ``score(question, baseline_answer, candidate_answer)`` returning
:class:`DeepResearchScoreResult`, the flipped-order position-bias control,
the multi-trial voting - and only swaps the judge's system prompt for one
anchored in a rubric built from the baseline report.

Because the parent's trials are already repeated measurements of the same
answer sheet, they host the paper's variance decomposition for free: how
much of the score variance is the answer being graded versus anything
else. The paper needed six judges across three effort levels to estimate
that; a single judge's repeated trials estimate the within-judge residual,
which is the half of the decomposition that rubric anchoring is supposed
to shrink.
"""

from contextvars import ContextVar
from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam

from evals.metrics.deep_research_pairwise_metric import (
    DEEP_RESEARCH_PAIRWISE_PROMPT,
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchPairwisePreferenceInput,
)
from evals.metrics.reference_rubric import (
    AnchorMode,
    build_rubric,
    render_rubric_anchor,
)

ANCHORED_SYSTEM_PROMPT = """You are an expert evaluator for reports to a research question. You'll be comparing two responses to a research question: report_a and report_b.

Grade both reports against the grading rubric supplied with the question. The rubric is the standard: work outward from its criteria and its official answer, and prefer the report that covers more of the ground the rubric names.

Evaluate both reports on these dimensions:
1. Instruction following: Evaluates response's fidelity to user specified instructions and constraints.
2. Comprehensiveness: Measures breadth and range of information covered in response, addressing the scope of user request.
3. Completeness: Measures the depth and thoroughness of information for topics addressed in the report.
4. Writing quality: Evaluates clarity, conciseness, logical organization and overall readability of the report.

For each dimension, indicate which report you prefer (either "a" or "b") and provide a concise explanation for your choice.
Your explanations should cite specific examples to justify your preference and point out what can be improved in the other report.
Also provide a gap score that measures the difference in quality between the two reports for that dimension.
The gap score should be a number from 0 to 5, where 0 indicates that both reports have similar quality and 5 is the maximum difference in quality.

Be fair and objective in your evaluation. Do not be biased towards either report A or B.
The length of a report is not necessarily an indicator of quality - focus on the substance and how well it meets the user's needs.
Cite the rubric criteria or the official answer when explaining a preference.
"""

#: The reference answer for the row currently being scored, in the original
#: (unflipped) orientation. Set by ``_get_pairwise_preference`` so the anchor
#: stays fixed while the parent metric swaps report_a/report_b.
_reference_answer: ContextVar[str] = ContextVar("reference_answer", default="")


class RubricAnchoredPairwiseMetric(DeepResearchPairwiseMetric):
    """Pairwise metric that grades against an explicit rubric.

    Args:
        anchor_mode: which parts of the rubric anchor the judge prompt.
            ``FULL`` is the paper's main arm; ``ANSWER_ONLY`` and ``NONE``
            reproduce its two ablations, with ``NONE`` reducing to the stock
            flat prompt.
        kwargs: forwarded to :class:`DeepResearchPairwiseMetric`.
    """

    def __init__(
        self,
        *args: Any,
        anchor_mode: AnchorMode = AnchorMode.FULL,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_mode = anchor_mode

    def _get_pairwise_preference(
        self, metric_input: DeepResearchPairwisePreferenceInput
    ) -> dict:
        """Run the stock trial loop with the reference answer pinned.

        The parent flips report_a/report_b between orderings to control
        position bias. An anchor built from ``metric_input.baseline_answer``
        would then describe the *candidate* under the flipped ordering, which
        would hand the position-bias control a confound. Pinning the true
        reference in a ContextVar keeps the anchor identical across both
        orderings while remaining safe under the evaluator's thread pool,
        where several rows share this metric instance.
        """
        token = _reference_answer.set(metric_input.baseline_answer)
        try:
            return super()._get_pairwise_preference(metric_input)
        finally:
            _reference_answer.reset(token)

    def _get_evaluation_messages(
        self, metric_input: DeepResearchPairwisePreferenceInput
    ) -> List[ChatCompletionMessageParam]:
        """Render the stock messages with the rubric anchor appended.

        The rubric is built from the reference answer pinned by
        :meth:`_get_pairwise_preference`, so it is identical for the original
        and the flipped ordering and cannot itself take a side.
        """
        rubric = build_rubric(
            question=metric_input.question,
            official_answer=_reference_answer.get(),
            mode=self.anchor_mode,
        )
        anchor = render_rubric_anchor(rubric)

        if not anchor:
            # AnchorMode.NONE reproduces the stock prompt byte-for-byte.
            return super()._get_evaluation_messages(metric_input)

        system = (
            ANCHORED_SYSTEM_PROMPT
            if self.anchor_mode is AnchorMode.FULL
            else DEEP_RESEARCH_PAIRWISE_PROMPT
        )

        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"""
<prompt>
{metric_input.question}
</prompt>

{anchor}

<report_a>
{metric_input.baseline_answer}
</report_a>

<report_b>
{metric_input.candidate_answer}
</report_b>
""",
            },
        ]

    def analyze_agreement(
        self, scores_list: List[Any]
    ) -> Dict[str, Any]:
        """Decompose score variance across the trials the metric already ran.

        The paper attributes 95.6% of score variance to the answer being
        graded and 0.2% to judge identity. Those figures need a judge panel;
        what a single judge's repeated trials give us is the complement -
        the residual spread on a fixed answer that anchoring is supposed to
        compress. The per-dimension spread reported here is that residual.

        Args:
            scores_list: score results from ``score()``, whose
                ``raw_preferences`` carry the original and flipped trials.

        Returns:
            Dictionary with per-dimension trial counts, mean trial score
            spread, preference agreement, and an ``icc`` estimated from the
            repeated trials, plus overall averages of the three.
        """
        per_dimension: Dict[str, Any] = {}
        all_pref_agreement: List[float] = []

        for dimension in DIMENSIONS:
            n_scores = 0
            spreads: List[float] = []
            agreements: List[float] = []
            # One row of trial scores per scored answer sheet; the panel
            # across rows is what the ICC is estimated from.
            panels: List[List[float]] = []

            for score_result in scores_list:
                dim = getattr(score_result, dimension)
                raw = dim.raw_preferences or {}
                original = raw.get("original") or []
                flipped = raw.get("flipped") or []

                if not original and not flipped:
                    continue

                # score_b is signed toward candidate (b) in the original
                # ordering; flipping the order flips the sign.
                trials = [p["score_b"] for p in original] + [
                    -p["score_b"] for p in flipped
                ]
                if trials:
                    n_scores += 1
                    spreads.append(max(trials) - min(trials))
                    panels.append(trials)

                # Preference agreement: how often the repeated measurements
                # of the *same* answer pair reached the same verdict once
                # the flipped ordering is re-aligned.
                prefs = [p["preferred"] for p in original] + [
                    "a" if p["preferred"] == "b" else "b" for p in flipped
                ]
                if len(prefs) > 1:
                    top = max(set(prefs), key=prefs.count)
                    agreements.append(prefs.count(top) / len(prefs))

            spread = sum(spreads) / len(spreads) if spreads else 0.0
            pref_agreement = (
                sum(agreements) / len(agreements) if agreements else 0.0
            )
            all_pref_agreement.extend(agreements)

            per_dimension[dimension] = {
                "num_scored": n_scores,
                "mean_trial_score_spread": spread,
                "preference_agreement": pref_agreement,
                "icc": _panel_icc(panels),
            }

        return {
            "anchor_mode": self.anchor_mode.value,
            "num_rows": len(scores_list),
            "per_dimension": per_dimension,
            "overall": {
                "mean_trial_score_spread": (
                    sum(d["mean_trial_score_spread"] for d in per_dimension.values())
                    / len(per_dimension)
                    if per_dimension
                    else 0.0
                ),
                "preference_agreement": (
                    sum(all_pref_agreement) / len(all_pref_agreement)
                    if all_pref_agreement
                    else 0.0
                ),
                "icc": _mean_icc(per_dimension.values()),
            },
        }


def _panel_icc(panels: List[List[float]]) -> float:
    """Two-way random-effects ICC(2,1) over a rows-by-trials score panel.

    Rows are the answer sheets being graded - the paper's targets, which
    explain 95.6% of its score variance. Columns are the repeated trials
    (original plus order-flipped). The estimate is the fraction of total
    variance attributable to the answer being graded rather than to trial
    noise, which is the quantity the paper reports collapsing when the
    rubric's official answer is removed.

    Returns ``None`` when the panel cannot support an estimate (fewer than
    two rows, or ragged panels, which the flipped trials can produce).
    """
    rows = [r for r in panels if len(r) > 1]
    if len(rows) < 2:
        return None

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        return None

    k = widths.pop()
    n = len(rows)
    grand_mean = sum(sum(r) for r in rows) / (n * k)

    row_means = [sum(r) / k for r in rows]
    col_means = [sum(r[i] for r in rows) / n for i in range(k)]

    ms_rows = k * sum((m - grand_mean) ** 2 for m in row_means) / (n - 1)
    ms_cols = n * sum((m - grand_mean) ** 2 for m in col_means) / (k - 1)
    ss_residual = sum(
        (r[i] - row_means[j] - col_means[i] + grand_mean) ** 2
        for j, r in enumerate(rows)
        for i in range(k)
    )
    ms_residual = ss_residual / ((n - 1) * (k - 1))

    if ms_residual == 0:
        # Trials repeat exactly: perfectly reliable by construction.
        return 1.0

    # ICC(2,1) for absolute agreement, single measure.
    denom = (
        ms_rows
        + (k - 1) * ms_residual
        + k * (ms_cols - ms_residual) / n
    )
    if denom <= 0:
        return None
    return (ms_rows - ms_residual) / denom


def _mean_icc(dimensions: Any) -> float:
    """Mean of the per-dimension ICCs that could be estimated."""
    values = [d["icc"] for d in dimensions if d.get("icc") is not None]
    return sum(values) / len(values) if values else None
