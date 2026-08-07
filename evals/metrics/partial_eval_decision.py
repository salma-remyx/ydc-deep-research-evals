"""Partial-evaluation decision layer for the pairwise pipeline.

A pre-registered comparison policy that, given the paired per-question
outcomes a pairwise run has collected so far, returns one of four verdicts:

* ``better``         - the candidate beats the baseline by the required margin
* ``not_better``     - the baseline beats the candidate by the required margin
* ``abstain``        - enough evidence to say the two systems are within the
                       required margin (no meaningful difference)
* ``needs_evidence`` - the observed outcomes cannot yet support a decision

This is the decision layer of *ParEvalLayer: When Partial LLM-Agent
Evaluations Support a Decision* (arXiv:2608.02444). Instead of reporting a
partial score and hoping it agrees with the completed run, ParEvalLayer fixes
a comparison policy in advance and, at each point in a possibly incomplete
run, records whether the tested system is better by the required amount, not
better by that amount, needs more evidence, or should abstain -- and reports
how many comparisons remain without a decision.

Mode 2 (adapted port). ParEvalLayer consumes generic paired per-system task
outcomes ``D_i``. In this repo those outcomes are exactly the per-question
``is_win`` / ``is_lose`` / ``is_tie`` the pairwise metric already produces, so
the decision layer attaches directly to ``DeepResearchPairwiseMetric.aggregate``.
The paper's policy is reproduced faithfully -- a pre-registered margin and
confidence level over a Wilson score interval on the observed win proportion,
with an indifference band and a sequential stopping replay. The paper has no
learned estimator, bespoke optimizer, or external benchmark suite, so none are
substituted or cut; only the outcome source is target-native.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

Outcome = Tuple[bool, bool, bool]
"""A per-question paired outcome: ``(is_win, is_lose, is_tie)`` for the
candidate relative to the baseline. Exactly one element should be true."""

# ParEvalLayer's four decision-layer states.
BETTER = "better"
NOT_BETTER = "not_better"
ABSTAIN = "abstain"
NEEDS_EVIDENCE = "needs_evidence"

Verdict = str  # one of the four constants above


@dataclass(frozen=True)
class ComparisonPolicy:
    """A comparison policy chosen in advance (ParEvalLayer's pre-registered rule).

    Attributes:
        margin: the required amount. The candidate's win proportion must exceed
            ``0.5 + margin`` (or fall below ``0.5 - margin``) for a decision;
            outcomes pinned inside ``[0.5 - margin, 0.5 + margin]`` are treated
            as practically equivalent and lead to ``abstain``.
        confidence: coverage of the Wilson score interval on the win proportion.
        min_decisions: floor on observed win/lose outcomes before any verdict
            other than ``needs_evidence`` may be returned.
    """

    margin: float = 0.1
    confidence: float = 0.95
    min_decisions: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.margin < 0.5:
            raise ValueError("margin must be in (0, 0.5)")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.min_decisions < 1:
            raise ValueError("min_decisions must be >= 1")

    @property
    def z(self) -> float:
        """Two-sided z-score for the configured confidence level."""
        return NormalDist().inv_cdf((1.0 + self.confidence) / 2.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "margin": self.margin,
            "confidence": self.confidence,
            "min_decisions": self.min_decisions,
        }


def _counts(outcomes: Sequence[Outcome]) -> Tuple[int, int, int]:
    wins = sum(1 for w, _, _ in outcomes if w)
    losses = sum(1 for _, l, _ in outcomes if l)
    ties = sum(1 for _, _, t in outcomes if t)
    return wins, losses, ties


def _wilson(
    wins: int, losses: int, policy: ComparisonPolicy
) -> Tuple[float, float, float, int]:
    """Wilson score interval for the candidate win proportion among decided
    outcomes. Returns ``(low, high, proportion, decisions)``."""
    decisions = wins + losses
    if decisions <= 0:
        return 0.5, 0.5, 0.5, 0
    proportion = wins / decisions
    z = policy.z
    z2 = z * z
    denom = 1.0 + z2 / decisions
    center = (proportion + z2 / (2.0 * decisions)) / denom
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / decisions + z2 / (4.0 * decisions**2)
    ) / denom
    return max(0.0, center - half), min(1.0, center + half), proportion, decisions


def decide(
    outcomes: Sequence[Outcome], policy: Optional[ComparisonPolicy] = None
) -> Verdict:
    """Apply the comparison policy to the outcomes observed so far.

    Returns one of ``BETTER``, ``NOT_BETTER``, ``ABSTAIN``, ``NEEDS_EVIDENCE``.
    """
    policy = policy or ComparisonPolicy()
    wins, losses, _ = _counts(outcomes)
    low, high, _, decisions = _wilson(wins, losses, policy)
    upper = 0.5 + policy.margin
    lower = 0.5 - policy.margin
    if decisions < policy.min_decisions:
        return NEEDS_EVIDENCE
    if low > upper:
        return BETTER
    if high < lower:
        return NOT_BETTER
    if low >= lower and high <= upper:
        return ABSTAIN
    return NEEDS_EVIDENCE


def summarize_run(
    outcomes: Sequence[Outcome], policy: Optional[ComparisonPolicy] = None
) -> Dict[str, object]:
    """Verdict for the observed outcomes plus a sequential stopping replay.

    ``questions_to_decision`` is the earliest number of outcomes (in collection
    order) after which the verdict already matched the final verdict -- i.e. how
    much of the run was actually required to reach the same conclusion the full
    run reaches. ``None`` means the run never reached a decisive verdict.
    ``fraction_needed`` is that count over the run length.
    """
    policy = policy or ComparisonPolicy()
    wins, losses, ties = _counts(outcomes)
    low, high, proportion, decisions = _wilson(wins, losses, policy)
    total = len(outcomes)
    final = decide(outcomes, policy)

    questions_to_decision: Optional[int] = None
    if final != NEEDS_EVIDENCE and total >= policy.min_decisions:
        prefix = list(outcomes)
        for k in range(policy.min_decisions, total + 1):
            if decide(prefix[:k], policy) == final:
                questions_to_decision = k
                break
    fraction_needed = (
        questions_to_decision / total if (questions_to_decision and total) else 1.0
    )

    return {
        "verdict": final,
        "support": total,
        "decided": decisions,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_proportion": proportion,
        "ci": [low, high],
        "questions_to_decision": questions_to_decision,
        "fraction_needed": fraction_needed,
    }


def majority_outcomes(per_question: Sequence[Sequence[Outcome]]) -> List[Outcome]:
    """Collapse each question's dimension outcomes into one (win/lose/tie) by
    majority vote, so the decision layer can be replayed in question units."""
    summary: List[Outcome] = []
    for question in per_question:
        wins, losses, ties = _counts(question)
        if wins > losses:
            summary.append((True, False, False))
        elif losses > wins:
            summary.append((False, True, False))
        else:
            summary.append((False, False, True))
    return summary
