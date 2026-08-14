"""Pivotal-vote (affected-set) analysis for pairwise LLM-judge evaluations.

Adapted from "Blind to the Pivotal Vote: Aggregate Independence Metrics
Miss Where Verification Actually Helps" (arXiv:2608.06940).

The paper's core result, in one line: for an LLM-judge panel, a single
substituted (or added) ballot can ONLY change decisions that were decided
by a one-vote margin -- the "pivotal" set. The entire accuracy gain from
an external verification signal (e.g. executing a test suite, or a human
fact-check) concentrates on those pivotal decisions and is essentially
zero on wider margins. Population-level panel-independence diagnostics and
this margin-stratified utility are complementary questions.

This module applies that affected-set arithmetic to the repo's pairwise
metric. Each ``(query, dimension)`` decision is reached by majority vote
over the flipped-order trial ballots already tallied on
``DimensionResult.preferred`` (``"a"``/``"b"`` votes from
``DeepResearchPairwiseMetric._get_pairwise_preference``).
:func:`ballot_margin` / :func:`is_pivotal_ballot` characterize a single
decision; :func:`pivotal_breakdown_across_dimensions` reports, per
dimension and overall, what fraction of decisions are pivotal -- i.e. the
share where a targeted re-trial or human fact-check (the team's
verification signal on DeepConsult reports) could plausibly move the
verdict. That fraction is the paper's call-reduction rule: spend the
verification budget on the pivotal queries, not the whole dataset.

Parity note. The paper's panels are odd-sized, so "one-vote margin" is a
margin of 1. This repo's tally is even-sized (``2 * num_trials`` original
+ flipped ballots), so the smallest *decisive* margin is 2 (a single
ballot substitution collapses it to a tie) and a tie (margin 0) becomes a
decision under a single added ballot. We therefore treat a decision as
pivotal when ``ballot_margin <= pivotal_margin`` (default 2), which is the
parity-agnostic statement of "a single-ballot substitution could change
the grade": margins {0, 1, 2} are pivotal, wider margins are stable.
"""

from typing import Any, Dict, Iterable, List, Sequence

# Default maximum margin (inclusive) for a decision to count as "pivotal":
# a single-ballot substitution or added ballot can change its grade. See
# the parity note in the module docstring.
DEFAULT_PIVOTAL_MARGIN = 2


def ballot_margin(preferred: Sequence[str]) -> int:
    """Absolute vote margin ``|num_b - num_a|`` over the recorded ballots.

    A tie has margin 0; wider majorities have larger margins. ``preferred``
    is the per-trial list of ``"a"``/``"b"`` votes stored on
    ``DimensionResult.preferred``.
    """
    num_b = sum(1 for vote in preferred if vote == "b")
    num_a = sum(1 for vote in preferred if vote == "a")
    return abs(num_b - num_a)


def is_pivotal_ballot(
    preferred: Sequence[str], *, pivotal_margin: int = DEFAULT_PIVOTAL_MARGIN
) -> bool:
    """Whether a single-ballot substitution could change this decision.

    True iff ``ballot_margin(preferred) <= pivotal_margin``. At or below
    the threshold a single substituted/added ballot crosses a grade
    boundary (win/lose -> tie, or tie -> decision); above it the majority
    is stable under any single-ballot change.
    """
    return ballot_margin(preferred) <= pivotal_margin


def _margin_histogram(margins: Sequence[int]) -> Dict[str, int]:
    """Counts of decisions at each observed margin, keyed by margin."""
    histogram: Dict[int, int] = {}
    for margin in margins:
        histogram[margin] = histogram.get(margin, 0) + 1
    return {str(key): histogram[key] for key in sorted(histogram)}


def pivotal_breakdown(
    dim_results: Iterable[Any],
    *,
    pivotal_margin: int = DEFAULT_PIVOTAL_MARGIN,
) -> Dict[str, Any]:
    """Pivotal-vote statistics over one group of decisions.

    Each item is duck-typed as having ``.preferred`` (ballot list) and
    ``.grade`` (``"win"``/``"lose"``/``"tie"``), so this works directly
    with ``DimensionResult`` objects without importing them (avoids a
    circular import with the metric module that hosts this analysis).

    Returns counts and rates. ``pivotal_rate`` is the affected-set
    fraction -- the call-reduction share of decisions a single
    substituted/added ballot could move.
    """
    items = list(dim_results)
    total = len(items)
    if total == 0:
        return {
            "support": 0,
            "num_pivotal": 0,
            "pivotal_rate": 0.0,
            "num_decided": 0,
            "decided_rate": 0.0,
            "num_ties": 0,
            "tie_rate": 0.0,
            "pivotal_margin": pivotal_margin,
            "margin_distribution": {},
        }

    margins = [ballot_margin(item.preferred) for item in items]
    num_pivotal = sum(1 for margin in margins if margin <= pivotal_margin)
    num_ties = sum(1 for item in items if item.grade == "tie")

    return {
        "support": total,
        "num_pivotal": num_pivotal,
        # Affected-set fraction == call-reduction fraction: the share of
        # decisions where a verification signal could change the grade.
        "pivotal_rate": num_pivotal / total,
        "num_decided": total - num_ties,
        "decided_rate": (total - num_ties) / total,
        "num_ties": num_ties,
        "tie_rate": num_ties / total,
        "pivotal_margin": pivotal_margin,
        "margin_distribution": _margin_histogram(margins),
    }


def pivotal_breakdown_across_dimensions(
    scores_list: Sequence[Any],
    dimensions: Sequence[str],
    *,
    pivotal_margin: int = DEFAULT_PIVOTAL_MARGIN,
) -> Dict[str, Any]:
    """Pivotal-vote breakdown per dimension and pooled overall.

    ``scores_list`` is a list of ``DeepResearchScoreResult``-like objects;
    ``dimensions`` is the list of attribute names to read off each (the
    repo's ``DIMENSIONS``). Passing ``dimensions`` in, rather than
    importing it, keeps this module free of a circular import with the
    metric module that calls it. The ``overall`` entry pools every
    ``(query, dimension)`` decision across dimensions.
    """
    breakdown: Dict[str, Any] = {}
    pooled: List[Any] = []

    for dimension in dimensions:
        dim_results = [getattr(score, dimension) for score in scores_list]
        breakdown[dimension] = pivotal_breakdown(
            dim_results, pivotal_margin=pivotal_margin
        )
        pooled.extend(dim_results)

    breakdown["overall"] = pivotal_breakdown(pooled, pivotal_margin=pivotal_margin)
    return breakdown
