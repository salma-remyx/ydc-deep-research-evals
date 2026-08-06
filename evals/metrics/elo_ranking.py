"""Elo-based ranking of deep-research pairwise evaluation outcomes.

Adapted from "(Towards) Scalable Reliable Automated Evaluation with Large
Language Models" (arXiv:2607.28282). The paper aggregates many LLM-judge
pairwise comparisons into *stable, interpretable* scalar ratings via an **Elo
rating system**, and lets the evaluator trade confidence for coverage through
an **adjustable agreement threshold** that ranges from full unanimity down to
simple majority voting.

This is a Mode 2 (adapted port):

* Core mechanism kept at full fidelity:
    - Logistic Elo updates over pairwise match outcomes (win / loss / tie).
    - An adjustable agreement threshold that decides whether a row's
      multi-trial preference is decisive enough to count as a win/loss or
      collapses to a low-confidence tie. The repo's existing per-dimension
      majority vote is exactly the ``threshold=0.5`` case; raising the
      threshold toward ``1.0`` requires near-unanimity.
* Auxiliary components substituted for target-native equivalents:
    - The paper's "multiple LLMs" judges -> this repo's existing multi-trial,
      flipped-order evaluations (the repo already aggregates many judgments
      per row to reduce bias).
    - The paper's competency-profile benchmark -> the repo's existing
      ``DeepResearchScoreResult`` rows (DeepConsult). No separate benchmark /
      evaluation framework is ported here; that belongs downstream.

The module is intentionally parameter-free (Elo is a closed-form iterative
update) and has no dependency on the metric's pydantic models: ``rank_with_elo``
reads ``getattr(row, dimension).preferred`` via duck typing, so it stays
decoupled and easy to test.
"""

from typing import Any, Dict, Literal, Sequence, Tuple

# Standard logistic-Elo hyperparameters.
DEFAULT_K = 32.0
DEFAULT_BASE_RATING = 1000.0

# Player labels for the repo's fixed baseline-vs-candidate comparison frame.
# `preferred == "b"` in the metric means the candidate (report_b) won.
BASELINE_PLAYER = "baseline"
CANDIDATE_PLAYER = "candidate"

MatchOutcome = Literal["win", "loss", "tie"]
Match = Tuple[str, str, MatchOutcome]


def resolve_outcome(
    votes: Sequence[str], agreement_threshold: float = 0.5
) -> MatchOutcome:
    """Decide a candidate-perspective outcome from per-trial ``"a"``/``"b"`` votes.

    ``agreement_threshold`` is the fraction of trials that must agree with the
    majority for the match to count as decided:

    * ``0.5`` (default) reproduces simple majority voting -- the repo's existing
      per-row grade.
    * ``1.0`` requires full unanimity -- rows without unanimous agreement
      collapse to ties, trading coverage for confidence.

    A tie in the raw vote count is always a tie, regardless of threshold.
    """
    if not votes:
        return "tie"
    candidate = sum(1 for v in votes if v == "b")
    baseline = sum(1 for v in votes if v == "a")
    agreement = max(candidate, baseline) / len(votes)
    if agreement < agreement_threshold:
        return "tie"
    if candidate > baseline:
        return "win"
    if baseline > candidate:
        return "loss"
    return "tie"


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard logistic expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    matches: Sequence[Match],
    k: float = DEFAULT_K,
    base_rating: float = DEFAULT_BASE_RATING,
) -> Dict[str, float]:
    """Run sequential Elo updates over a series of matches.

    Each match is ``(player_a, player_b, outcome_for_a)`` where the outcome is
    ``"win"`` / ``"loss"`` / ``"tie"`` from player A's perspective (a tie is
    worth 0.5). Players are seeded at ``base_rating`` on first appearance. The
    update order follows ``matches``; Elo converges to the same relative
    ordering regardless of order, so results are kept deterministic.
    """
    score_for_outcome = {"win": 1.0, "loss": 0.0, "tie": 0.5}
    ratings: Dict[str, float] = {}
    for player_a, player_b, outcome in matches:
        rating_a = ratings.setdefault(player_a, base_rating)
        rating_b = ratings.setdefault(player_b, base_rating)
        expected_a = expected_score(rating_a, rating_b)
        actual_a = score_for_outcome[outcome]
        ratings[player_a] = rating_a + k * (actual_a - expected_a)
        ratings[player_b] = rating_b + k * ((1.0 - actual_a) - (1.0 - expected_a))
    return ratings


def _matches_from_scores(
    scores_list: Sequence[Any],
    dimension: str,
    agreement_threshold: float,
) -> Sequence[Match]:
    """Build candidate-vs-baseline matches for one dimension across all rows."""
    matches: list[Match] = []
    for score_result in scores_list:
        votes = getattr(score_result, dimension).preferred
        outcome = resolve_outcome(votes, agreement_threshold)
        matches.append((CANDIDATE_PLAYER, BASELINE_PLAYER, outcome))
    return matches


def _summarize_ratings(
    ratings: Dict[str, float],
    matches: Sequence[Match],
) -> Dict[str, Any]:
    decided = sum(1 for _, _, outcome in matches if outcome != "tie")
    total = len(matches)
    return {
        "baseline_rating": ratings.get(BASELINE_PLAYER, DEFAULT_BASE_RATING),
        "candidate_rating": ratings.get(CANDIDATE_PLAYER, DEFAULT_BASE_RATING),
        "decided_matches": decided,
        "total_matches": total,
        "coverage": decided / total if total else 0.0,
    }


def rank_with_elo(
    scores_list: Sequence[Any],
    dimensions: Sequence[str],
    agreement_threshold: float = 0.5,
    k: float = DEFAULT_K,
    base_rating: float = DEFAULT_BASE_RATING,
) -> Dict[str, Any]:
    """Compute Elo ratings and threshold-adjustable coverage from score rows.

    Each row contributes one candidate-vs-baseline match per dimension. The
    outcome of each match is resolved from that row's per-trial ``preferred``
    votes at ``agreement_threshold`` (see :func:`resolve_outcome`).

    Returns per-dimension and overall ratings plus the number of matches that
    were decisive (``coverage``) at the requested threshold -- the knob the
    paper uses to trade confidence for coverage.
    """
    all_matches: list[Match] = []
    per_dimension: Dict[str, Dict[str, Any]] = {}

    for dimension in dimensions:
        matches = _matches_from_scores(scores_list, dimension, agreement_threshold)
        all_matches.extend(matches)
        ratings = compute_elo_ratings(matches, k=k, base_rating=base_rating)
        per_dimension[dimension] = _summarize_ratings(ratings, matches)

    overall_ratings = compute_elo_ratings(
        all_matches, k=k, base_rating=base_rating
    )
    overall = _summarize_ratings(overall_ratings, all_matches)

    return {
        "agreement_threshold": agreement_threshold,
        "k": k,
        "overall": overall,
        "dimensions": per_dimension,
    }
