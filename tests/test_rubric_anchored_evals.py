"""Tests for rubric-anchored pairwise evaluation.

These exercise the wiring end to end against the pre-existing pipeline
modules (``evals.deep_research_pairwise_evals`` and
``evals.metrics.deep_research_pairwise_metric``), with the OpenAI call
stubbed so no network access is required.
"""

from typing import List
from unittest.mock import patch

import pandas as pd

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DEEP_RESEARCH_PAIRWISE_PROMPT,
    DeepResearchPairwisePreferenceOutput,
    DeepResearchScoreResult,
    DimensionResult,
    Preference,
)
from evals.metrics.reference_rubric import AnchorMode
from evals.metrics.rubric_anchored_pairwise_metric import (
    ANCHORED_SYSTEM_PROMPT,
    RubricAnchoredPairwiseMetric,
)
from evals.rubric_anchored_evals import RubricAnchoredEvaluator


QUESTION = (
    "Evaluate the potential consequences of TikTok bans on investment risks. "
    "Consider how varying degrees of restrictions might impact business operations."
)

BASELINE = """# Investment Risk of TikTok Bans

## Overview
Advertisers face repricing risk on restricted inventory.

- Consider the effect on small-business customer acquisition costs
- Analyze exposure for consumer brands with concentrated Gen-Z reach
"""

CANDIDATE = """TikTok bans affect investment risk in several ways.
The main effect is on advertising reach for consumer brands.
"""


def _preference(preferred: str, gap_score: int) -> Preference:
    return Preference(
        explanation="stubbed", preferred=preferred, gap_score=gap_score
    )


def _output(
    preferred: str = "b", gap_score: int = 2
) -> DeepResearchPairwisePreferenceOutput:
    """A single judge verdict, identical across dimensions."""
    prefs = {d: _preference(preferred, gap_score) for d in DIMENSIONS}
    return DeepResearchPairwisePreferenceOutput(**prefs)


class _StubResponses:
    """Records the messages it saw and answers from the report contents.

    The judge call is not order-indexed: the parent metric submits the
    original and flipped trials to one thread pool and collects them with
    ``as_completed``, so stubbing by call position would be racy. Instead
    each stub reads which report is in which slot and returns the verdict a
    judge preferring the candidate would give in that orientation.
    """

    def __init__(self, candidate_answer: str, gap_score: int = 2):
        self.candidate_answer = candidate_answer.strip()
        self.gap_score = gap_score
        self.seen: List[List[dict]] = []

    def _slot(self, messages, tag):
        return messages[1]["content"].split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()

    def __call__(self, messages=None, output_class=None, **kwargs):
        self.seen.append(messages)
        # The candidate is report_b in the original ordering and report_a
        # once the parent flips the inputs.
        preferred = (
            "b" if self._slot(messages, "report_b") == self.candidate_answer else "a"
        )
        return _output(preferred=preferred, gap_score=self.gap_score)


def _run(metric, question=QUESTION, baseline=BASELINE, candidate=CANDIDATE):
    return metric.score(
        question=question,
        baseline_answer=baseline,
        candidate_answer=candidate,
    )


def test_anchored_metric_returns_existing_score_result_type():
    """The anchored metric keeps the stock score() contract."""
    metric = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.FULL)
    stub = _StubResponses(CANDIDATE)

    with patch.object(
        metric, "_query_evaluation_model", side_effect=stub
    ):
        result = _run(metric)

    for dimension in DIMENSIONS:
        dim = getattr(result, dimension)
        assert dim.grade in ("win", "lose", "tie")
        assert isinstance(dim.score, float)
        assert isinstance(dim.is_win, bool)


def test_anchor_block_reaches_the_judge_prompt():
    """The rubric anchor is actually passed to the judge."""
    metric = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.FULL)
    stub = _StubResponses(CANDIDATE)

    with patch.object(metric, "_query_evaluation_model", side_effect=stub):
        _run(metric)

    assert stub.seen, "judge was never called"
    user_contents = [m[1]["content"] for m in stub.seen]
    assert all("<grading_rubric>" in c for c in user_contents)
    assert all("<official_answer>" in c for c in user_contents)

    def anchor_block(content):
        return content.split("<grading_rubric>")[1].split("</grading_rubric>")[0]

    # The rubric is derived from the baseline, so it must be identical across
    # the original and flipped orderings and cannot itself take a side.
    assert len({anchor_block(c) for c in user_contents}) == 1


def test_anchored_swaps_system_prompt_and_none_mode_restores_stock():
    """FULL uses the anchored prompt; NONE reduces to the stock prompt."""
    full = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.FULL)
    full_stub = _StubResponses(CANDIDATE)
    with patch.object(full, "_query_evaluation_model", side_effect=full_stub):
        _run(full)
    assert all(m[0]["content"] == ANCHORED_SYSTEM_PROMPT for m in full_stub.seen)

    none_metric = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.NONE)
    none_stub = _StubResponses(CANDIDATE)
    with patch.object(none_metric, "_query_evaluation_model", side_effect=none_stub):
        _run(none_metric)
    assert all(
        m[0]["content"] == DEEP_RESEARCH_PAIRWISE_PROMPT for m in none_stub.seen
    )
    assert all("<grading_rubric>" not in m[1]["content"] for m in none_stub.seen)


def test_answer_only_mode_drops_criteria_keeps_official_answer():
    """The paper's first ablation: criteria gone, official answer retained."""
    metric = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.ANSWER_ONLY)
    stub = _StubResponses(CANDIDATE)

    with patch.object(metric, "_query_evaluation_model", side_effect=stub):
        _run(metric)

    for messages in stub.seen:
        content = messages[1]["content"]
        assert "<official_answer>" in content
        assert "Grade against these explicit criteria" not in content


def test_anchor_symmetry_survives_position_flip():
    """Anchoring must not leak which report is the reference.

    The parent metric flips report_a/report_b to control position bias. The
    anchor is built from the baseline, so under the flipped ordering the
    baseline appears as report_b and the rubric must still describe it -
    not the candidate.
    """
    metric = RubricAnchoredPairwiseMetric(anchor_mode=AnchorMode.FULL)
    stub = _StubResponses(CANDIDATE)

    with patch.object(metric, "_query_evaluation_model", side_effect=stub):
        _run(metric)

    def report_body(content, tag):
        return content.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()

    contents = [m[1]["content"] for m in stub.seen]
    original = [c for c in contents if report_body(c, "report_a") == BASELINE.strip()]
    flipped = [c for c in contents if report_body(c, "report_b") == BASELINE.strip()]

    assert original and flipped, "expected both orderings to be judged"

    # In the flipped ordering the baseline moved to report_b...
    assert report_body(original[0], "report_b") == report_body(flipped[0], "report_a")
    # ...and the anchor stayed put rather than following the baseline.
    assert report_body(original[0], "official_answer") == report_body(
        flipped[0], "official_answer"
    )
    assert report_body(original[0], "official_answer") == BASELINE.strip()


def test_evaluator_subclass_wires_anchored_metric_into_pipeline():
    """RubricAnchoredEvaluator is a DeepResearchEvaluator using the anchored metric."""
    evaluator = RubricAnchoredEvaluator(
        anchor_mode=AnchorMode.FULL, num_workers=1, metric_num_workers=1
    )
    assert isinstance(evaluator, DeepResearchEvaluator)
    assert isinstance(evaluator.pairwise_metric, RubricAnchoredPairwiseMetric)
    assert evaluator.pairwise_metric.anchor_mode is AnchorMode.FULL
    # Inherited behaviour is intact, not re-implemented.
    assert evaluator.pairwise_metric.num_trials == evaluator.metric_num_trials


def test_aggregate_results_reports_rubric_agreement():
    """aggregate_results() keeps the stock aggregates and adds the analysis."""
    evaluator = RubricAnchoredEvaluator(
        anchor_mode=AnchorMode.FULL, num_workers=1, metric_num_workers=1
    )
    stub = _StubResponses(CANDIDATE)
    results = []
    with patch.object(
        evaluator.pairwise_metric, "_query_evaluation_model", side_effect=stub
    ):
        results = evaluator.evaluate_batch(
            pd.DataFrame(
                [
                    {
                        "question": QUESTION,
                        "baseline_answer": BASELINE,
                        "candidate_answer": CANDIDATE,
                    }
                ]
            )
        )

    aggregate = evaluator.aggregate_results(results)
    assert aggregate["support"] == 1
    assert "overall" in aggregate
    assert "win_rate" in aggregate["comprehensiveness"]

    agreement = aggregate["rubric_agreement"]
    assert agreement["anchor_mode"] == "full"
    assert agreement["num_rows"] == 1
    assert set(agreement["per_dimension"]) == set(DIMENSIONS)
    # Stub verdicts are identical across trials, so agreement is maximal.
    assert agreement["overall"]["preference_agreement"] == 1.0
    assert agreement["overall"]["mean_trial_score_spread"] == 0.0


def test_agreement_distinguishes_consistent_from_noisy_trials():
    """Agreement falls when repeated trials of the same answer disagree."""
    evaluator = RubricAnchoredEvaluator(
        anchor_mode=AnchorMode.FULL, num_workers=1, metric_num_workers=1
    )
    metric = evaluator.pairwise_metric

    # One row whose trials all agree, versus one row whose trials split.
    # Under the flipped ordering the candidate is report_a, so a verdict
    # preferring the candidate is recorded as "a" there.
    consistent = _score_result_with_trials(
        original=[("b", 2), ("b", 2)], flipped=[("a", 2), ("a", 2)]
    )
    noisy = _score_result_with_trials(
        original=[("b", 4), ("a", 0)], flipped=[("a", 5), ("b", 1)]
    )

    consistent_stats = metric.analyze_agreement([consistent])
    noisy_stats = metric.analyze_agreement([noisy])

    assert consistent_stats["overall"]["preference_agreement"] == 1.0
    assert consistent_stats["overall"]["mean_trial_score_spread"] == 0.0

    assert noisy_stats["overall"]["preference_agreement"] < 1.0
    assert noisy_stats["overall"]["mean_trial_score_spread"] > 0.0


def _trial(preferred: str, gap_score: int) -> dict:
    """One stored trial, matching what ``_get_pairwise_preference`` saves."""
    return {
        "preferred": preferred,
        "gap_score": gap_score,
        "explanation": "s",
        "score_b": gap_score if preferred == "b" else -gap_score,
    }


def _score_result_with_trials(*, original, flipped) -> DeepResearchScoreResult:
    """Build a DeepResearchScoreResult from raw trial verdicts.

    ``original`` verdicts are in report_a=baseline, report_b=candidate
    order; ``flipped`` verdicts come from the reversed ordering, exactly as
    ``get_pairwise_preference`` stores them.
    """
    dimension_results = {}
    for dimension in DIMENSIONS:
        original_prefs = [
            _trial(p, g) for p, g in original
        ]
        flipped_prefs = [
            _trial(p, g) for p, g in flipped
        ]
        # Re-align flipped verdicts to the original orientation.
        realigned = [
            "a" if p == "b" else "b" for p in (t["preferred"] for t in flipped_prefs)
        ]
        num_wins = sum(1 for p in realigned if p == "b")
        num_losses = sum(1 for p in realigned if p == "a")
        grade = (
            "win" if num_wins > num_losses
            else "lose" if num_wins < num_losses
            else "tie"
        )

        dimension_results[dimension] = DimensionResult(
            grade=grade,
            is_win=grade == "win",
            is_tie=grade == "tie",
            is_lose=grade == "lose",
            score=5.0,
            preferred=realigned,
            raw_preferences={"original": original_prefs, "flipped": flipped_prefs},
        )

    return DeepResearchScoreResult(**dimension_results)
