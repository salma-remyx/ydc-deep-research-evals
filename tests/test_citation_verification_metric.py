"""Integration tests for the citation-verification metric.

These exercise the wiring against NON-NEW modules:

  - ``evals.utils.replace_markdown_links_with_text`` and the existing
    ``DeepResearchPairwisePreferenceInput`` (which strips citation links) are
    contrasted with the new citation-preserving input.
  - The new metric reuses the existing ``Preference`` rubric primitive and the
    flipped-order consensus pattern; we exercise the pure consensus /
    aggregation logic without hitting the network.
"""

from evals.metrics.citation_verification_metric import (
    CITATION_DIMENSIONS,
    CitationDimensionResult,
    CitationVerificationInput,
    CitationVerificationMetric,
    CitationVerificationScoreResult,
)
from evals.metrics.deep_research_pairwise_metric import (
    DeepResearchPairwisePreferenceInput,
    Preference,
)

CITED = "The market grew 5% [report](http://example.com/report.pdf) last year."


def test_citation_input_preserves_markdown_links_that_base_strips():
    """The existing input strips citations; the citation metric must keep them.

    This is the integration point with ``utils.replace_markdown_links_with_text``:
    verifying citation quality is only possible while the links survive.
    """
    base = DeepResearchPairwisePreferenceInput(
        question="q", baseline_answer=CITED, candidate_answer="other"
    )
    assert "](http" not in base.baseline_answer
    assert "[report]" not in base.baseline_answer

    cited = CitationVerificationInput(
        question="q", baseline_answer=CITED, candidate_answer="other"
    )
    assert "[report](http://example.com/report.pdf)" in cited.baseline_answer


def test_consensus_mitigates_position_bias_and_flags_flip_disagreement():
    metric = CitationVerificationMetric()

    # Candidate wins every original trial, but the judge flips to baseline when
    # the report order is reversed -> a position-biased, contradictory verdict.
    biased_original = [Preference(explanation="e", preferred="b", gap_score=2)] * 3
    # In flipped order 'b' is the baseline; canonicalized back to 'a'.
    biased_flipped = [Preference(explanation="e", preferred="b", gap_score=2)] * 3
    biased = metric._consensus_for_dimension(biased_original, biased_flipped)
    assert biased["consensus_grade"] == "tie"  # 3 'b' vs 3 'a' cancel out
    assert biased["flip_disagreement"] is True

    # A consistent judge picks the candidate regardless of order -> win, no drift.
    consistent_original = [Preference(explanation="e", preferred="b", gap_score=1)] * 3
    consistent_flipped = [Preference(explanation="e", preferred="a", gap_score=1)] * 3
    consistent = metric._consensus_for_dimension(
        consistent_original, consistent_flipped
    )
    assert consistent["consensus_grade"] == "win"
    assert consistent["flip_disagreement"] is False


def test_score_result_has_paper_dimensions():
    assert CITATION_DIMENSIONS == ["source_relevance", "factual_support"]
    fields = set(CitationDimensionResult.model_fields)
    assert "flip_disagreement" in fields  # paper's directional-bias signal
    assert {
        "grade",
        "is_win",
        "is_tie",
        "is_lose",
        "score",
        "preferred",
        "raw_preferences",
    } <= fields


def _result(sr_grade, fs_grade, sr_flip, fs_flip):
    def _dim(grade, flip):
        return CitationDimensionResult(
            grade=grade,
            is_win=grade == "win",
            is_tie=grade == "tie",
            is_lose=grade == "lose",
            score=5.0,
            preferred=[],
            raw_preferences={},
            flip_disagreement=flip,
        )

    return CitationVerificationScoreResult(
        source_relevance=_dim(sr_grade, sr_flip),
        factual_support=_dim(fs_grade, fs_flip),
    )


def test_aggregate_reports_directional_bias_rate():
    metric = CitationVerificationMetric()
    scores = [
        _result("win", "lose", True, False),
        _result("win", "lose", False, True),
    ]
    agg = metric.aggregate(scores)

    assert agg["support"] == 2
    assert agg["source_relevance"]["win_rate"] == 1.0
    assert agg["factual_support"]["lose_rate"] == 1.0
    # One of two rows disagreed on each dimension -> 0.5 per dimension, 0.5 overall.
    assert agg["overall"]["flip_disagreement_rate"] == 0.5
    assert agg["overall"]["net_winrate"] == 0.5
