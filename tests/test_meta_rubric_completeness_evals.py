"""Tests for the two-level meta-rubric factual-completeness scorer.

These cover GAMUT's distinct mechanism (structured meta-rubric -> compiled flat weighted
checklist -> weighted coverage) and confirm the new scorer is wired into the repo's
existing judge path. The OpenAI client in ``evals.utils`` is constructed at import time,
so dummy credentials are set before any ``evals`` import.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from evals.meta_rubric_completeness_evals import (  # noqa: E402
    AuthoredCriterion,
    AuthoredGroup,
    AuthoredMetaRubric,
    CheckItem,
    CheckVerdict,
    ChecklistVerdicts,
    MetaRubricCompletenessScorer,
    aggregate_coverage,
    compile_meta_rubric,
)
from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DEFAULT_EVAL_MODEL,
    DIMENSIONS,
)


def test_scorer_enriches_existing_completeness_dimension():
    """The new capability targets the repo's existing 'completeness' dimension and
    reuses its default judge model constant (proof of integration wiring)."""
    assert "completeness" in DIMENSIONS
    scorer = MetaRubricCompletenessScorer()
    assert scorer.model == DEFAULT_EVAL_MODEL


def _sample_meta_rubric() -> AuthoredMetaRubric:
    return AuthoredMetaRubric(
        groups=[
            AuthoredGroup(
                name="market",
                importance=3.0,
                criteria=[
                    AuthoredCriterion(
                        description="Mentions TikTok", kind="atomic", items=[]
                    ),
                    AuthoredCriterion(
                        description="Risks covered",
                        kind="coverage",
                        items=["regulatory risk", "revenue risk", "supply-chain risk"],
                    ),
                ],
            ),
            AuthoredGroup(
                name="strategy",
                importance=1.0,
                criteria=[
                    AuthoredCriterion(
                        description="Mitigation sequence",
                        kind="ordered",
                        items=["diversify", "lobby", "pivot"],
                    ),
                ],
            ),
        ]
    )


def test_compile_meta_rubric_flattens_and_weights():
    checks = compile_meta_rubric(_sample_meta_rubric())

    # 1 atomic + 3 coverage in group "market", 3 ordered in group "strategy" = 7 checks.
    assert len(checks) == 7

    market = [c for c in checks if c.group == "market"]
    strategy = [c for c in checks if c.group == "strategy"]
    assert len(market) == 4 and len(strategy) == 3

    # Group importance is distributed evenly across its checks.
    assert all(abs(c.weight - 3.0 / 4) < 1e-9 for c in market)
    assert all(abs(c.weight - 1.0 / 3) < 1e-9 for c in strategy)

    # Coverage criterion expands to one check per member item.
    assert {c.description for c in market} == {
        "Mentions TikTok",
        "regulatory risk",
        "revenue risk",
        "supply-chain risk",
    }

    # Ordered steps carry their sequence index; others do not.
    ordered = sorted(strategy, key=lambda c: c.order_index)
    assert [c.order_index for c in ordered] == [0, 1, 2]
    assert all(c.order_index is None for c in market)


def test_aggregate_coverage_is_weighted():
    checks = [
        CheckItem(group="g1", kind="atomic", description="a", weight=2.0),
        CheckItem(group="g1", kind="coverage", description="b", weight=2.0),
        CheckItem(group="g2", kind="coverage", description="c", weight=1.0),
    ]
    verdicts = [
        CheckVerdict(description="a", met=True, justification="ok"),
        CheckVerdict(description="b", met=False, justification="missing"),
        CheckVerdict(description="c", met=True, justification="ok"),
    ]

    result = aggregate_coverage(verdicts, checks)

    # met weight 2.0 + 1.0 = 3.0 over total 5.0.
    assert abs(result["factual_completeness_coverage"] - 0.6) < 1e-9
    assert result["checks_total"] == 3 and result["checks_met"] == 2
    assert abs(result["coverage_by_group"]["g1"]["coverage"] - 0.5) < 1e-9
    assert result["coverage_by_group"]["g2"]["coverage"] == 1.0
    assert abs(result["coverage_by_kind"]["coverage"]["coverage"] - 1.0 / 3) < 1e-9


def test_aggregate_coverage_rejects_mismatched_lengths():
    checks = [CheckItem(group="g", kind="atomic", description="a", weight=1.0)]
    verdicts = [
        CheckVerdict(description="a", met=True, justification="ok"),
        CheckVerdict(description="b", met=False, justification="extra"),
    ]
    try:
        aggregate_coverage(verdicts, checks)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched verdict/check counts")


def test_scorer_end_to_end_wires_author_compile_score(monkeypatch):
    """Author -> compile -> score -> aggregate using the repo's judge call path,
    with the structured-output query stubbed to deterministic models."""
    calls = []

    def fake_query(messages, output_class, model, temperature, max_completion_tokens):
        calls.append(output_class)
        if output_class is AuthoredMetaRubric:
            return _sample_meta_rubric()
        if output_class is ChecklistVerdicts:
            # Mark every other check as met so coverage is non-trivial.
            return ChecklistVerdicts(
                verdicts=[
                    CheckVerdict(
                        description="x",
                        met=(i % 2 == 0),
                        justification="stub",
                    )
                    for i in range(7)
                ]
            )
        raise AssertionError(f"unexpected output_class: {output_class}")

    monkeypatch.setattr(
        "evals.meta_rubric_completeness_evals.query_openai_model_structured_outputs",
        fake_query,
    )

    scorer = MetaRubricCompletenessScorer(model=DEFAULT_EVAL_MODEL)
    result = scorer.score(
        question="Evaluate the consequences of TikTok bans on investment risk.",
        baseline_answer="A complete reference report covering risks and mitigations.",
        candidate_answer="A shorter report that misses several risks.",
    )

    # Both authoring and scoring went through the repo's structured-output helper.
    assert calls == [AuthoredMetaRubric, ChecklistVerdicts]
    assert 0.0 <= result["factual_completeness_coverage"] <= 1.0
    assert result["checks_total"] == 7
    assert result["checks_met"] == 4  # indices 0,2,4,6 of 7
    assert result["meta_rubric"]["groups"][0]["name"] == "market"
