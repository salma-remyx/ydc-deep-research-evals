"""Tests for the judge-reliability audit.

These exercise the integration with the existing pairwise pipeline rather
than self-testing the new module in isolation: the audit's bias-probe and
drift helpers consume ``DeepResearchScoreResult`` / aggregate / per-row
structures produced by ``evals.metrics.deep_research_pairwise_metric`` and
``evals.deep_research_pairwise_evals``, and the auditor subclasses the
existing ``DeepResearchEvaluator`` call site. No network calls are made --
the tests build metric-shaped synthetic data offline.
"""

import pandas as pd
import pytest

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.judge_reliability_audit import (
    JudgeSwapAuditor,
    aggregate_drift,
    position_bias_probe,
    protocol_audit,
    verdict_flip_rate,
)
from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)


def _pref(preferred: str, gap: int = 2) -> dict:
    """Build a raw preference dict in the shape the metric records."""
    return {
        "preferred": preferred,
        "gap_score": gap,
        "score_b": -gap if preferred == "a" else gap,
    }


def _score_result(
    original_per_dim: dict,
    flipped_per_dim: dict,
    grades: dict | None = None,
) -> DeepResearchScoreResult:
    """Assemble a DeepResearchScoreResult exactly as the metric would.

    ``preferred`` collects original preferences plus the position-corrected
    flipped preferences (matching the metric's own correction).
    """
    grades = grades or {d: "tie" for d in DIMENSIONS}
    dims = {}
    for d in DIMENSIONS:
        original = original_per_dim[d]
        flipped = flipped_per_dim[d]
        all_pref = [p["preferred"] for p in original] + [
            "a" if p["preferred"] == "b" else "b" for p in flipped
        ]
        dims[d] = DimensionResult(
            grade=grades[d],
            is_win=grades[d] == "win",
            is_tie=grades[d] == "tie",
            is_lose=grades[d] == "lose",
            score=5.0,
            preferred=all_pref,
            raw_preferences={"original": original, "flipped": flipped},
        )
    return DeepResearchScoreResult(**dims)


def test_position_bias_probe_distinguishes_biased_and_unbiased():
    # Unbiased: candidate preferred in both orders -> delta 0, no disagreement.
    unbiased = _score_result(
        {d: [_pref("b"), _pref("b")] for d in DIMENSIONS},
        {d: [_pref("a"), _pref("a")] for d in DIMENSIONS},
    )
    # Position-biased toward slot b: always prefers slot b regardless of content.
    biased = _score_result(
        {d: [_pref("b"), _pref("b")] for d in DIMENSIONS},
        {d: [_pref("b"), _pref("b")] for d in DIMENSIONS},
    )

    clean = position_bias_probe([unbiased])
    dirty = position_bias_probe([biased])

    assert clean["overall"]["mean_abs_delta"] == pytest.approx(0.0)
    assert clean["overall"]["disagreement_rate"] == pytest.approx(0.0)
    assert dirty["overall"]["mean_abs_delta"] == pytest.approx(1.0)
    assert dirty["overall"]["disagreement_rate"] == pytest.approx(1.0)
    for d in DIMENSIONS:
        assert dirty[d]["mean_abs_delta"] == pytest.approx(1.0)
        assert dirty[d]["support"] == 1


def test_aggregate_drift_measures_judge_swap_movement():
    reference = {
        "overall": {"win_rate": 0.4, "avg_score": 5.0, "net_winrate": 0.4},
        "instruction_following": {
            "win_rate": 0.3,
            "avg_score": 4.0,
            "net_winrate": 0.3,
        },
    }
    candidate = {
        "overall": {"win_rate": 0.6, "avg_score": 6.0, "net_winrate": 0.7},
        "instruction_following": {
            "win_rate": 0.5,
            "avg_score": 4.5,
            "net_winrate": 0.5,
        },
    }

    drift = aggregate_drift(reference, candidate, "o3-mini", "gpt-4o")

    assert drift["reference"] == "o3-mini"
    assert drift["candidate"] == "gpt-4o"
    assert drift["overall"]["win_rate_delta"] == pytest.approx(0.2)
    assert drift["overall"]["avg_score_delta"] == pytest.approx(1.0)
    assert drift["overall"]["net_winrate_delta"] == pytest.approx(0.3)
    assert drift["instruction_following"]["avg_score_delta"] == pytest.approx(0.5)


def test_verdict_flip_rate_counts_grade_changes():
    # Reference judge grades everything "tie"; the candidate judge flips
    # instruction_following to "win" on the first row only.
    ref_rows = [
        {"success": True, **{f"{d}_grade": "tie" for d in DIMENSIONS}},
        {"success": True, **{f"{d}_grade": "tie" for d in DIMENSIONS}},
    ]
    cand_rows = [
        {
            "success": True,
            **{
                f"{d}_grade": "win" if d == "instruction_following" else "tie"
                for d in DIMENSIONS
            },
        },
        {"success": True, **{f"{d}_grade": "tie" for d in DIMENSIONS}},
    ]

    rate = verdict_flip_rate(ref_rows, cand_rows)

    assert rate["instruction_following"] == pytest.approx(0.5)
    assert rate["comprehensiveness"] == pytest.approx(0.0)
    assert rate["overall"] == pytest.approx(0.5)
    assert rate["support"] == 2


def test_protocol_audit_counts_success_and_trials():
    sr = _score_result(
        {d: [_pref("b"), _pref("b"), _pref("b")] for d in DIMENSIONS},
        {d: [_pref("a"), _pref("a"), _pref("a")] for d in DIMENSIONS},
    ).model_dump()
    rows = [
        {"success": True, "score_result": sr},
        {"success": True, "score_result": sr},
        {"success": False, "error": "boom"},
    ]

    audit = protocol_audit(rows)

    assert audit["rows_attempted"] == 3
    assert audit["rows_succeeded"] == 2
    assert audit["rows_failed"] == 1
    # 3 original + 3 flipped trials survive per row-dimension -> mean 6.0
    assert audit["mean_trials_collected_per_dimension"] == pytest.approx(6.0)
    assert audit["bias_probe_coverage"] == 2


def test_judge_swap_auditor_is_wired_into_the_call_site():
    # Integration contract: the auditor subclasses the existing evaluator
    # and inherits its call-site machinery, wired for the reference judge.
    auditor = JudgeSwapAuditor(
        judges=["o3-mini-2025-01-31", "gpt-4o-2024-08-06"]
    )

    assert isinstance(auditor, DeepResearchEvaluator)
    assert auditor.judges == ["o3-mini-2025-01-31", "gpt-4o-2024-08-06"]
    assert auditor.model == "o3-mini-2025-01-31"
    assert hasattr(auditor, "evaluate_batch")
    assert hasattr(auditor, "aggregate_results")
    assert auditor.pairwise_metric.eval_model == "o3-mini-2025-01-31"


def test_judge_swap_auditor_requires_at_least_one_judge():
    with pytest.raises(ValueError):
        JudgeSwapAuditor(judges=[])


def test_audit_assembles_report_with_network_stubbed(monkeypatch):
    # Drive the full audit orchestration offline by stubbing the inherited
    # network-backed methods; this proves bias probe + drift + verdict-flip
    # rate are assembled from the existing pipeline's output shapes.
    auditor = JudgeSwapAuditor(judges=["judge-a", "judge-b"])

    grades_a = {d: "tie" for d in DIMENSIONS}
    grades_b = {"instruction_following": "win"}
    grades_b.update({d: "tie" for d in DIMENSIONS if d != "instruction_following"})

    res_a = _score_result(
        {d: [_pref("b"), _pref("b")] for d in DIMENSIONS},
        {d: [_pref("a"), _pref("a")] for d in DIMENSIONS},
        grades=grades_a,
    )
    res_b = _score_result(
        {d: [_pref("b"), _pref("b")] for d in DIMENSIONS},
        {d: [_pref("a"), _pref("a")] for d in DIMENSIONS},
        grades=grades_b,
    )

    def to_row(res):
        dump = res.model_dump()
        row = {"success": True, "score_result": dump}
        for d in DIMENSIONS:
            row[f"{d}_grade"] = dump[d]["grade"]
        return row

    rows = {"judge-a": [to_row(res_a)], "judge-b": [to_row(res_b)]}
    aggregate = {
        "support": 1,
        "overall": {
            "win_rate": 0.0,
            "tie_rate": 1.0,
            "lose_rate": 0.0,
            "avg_score": 5.0,
            "net_winrate": 0.0,
        },
    }
    for d in DIMENSIONS:
        aggregate[d] = aggregate["overall"]

    seen = []

    def fake_batch(self, data):
        seen.append(self.model)
        return rows[self.model]

    def fake_aggregate(self, results):
        return aggregate

    monkeypatch.setattr(JudgeSwapAuditor, "evaluate_batch", fake_batch)
    monkeypatch.setattr(JudgeSwapAuditor, "aggregate_results", fake_aggregate)

    df = pd.DataFrame(
        {"question": ["q"], "baseline_answer": ["a"], "candidate_answer": ["b"]}
    )
    report = auditor.audit(df)

    assert report["reference_judge"] == "judge-a"
    assert report["judges"] == ["judge-a", "judge-b"]
    assert seen == ["judge-a", "judge-b"]  # each judge evaluated exactly once
    assert (
        report["per_judge"]["judge-a"]["bias_probe"]["overall"]["mean_abs_delta"]
        == pytest.approx(0.0)
    )
    drift = report["drift"]["judge-a->judge-b"]
    flip = drift["verdict_flip_rate"]
    assert flip["instruction_following"] == pytest.approx(1.0)
    assert flip["comprehensiveness"] == pytest.approx(0.0)
    assert flip["overall"] == pytest.approx(1.0)
    assert "aggregate_drift" in drift
