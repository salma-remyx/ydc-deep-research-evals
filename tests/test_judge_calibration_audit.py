"""Integration tests for the judge calibration & sensitivity audit.

These build ``DeepResearchScoreResult`` objects from the *existing*
``evals.metrics.deep_research_pairwise_metric`` module (the call-site contract)
and feed them to the new ``evals.metrics.judge_calibration_audit`` module,
proving the audit consumes the pipeline's real data structures. No network is
used: the audit operates only on already-emitted per-trial preferences.
"""

import json
import os

# evals.utils constructs the OpenAI client at import time; the audit is fully
# offline, so dummy credentials are enough for the import to succeed.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.metrics.judge_calibration_audit import (  # noqa: E402
    audit,
    audit_results_jsonl,
)


def _pref(preferred: str, gap: int) -> dict:
    """A Preference.model_dump()-shaped dict (preferred + gap_score + score_b)."""
    return {
        "preferred": preferred,
        "gap_score": gap,
        "score_b": gap if preferred == "b" else -gap,
        "explanation": "",
    }


def _dim(grade: str, original: list, flipped: list) -> DimensionResult:
    """Build a DimensionResult whose preferred list is canonicalized exactly as
    DeepResearchPairwiseMetric._get_pairwise_preference does (original as-is,
    flipped inverted, so "b" always denotes the candidate)."""
    preferred = [p["preferred"] for p in original] + [
        "a" if p["preferred"] == "b" else "b" for p in flipped
    ]
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=5.0,
        preferred=preferred,
        raw_preferences={"original": original, "flipped": flipped},
    )


def _result(grade: str, original: list, flipped: list) -> DeepResearchScoreResult:
    """A full DeepResearchScoreResult with the same dimension repeated."""
    dimension = _dim(grade, original, flipped)
    return DeepResearchScoreResult(**{name: dimension for name in DIMENSIONS})


def test_position_stable_decisive_win_has_zero_flip_rate():
    # Candidate consistently wins in BOTH orderings -> no position sensitivity.
    result = _result(
        "win",
        original=[_pref("b", 3), _pref("b", 3), _pref("b", 3)],
        flipped=[_pref("a", 3), _pref("a", 3), _pref("a", 3)],
    )
    report = audit([result])
    dim = report["dimensions"]["comprehensiveness"]
    assert dim["position_flip_rate"] == 0.0
    assert dim["decision_stability_rate"] == 1.0
    # gap of 3 is well above the generosity threshold -> not over-credited.
    assert dim["over_credited_win_rate"] == 0.0
    assert "position-stable" in report["recommendation"]


def test_pure_position_bias_maximizes_flip_rate():
    # Judge always prefers report_b regardless of who is in it -> pure position
    # bias. Original: candidate is b -> wins. Flipped: candidate is a -> loses.
    result = _result(
        "tie",
        original=[_pref("b", 2), _pref("b", 2), _pref("b", 2)],
        flipped=[_pref("b", 2), _pref("b", 2), _pref("b", 2)],
    )
    report = audit([result])
    dim = report["dimensions"]["instruction_following"]
    assert dim["position_flip_rate"] == 1.0
    assert "sensitive to answer order" in report["recommendation"]


def test_over_generous_win_on_zero_gap_is_flagged():
    # A decisive 'win' awarded despite a zero quality gap -> over-credited.
    result = _result(
        "win",
        original=[_pref("b", 0), _pref("b", 0)],
        flipped=[_pref("a", 0), _pref("a", 0)],
    )
    report = audit([result])
    dim = report["dimensions"]["completeness"]
    assert dim["over_credited_win_rate"] == 1.0
    assert "over-generous" in report["recommendation"]


def test_empty_input_is_handled():
    report = audit([])
    assert report["support"] == 0
    assert "error" in report


def test_audit_results_jsonl_consumes_pipeline_output(tmp_path):
    """The sibling CLI reads the exact JSONL format the pairwise eval writes:
    one row dict per line, each carrying a `score_result` field that is the
    model_dump() of a DeepResearchScoreResult."""
    good = _result(
        "win",
        original=[_pref("b", 3), _pref("b", 3)],
        flipped=[_pref("a", 3), _pref("a", 3)],
    )
    rows = [
        {"success": True, "question": "q1", "score_result": good.model_dump()},
        {"success": False, "error": "boom"},  # skipped
    ]
    path = tmp_path / "deep_research_results_test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    report = audit_results_jsonl(path)
    assert report["support"] == 1  # only the successful row was audited
    assert report["dimensions"]["writing_quality"]["position_flip_rate"] == 0.0
