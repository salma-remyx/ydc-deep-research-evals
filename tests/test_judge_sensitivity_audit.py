"""Integration tests for evals.judge_sensitivity_audit.

These tests import the *existing* pairwise metric contract
(``DeepResearchScoreResult`` / ``DimensionResult`` / ``DIMENSIONS`` from
``evals.metrics.deep_research_pairwise_metric``) and exercise the new audit on
real ``DimensionResult.preferred`` vote vectors -- proving the new module
consumes the existing pipeline's emitted contract correctly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the repo root importable regardless of how pytest is invoked, and provide
# dummy OpenAI credentials so importing evals.utils (which constructs a client at
# import time) does not fail in CI without real credentials.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_ORGANIZATION_ID", "test-org")

from evals.judge_sensitivity_audit import (  # noqa: E402
    JudgeRun,
    JudgeSensitivityAudit,
    infer_provider,
    main,
)
from evals.metrics.deep_research_pairwise_metric import (  # noqa: E402
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)

GRADE_PREFS = {
    "win": ["b", "b", "b", "b", "b", "b"],
    "lose": ["a", "a", "a", "a", "a", "a"],
    "tie": ["a", "a", "a", "b", "b", "b"],
}


def _dim(preferred, grade=None):
    b = preferred.count("b")
    a = preferred.count("a")
    if grade is None:
        grade = "win" if b > a else ("lose" if a > b else "tie")
    return DimensionResult(
        grade=grade,
        is_win=grade == "win",
        is_tie=grade == "tie",
        is_lose=grade == "lose",
        score=0.0,
        preferred=list(preferred),
        raw_preferences={},
    )


def _score(per_dim):
    """per_dim: {dimension: preferred_list} (grade derived from the votes)."""
    return DeepResearchScoreResult(**{dim: _dim(per_dim[dim]) for dim in DIMENSIONS})


def _run_from_grades(name, row_grades):
    return JudgeRun(
        name=name,
        provider=infer_provider(name),
        scores=[
            _score({dim: GRADE_PREFS[g] for dim in DIMENSIONS}) for g in row_grades
        ],
    )


# --------------------------------------------------------------------------- #
# Provider inference.
# --------------------------------------------------------------------------- #


def test_infer_provider_maps_paper_panel():
    assert infer_provider("gpt-5.5") == "openai"
    assert infer_provider("o3-mini-2025-01-31") == "openai"
    assert infer_provider("claude-opus-4.8") == "anthropic"
    assert infer_provider("gemini-3.5-flash") == "google"
    assert infer_provider("grok-4.3") == "xai"
    assert infer_provider("mystery-model") == "unknown"


# --------------------------------------------------------------------------- #
# Inter-judge agreement (Fleiss' kappa) + judge-vs-consensus (Cohen's kappa).
# --------------------------------------------------------------------------- #


def test_fleiss_kappa_perfect_agreement_is_one():
    grades = ["win", "lose", "win", "tie", "lose", "win"]
    runs = [
        _run_from_grades("gpt-5", grades),
        _run_from_grades("claude-opus", grades),
        _run_from_grades("gemini-flash", grades),
    ]
    audit = JudgeSensitivityAudit(runs, n_perm=1, n_bootstrap=1, seed=0)
    for dim in DIMENSIONS:
        assert audit.fleiss_kappa(dim) == 1.0


def test_fleiss_kappa_partial_agreement_below_one():
    runs = [
        _run_from_grades("gpt-5", ["win", "win", "lose", "tie"]),
        _run_from_grades("claude-opus", ["win", "lose", "lose", "tie"]),
        _run_from_grades("gemini-flash", ["win", "win", "win", "tie"]),
    ]
    audit = JudgeSensitivityAudit(runs, n_perm=1, n_bootstrap=1, seed=0)
    k = audit.fleiss_kappa("instruction_following")
    assert k is not None
    assert 0.0 <= k < 1.0


def test_judge_vs_consensus_perfect_agreement_is_one():
    grades = ["win", "lose", "win", "tie", "lose", "win"]
    runs = [
        _run_from_grades("gpt-5", grades),
        _run_from_grades("claude-opus", grades),
        _run_from_grades("gemini-flash", grades),
    ]
    audit = JudgeSensitivityAudit(runs, n_perm=1, n_bootstrap=1, seed=0)
    agreement = audit.judge_vs_consensus_agreement()
    assert agreement["mean"] == 1.0
    for dim in DIMENSIONS:
        for judge_kappa in agreement["per_dimension"][dim]["per_judge"].values():
            assert judge_kappa == 1.0


# --------------------------------------------------------------------------- #
# Consensus rates + confidence intervals.
# --------------------------------------------------------------------------- #


def test_consensus_rates_and_wilson_cis_bracket_proportions():
    grades = ["win", "lose", "win", "tie", "lose", "win", "win", "lose"]
    runs = [
        _run_from_grades("gpt-5", grades),
        _run_from_grades("claude-opus", grades),
    ]
    audit = JudgeSensitivityAudit(runs, n_perm=1, n_bootstrap=50, seed=0)
    result = audit.consensus_with_cis("completeness")
    rates = result["rates"]
    assert abs(sum(rates.values()) - 1.0) < 1e-9
    for grade in ("win", "tie", "lose"):
        lo, hi = result["wilson"][grade]
        assert lo is not None and hi is not None
        assert lo <= rates[grade] <= hi
        blo, bhi = result["bootstrap"][grade]
        assert blo is not None and bhi is not None
        assert blo <= bhi


def test_leniency_favor_rates_in_unit_interval():
    # gpt-5 favors candidate (all win); claude-opus favors baseline (all lose).
    runs = [
        _run_from_grades("gpt-5", ["win"] * 5),
        _run_from_grades("claude-opus", ["lose"] * 5),
    ]
    audit = JudgeSensitivityAudit(runs, n_perm=1, n_bootstrap=1, seed=0)
    leniency = audit.leniency("writing_quality")
    assert leniency["gpt-5"]["favor_rate"] == 1.0
    assert leniency["claude-opus"]["favor_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Same-provider self-preference (leniency-adjusted permutation test).
# --------------------------------------------------------------------------- #


def test_same_provider_not_identifiable_single_candidate():
    # One candidate provider: same-provider label is constant within each judge,
    # so it is confounded with judge identity and cannot be isolated.
    runs = [
        _run_from_grades("gpt-5", ["tie"] * 6),
        _run_from_grades("claude-opus", ["tie"] * 6),
        _run_from_grades("gemini-flash", ["tie"] * 6),
    ]
    audit = JudgeSensitivityAudit(runs, candidate_provider="openai", n_perm=19, seed=0)
    sp = audit.same_provider_effect("instruction_following")
    assert sp["identifiable"] is False


def test_same_provider_detected_in_candidate_panel():
    # Multi-candidate panel: candidate provider varies by row, and each judge
    # systematically favors the candidate when providers match (same-provider
    # bias). The residualized permutation test should flag a strong effect.
    judges = [("gpt-5", "openai"), ("claude-opus", "anthropic"), ("gemini-flash", "google")]
    n_half = 10
    candidate_providers = ["openai"] * n_half + ["anthropic"] * n_half
    runs = []
    for jname, jprov in judges:
        scores = []
        for cprov in candidate_providers:
            same = jprov == cprov
            vec = ["b"] * 6 if same else ["a"] * 6
            scores.append(_score({dim: vec for dim in DIMENSIONS}))
        runs.append(JudgeRun(name=jname, provider=jprov, scores=scores))
    audit = JudgeSensitivityAudit(
        runs, candidate_providers=candidate_providers, n_perm=499, seed=0
    )
    sp = audit.same_provider_effect(None)  # pooled across dimensions
    assert sp["identifiable"] is True
    assert sp["same_provider_advantage"] > 0.5
    assert sp["p_value"] < 0.05


# --------------------------------------------------------------------------- #
# Full audit shape + CLI wiring (rehydrates existing JSONL contract).
# --------------------------------------------------------------------------- #


def test_audit_report_well_formed():
    runs = [
        _run_from_grades("gpt-5", ["win", "lose", "tie"]),
        _run_from_grades("claude-opus", ["win", "win", "tie"]),
    ]
    audit = JudgeSensitivityAudit(
        runs, candidate_provider="openai", n_perm=19, n_bootstrap=20, seed=0
    )
    report = audit.audit()
    assert report["config"]["n_judges"] == 2
    assert report["config"]["n_rows"] == 3
    assert set(report["per_dimension"]) == set(DIMENSIONS)
    assert "same_provider_effect_pooled" in report["overall"]


def test_cli_consumes_existing_jsonl_outputs(tmp_path):
    # Build two judge JSONL files in the exact shape deep_research_pairwise_evals
    # emits (a top-level "score_result" per line), then run the CLI end-to-end.
    judges = [("gpt-5", ["win", "lose", "tie"]), ("claude-opus", ["win", "win", "tie"])]
    paths = []
    for jname, grades in judges:
        run = _run_from_grades(jname, grades)
        path = tmp_path / f"deep_research_results_{jname}.jsonl"
        with open(path, "w") as f:
            for sr in run.scores:
                f.write(json.dumps({"question": "q", "score_result": sr.model_dump()}) + "\n")
        paths.append(str(path))

    report = main(
        [
            "--judge-results", *paths,
            "--judge-names", "gpt-5", "claude-opus",
            "--candidate-provider", "openai",
            "--output-dir", str(tmp_path),
            "--n-perm", "19",
            "--n-bootstrap", "20",
        ]
    )
    assert report["config"]["n_judges"] == 2
    assert report["config"]["n_rows"] == 3
    written = json.loads((tmp_path / "judge_sensitivity_audit.json").read_text())
    assert set(written["per_dimension"]) == set(DIMENSIONS)
