"""Calibration & sensitivity audit for the pairwise deep-research judge.

Adapted (Mode 2) from "LLM Judges Can Be Too Generous When There Is No
Reference Answer" (arxiv:2607.12885). The paper runs a two-stage audit of an
LLM-as-a-judge -- (a) *calibration*: does the judge over-credit answers, and
(b) *sensitivity*: do its decisions flip when the reference answer is added or
repositioned -- and concludes a judge must be calibrated against a
reference-aware sample before it can be trusted in reference-free settings.

This module keeps that two-stage audit shape but feeds it signals the repo
already emits, so the judge is never re-queried:

* Sensitivity -> the metric's own position-debiasing probe. Every scored row
  already runs the judge in *original* (baseline=A, candidate=B) and *flipped*
  (candidate=A, baseline=B) order. The disagreement between those two views is
  the position-sensitivity signal -- the on-repo analog of the paper's
  "reference positioning flips decisions by up to 85%".
* Calibration -> a gap-score / grade consistency check on the emitted
  per-trial preferences. A decisive win/loss awarded on a near-zero reported
  quality gap (or a near-even trial split) is the on-repo analog of the paper's
  "over-credit incorrect answers" finding.

Substitutions vs. the paper: (1) reference-presence re-querying is replaced by
the emitted original/flipped trial pair; (2) human-annotation alignment is cut
(no human labels ship with the repo -- a downstream PR); (3) the multilingual
benchmark and the reported 85% figure are the paper's empirical results, not
the method, and are intentionally not reproduced here. This module is the
*measurement* tool; the input contract matches
``DeepResearchPairwiseMetric.aggregate`` (``List[DeepResearchScoreResult]``).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evals.metrics.deep_research_pairwise_metric import (
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)


def _candidate_wins_original(prefs: List[Dict[str, Any]]) -> List[bool]:
    """Candidate is report_b in original order -> wins when preferred == 'b'."""
    return [p.get("preferred") == "b" for p in prefs]


def _candidate_wins_flipped(prefs: List[Dict[str, Any]]) -> List[bool]:
    """Candidate is report_a in flipped order -> wins when preferred == 'a'."""
    return [p.get("preferred") == "a" for p in prefs]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _gap_magnitudes(prefs: List[Dict[str, Any]]) -> List[float]:
    return [abs(float(p.get("gap_score", 0))) for p in prefs]


def _dimension_audit(
    dim_results: List[DimensionResult],
    *,
    generosity_gap_threshold: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    flip_rates: List[float] = []
    gap_scores: List[float] = []
    low_margin_decisive: List[bool] = []
    over_credited_wins = 0
    total_wins = 0
    grades: Dict[str, int] = {"win": 0, "tie": 0, "lose": 0}

    for res in dim_results:
        raw = res.raw_preferences or {}
        orig = raw.get("original", []) or []
        flipped = raw.get("flipped", []) or []

        wins_o = _candidate_wins_original(orig)
        wins_f = _candidate_wins_flipped(flipped)
        if wins_o and wins_f:
            p_orig = _mean(wins_o)
            p_flip = _mean(wins_f)
            # Analytic probability that a decision flips purely due to position.
            flip_rates.append(p_orig * (1 - p_flip) + (1 - p_orig) * p_flip)

        gaps = _gap_magnitudes(orig) + _gap_magnitudes(flipped)
        if gaps:
            gap_scores.append(_mean(gaps))

        grades[res.grade] = grades.get(res.grade, 0) + 1

        preferred = list(res.preferred or [])
        decisive = res.grade in ("win", "lose")
        if decisive and preferred:
            frac_candidate = _mean(p == "b" for p in preferred)
            margin = abs(frac_candidate - 0.5) * 2.0
            low_margin_decisive.append(margin < margin_threshold)

        if res.grade == "win":
            total_wins += 1
            mean_gap = _mean(gaps) if gaps else 0.0
            if mean_gap <= generosity_gap_threshold:
                over_credited_wins += 1

    position_flip_rate = _mean(flip_rates)
    return {
        "support": len(dim_results),
        "position_flip_rate": position_flip_rate,
        "decision_stability_rate": 1.0 - position_flip_rate,
        "mean_gap_score": _mean(gap_scores) if gap_scores else 0.0,
        "low_margin_decisive_rate": _mean(low_margin_decisive),
        "over_credited_win_rate": (
            over_credited_wins / total_wins if total_wins else 0.0
        ),
        "grade_counts": grades,
    }


def audit(
    scores_list: List[DeepResearchScoreResult],
    *,
    generosity_gap_threshold: float = 1.0,
    margin_threshold: float = 0.34,
) -> Dict[str, Any]:
    """Run the calibration + sensitivity audit over scored rows.

    Args mirror the contract of ``DeepResearchPairwiseMetric.aggregate``: a
    list of per-row ``DeepResearchScoreResult`` objects. Returns per-dimension
    and overall sensitivity (position-flip) and calibration (over-generosity)
    rates, plus a deterministic recommendation in the spirit of the paper.
    """
    if not scores_list:
        return {"support": 0, "error": "No scored rows to audit"}

    report: Dict[str, Any] = {"support": len(scores_list), "dimensions": {}}
    for dimension in DIMENSIONS:
        dim_results = [getattr(score, dimension) for score in scores_list]
        report["dimensions"][dimension] = _dimension_audit(
            dim_results,
            generosity_gap_threshold=generosity_gap_threshold,
            margin_threshold=margin_threshold,
        )

    dims = report["dimensions"].values()
    report["overall"] = {
        "position_flip_rate": _mean(d["position_flip_rate"] for d in dims),
        "over_credited_win_rate": _mean(d["over_credited_win_rate"] for d in dims),
        "low_margin_decisive_rate": _mean(d["low_margin_decisive_rate"] for d in dims),
        "mean_gap_score": _mean(d["mean_gap_score"] for d in dims),
    }
    report["recommendation"] = _recommendation(report["overall"])
    return report


def _recommendation(overall: Dict[str, Any]) -> str:
    flip = overall.get("position_flip_rate", 0.0)
    over = overall.get("over_credited_win_rate", 0.0)
    notes: List[str] = []
    if flip >= 0.3:
        notes.append(
            f"position_flip_rate={flip:.2f} is high: judge decisions are "
            "sensitive to answer order. Per the paper, calibrate against a "
            "reference-aware sample before trusting these reference-free "
            "judgments."
        )
    if over >= 0.3:
        notes.append(
            f"over_credited_win_rate={over:.2f} is high: the judge awards wins "
            "on near-zero quality gaps (over-generous). Treat decisive wins with "
            "mean_gap_score~0 as unreliable."
        )
    if not notes:
        notes.append(
            "No strong calibration/sensitivity red flags: the judge's decisions "
            "are largely position-stable and gap-consistent."
        )
    return " ".join(notes)


def audit_results_jsonl(path: Path, **audit_kwargs: Any) -> Dict[str, Any]:
    """Audit a results JSONL produced by ``evals.deep_research_pairwise_evals``.

    Each line is a row dict with a ``score_result`` field (the ``model_dump``
    of a ``DeepResearchScoreResult``). Failed / unparseable rows are skipped.
    """
    scores: List[DeepResearchScoreResult] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("success", False) or "score_result" not in row:
                continue
            try:
                scores.append(
                    DeepResearchScoreResult.model_validate(row["score_result"])
                )
            except Exception as exc:
                print(f"Skipping unparseable row: {exc}")
    return audit(scores, **audit_kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pairwise judge for calibration (over-generosity) and "
            "sensitivity (position-flip). Adapted from arxiv:2607.12885."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help=(
            "Path to a results JSONL from "
            "`python -m evals.deep_research_pairwise_evals`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the audit report as JSON.",
    )
    parser.add_argument(
        "--generosity-gap-threshold",
        type=float,
        default=1.0,
        help=(
            "A candidate 'win' whose mean per-trial gap score is at or below "
            "this is flagged as over-credited."
        ),
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.34,
        help=(
            "Decisive (win/lose) rows whose trial vote split is within this "
            "margin of even are flagged as low-confidence."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_results_jsonl(
        args.results,
        generosity_gap_threshold=args.generosity_gap_threshold,
        margin_threshold=args.margin_threshold,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
