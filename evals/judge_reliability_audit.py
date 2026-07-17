"""Judge-reliability audit for the deep-research pairwise pipeline.

Runs the repo's existing pairwise evaluation across multiple LLM judges
and audits the *measurement* consequences of swapping the judge: residual
position bias, score drift, and a protocol audit trail. The pairwise
metric already runs flipped-order trials to mitigate position bias; this
module is the first thing in the repo that quantifies how much bias
*remains* after that mitigation and how much the verdict moves when the
judge changes.

Adapted from "When the Judge Changes, So Does the Measurement: Auditing
LLM-as-Judge Reliability" (arXiv:2607.08535v1).

Mode 3 (inspired experiment): the paper is an audit *study*, not a model
or trainer, so there is nothing to port -- its measurement-validity
framing (bias probes, evaluator-replacement score drift, protocol audit
trails) is applied to this repo's existing flipped-trial pairwise
pipeline. Out of scope: the paper's Qwen3 / MiniMax judge-comparison
benchmark suite (no multi-judge dataset ships here) and its
structured-debate / repeated-sample-jury probes. The position-bias probe
is a parameter-free, target-native analysis of the original-vs-flipped
preference data the metric already records.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from tqdm import tqdm

from evals.deep_research_pairwise_evals import DeepResearchEvaluator
from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
)

_BIAS_METRICS = ["mean_abs_delta", "mean_signed_delta", "disagreement_rate"]
_AGG_METRICS = ["win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"]


def position_bias_probe(scores_list: List[DeepResearchScoreResult]) -> Dict[str, Any]:
    """Quantify residual position bias from original-vs-flipped preferences.

    The metric scores each row in the original order (baseline=a,
    candidate=b) and a flipped order, recording per-trial preferences in
    ``raw_preferences``. An unbiased judge prefers the candidate at the same
    rate regardless of position, so the gap between the candidate-preference
    rate in the original order and the (position-corrected) flipped order is
    a direct measure of residual position bias -- the paper's "bias probe".

    Returns per-dimension and overall: ``mean_abs_delta`` (mean |bias|),
    ``mean_signed_delta`` (>0 favors position b), ``disagreement_rate``
    (rows whose verdict flips with order), and ``support``.
    """
    probe: Dict[str, Any] = {"support": len(scores_list)}
    for dimension in DIMENSIONS:
        abs_deltas: List[float] = []
        signed_deltas: List[float] = []
        disagreements = 0
        usable = 0
        for score_result in scores_list:
            raw = getattr(score_result, dimension).raw_preferences or {}
            original = raw.get("original") or []
            flipped = raw.get("flipped") or []
            if not original or not flipped:
                continue
            usable += 1
            # Original order: candidate preferred iff "b". Flipped order:
            # candidate sits in position a, so preferred iff "a".
            p_orig = sum(p.get("preferred") == "b" for p in original) / len(original)
            p_flip = sum(p.get("preferred") == "a" for p in flipped) / len(flipped)
            delta = p_orig - p_flip
            signed_deltas.append(delta)
            abs_deltas.append(abs(delta))
            if (p_orig > 0.5) != (p_flip > 0.5):
                disagreements += 1
        probe[dimension] = {
            "mean_abs_delta": sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0,
            "mean_signed_delta": sum(signed_deltas) / len(signed_deltas)
            if signed_deltas
            else 0.0,
            "disagreement_rate": disagreements / usable if usable else 0.0,
            "support": usable,
        }
    _fill_overall(probe)
    return probe


def _fill_overall(probe: Dict[str, Any]) -> None:
    """Average the per-dimension bias metrics into an ``overall`` block."""
    probe["overall"] = {}
    for metric in _BIAS_METRICS:
        values = [
            probe[d][metric]
            for d in DIMENSIONS
            if isinstance(probe.get(d, {}).get(metric), (int, float))
        ]
        probe["overall"][metric] = sum(values) / len(values) if values else 0.0


def aggregate_drift(
    reference: Dict[str, Any],
    candidate: Dict[str, Any],
    ref_name: str = "reference",
    cand_name: str = "candidate",
) -> Dict[str, Any]:
    """Measure evaluator-replacement ambiguity between two judges.

    Given two ``aggregate()`` outputs from running the *same* data under two
    judges, report the per-dimension and overall change in each headline
    metric -- the paper's central observation that a score can move even
    though the candidate responses are fixed.
    """
    drift: Dict[str, Any] = {"reference": ref_name, "candidate": cand_name}
    for group in list(DIMENSIONS) + ["overall"]:
        ref_block = reference.get(group, {}) if isinstance(reference, dict) else {}
        cand_block = candidate.get(group, {}) if isinstance(candidate, dict) else {}
        entry = {
            f"{k}_delta": cand_block[k] - ref_block[k]
            for k in _AGG_METRICS
            if k in ref_block and k in cand_block
        }
        if entry:
            drift[group] = entry
    return drift


def verdict_flip_rate(
    reference_results: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fraction of rows whose dimension grade changes across a judge swap.

    Operates on per-row result dicts from ``evaluate_single`` (which carry
    ``<dim>_grade`` keys), aligned by index. A high flip rate means the swap
    moved real decisions, not just score magnitudes.
    """
    n = min(len(reference_results), len(candidate_results))
    flips = {d: 0 for d in DIMENSIONS}
    any_flip = 0
    for i in range(n):
        ref_row, cand_row = reference_results[i], candidate_results[i]
        row_any = False
        for dimension in DIMENSIONS:
            key = f"{dimension}_grade"
            if ref_row.get(key) != cand_row.get(key):
                flips[dimension] += 1
                row_any = True
        if row_any:
            any_flip += 1
    rate: Dict[str, Any] = {d: (flips[d] / n if n else 0.0) for d in DIMENSIONS}
    rate["overall"] = (any_flip / n) if n else 0.0
    rate["support"] = n
    return rate


def protocol_audit(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a protocol audit trail for one judge's run.

    Records throughput, success/failure counts, and how many preference
    trials survived per row-dimension (failed trials silently shrink the
    bias-probe sample). The paper argues reports need these trails so score
    shifts can be attributed to the protocol rather than the data.
    """
    attempted = len(results)
    succeeded = sum(1 for r in results if r.get("success", False))
    trial_counts: List[int] = []
    for r in results:
        score_result = r.get("score_result")
        if not score_result:
            continue
        for dimension in DIMENSIONS:
            raw = (score_result.get(dimension, {}) or {}).get("raw_preferences", {}) or {}
            collected = len(raw.get("original") or []) + len(raw.get("flipped") or [])
            if collected:
                trial_counts.append(collected)
    return {
        "rows_attempted": attempted,
        "rows_succeeded": succeeded,
        "rows_failed": attempted - succeeded,
        "mean_trials_collected_per_dimension": sum(trial_counts) / len(trial_counts)
        if trial_counts
        else 0.0,
        "bias_probe_coverage": succeeded,
    }


class JudgeSwapAuditor(DeepResearchEvaluator):
    """Run the existing pairwise eval across multiple judges and audit it.

    Subclasses ``DeepResearchEvaluator`` and reuses its ``evaluate_batch`` /
    ``aggregate_results`` for every judge (re-pointing ``self.model`` and
    ``self.pairwise_metric`` per judge), then layers the paper's audit on
    top: a per-judge position-bias probe, a protocol audit trail, and the
    score drift + verdict-flip rate between the reference judge (first in
    ``judges``) and each other judge.
    """

    def __init__(
        self,
        judges: Sequence[str],
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
    ):
        if not judges:
            raise ValueError("judges must contain at least one model name")
        # Initialise the parent against the reference judge so the inherited
        # evaluate_batch / aggregate_results are usable as-is.
        super().__init__(
            model=judges[0],
            output_path=output_path,
            num_workers=num_workers,
            metric_num_workers=metric_num_workers,
            metric_num_trials=metric_num_trials,
        )
        self.judges: List[str] = list(judges)
        self.metric_num_workers = metric_num_workers

    def _run_one_judge(self, judge: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate ``data`` under one judge via the inherited parent path."""
        self.model = judge
        self.pairwise_metric = DeepResearchPairwiseMetric(
            eval_model=judge,
            num_trials=self.metric_num_trials,
            num_workers=self.metric_num_workers,
        )
        results = self.evaluate_batch(data)
        return {
            "judge": judge,
            "results": results,
            "aggregate": self.aggregate_results(results),
            "bias_probe": position_bias_probe(self._score_results(results)),
            "protocol_audit": protocol_audit(results),
        }

    @staticmethod
    def _score_results(
        results: List[Dict[str, Any]],
    ) -> List[DeepResearchScoreResult]:
        parsed: List[DeepResearchScoreResult] = []
        for r in results:
            sr = r.get("score_result")
            if not sr:
                continue
            try:
                parsed.append(DeepResearchScoreResult.model_validate(sr))
            except Exception as exc:  # keep auditing past one bad row
                print(f"Skipping un-parseable score result: {exc}")
        return parsed

    def audit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run every judge over ``data`` and assemble the reliability report."""
        per_judge: Dict[str, Dict[str, Any]] = {}
        for judge in tqdm(self.judges, desc="Judges"):
            per_judge[judge] = self._run_one_judge(judge, data)

        reference = self.judges[0]
        report: Dict[str, Any] = {
            "judges": self.judges,
            "reference_judge": reference,
            "support": len(data),
            "per_judge": {
                j: {
                    "aggregate": per_judge[j]["aggregate"],
                    "bias_probe": per_judge[j]["bias_probe"],
                    "protocol_audit": per_judge[j]["protocol_audit"],
                }
                for j in self.judges
            },
            "drift": {},
        }
        ref_agg = per_judge[reference]["aggregate"]
        ref_results = per_judge[reference]["results"]
        for judge in self.judges[1:]:
            report["drift"][f"{reference}->{judge}"] = {
                "aggregate_drift": aggregate_drift(
                    ref_agg, per_judge[judge]["aggregate"], reference, judge
                ),
                "verdict_flip_rate": verdict_flip_rate(
                    ref_results, per_judge[judge]["results"]
                ),
            }
        return report


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit LLM-as-judge reliability: run the pairwise eval across "
            "multiple judges and report position bias, score drift, and a "
            "protocol audit trail."
        )
    )
    parser.add_argument(
        "--input-data",
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Input CSV with 'question', 'baseline_answer', 'candidate_answer'.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the report")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=[DEFAULT_EVAL_MODEL],
        help="Judge models to compare; the first is the reference judge.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-num-workers", type=int, default=1)
    parser.add_argument("--metric-num-trials", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    auditor = JudgeSwapAuditor(
        judges=args.judges,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    print(f"Auditing judges: {', '.join(args.judges)}")
    report = auditor.audit(df)

    report_path = output_dir / "judge_reliability_audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Audit report saved to {report_path}")

    print("\nPer-judge residual position bias (overall):")
    for judge, block in report["per_judge"].items():
        overall = block["bias_probe"].get("overall", {})
        print(
            f"  {judge}: mean|delta|={overall.get('mean_abs_delta', 0.0):.4f} "
            f"disagreement={overall.get('disagreement_rate', 0.0):.4f}"
        )

    for pair, drift in report["drift"].items():
        net = drift["aggregate_drift"].get("overall", {}).get("net_winrate_delta", 0.0)
        flip = drift["verdict_flip_rate"].get("overall", 0.0)
        print(f"\nDrift {pair}: overall net_winrate_delta={net:.4f}, verdict_flip_rate={flip:.4f}")


if __name__ == "__main__":
    main()
