"""Rubric-based scoring for deep research reports.

Scores a candidate research report against an atomic, weighted, MECE rubric of
pass/fail criteria and breaks pass rates down by criterion *weight*. This
surfaces a diagnostic the repo's aggregate pairwise dimension scores hide:
systematic failure on the highest-weighted (critical) criteria. A report can
look strong on average while quietly missing every weight-5 requirement.

Adapted (Mode 2 port) from "A rubric-based controlled comparison of frontier
language models on expert-authored clinical reasoning tasks"
(arXiv:2607.02175v1). The paper's core mechanism — an atomic, weighted, MECE
rubric graded criterion-by-criterion and aggregated by weight tier — is kept at
full fidelity. Two auxiliary components are substituted with target-native
equivalents: expert-authored clinical rubrics -> an LLM-generated rubric derived
from the DeepConsult ``baseline_answer`` (no clinician required); and the
paper's three-frontier-model comparison -> the repo's existing single
``o3-mini`` judge path. The paper's standalone benchmark dataset is out of
scope; this slots into the existing ``(question, baseline_answer,
candidate_answer) -> score`` forward path used by ``DeepResearchPairwiseMetric``.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from retry import retry
from tqdm import tqdm

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DIMENSIONS,
)
from evals.utils import query_openai_model_structured_outputs

# Per the paper: integer importance weights, 1 (low-stakes) .. 5 (critical).
MIN_WEIGHT = 1
MAX_WEIGHT = 5
DEFAULT_NUM_CRITERIA = 25

DimensionName = Literal[
    "instruction_following",
    "comprehensiveness",
    "completeness",
    "writing_quality",
]


class RubricCriterion(BaseModel):
    """One atomic, weighted, pass/fail rubric criterion."""

    dimension: DimensionName = Field(
        description="DeepResearch dimension: instruction_following, "
        "comprehensiveness, completeness, or writing_quality."
    )
    weight: int = Field(
        description="Importance from 1 (nice-to-have) to 5 (critical).",
        ge=MIN_WEIGHT,
        le=MAX_WEIGHT,
    )
    criterion: str = Field(
        description="A single atomic, independently checkable requirement "
        "the report should satisfy; gradable met/not-met on its own."
    )


class Rubric(BaseModel):
    """An atomic, weighted, MECE rubric for one research question."""

    criteria: List[RubricCriterion] = Field(default_factory=list)


class RubricCriterionJudgment(BaseModel):
    """The judge's met/not-met verdict on a single rubric criterion."""

    dimension: DimensionName
    weight: int
    criterion: str
    met: bool
    explanation: str = Field(description="One sentence citing the evidence.")


class RubricJudgmentOutput(BaseModel):
    """Structured output the judge fills in for every rubric criterion."""

    judgments: List[RubricCriterionJudgment]


class RubricScoreResult(BaseModel):
    """Complete rubric scoring result for one (question, candidate) pair."""

    question: str
    criteria: List[RubricCriterionJudgment]

    @property
    def mean_pass_rate(self) -> float:
        if not self.criteria:
            return 0.0
        return sum(1 for c in self.criteria if c.met) / len(self.criteria)

    @property
    def weighted_pass_rate(self) -> float:
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight == 0:
            return 0.0
        passed_weight = sum(c.weight for c in self.criteria if c.met)
        return passed_weight / total_weight


RUBRIC_GENERATION_PROMPT = """\
You are an expert evaluator authoring a grading rubric for a deep research \
report. Given the research question and a reference answer, produce an atomic, \
weighted, MECE rubric of pass/fail criteria that any report answering this \
question should satisfy. The reference answer is a guide to what a strong \
report covers, not a script to copy.

Rules:
- ATOMIC: each criterion is a single, independently checkable requirement a \
grader can mark met or not-met on its own.
- MECE: mutually exclusive and collectively exhaustive; together the criteria \
fully characterize a high-quality answer with no overlap.
- WEIGHT 1-5: 1 = nice-to-have, 5 = critical (failure would make the report \
unacceptable regardless of other strengths). Reserve weight 5 for requirements \
of that severity.
- Spread criteria across all four dimensions (instruction_following, \
comprehensiveness, completeness, writing_quality), weighting the distribution \
toward the substance dimensions (comprehensiveness, completeness).
- Aim for roughly {num_criteria} criteria.
"""


RUBRIC_JUDGING_PROMPT = """\
You are a strict but fair grader. For each rubric criterion, decide whether the \
candidate report SATISFIES it (met=true) or NOT (met=false), based ONLY on the \
candidate report's actual content. A criterion is met only if the report \
concretely satisfies it; do not give credit for vague gestures. Critical \
(weight-5) criteria must be satisfied in full, not partially. Return exactly \
one judgment per criterion, preserving each criterion's dimension, weight, and \
text, with a one-sentence explanation citing the evidence.
"""


class RubricScoringMetric:
    """Rubric-based scoring metric with a weight-tier pass-rate breakdown."""

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_criteria: int = DEFAULT_NUM_CRITERIA,
        num_workers: int = 4,
    ):
        self.eval_model = eval_model
        self.num_criteria = num_criteria
        self.num_workers = num_workers

    @retry(tries=3, delay=1, backoff=2)
    def generate_rubric(self, question: str, baseline_answer: str) -> Rubric:
        """Generate an atomic weighted MECE rubric from the reference answer."""
        messages = [
            {
                "role": "system",
                "content": RUBRIC_GENERATION_PROMPT.format(
                    num_criteria=self.num_criteria
                ),
            },
            {
                "role": "user",
                "content": (
                    f"<question>\n{question}\n</question>\n\n"
                    f"<reference_answer>\n{baseline_answer}\n</reference_answer>\n\n"
                    "Produce the rubric now. The four allowed dimension values "
                    "are: instruction_following, comprehensiveness, "
                    "completeness, writing_quality."
                ),
            },
        ]
        rubric = query_openai_model_structured_outputs(
            messages=messages,
            output_class=Rubric,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if rubric is None:
            raise ValueError("Failed to generate rubric from evaluation model")
        return Rubric.model_validate(rubric)

    @retry(tries=3, delay=1, backoff=2)
    def grade_against_rubric(
        self, question: str, candidate_answer: str, rubric: Rubric
    ) -> RubricJudgmentOutput:
        """Grade a candidate report against every criterion in the rubric."""
        criteria_block = "\n".join(
            f"- [dimension={c.dimension} | weight={c.weight}] {c.criterion}"
            for c in rubric.criteria
        )
        messages = [
            {"role": "system", "content": RUBRIC_JUDGING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<question>\n{question}\n</question>\n\n"
                    f"<candidate_report>\n{candidate_answer}\n</candidate_report>\n\n"
                    f"<rubric_criteria>\n{criteria_block}\n</rubric_criteria>"
                ),
            },
        ]
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=RubricJudgmentOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to grade candidate against rubric")
        return RubricJudgmentOutput.model_validate(output)

    def score(
        self, question: str, baseline_answer: str, candidate_answer: str
    ) -> RubricScoreResult:
        """Score a candidate report: build a rubric, then grade it.

        Mirrors the ``DeepResearchPairwiseMetric.score`` forward path
        ``(question, baseline_answer, candidate_answer) -> score``.
        """
        rubric = self.generate_rubric(question, baseline_answer)
        judged = self.grade_against_rubric(question, candidate_answer, rubric)
        return RubricScoreResult(question=question, criteria=judged.judgments)

    def aggregate(
        self, scores_list: List[RubricScoreResult]
    ) -> Dict[str, Any]:
        """Aggregate rubric scores into a weight-tier pass-rate breakdown.

        ``pass_rate_by_weight`` is the headline: the paper finds critical
        (weight-5) criteria pass at far lower rates than weight-1 criteria even
        when average scores look healthy. Pure (no model calls) so it is
        unit-testable directly.
        """
        all_criteria = [c for s in scores_list for c in s.criteria]
        if not all_criteria:
            return {"support": len(scores_list), "total_criteria": 0}

        # Pass rate per weight tier (the priority-inversion signal).
        by_weight: Dict[int, List[bool]] = {}
        by_dimension: Dict[str, List[bool]] = {dim: [] for dim in DIMENSIONS}
        for c in all_criteria:
            by_weight.setdefault(c.weight, []).append(c.met)
            if c.dimension in by_dimension:
                by_dimension[c.dimension].append(c.met)

        pass_rate_by_weight = {
            w: sum(mets) / len(mets) for w, mets in sorted(by_weight.items())
        }
        pass_rate_by_dimension = {
            dim: (sum(mets) / len(mets) if mets else 0.0)
            for dim, mets in by_dimension.items()
        }

        total_weight = sum(c.weight for c in all_criteria)
        passed_weight = sum(c.weight for c in all_criteria if c.met)
        weighted_pass_rate = (
            passed_weight / total_weight if total_weight else 0.0
        )

        critical_total = sum(1 for c in all_criteria if c.weight == MAX_WEIGHT)
        critical_met = sum(
            1 for c in all_criteria if c.weight == MAX_WEIGHT and c.met
        )
        critical_unsatisfied_rate = (
            1.0 - (critical_met / critical_total) if critical_total else None
        )

        return {
            "support": len(scores_list),
            "total_criteria": len(all_criteria),
            "mean_pass_rate": sum(c.met for c in all_criteria)
            / len(all_criteria),
            "weighted_pass_rate": weighted_pass_rate,
            "pass_rate_by_weight": pass_rate_by_weight,
            "pass_rate_by_dimension": pass_rate_by_dimension,
            "critical_unsatisfied_rate": critical_unsatisfied_rate,
        }


class RubricScoringEvaluator:
    """Evaluator wrapping :class:`RubricScoringMetric` for batch CSV runs."""

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        num_criteria: int = DEFAULT_NUM_CRITERIA,
        num_workers: int = 4,
    ):
        self.model = model
        self.metric = RubricScoringMetric(
            eval_model=model,
            num_criteria=num_criteria,
            num_workers=num_workers,
        )
        self.num_workers = num_workers

    def evaluate_single(
        self, question: str, baseline_answer: str, candidate_answer: str
    ) -> Dict[str, Any]:
        """Score one (question, baseline, candidate) row."""
        result: Dict[str, Any] = {
            "question": question,
            "baseline_answer": baseline_answer,
            "candidate_answer": candidate_answer,
        }
        try:
            score_result = self.metric.score(
                question=question,
                baseline_answer=baseline_answer,
                candidate_answer=candidate_answer,
            )
            result["success"] = True
            result["score_result"] = score_result.model_dump()
            result["mean_pass_rate"] = score_result.mean_pass_rate
            result["weighted_pass_rate"] = score_result.weighted_pass_rate
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        return result

    def evaluate_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Score every row in ``data`` in parallel."""
        required = ["question", "baseline_answer", "candidate_answer"]
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Input data must contain column: {col}")

        rows = data[required].to_dict("records")
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(
                    self.evaluate_single,
                    row["question"],
                    row["baseline_answer"],
                    row["candidate_answer"],
                )
                for row in rows
            ]
            for future in tqdm(
                futures, total=len(futures), desc="Rubric scoring"
            ):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing task: {e}")
                    results.append({"success": False, "error": str(e)})
        return results

    def aggregate_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rebuild score objects from row results and aggregate them."""
        score_results: List[RubricScoreResult] = []
        for result in results:
            if not result.get("success"):
                continue
            try:
                score_results.append(
                    RubricScoreResult.model_validate(result["score_result"])
                )
            except Exception as e:
                print(f"Error parsing score result: {e}")
        return self.metric.aggregate(score_results)


def run(
    input_data: str,
    output_dir: str,
    model: str = DEFAULT_EVAL_MODEL,
    num_workers: int = 4,
    num_criteria: int = DEFAULT_NUM_CRITERIA,
) -> Dict[str, Any]:
    """Run rubric scoring over a CSV; write JSONL rows + aggregate JSON."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"rubric_scoring_results_{model}.jsonl"

    print(f"Loading data from {input_data}")
    df = pd.read_csv(input_data)
    print(f"Loaded {len(df)} examples")

    evaluator = RubricScoringEvaluator(
        model=model, num_criteria=num_criteria, num_workers=num_workers
    )
    print(f"Starting rubric scoring with model {model}...")
    results = evaluator.evaluate_batch(df)

    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)
    print(f"Results saved to {output_path}")

    aggregate_metrics = evaluator.aggregate_results(results)
    aggregate_path = output_dir_path / f"rubric_scoring_aggregate_{model}.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"Aggregate metrics saved to {aggregate_path}")
    return aggregate_metrics


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rubric-based scoring of deep research reports"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="CSV with 'question', 'baseline_answer', 'candidate_answer'.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_EVAL_MODEL, help="Judge model"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Parallel worker threads"
    )
    parser.add_argument(
        "--num-criteria",
        type=int,
        default=DEFAULT_NUM_CRITERIA,
        help="Target number of rubric criteria per question",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    return run(
        input_data=args.input_data,
        output_dir=args.output_dir,
        model=args.model,
        num_workers=args.num_workers,
        num_criteria=args.num_criteria,
    )


if __name__ == "__main__":
    main()

