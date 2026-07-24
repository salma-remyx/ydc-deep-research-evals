"""Two-level meta-rubric factual-completeness scoring for deep research reports.

Capability: author a structured (grouped, importance-weighted, multi-kind) meta-rubric
for what a *complete* answer should contain, mechanically compile it into a flat
checklist of binary checks, have the existing OpenAI LLM judge score each check, and
aggregate a weighted *factual-completeness coverage* score. This complements the repo's
pairwise ``completeness`` dimension with a structured, coverage-based signal.

Adapted (Mode 2) port of GAMUT -- "Two-Level Meta-Rubrics for Evaluating Open-Ended
Generation: GAMUT, a Benchmark for Factual Completeness" (arXiv:2607.19322).

Core mechanism preserved at full fidelity:
  * Two-level representation -- a structured meta-rubric (Level 1) compiled into a flat
    checklist of binary, machine-gradable rubrics (Level 2).
  * Criteria kinds beyond a flat list: atomic facts, open-ended *coverage* sets, ordered
    processes, and relationships -- exactly the cases a list of independent booleans
    misses, and the heart of factual completeness.
  * Group importance propagates to compiled checks; the score is the weighted fraction of
    required content present (coverage) -- the "missing half of factuality".

Target-native substitutions (Mode 2):
  * Rubric authoring -- GAMUT uses expert human annotators; here the same OpenAI judge
    authors the meta-rubric from the question + reference answer (the fork's contract).
  * Benchmark dropped -- GAMUT's 1,813 wearable-image questions, 10-domain suite,
    14-model sweep, multimodal variant, and pointwise ground-truth verification are out
    of scope; this scorer runs over the repo's existing DeepConsult CSV contract instead.
"""

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry
from tqdm import tqdm

from evals.metrics.deep_research_pairwise_metric import DEFAULT_EVAL_MODEL
from evals.utils import query_openai_model_structured_outputs, replace_markdown_links_with_text

CHECK_KINDS = ("atomic", "coverage", "ordered", "relationship")
Kind = Literal["atomic", "coverage", "ordered", "relationship"]


class AuthoredCriterion(BaseModel):
    """A single required-content criterion in a meta-rubric (Level 1)."""

    description: str
    kind: Kind
    items: List[str] = Field(
        default_factory=list,
        description=(
            "Member facts/steps/relations to check. Required for coverage, ordered and "
            "relationship kinds (one check is compiled per item). Empty for atomic."
        ),
    )


class AuthoredGroup(BaseModel):
    """A grouped area of required content with a relative importance weight."""

    name: str
    importance: float = Field(ge=0.0, description="Relative importance (>= 0).")
    criteria: List[AuthoredCriterion]


class AuthoredMetaRubric(BaseModel):
    """A structured meta-rubric (Level 1) authored for one question."""

    groups: List[AuthoredGroup]


class CheckItem(BaseModel):
    """A single flat binary check (Level 2) compiled from a meta-rubric."""

    group: str
    kind: Kind
    description: str
    weight: float
    order_index: Optional[int] = None


class CheckVerdict(BaseModel):
    """The judge's binary verdict for one compiled check."""

    description: str
    met: bool
    justification: str


class ChecklistVerdicts(BaseModel):
    """Container for the judge's per-check binary verdicts."""

    verdicts: List[CheckVerdict]


AUTHORING_PROMPT = """\
You are authoring a factual-completeness meta-rubric for a research report.

Given the research QUESTION and a REFERENCE report, enumerate the full set of facts a \
complete answer should contain, organized into grouped areas. For each group give a \
short name, a relative importance (>= 0, higher = more central to completeness), and a \
list of criteria. Each criterion is one of:
  - "atomic": a single required fact (leave items empty).
  - "coverage": an open-ended set where coverage matters; list every member item that
    should be present.
  - "ordered": a process or sequence; list the steps in order.
  - "relationship": a required relationship among facts; list the relationships.

Focus on completeness (what could be missing), not writing style. Do not invent facts
unsupported by the question or reference. Prefer concrete, individually checkable items.
"""


SCORING_PROMPT = """\
You are scoring factual completeness against a flat checklist of binary checks.

For each check, decide whether the CANDIDATE report satisfies it and answer with a single \
boolean "met" plus a one-sentence justification citing the report (or noting its absence). \
Score strictly: a check is "met" only if the candidate clearly contains the required \
content. Return exactly one verdict per check, in the order given.
"""


def compile_meta_rubric(meta_rubric: AuthoredMetaRubric) -> List[CheckItem]:
    """Mechanically flatten a structured meta-rubric into a flat weighted checklist.

    GAMUT's Level-2 compilation: each atomic criterion yields one check; each
    coverage/ordered/relationship criterion expands to one check per member item. A
    group's importance is distributed evenly across its checks so every group contributes
    to the score in proportion to its share of required content.
    """
    checks: List[CheckItem] = []
    for group in meta_rubric.groups:
        raw: List[tuple] = []
        for criterion in group.criteria:
            if criterion.kind == "atomic" or not criterion.items:
                raw.append((criterion.kind, criterion.description, None))
            else:
                for i, item in enumerate(criterion.items):
                    order_index = i if criterion.kind == "ordered" else None
                    raw.append((criterion.kind, item, order_index))

        per_check_weight = group.importance / len(raw) if raw else 0.0
        for kind, description, order_index in raw:
            checks.append(
                CheckItem(
                    group=group.name,
                    kind=kind,
                    description=description,
                    weight=per_check_weight,
                    order_index=order_index,
                )
            )
    return checks


def aggregate_coverage(
    verdicts: List[CheckVerdict], checks: List[CheckItem]
) -> Dict[str, Any]:
    """Aggregate binary verdicts into a weighted factual-completeness coverage score.

    Coverage is the weight-weighted fraction of required content present. If every check
    carries zero weight (degenerate meta-rubric), weights fall back to uniform so the
    score remains well-defined.
    """
    if len(verdicts) != len(checks):
        raise ValueError(
            f"Expected {len(checks)} verdicts, received {len(verdicts)}"
        )

    total_weight = sum(check.weight for check in checks)
    if total_weight > 0:
        weights = [check.weight for check in checks]
    else:
        weights = [1.0] * len(checks)
        total_weight = float(len(checks))

    met_weight = sum(
        weight for verdict, weight in zip(verdicts, weights) if verdict.met
    )
    coverage = met_weight / total_weight if total_weight > 0 else 0.0

    by_group: Dict[str, Dict[str, float]] = {}
    by_kind: Dict[str, Dict[str, float]] = {}
    for check, verdict, weight in zip(checks, verdicts, weights):
        for bucket, key in ((by_group, check.group), (by_kind, check.kind)):
            slot = bucket.setdefault(
                key, {"met_weight": 0.0, "total_weight": 0.0, "met": 0, "total": 0}
            )
            slot["total_weight"] += weight
            slot["total"] += 1
            if verdict.met:
                slot["met_weight"] += weight
                slot["met"] += 1

    def _coverage(slot: Dict[str, float]) -> float:
        return (
            slot["met_weight"] / slot["total_weight"]
            if slot["total_weight"] > 0
            else 0.0
        )

    return {
        "factual_completeness_coverage": coverage,
        "checks_total": len(checks),
        "checks_met": sum(1 for verdict in verdicts if verdict.met),
        "coverage_by_group": {
            key: {
                "coverage": _coverage(slot),
                "met": slot["met"],
                "total": slot["total"],
            }
            for key, slot in by_group.items()
        },
        "coverage_by_kind": {
            key: {
                "coverage": _coverage(slot),
                "met": slot["met"],
                "total": slot["total"],
            }
            for key, slot in by_kind.items()
        },
    }


class MetaRubricCompletenessScorer:
    """Scores factual-completeness coverage via the two-level meta-rubric pipeline."""

    def __init__(self, model: str = DEFAULT_EVAL_MODEL):
        self.model = model

    def _authoring_messages(
        self, question: str, reference_answer: str
    ) -> List[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": AUTHORING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<question>\n{question}\n</question>\n\n"
                    f"<reference_report>\n"
                    f"{replace_markdown_links_with_text(reference_answer, '')}\n"
                    f"</reference_report>"
                ),
            },
        ]

    def _scoring_messages(
        self, checks: List[CheckItem], candidate_answer: str
    ) -> List[ChatCompletionMessageParam]:
        checklist = "\n".join(
            f"{i}. [{c.kind}] ({c.group}) {c.description}"
            for i, c in enumerate(checks, start=1)
        )
        return [
            {"role": "system", "content": SCORING_PROMPT},
            {
                "role": "user",
                "content": (
                    f"<candidate_report>\n"
                    f"{replace_markdown_links_with_text(candidate_answer, '')}\n"
                    f"</candidate_report>\n\n"
                    f"<checklist>\n{checklist}\n</checklist>"
                ),
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query_structured(
        self,
        messages: List[ChatCompletionMessageParam],
        output_class: type[BaseModel],
    ) -> BaseModel:
        parsed = query_openai_model_structured_outputs(
            messages=messages,
            output_class=output_class,
            model=self.model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if parsed is None:
            raise ValueError("Failed to get structured output from evaluation model")
        return output_class.model_validate(parsed)

    def author_meta_rubric(
        self, question: str, reference_answer: str
    ) -> AuthoredMetaRubric:
        return self._query_structured(
            self._authoring_messages(question, reference_answer), AuthoredMetaRubric
        )

    def score_checklist(
        self, checks: List[CheckItem], candidate_answer: str
    ) -> List[CheckVerdict]:
        verdicts = self._query_structured(
            self._scoring_messages(checks, candidate_answer), ChecklistVerdicts
        ).verdicts
        if len(verdicts) != len(checks):
            raise ValueError(
                f"Expected {len(checks)} verdicts, received {len(verdicts)}"
            )
        return verdicts

    def score(
        self, question: str, baseline_answer: str, candidate_answer: str
    ) -> Dict[str, Any]:
        """Author -> compile -> score -> aggregate for one question-answer row."""
        meta_rubric = self.author_meta_rubric(question, baseline_answer)
        checks = compile_meta_rubric(meta_rubric)
        verdicts = self.score_checklist(checks, candidate_answer)
        result = aggregate_coverage(verdicts, checks)
        result["meta_rubric"] = meta_rubric.model_dump()
        return result


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-row coverage results into corpus-level factual completeness."""
    scored = [r for r in results if r.get("success", False)]
    if not scored:
        return {"support": 0, "error": "No successful evaluations found"}

    coverages = [r["factual_completeness_coverage"] for r in scored]
    return {
        "support": len(scored),
        "mean_factual_completeness_coverage": sum(coverages) / len(coverages),
        "total_checks": sum(r["checks_total"] for r in scored),
        "total_checks_met": sum(r["checks_met"] for r in scored),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run two-level meta-rubric factual-completeness evaluations "
            "(adapted from GAMUT, arXiv:2607.19322)"
        )
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="CSV with 'question', 'baseline_answer', 'candidate_answer' columns.",
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
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meta_rubric_completeness_{args.model}.jsonl"

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    scorer = MetaRubricCompletenessScorer(model=args.model)
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                scorer.score,
                row["question"],
                row["baseline_answer"],
                row["candidate_answer"],
            ): index
            for index, (_, row) in enumerate(df.iterrows())
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Scoring"
        ):
            index = futures[future]
            try:
                result = future.result()
                result["success"] = True
                results.append(result)
            except Exception as exc:  # mirror existing CLI's per-row guard
                print(f"Error scoring row {index}: {exc}")
                results.append({"success": False, "error": str(exc), "row_index": index})

    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)
    print(f"Results saved to {output_path}")

    aggregate_metrics = aggregate_results(results)
    aggregate_path = output_dir / f"meta_rubric_completeness_aggregate_{args.model}.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)

    print("\nFactual-completeness coverage:")
    print(f"Support: {aggregate_metrics.get('support', 0)}")
    if "mean_factual_completeness_coverage" in aggregate_metrics:
        print(
            "Mean coverage: "
            f"{aggregate_metrics['mean_factual_completeness_coverage']:.4f}"
        )
        print(f"Checks met: {aggregate_metrics['total_checks_met']}")
        print(f"Checks total: {aggregate_metrics['total_checks']}")


if __name__ == "__main__":
    main()
