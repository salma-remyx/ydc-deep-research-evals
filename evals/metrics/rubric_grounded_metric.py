"""Rubric-grounded deep research metric.

Adapted (Mode 2) from "From Simple QA to Deep Research: A Verifiable
Benchmark Constructed through Iterative Task Evolution" (arXiv:2608.02163v1).

The paper's portable core is a *DAG of atomic research steps*, each carrying
verifiable, fact-grounded checkpoints shared between task synthesis and reward.
We port that core and substitute the auxiliaries this repo cannot host:

  * Grounding -- the paper anchors checkpoints to a curated per-task knowledge
    base (none here), so we ground them to the *question*: both reports are
    scored against the same objective checkpoints, keeping win/tie/lose fair.
  * Evolution loop -- the iterative Explorer->Formalizer->Challenger loop that
    *builds* a 500-task benchmark is collapsed to one synthesize + one refine
    pass plus a deterministic dedupe; benchmark construction is out of scope.
  * Eval framework -- folded into the existing ``DeepResearchPairwiseMetric``
    interface so the result is a drop-in ``DeepResearchScoreResult``.

Payoff: completeness/comprehensiveness are scored checkpoint-by-checkpoint with
structure weighting (a step that unlocks more descendants weighs more) instead
of a single holistic preference.
"""

import concurrent.futures
from typing import Any, Dict, List, Literal, Tuple, Type

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DIMENSIONS,
    DeepResearchScoreResult,
    DimensionResult,
)
from evals.utils import (
    query_openai_model_structured_outputs,
    replace_markdown_links_with_text,
)

Dimension = Literal[
    "instruction_following",
    "comprehensiveness",
    "completeness",
    "writing_quality",
]

_COVERAGE_VALUE = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
# |delta| at or below this is reported as a tie (noise guard for pointwise).
_TIE_EPS = 0.05


class ResearchStep(BaseModel):
    """One atomic research step in the DAG and its verifiable checkpoints."""

    id: str = Field(description="Short stable id, e.g. 's1'.")
    description: str = Field(description="The atomic sub-question this step resolves.")
    depends_on: List[str] = Field(
        default_factory=list,
        description="Ids of steps whose results this step builds on (DAG parents).",
    )
    dimension: Dimension = Field(
        description="Which evaluation dimension this step's checkpoints ground."
    )
    checkpoints: List[str] = Field(
        description="Concrete, independently verifiable rubric items a complete "
        "answer must satisfy for this step."
    )


class ResearchDAG(BaseModel):
    """A research question decomposed into a DAG of checkpoint-bearing steps."""

    question: str
    steps: List[ResearchStep]

    def checkpoints(self) -> List[Tuple[str, str, str]]:
        """Flatten to (step_id, dimension, checkpoint_text) triples."""
        return [
            (step.id, step.dimension, checkpoint)
            for step in self.steps
            for checkpoint in step.checkpoints
        ]


class CheckpointVerdict(BaseModel):
    """Pointwise verdict for one checkpoint against a single report."""

    step_id: str
    dimension: Dimension
    checkpoint: str
    coverage: Literal["covered", "partial", "missing"]
    evidence: str = Field(description="Short note or snippet justifying the call.")


class _VerdictBatch(BaseModel):
    """Envelope so structured-output returns a list of verdicts."""

    verdicts: List[CheckpointVerdict]


_EXPLORER_FORMALIZER_PROMPT = """\
You are decomposing a deep-research question into a verifiable evaluation rubric.

Build a directed acyclic graph (DAG) of atomic research steps that a complete, \
high-quality answer must perform. For every step write concrete, fact-grounded \
checkpoints -- specific, checkable claims a correct answer must establish. Each \
checkpoint must be objective and independently verifiable (never stylistic), so \
it can be scored covered / partial / missing.

Assign each step to exactly one dimension:
- comprehensiveness: breadth / scope of what the answer must cover.
- completeness: depth / thoroughness for the topics addressed.
- instruction_following: fidelity to explicit constraints stated in the question.
- writing_quality: clarity, organization, and readability.
Most steps should be comprehensiveness or completeness; add the others only where \
the question explicitly demands them. Keep dependencies acyclic (depends_on may \
only reference earlier ids). Produce 4-8 steps total.
"""

_CHALLENGER_PROMPT = """\
You are hardening an evaluation rubric. Given a draft DAG, return a refined \
version that (a) merges near-duplicate steps, (b) drops vague or non-verifiable \
checkpoints, and (c) keeps dependencies acyclic. Do not invent sub-topics beyond \
what the question requires.
"""

_SCORER_PROMPT = """\
You are scoring a research report against an objective, checkpoint-based rubric.

For every checkpoint, judge whether the report establishes it and return one \
verdict with coverage in {covered, partial, missing}:
- covered: fully established; partial: touched but incomplete; missing: absent.
Cite a short snippet or note as evidence. Return exactly one verdict per \
checkpoint, preserving each checkpoint's step_id and dimension.
"""


def _clean(text: str) -> str:
    """Normalize an answer the same way the pairwise input does."""
    if not text or not text.strip():
        raise ValueError("Answer cannot be empty")
    return replace_markdown_links_with_text(text.strip(), "")


class RubricGroundedMetric:
    """Deep research metric that scores checkpoint-by-checkpoint over a DAG.

    Drop-in alternative to :class:`DeepResearchPairwiseMetric`: ``score`` returns
    a :class:`DeepResearchScoreResult` and ``aggregate`` returns the same shape,
    so the existing evaluator and CLI summary work unchanged.
    """

    def __init__(self, eval_model: str = DEFAULT_EVAL_MODEL, num_workers: int = 3):
        self.eval_model = eval_model
        self.num_workers = max(num_workers, 1)

    # -- LLM-backed synthesis (Explorer + Formalizer + Challenger) -- #

    @retry(tries=3, delay=1, backoff=2)
    def _query_structured(
        self,
        messages: List[ChatCompletionMessageParam],
        output_class: Type[BaseModel],
        max_completion_tokens: int,
        fail_msg: str,
    ) -> BaseModel:
        parsed = query_openai_model_structured_outputs(
            messages=messages,
            output_class=output_class,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=max_completion_tokens,
        )
        if parsed is None:
            raise ValueError(fail_msg)
        return output_class.model_validate(parsed)

    def build_research_dag(self, question: str) -> ResearchDAG:
        """Explorer + Formalizer (decompose) then Challenger (refine)."""
        common_user = f"<question>\n{question}\n</question>"
        draft = self._query_structured(
            [{"role": "system", "content": _EXPLORER_FORMALIZER_PROMPT},
             {"role": "user", "content": common_user}],
            ResearchDAG,
            8000,
            "Failed to synthesize research DAG",
        )
        refined = self._query_structured(
            [{"role": "system", "content": _CHALLENGER_PROMPT},
             {"role": "user",
              "content": f"{common_user}\n\n<draft_dag>\n{draft.model_dump_json()}\n</draft_dag>"}],
            ResearchDAG,
            8000,
            "Failed to refine research DAG",
        )
        dag = refined if isinstance(refined, ResearchDAG) and refined.steps else draft
        return self._dedupe_checkpoints(dag)

    @staticmethod
    def _dedupe_checkpoints(dag: ResearchDAG) -> ResearchDAG:
        """Challenger post-process: drop exact-duplicate checkpoints per step."""
        for step in dag.steps:
            seen: set = set()
            kept: List[str] = []
            for checkpoint in step.checkpoints:
                key = checkpoint.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    kept.append(checkpoint)
            step.checkpoints = kept
        return dag

    def score_checkpoints(
        self, dag: ResearchDAG, answer: str, label: str
    ) -> List[CheckpointVerdict]:
        """Pointwise-score every checkpoint against one report."""
        items = "\n".join(
            f"- [{sid}|{dim}] {checkpoint}"
            for sid, dim, checkpoint in dag.checkpoints()
        )
        batch = self._query_structured(
            [{"role": "system", "content": _SCORER_PROMPT},
             {"role": "user",
              "content": f"<question>\n{dag.question}\n</question>\n\n<{label}>\n{answer}\n</{label}>\n\n<checkpoints>\n{items}\n</checkpoints>"}],
            _VerdictBatch,
            10000,
            "Failed to score checkpoints",
        )
        return _VerdictBatch.model_validate(batch).verdicts

    # -- Pure structure / aggregation logic (no LLM, unit-testable) -- #

    @staticmethod
    def structure_weights(dag: ResearchDAG) -> Dict[str, float]:
        """Weight each step by ``1 + descendant_count`` then normalize.

        A step that unlocks more downstream work weighs more: missing it likely
        cascades. This is the paper's structure weighting on per-step scores.
        """
        children: Dict[str, List[str]] = {step.id: [] for step in dag.steps}
        for step in dag.steps:
            for parent in step.depends_on:
                if parent in children:
                    children[parent].append(step.id)

        def descendants(node: str) -> int:
            seen, stack = set(), list(children.get(node, []))
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(children.get(current, []))
            return len(seen)

        raw = {step.id: 1.0 + descendants(step.id) for step in dag.steps}
        total = sum(raw.values()) or 1.0
        return {node: value / total for node, value in raw.items()}

    @staticmethod
    def aggregate_verdicts_to_dimensions(
        dag: ResearchDAG,
        baseline_verdicts: List[CheckpointVerdict],
        candidate_verdicts: List[CheckpointVerdict],
        weights: Dict[str, float],
    ) -> DeepResearchScoreResult:
        """Map pointwise verdicts + structure weights to a 4-dimension result."""
        per_step = {step.id: max(len(step.checkpoints), 1) for step in dag.steps}

        def weighted_coverage(verdicts: List[CheckpointVerdict]) -> Dict[str, float]:
            acc = {dim: 0.0 for dim in DIMENSIONS}
            weight_total = {dim: 0.0 for dim in DIMENSIONS}
            for verdict in verdicts:
                share = weights.get(verdict.step_id, 0.0) / per_step.get(verdict.step_id, 1)
                acc[verdict.dimension] += share * _COVERAGE_VALUE[verdict.coverage]
                weight_total[verdict.dimension] += share
            return {
                dim: (acc[dim] / weight_total[dim] if weight_total[dim] > 0 else 0.0)
                for dim in DIMENSIONS
            }

        baseline_cov = weighted_coverage(baseline_verdicts)
        candidate_cov = weighted_coverage(candidate_verdicts)

        dimension_results: Dict[str, DimensionResult] = {}
        for dim in DIMENSIONS:
            delta = candidate_cov[dim] - baseline_cov[dim]
            if delta > _TIE_EPS:
                grade, preferred = "win", ["b"]
            elif delta < -_TIE_EPS:
                grade, preferred = "lose", ["a"]
            else:
                grade, preferred = "tie", ["a", "b"]
            dimension_results[dim] = DimensionResult(
                grade=grade,
                is_win=grade == "win",
                is_tie=grade == "tie",
                is_lose=grade == "lose",
                score=5.0 + delta * 5.0,
                preferred=preferred,
                raw_preferences={
                    "baseline_coverage": baseline_cov[dim],
                    "candidate_coverage": candidate_cov[dim],
                    "num_checkpoints": sum(
                        1 for v in candidate_verdicts if v.dimension == dim
                    ),
                    "verdicts": [
                        v.model_dump() for v in candidate_verdicts if v.dimension == dim
                    ],
                },
            )
        return DeepResearchScoreResult(**dimension_results)

    # -- Public metric interface (mirrors DeepResearchPairwiseMetric) -- #

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> DeepResearchScoreResult:
        """Score a question-answer pair via DAG construction + checkpoint scoring."""
        clean_question = question.strip() or question
        baseline_clean = _clean(baseline_answer)
        candidate_clean = _clean(candidate_answer)

        dag = self.build_research_dag(clean_question)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            baseline_future = executor.submit(
                self.score_checkpoints, dag, baseline_clean, "report_a"
            )
            candidate_future = executor.submit(
                self.score_checkpoints, dag, candidate_clean, "report_b"
            )
            baseline_verdicts = baseline_future.result()
            candidate_verdicts = candidate_future.result()

        weights = self.structure_weights(dag)
        return self.aggregate_verdicts_to_dimensions(
            dag, baseline_verdicts, candidate_verdicts, weights
        )

    def aggregate(
        self, scores_list: List[DeepResearchScoreResult]
    ) -> Dict[str, Any]:
        """Aggregate scored rows into the same shape as the pairwise metric."""
        aggregated: Dict[str, Any] = {"support": len(scores_list)}
        for dim in DIMENSIONS:
            results = [getattr(score, dim) for score in scores_list]
            count = len(results) or 1
            wins = sum(r.is_win for r in results)
            losses = sum(r.is_lose for r in results)
            aggregated[dim] = {
                "win_rate": wins / count,
                "tie_rate": sum(r.is_tie for r in results) / count,
                "lose_rate": losses / count,
                "avg_score": sum(r.score for r in results) / count,
                "net_winrate": wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            }

        aggregated["overall"] = {
            metric: sum(aggregated[dim][metric] for dim in DIMENSIONS) / len(DIMENSIONS)
            for metric in ["win_rate", "tie_rate", "lose_rate", "avg_score", "net_winrate"]
        }

        # Rubric-specific transparency: how completely each report covered the
        # constructed checkpoints, and how many were assessed.
        candidate_cov = baseline_cov = 0.0
        total_checkpoints = 0
        for score in scores_list:
            for dim in DIMENSIONS:
                raw = getattr(score, dim).raw_preferences
                candidate_cov += raw.get("candidate_coverage", 0.0)
                baseline_cov += raw.get("baseline_coverage", 0.0)
                total_checkpoints += raw.get("num_checkpoints", 0)
        denom = (len(scores_list) * len(DIMENSIONS)) or 1
        aggregated["overall"]["avg_candidate_checkpoint_coverage"] = candidate_cov / denom
        aggregated["overall"]["avg_baseline_checkpoint_coverage"] = baseline_cov / denom
        aggregated["overall"]["total_checkpoints"] = total_checkpoints
        return aggregated
