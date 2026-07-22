"""Multi-agent debate judge architecture for deep-research pairwise evaluation.

This module implements multi-agent debate as an alternative *judge architecture*
that produces the same per-dimension ``Preference(preferred, gap_score,
explanation)`` verdict as the repo's single-pass ``DeepResearchPairwiseMetric``.

Adapted (Mode 2) from:

    "Does Multi-Agent Debate Improve AI Feedback on Research Papers?"
    arxiv: https://arxiv.org/abs/2607.14713v1

The paper's headline finding is that, for feedback on research papers, human
authors *preferred* a single frontier-model pass over two multi-agent debate
tools, even though one debate tool spent roughly thirty times the tokens, and
that an AI judge's ranking can reverse the author's ranking. Rather than reproduce
the paper's separate benchmark (author usefulness rankings, the journal
referee-report comparison, and the AI-judge-vs-author analysis -- those are a
downstream-eval concern), this module delivers the *judge architecture* the paper
tested so the team can run debate and single-pass side-by-side on the DeepConsult
dataset and measure whether the paper's "single-pass stays competitive" result
holds for deep-research reports.

Core mechanism kept at fidelity
--------------------------------
A panel of debater agents independently assesses the report pair from distinct
perspectives (so the panel is genuinely diverse even at ``temperature=0``), they
revise their verdicts over a fixed number of debate rounds in light of one
another's evidence, and a final *chair* agent synthesizes a single consolidated
verdict -- producing exactly the ``DeepResearchPairwisePreferenceOutput`` the
single-pass judge produces.

Auxiliary components substituted for target-native equivalents (Mode 2)
----------------------------------------------------------------------
* The repo's existing 4-dimension pairwise rubric + ``Preference`` schema stand
  in for the paper's research-paper-feedback template.
* The repo's existing ``query_openai_model_structured_outputs`` OpenAI helper
  stands in for a bespoke multi-provider orchestrator (the repo is OpenAI-only).
* The repo's existing flipped-order trial + majority-vote consensus + ``aggregate``
  machinery is reused unchanged (debate only overrides ``_query_evaluation_model``,
  the single-pass call site), so debate and single-pass results are directly
  comparable.
"""

from typing import List, Optional

from openai.types.chat import ChatCompletionMessageParam
from retry import retry

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DEEP_RESEARCH_PAIRWISE_PROMPT,
    DIMENSIONS,
    DeepResearchPairwiseMetric,
    DeepResearchPairwisePreferenceOutput,
)
from evals.utils import query_openai_model_structured_outputs

# A perspective-diverse panel: each debater emphasizes a different evaluation
# lens (and one plays devil's advocate) so the opening verdicts differ even at
# temperature 0. All debaters still emit the full 4-dimension verdict.
DEFAULT_DEBATER_PERSPECTIVES: List[str] = [
    (
        "You are Debater 1, an expert in instruction following and writing "
        "quality. Weigh surface fidelity, clarity, and adherence to the user's "
        "constraints heavily, but you must still judge all four dimensions."
    ),
    (
        "You are Debater 2, an expert in comprehensiveness and completeness. "
        "Prioritize the breadth and depth of evidence and analysis, but you "
        "must still judge all four dimensions."
    ),
    (
        "You are Debater 3, a devil's advocate. Make the strongest case against "
        "whichever report currently looks preferred, surfacing counter-evidence "
        "and overlooked weaknesses, but you must still judge all four dimensions."
    ),
]

CHAIR_SYSTEM_PROMPT = (
    "You are the chair of a panel of expert evaluators who have just debated "
    "which of two research reports (report_a vs report_b) is stronger. Weigh "
    "the debaters' evidence below, resolve disagreements on the merits (do not "
    "simply follow the majority where a minority argument is stronger), and "
    "produce the final consolidated verdict across all four dimensions."
)


class MultiAgentDebateMetric(DeepResearchPairwiseMetric):
    """Pairwise metric that scores a report pair via multi-agent debate.

    Overrides :meth:`_query_evaluation_model` -- the single-pass call site of
    :class:`DeepResearchPairwiseMetric` -- so the entire inherited pipeline
    (flipped-order trials, majority-vote consensus, normalization, and
    :meth:`aggregate`) runs unchanged against the debate verdicts. The result is
    a second judge architecture whose output is directly comparable to the
    single-pass judge, letting the team test whether debate beats single-pass on
    the DeepConsult dataset.
    """

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
        n_debaters: int = 3,
        n_rounds: int = 1,
        perspectives: Optional[List[str]] = None,
    ):
        """
        Args:
            eval_model: The model to use for every debater and chair call.
            num_trials: Number of (original + flipped) trials; inherited from
                the single-pass metric so the two are comparable.
            num_workers: Parallel workers for the inherited trial fan-out.
            n_debaters: Number of debating agents on the panel.
            n_rounds: Number of debate revision rounds after the opening verdicts.
            perspectives: Optional custom debater role prompts. When omitted, the
                first ``n_debaters`` default perspectives are used.
        """
        super().__init__(
            eval_model=eval_model,
            num_trials=num_trials,
            num_workers=num_workers,
        )
        self.n_rounds = n_rounds
        if perspectives is not None:
            self.debater_perspectives = list(perspectives)
        else:
            self.debater_perspectives = [
                DEFAULT_DEBATER_PERSPECTIVES[i % len(DEFAULT_DEBATER_PERSPECTIVES)]
                for i in range(n_debaters)
            ]
        # Total structured-output model calls issued by this metric. Single-pass
        # issues one call per trial-half; debate issues many more -- surfacing
        # the token-cost trade-off the paper highlights.
        self.model_call_count = 0

    @retry(tries=3, delay=1, backoff=2)
    def _structured_call(
        self, messages: List[ChatCompletionMessageParam]
    ) -> DeepResearchPairwisePreferenceOutput:
        """Single structured-output call with retry; the debate's model seam."""
        self.model_call_count += 1
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=DeepResearchPairwisePreferenceOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from debate model")
        return DeepResearchPairwisePreferenceOutput.model_validate(output)

    def _query_evaluation_model(
        self, messages: List[ChatCompletionMessageParam]
    ) -> DeepResearchPairwisePreferenceOutput:
        """Run the multi-agent debate and return the chair's consolidated verdict.

        Replaces the single-pass call of the base metric. ``messages`` is the
        [system=rubric, user=reports] pair built by ``_get_evaluation_messages``;
        the user block (containing the prompt and the two reports) is reused
        verbatim for every debater and the chair, while the system prompt carries
        each role's perspective.
        """
        user_content = messages[-1]["content"]

        # 1. Independent opening verdicts from each debater's perspective.
        verdicts = [
            self._debater_turn(perspective, user_content, prior_verdicts=None)
            for perspective in self.debater_perspectives
        ]

        # 2. Debate rounds: each debater revises in light of the others.
        for _ in range(self.n_rounds):
            verdicts = [
                self._debater_turn(perspective, user_content, prior_verdicts=verdicts)
                for perspective in self.debater_perspectives
            ]

        # 3. The chair synthesizes a single consolidated verdict.
        return self._chair_turn(user_content, verdicts)

    def _debater_turn(
        self,
        perspective: str,
        user_content: str,
        prior_verdicts: Optional[List[DeepResearchPairwisePreferenceOutput]],
    ) -> DeepResearchPairwisePreferenceOutput:
        """One debater's turn: open independently, or revise against peers."""
        system = f"{DEEP_RESEARCH_PAIRWISE_PROMPT}\n\n{perspective}"
        if prior_verdicts:
            system += (
                "\n\nBelow are the other debaters' current verdicts "
                "(anonymized). Reconsider your position in light of their "
                "evidence: revise where their arguments are stronger and hold "
                "firm where yours are. Do not simply copy the majority.\n\n"
                f"{self._render_verdicts(prior_verdicts)}"
            )
        return self._structured_call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        )

    def _chair_turn(
        self,
        user_content: str,
        verdicts: List[DeepResearchPairwisePreferenceOutput],
    ) -> DeepResearchPairwisePreferenceOutput:
        """The chair's synthesis turn: produce the final consolidated verdict."""
        system = (
            f"{DEEP_RESEARCH_PAIRWISE_PROMPT}\n\n{CHAIR_SYSTEM_PROMPT}\n\n"
            f"{self._render_verdicts(verdicts)}"
        )
        return self._structured_call(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        )

    @staticmethod
    def _render_verdicts(
        verdicts: List[DeepResearchPairwisePreferenceOutput],
    ) -> str:
        """Render debater verdicts as a compact, anonymized per-dimension block."""
        blocks = []
        for index, verdict in enumerate(verdicts, start=1):
            lines = [f"Debater {index}:"]
            for dimension in DIMENSIONS:
                preference = getattr(verdict, dimension)
                lines.append(
                    f"  - {dimension}: prefers {preference.preferred} "
                    f"(gap {preference.gap_score}/5) - {preference.explanation}"
                )
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
