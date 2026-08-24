"""Reference-anchored grading rubrics for pairwise deep research evaluation.

Adapted from "Grading Needs a Rubric, Not Intelligence" (arXiv:2608.17938).
The paper's central result is that grading reliability is decoupled from
judge intelligence once the judge is anchored in an explicit rubric, and
that *within* the rubric the official answer does nearly all the work:
dropping the rubric's criteria/levels while keeping the official answer
changes nothing measurable, whereas dropping the official answer too
collapses reliability (ICC 0.888 -> 0.628) and makes judge effort matter
again.

This module ports that anchor as a deterministic, parameter-free
construction over the reference (baseline) report the pairwise pipeline
already receives. The paper builds rubrics with a frontier model reading
source documents once at ingestion; here the reference answer *is* the
official answer, so the extraction is a cheap structural one and the
expensive frontier pass is unnecessary.

The three ``AnchorMode`` values reproduce the paper's ablation arms so a
single run can measure how much of the agreement is carried by the
official answer versus the criteria.
"""

import re
from enum import Enum
from typing import Any, Dict, List

# Short cap on the extracted criteria list. The paper's rubrics enumerate a
# handful of criteria per question; an unbounded list would let a long
# reference dominate the judge prompt.
MAX_CRITERIA = 8

# Cap on how much of the official answer is echoed into the judge prompt.
# The reference report is already present in full as one of the two reports
# being compared, so echoing all of it would roughly double the prompt for
# no new signal; an excerpt carries the anchor at bounded cost.
MAX_OFFICIAL_ANSWER_CHARS = 6000

_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.{6,})$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.{3,})$")

# Lead words that mark a topic sentence as a criterion worth covering.
_CRITERION_HINTS = (
    "consider",
    "analyze",
    "evaluate",
    "assess",
    "explore",
    "identify",
    "compare",
    "include",
    "how",
    "what",
    "why",
    "impact",
    "risk",
    "trend",
    "opportunit",
    "challenge",
    "strateg",
)


class AnchorMode(str, Enum):
    """Which parts of the rubric the judge prompt is anchored to."""

    #: Full rubric: criteria/levels plus the official answer (paper's main arm).
    FULL = "full"
    #: Official answer only, criteria removed (paper's first ablation).
    ANSWER_ONLY = "answer_only"
    #: No anchor at all - the flat dimension descriptions (paper's second
    #: ablation, and the behaviour of the stock pairwise prompt).
    NONE = "none"


def _clean(text: str) -> str:
    """Normalise a fragment of markdown into a single-line criterion."""
    text = re.sub(r"\s+", " ", text).strip()
    # Strip markdown links, which the metric's input validator also removes
    # from the answers themselves - a criterion should read as a claim, not
    # as a citation payload.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    # Drop escaped punctuation and inline emphasis produced by markdown
    # rendering, and any bold/emphasis markers mid-criterion.
    text = text.replace("\\.", ".").replace("\\,", ",")
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    return text.strip(" -–—:*#").strip()


def _is_criterion(text: str) -> bool:
    """Whether a fragment reads as a checkable requirement."""
    lowered = text.lower()
    if len(lowered) < 8:
        return False
    return any(hint in lowered for hint in _CRITERION_HINTS)


def extract_criteria(question: str, official_answer: str) -> List[str]:
    """Extract checkable criteria for one question from its official answer.

    Sources, in order of confidence: explicit imperatives in the question
    ("Consider...", "Analyze..."), markdown headings and bullets in the
    official answer, and topic sentences. Deterministic and
    parameter-free - the same input always yields the same rubric.
    """
    criteria: List[str] = []

    # 1. Imperatives from the question. These are the asker's own
    # requirements, so they outrank anything we can infer from prose.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", question or ""):
        fragment = _clean(sentence)
        if fragment and _is_criterion(fragment):
            criteria.append(fragment)

    # 2. Structure of the official answer: headings name the ground the
    # report chose to cover, bullets name the claims it made there.
    for line in (official_answer or "").splitlines():
        heading = _HEADING_RE.match(line)
        bullet = _BULLET_RE.match(line)
        if not heading and not bullet:
            continue
        fragment = _clean(heading.group(1) if heading else bullet.group(1))
        if fragment and _is_criterion(fragment):
            criteria.append(fragment)

    # 3. Fallback: leading topic sentences, so an unstructured reference
    # still contributes an anchor.
    if not criteria:
        paragraphs = [p for p in (official_answer or "").split("\n\n") if p.strip()]
        for paragraph in paragraphs:
            fragment = _clean(paragraph.split(".")[0])
            if fragment and _is_criterion(fragment):
                criteria.append(fragment)

    # De-duplicate case-insensitively, preserving first occurrence.
    seen = set()
    unique = []
    for criterion in criteria:
        key = criterion.lower()
        if key not in seen:
            seen.add(key)
            unique.append(criterion)

    return unique[:MAX_CRITERIA]


def build_rubric(
    question: str,
    official_answer: str,
    mode: AnchorMode = AnchorMode.FULL,
) -> Dict[str, Any]:
    """Build the grading rubric for one question.

    Returns a dict with ``mode``, ``criteria`` and ``official_answer`` keys.
    Criteria are dropped entirely under ``ANSWER_ONLY`` and the official
    answer under ``NONE``, mirroring the paper's ablation arms.
    """
    rubric: Dict[str, Any] = {"mode": mode.value, "criteria": [], "official_answer": ""}
    if mode is AnchorMode.NONE:
        return rubric

    if mode is AnchorMode.FULL:
        rubric["criteria"] = extract_criteria(question, official_answer)

    rubric["official_answer"] = _excerpt(official_answer)
    return rubric


def _excerpt(official_answer: str, limit: int = MAX_OFFICIAL_ANSWER_CHARS) -> str:
    """Bound the official answer echoed into the judge prompt.

    Takes a head and a tail around the midpoint so long references still
    show both their framing and their conclusions rather than only an
    opening. Unchanged when already within the limit.
    """
    text = (official_answer or "").strip()
    if len(text) <= limit:
        return text

    half = limit // 2
    return f"{text[:half]}\n\n[... reference answer truncated ...]\n\n{text[-half:]}"


def render_rubric_anchor(rubric: Dict[str, Any]) -> str:
    """Render a rubric as the prompt fragment handed to the judge.

    An empty string means "no anchor" - callers append the fragment only if
    non-empty, so ``AnchorMode.NONE`` reproduces the stock flat prompt.
    """
    criteria = rubric.get("criteria") or []
    official_answer = rubric.get("official_answer") or ""

    if not criteria and not official_answer:
        return ""

    parts: List[str] = ["<grading_rubric>"]

    if criteria:
        parts.append("Grade against these explicit criteria, in order:")
        parts.extend(f"{i}. {criterion}" for i, criterion in enumerate(criteria, 1))
        parts.append("")
    else:
        parts.append("Grade against the official answer below.")

    if official_answer:
        parts.extend(
            [
                "<official_answer>",
                official_answer.strip(),
                "</official_answer>",
                "",
                "The official answer is the reference standard of coverage. A report is",
                "more comprehensive and complete when it covers ground the official",
                "answer covers, and weaker when it omits that ground. Do not reward a",
                "report merely for being longer than the official answer.",
            ]
        )

    parts.append("</grading_rubric>")
    return "\n".join(parts)
