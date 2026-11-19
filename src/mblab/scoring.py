"""Evaluation engine: deterministic scoring with pre-registered rules.

Automatic methods (``exact_match``, ``regex``, ``numeric``) read an answer out of
the response and compare it with the rule in the experiment definition. The
``manual`` method uses points from human graders (see :mod:`mblab.grading`).

Outcomes are kept apart on purpose:

* ``scored`` — an answer was found and judged (correct or incorrect).
* ``format_failure`` — no usable answer could be extracted. Counted as
  incorrect in accuracy, but reported separately, because a formatting habit is
  not the same thing as a wrong answer.
* ``pending_manual`` — a manually graded task without a grade yet. Excluded.
* ``not_scorable`` — the client returned an error, a refusal, or an incomplete
  response. Excluded from accuracy, counted as a failure in task success rate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from mblab.schema import (
    ExactMatchScoring,
    ManualGrade,
    NumericScoring,
    RegexScoring,
    RunRecord,
    Task,
)

ScoreStatus = Literal["scored", "format_failure", "pending_manual", "not_scorable"]
ScoreSource = Literal["automatic", "manual", "none"]

_EDGE_MARKDOWN = "*_~`"
_EDGE_PUNCTUATION = " \t\"'“”‘’.,;:!?"
_NUMBER_RE = re.compile(r"[-−]?\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


@dataclass(frozen=True)
class ScoreResult:
    status: ScoreStatus
    correct: bool | None
    score: float | None
    extracted_answer: str | None
    detail: str
    source: ScoreSource
    n_graders: int = 0
    grader_disagreement: bool = False


def _answer_line_re(label: str) -> re.Pattern[str]:
    return re.compile(
        rf"^[ \t>*_#`~-]*{re.escape(label)}[ \t*_`~]*[:：](?P<value>.*)$",
        re.IGNORECASE | re.MULTILINE,
    )


def strip_markdown_edges(text: str) -> str:
    return text.strip(" \t\r\n" + _EDGE_MARKDOWN)


def extract_answer(response_text: str, mode: str, label: str = "ANSWER") -> tuple[str | None, str]:
    """Return ``(answer, detail)``. ``answer`` is ``None`` when nothing usable was found."""
    if mode == "full_response":
        text = response_text.strip()
        return (text or None), "full response"
    if mode == "last_line":
        lines = [ln for ln in response_text.splitlines() if ln.strip()]
        if not lines:
            return None, "empty response"
        return strip_markdown_edges(lines[-1]), "last non-empty line"

    matches = list(_answer_line_re(label).finditer(response_text))
    if not matches:
        return None, f"no '{label}:' line found"
    last = matches[-1]
    value = strip_markdown_edges(last.group("value"))
    if not value:
        # "ANSWER:" alone on a line, value on the next non-empty line.
        rest = response_text[last.end() :].splitlines()
        following = [ln for ln in rest if ln.strip()]
        if following:
            value = strip_markdown_edges(following[0])
    if not value:
        return None, f"'{label}:' line is empty"
    detail = f"last of {len(matches)} '{label}:' lines" if len(matches) > 1 else f"'{label}:' line"
    return value, detail


def normalize_text(text: str, steps: Sequence[str]) -> str:
    for step in steps:
        if step == "strip_markdown":
            text = strip_markdown_edges(text)
        elif step == "lowercase":
            text = text.casefold()
        elif step == "collapse_whitespace":
            text = " ".join(text.split())
        elif step == "remove_whitespace":
            text = "".join(text.split())
        elif step == "strip_edge_punctuation":
            text = text.strip(_EDGE_PUNCTUATION)
        else:  # pragma: no cover - guarded by the schema Literal
            raise ValueError(f"unknown normalization step: {step}")
    return text


def parse_single_number(text: str) -> tuple[float | None, str]:
    """Parse exactly one distinct number from ``text`` (thousands separators allowed)."""
    cleaned = _THOUSANDS_RE.sub("", text)
    values = {float(m.replace("−", "-")) for m in _NUMBER_RE.findall(cleaned)}
    if not values:
        return None, "no number in the answer"
    if len(values) > 1:
        shown = ", ".join(f"{v:g}" for v in sorted(values))
        return None, f"more than one number in the answer ({shown})"
    return values.pop(), "one number"


def score_automatic(task: Task, response_text: str) -> ScoreResult:
    rule = task.scoring
    if not isinstance(rule, ExactMatchScoring | RegexScoring | NumericScoring):
        raise TypeError(f"task '{task.id}' is not automatically scored")

    answer, detail = extract_answer(response_text, rule.extract, rule.answer_label)
    if answer is None:
        return ScoreResult("format_failure", False, 0.0, None, detail, "automatic")

    if isinstance(rule, ExactMatchScoring):
        got = normalize_text(answer, rule.normalize)
        accepted = {normalize_text(a, rule.normalize) for a in rule.accepted}
        ok = got in accepted
        why = "matches an accepted answer" if ok else "does not match any accepted answer"
        return ScoreResult("scored", ok, float(ok), answer, why, "automatic")

    if isinstance(rule, RegexScoring):
        flags = re.IGNORECASE if rule.ignore_case else 0
        ok = re.search(rule.pattern, answer, flags) is not None
        why = "matches the pattern" if ok else "does not match the pattern"
        return ScoreResult("scored", ok, float(ok), answer, why, "automatic")

    value, parse_detail = parse_single_number(answer)
    if value is None:
        return ScoreResult("format_failure", False, 0.0, answer, parse_detail, "automatic")
    ok = abs(value - rule.expected) <= rule.abs_tolerance
    why = f"parsed {value:g}; expected {rule.expected:g} ± {rule.abs_tolerance:g}"
    return ScoreResult("scored", ok, float(ok), answer, why, "automatic")


def score_record(task: Task, record: RunRecord, grades: Sequence[ManualGrade] = ()) -> ScoreResult:
    if record.response_status != "completed":
        return ScoreResult(
            "not_scorable", None, None, None, f"response_status={record.response_status}", "none"
        )
    if task.scoring.method != "manual":
        return score_automatic(task, record.response_text)

    rule = task.scoring
    if not grades:
        return ScoreResult("pending_manual", None, None, None, "awaiting manual grade", "none")
    mean_points = fmean(g.points for g in grades)
    disagreement = len({g.points for g in grades}) > 1
    return ScoreResult(
        status="scored",
        correct=mean_points >= rule.pass_threshold,
        score=mean_points / rule.max_points,
        extracted_answer=None,
        detail=f"mean {mean_points:g}/{rule.max_points:g} from {len(grades)} grade(s)",
        source="manual",
        n_graders=len(grades),
        grader_disagreement=disagreement,
    )


def check_reference_answer(task: Task) -> str | None:
    """Return an error message if the reference answer fails its own scoring rule."""
    rule = task.scoring
    if rule.method == "manual" or task.reference_answer is None:
        return None
    if rule.extract == "answer_line":
        response = f"{rule.answer_label}: {task.reference_answer}"
    else:
        response = task.reference_answer
    result = score_automatic(task, response)
    if not result.correct:
        return (
            f"reference_answer {task.reference_answer!r} is not scored correct by its own "
            f"rule ({result.detail})"
        )
    return None
