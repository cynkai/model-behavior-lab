"""Evaluation engine: deterministic scoring with pre-registered rules.

Automatic methods (``exact_match``, ``regex``, ``numeric``) read an answer out of
the response and compare it with the rule in the experiment definition. The
``manual`` method uses points from human graders (see :mod:`mblab.grading`).

Outcomes are kept apart on purpose:

* ``scored`` — an answer was found and judged (correct or incorrect).
* ``format_failure`` — no usable answer could be extracted or read (no answer
  line, no number, two different dates, a name missing from a sequence). Counted
  as incorrect in accuracy, but reported separately, because a formatting habit is
  not the same thing as a wrong answer.
* ``pending_manual`` — a manually graded task without a grade yet. Excluded.
* ``not_scorable`` — the client returned an error, a refusal, or an incomplete
  response. Excluded from accuracy, counted as a failure in task success rate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean
from typing import Literal

from mblab.schema import (
    DateScoring,
    ExactMatchScoring,
    ManualGrade,
    NumericScoring,
    RegexScoring,
    RunRecord,
    SequenceScoring,
    Task,
)

ScoreStatus = Literal["scored", "format_failure", "pending_manual", "not_scorable"]
ScoreSource = Literal["automatic", "manual", "none"]

_EDGE_MARKDOWN = "*_~`"
_EDGE_PUNCTUATION = " \t\"'“”‘’.,;:!?"
_NUMBER_RE = re.compile(r"[-−]?\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_ISO_DATE_RE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
AUTOMATIC_RULES = (ExactMatchScoring, RegexScoring, NumericScoring, DateScoring, SequenceScoring)


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
    #: The answer as the rule read it ("265", "2027-02-27", "Ben, Cho, Eun, Dev, Ana"), so
    #: answers that differ only in wording count as one when distinct answers are counted.
    canonical: str | None = None


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


def parse_single_decimal(text: str) -> tuple[Decimal | None, str]:
    """Read the numeric rule's human formats without rounding exact benchmark counts."""
    cleaned = _THOUSANDS_RE.sub("", text)
    values = {Decimal(m.replace("−", "-")) for m in _NUMBER_RE.findall(cleaned)}
    if not values:
        return None, "no number in the answer"
    if len(values) > 1:
        shown = ", ".join(str(value) for value in sorted(values))
        return None, f"more than one number in the answer ({shown})"
    return values.pop(), "one number"


def parse_numbers(text: str) -> list[float]:
    """Every number in ``text``, in order (thousands separators allowed)."""
    cleaned = _THOUSANDS_RE.sub("", text)
    return [float(m.replace("−", "-")) for m in _NUMBER_RE.findall(cleaned)]


def find_sequence(text: str, items: Sequence[str]) -> tuple[list[str] | None, str]:
    """The items in the order ``text`` names them, or ``None`` if one is missing or repeated."""
    positions: dict[str, int] = {}
    problems = []
    for item in items:
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, item.split())) + r"(?!\w)"
        found = [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
        if len(found) != 1:
            problems.append(f"{item} named {len(found)} times")
        else:
            positions[item] = found[0]
    if problems:
        return None, "; ".join(problems)
    return sorted(items, key=positions.__getitem__), "each item named once"


def _scored(ok: bool, answer: str, why: str, canonical: str) -> ScoreResult:
    return ScoreResult("scored", ok, float(ok), answer, why, "automatic", canonical=canonical)


def _unreadable(answer: str | None, why: str) -> ScoreResult:
    return ScoreResult("format_failure", False, 0.0, answer, why, "automatic")


def score_automatic(task: Task, response_text: str) -> ScoreResult:
    rule = task.scoring
    if not isinstance(rule, AUTOMATIC_RULES):
        raise TypeError(f"task '{task.id}' is not automatically scored")

    answer, detail = extract_answer(response_text, rule.extract, rule.answer_label)
    if answer is None:
        return _unreadable(None, detail)

    if isinstance(rule, ExactMatchScoring):
        got = normalize_text(answer, rule.normalize)
        accepted = {normalize_text(a, rule.normalize) for a in rule.accepted}
        ok = got in accepted
        why = "matches an accepted answer" if ok else "does not match any accepted answer"
        return _scored(ok, answer, why, got)

    if isinstance(rule, RegexScoring):
        flags = re.IGNORECASE if rule.ignore_case else 0
        ok = re.search(rule.pattern, answer, flags) is not None
        why = "matches the pattern" if ok else "does not match the pattern"
        return _scored(ok, answer, why, " ".join(answer.split()).casefold())

    if isinstance(rule, DateScoring):
        dates = sorted(set(_ISO_DATE_RE.findall(answer)))
        if not dates:
            return _unreadable(answer, "no YYYY-MM-DD date in the answer")
        if len(dates) > 1:
            return _unreadable(answer, f"more than one date in the answer ({', '.join(dates)})")
        expected = rule.expected.isoformat()
        why = f"read {dates[0]}; expected {expected}"
        return _scored(dates[0] == expected, answer, why, dates[0])

    if isinstance(rule, SequenceScoring):
        order, why = find_sequence(answer, rule.expected)
        if order is None:
            return _unreadable(answer, why)
        got = ", ".join(order)
        ok = [i.casefold() for i in order] == [i.casefold() for i in rule.expected]
        return _scored(ok, answer, f"read {got}; expected {', '.join(rule.expected)}", got)

    if isinstance(rule.expected, list):
        values = parse_numbers(answer)
        if len(values) != len(rule.expected):
            return _unreadable(
                answer, f"expected {len(rule.expected)} numbers, found {len(values)}"
            )
        ok = all(
            abs(v - e) <= rule.abs_tolerance for v, e in zip(values, rule.expected, strict=True)
        )
        got = ", ".join(f"{v:g}" for v in values)
        expected = ", ".join(f"{e:g}" for e in rule.expected)
        return _scored(ok, answer, f"read {got}; expected {expected} ± {rule.abs_tolerance:g}", got)

    value, parse_detail = parse_single_number(answer)
    if value is None:
        return _unreadable(answer, parse_detail)
    ok = abs(value - rule.expected) <= rule.abs_tolerance
    why = f"parsed {value:g}; expected {rule.expected:g} ± {rule.abs_tolerance:g}"
    return _scored(ok, answer, why, f"{value:g}")


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


def _as_response(task: Task, answer: str) -> str:
    rule = task.scoring
    return f"{rule.answer_label}: {answer}" if rule.extract == "answer_line" else answer


def check_reference_answer(task: Task) -> str | None:
    """An error message if the reference answer or an answer example breaks the rule.

    The reference answer and every ``answer_examples.correct`` entry must score correct,
    every ``incorrect`` entry must be read and scored wrong, and every ``unreadable``
    entry must be a format failure.
    """
    rule = task.scoring
    if rule.method == "manual" or task.reference_answer is None:
        return None
    examples = task.answer_examples
    expectations = [(task.reference_answer, "correct", "reference_answer")]
    if examples is not None:
        expectations += [(a, "correct", "correct example") for a in examples.correct]
        expectations += [(a, "incorrect", "incorrect example") for a in examples.incorrect]
        expectations += [(a, "unreadable", "unreadable example") for a in examples.unreadable]
    for answer, wanted, what in expectations:
        result = score_automatic(task, _as_response(task, answer))
        got = (
            "unreadable"
            if result.status == "format_failure"
            else ("correct" if result.correct else "incorrect")
        )
        if got != wanted:
            return (
                f"{what} {answer!r} is not scored {wanted} by its own rule "
                f"(it is {got}: {result.detail})"
            )
    return None
