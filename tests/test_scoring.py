from __future__ import annotations

import pytest
from pydantic import ValidationError

from mblab.records import parse_thinking_seconds
from mblab.schema import ManualGrade, NumericScoring, SequenceScoring, Task
from mblab.scoring import (
    check_reference_answer,
    extract_answer,
    normalize_text,
    parse_single_number,
    score_record,
)

from .conftest import FIXED_NOW, make_record


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\nANSWER: 265", "265"),
        ("**ANSWER:** `265`", "265"),
        ("> answer : 265 minutes", "265 minutes"),
        ("ANSWER: 1\nmore thinking\nANSWER: 2", "2"),  # last answer line wins
        ("ANSWER:\n\n  42  \n", "42"),  # value on the following line
        ("ANSWER： 7", "7"),  # full-width colon
        ("The answer: 265", None),  # must start the line
        ("no marker at all", None),
    ],
)
def test_extract_answer_line(text, expected):
    answer, _ = extract_answer(text, "answer_line")
    assert answer == expected


def test_extract_other_modes():
    assert extract_answer("a\n\n b \n", "last_line")[0] == "b"
    assert extract_answer("  whole  ", "full_response")[0] == "whole"
    assert extract_answer("   ", "full_response")[0] is None


@pytest.mark.parametrize(
    ("text", "value"),
    [("265", 265), ("1,234", 1234), ("−3", -3), ("265 minutes (265)", 265), ("2.5 hours", 2.5)],
)
def test_parse_single_number(text, value):
    assert parse_single_number(text)[0] == value


@pytest.mark.parametrize("text", ["", "about two hundred", "4 hours 25 minutes", "1, 2"])
def test_parse_single_number_rejects_missing_or_ambiguous(text):
    assert parse_single_number(text)[0] is None


def test_normalize_text_steps():
    assert (
        normalize_text(
            " **Ben,  CHO.** ",
            ["strip_markdown", "collapse_whitespace", "lowercase", "strip_edge_punctuation"],
        )
        == "ben, cho"
    )
    assert normalize_text("(37, 13)", ["remove_whitespace", "strip_edge_punctuation"]) == "(37,13)"


@pytest.mark.parametrize(
    ("task_id", "answer", "correct", "status"),
    [
        ("T1", "265", True, "scored"),
        ("T1", "265 minutes", True, "scored"),
        ("T1", "4 hours 25 minutes", False, "format_failure"),
        ("T1", "264", False, "scored"),
        ("T2", "2027-02-27", True, "scored"),
        ("T2", "2027-02-27 (Saturday)", True, "scored"),
        ("T2", "2027-02-27 or 2027-02-28", False, "format_failure"),  # two dates: unreadable
        ("T3", "22", True, "scored"),
        ("T3", "21", False, "scored"),
        ("T4", "Ben, Cho, Eun, Dev, Ana", True, "scored"),
        ("T4", "ben - cho - eun - dev - ana.", True, "scored"),
        ("T4", "Ben, Cho, Eun, Ana, Dev", False, "scored"),
        ("T5", "633", True, "scored"),
        ("T6", "(37, 13)", True, "scored"),
        ("T6", "`(37,13)`", True, "scored"),
        ("T6", "(1, 6)", False, "scored"),
        # The review's correct answers in other wordings, scored wrong by the 0.1.0 rules:
        ("T2", "2027-02-27 (Saturday, 27 February 2027)", True, "scored"),
        ("T4", "Ben, Cho, Eun, Dev, and Ana", True, "scored"),
        ("T4", "1. Ben, 2. Cho, 3. Eun, 4. Dev, 5. Ana", True, "scored"),
        ("T6", "It prints (37, 13)", True, "scored"),
        # Unreadable is not wrong: a format failure, reported apart from wrong answers.
        ("T2", "27 February 2027", False, "format_failure"),
        ("T4", "Ben, Cho, Eun, Dev", False, "format_failure"),
        ("T6", "(37, 13, 2027)", False, "format_failure"),
        ("T6", "(13, 37)", False, "scored"),
    ],
)
def test_template_scoring_rules(definition, plan, task_id, answer, correct, status):
    run_id = f"A:{task_id}:r1"
    rec = make_record(definition, plan, run_id, response_text=f"reasoning...\nANSWER: {answer}")
    result = score_record(definition.task(task_id), rec)
    assert (result.correct, result.status) == (correct, status)
    assert result.source == "automatic"


def test_missing_answer_line_is_a_format_failure(definition, plan):
    rec = make_record(definition, plan, "A:T1:r1", response_text="It takes 265 minutes.")
    result = score_record(definition.task("T1"), rec)
    assert result.status == "format_failure"
    assert result.correct is False and result.score == 0.0


def test_non_completed_responses_are_not_scorable(definition, plan):
    rec = make_record(definition, plan, "A:T1:r1", response_status="refused", response_text="")
    result = score_record(definition.task("T1"), rec)
    assert result.status == "not_scorable" and result.correct is None


def _manual_task() -> Task:
    return Task.model_validate(
        {
            "id": "M1",
            "title": "Explain",
            "prompt": "Explain X.",
            "scoring": {
                "method": "manual",
                "rubric": "2 = complete",
                "max_points": 2,
                "pass_threshold": 2,
            },
        }
    )


def test_manual_scoring_uses_mean_of_grades(definition, plan):
    rec = make_record(definition, plan, "A:T1:r1")
    task = _manual_task()
    assert score_record(task, rec).status == "pending_manual"
    grades = [
        ManualGrade(
            run_id=rec.run_id, points=p, max_points=2, grader_id=g, blind=True, graded_at=FIXED_NOW
        )
        for p, g in ((2, "g1"), (1, "g2"))
    ]
    result = score_record(task, rec, grades)
    assert result.status == "scored" and result.source == "manual"
    assert result.score == pytest.approx(0.75)
    assert result.correct is False  # mean 1.5 < pass threshold 2
    assert result.grader_disagreement and result.n_graders == 2


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("Thought for 37s", 37),
        ("Thought for 1m 12s", 72),
        ("Thought for 2 minutes", 120),
        ("Thought for 1h 2m", 3720),
        ("1분 12초 동안 생각함", 72),
        ("Thought for a few seconds", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_thinking_seconds(text, seconds):
    assert parse_thinking_seconds(text) == seconds


def _task(scoring, reference, **extra):
    return Task(id="X", title="x", prompt="p", reference_answer=reference, scoring=scoring, **extra)


def test_answer_examples_are_checked_against_the_rule():
    rule = {"method": "numeric", "expected": [37, 13]}
    examples = {"correct": ["37 and 13"], "unreadable": ["37"]}
    assert check_reference_answer(_task(rule, "(37, 13)", answer_examples=examples)) is None
    bad = _task(rule, "(37, 13)", answer_examples={"incorrect": ["37"]})
    assert "incorrect example '37' is not scored incorrect" in check_reference_answer(bad)


def test_new_rules_reject_bad_definitions():
    with pytest.raises(ValidationError, match="at least two"):
        NumericScoring(method="numeric", expected=[1])
    with pytest.raises(ValidationError, match="distinct"):
        SequenceScoring(method="sequence", expected=["Ana", "ana"])
