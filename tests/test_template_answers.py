"""Independently recompute every reference answer in the bundled template.

If a task prompt changes, its answer has to be re-derived here as well; the
experiment must never ship an unverified answer key.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import re
from datetime import date, datetime, timedelta

from mblab.scoring import check_reference_answer


def test_every_task_has_a_self_consistent_reference_answer(definition):
    assert len(definition.tasks) == 6
    for task in definition.tasks:
        assert task.reference_answer
        assert check_reference_answer(task) is None


def test_t1_elapsed_minutes(definition):
    start = datetime(2026, 1, 1, 9, 47)
    end = datetime(2026, 1, 1, 14, 12)
    assert "09:47" in definition.task("T1").prompt and "14:12" in definition.task("T1").prompt
    assert int((end - start).total_seconds() // 60) == 265
    assert definition.task("T1").scoring.expected == 265


def test_t2_calendar_offset(definition):
    assert "19 November 2026" in definition.task("T2").prompt
    assert (date(2026, 11, 19) + timedelta(days=100)).isoformat() == "2027-02-27"
    assert definition.task("T2").reference_answer == "2027-02-27"


def test_t3_letter_count(definition):
    sentence = "Seventeen elegant elephants entered the emerald temple, never resting even once."
    assert sentence in definition.task("T3").prompt
    assert sentence.lower().count("e") == 22
    assert definition.task("T3").scoring.expected == 22


def test_t4_seating_puzzle_has_exactly_one_solution(definition):
    names = ["Ana", "Ben", "Cho", "Dev", "Eun"]
    solutions = []
    for chairs in itertools.permutations(range(1, 6)):
        s = dict(zip(names, chairs, strict=True))
        if (
            s["Ben"] < s["Eun"]
            and s["Cho"] % 2 == 0
            and abs(s["Ana"] - s["Dev"]) == 1
            and s["Eun"] != 5
            and s["Dev"] not in (1, 5)
        ):
            solutions.append(sorted(names, key=s.get))
    assert solutions == [["Ben", "Cho", "Eun", "Dev", "Ana"]]
    assert definition.task("T4").reference_answer == "Ben, Cho, Eun, Dev, Ana"


def test_t5_digit_sum_count(definition):
    count = sum(1 for n in range(1, 10_000) if sum(map(int, str(n))) == 20)
    assert count == 633
    assert definition.task("T5").scoring.expected == 633


def test_t6_code_trace_matches_real_output(definition):
    prompt = definition.task("T6").prompt
    code = re.search(r"(def f\(n\):.*?print\(f\(2027\)\))", prompt, re.DOTALL).group(1)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, {})  # the snippet is a fixed string from our own template
    assert buffer.getvalue().strip() == "(37, 13)"
    assert definition.task("T6").reference_answer == "(37, 13)"


def test_rule_fields_match_the_recomputed_answers(definition):
    # The 0.2 rules state the answers in their own fields; they must agree with the
    # independently recomputed values above.
    assert definition.task("T2").scoring.expected == date(2026, 11, 19) + timedelta(days=100)
    assert definition.task("T4").scoring.expected == ["Ben", "Cho", "Eun", "Dev", "Ana"]
    assert definition.task("T6").scoring.expected == [37, 13]
    for task in definition.tasks:
        assert task.answer_examples is not None and task.answer_examples.correct
