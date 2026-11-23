from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from mblab.agreement import TaskAgreement, summarize_agreement
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition, ManualGrade, ManualScoring, ResponseStatus, Task
from mblab.study import Study

from .conftest import FIXED_NOW, make_record


@pytest.fixture
def agreement_study(tmp_path: Path, definition: ExperimentDefinition) -> Study:
    tasks = [
        Task(
            id=task_id,
            title=task_id,
            prompt="Explain",
            scoring=ManualScoring(
                method="manual",
                rubric="Award points",
                max_points=maximum,
                pass_threshold=threshold,
            ),
        )
        for task_id, maximum, threshold in (("Z", 4, 2), ("M", 10, 8))
    ]
    definition = definition.model_copy(update={"tasks": [tasks[0], definition.tasks[0], tasks[1]]})
    plan = generate_plan(definition, now=FIXED_NOW)
    records = [
        make_record(definition, plan, f"A:{task.id}:r{repeat}", response_text="Private answer")
        for task in tasks
        for repeat in (1, 2, 3)
    ]
    return Study(tmp_path, definition, plan, records, [])


def grade(run_id: str, grader_id: str, points: float) -> ManualGrade:
    return ManualGrade(
        run_id=run_id,
        grader_id=grader_id,
        points=points,
        max_points=10 if ":M:" in run_id else 4,
        blind=False,
        graded_at=FIXED_NOW,
        notes="Private notes",
    )


def test_no_grades_keeps_manual_tasks_in_definition_order(agreement_study: Study) -> None:
    # Given: manual tasks with eligible responses but no grades.
    summaries = summarize_agreement(agreement_study, set())
    assert summaries == [TaskAgreement("Z", 3, 0, 0, 0, 0, 0), TaskAgreement("M", 3, 0, 0, 0, 0, 0)]


def test_one_grade_per_response_does_not_create_cross_response_pairs(
    agreement_study: Study,
) -> None:
    agreement_study.grades = [grade(f"A:Z:r{n}", f"grader{n}", 2) for n in (1, 2, 3)]
    summaries = summarize_agreement(agreement_study, set())
    assert summaries[0] == TaskAgreement("Z", 3, 3, 0, 0, 0, 0)


def test_pairs_use_each_task_threshold_and_include_unblinded_grades(agreement_study: Study) -> None:
    # Given: three graders, one exact match and two passing scores per response.
    agreement_study.grades = [
        grade(f"A:{task_id}:r1", f"grader{n}", points)
        for task_id, scores in (("Z", (1, 2, 2)), ("M", (7, 8, 9)))
        for n, points in enumerate(scores)
    ]
    summaries = summarize_agreement(agreement_study, set())
    assert summaries == [TaskAgreement("Z", 3, 1, 1, 3, 1, 1), TaskAgreement("M", 3, 1, 1, 3, 0, 1)]


@pytest.mark.parametrize("status", ["error", "refused", "incomplete"])
def test_excluded_unplaced_and_incomplete_responses_do_not_contribute(
    agreement_study: Study,
    status: ResponseStatus,
) -> None:
    agreement_study.records[0] = agreement_study.records[0].model_copy(
        update={"protocol_deviation": "Changed setup"}
    )
    agreement_study.records[1] = agreement_study.records[1].model_copy(
        update={"response_status": status}
    )
    agreement_study.grades = [
        grade(f"A:Z:r{n}", grader_id, 2) for n in (1, 2, 3) for grader_id in ("one", "two")
    ]
    summaries = summarize_agreement(agreement_study, {"A:Z:r3"})
    assert summaries[0] == TaskAgreement("Z", 0, 0, 0, 0, 0, 0)


def test_two_graders_per_response_add_pairs_without_cross_response_comparisons(
    agreement_study: Study,
) -> None:
    # Given: the same two graders assess two responses differently.
    agreement_study.grades = [
        grade(f"A:Z:r{repeat}", f"grader{n}", points)
        for repeat, scores in ((1, (1, 1)), (2, (2, 3)))
        for n, points in enumerate(scores)
    ]
    summaries = summarize_agreement(agreement_study, set())
    assert summaries[0] == TaskAgreement("Z", 3, 2, 2, 2, 1, 2)


@pytest.mark.parametrize("duplicate_points", [2, 3])
def test_duplicate_graders_and_wrong_scale_are_unusable(
    agreement_study: Study,
    duplicate_points: float,
) -> None:
    # Given: duplicates are unusable even if identical; wrong scale cannot add a rater.
    agreement_study.grades = [
        grade("A:Z:r1", "duplicate", 2),
        grade("A:Z:r1", "duplicate", duplicate_points),
        grade("A:Z:r1", "valid", 2),
        grade("A:Z:r1", "wrong-scale", 2).model_copy(update={"max_points": 10}),
        grade("unrecorded", "orphan", 2),
    ]
    summaries = summarize_agreement(agreement_study, set())
    assert summaries[0] == TaskAgreement("Z", 3, 1, 0, 0, 0, 0)


def test_many_graders_count_unordered_pairs_per_response(agreement_study: Study) -> None:
    # Given: 100 graders: 25 fail, 25 score two points, 50 score three points.
    agreement_study.grades = [
        grade("A:Z:r1", f"grader{n}", points)
        for n, points in enumerate([1] * 25 + [2] * 25 + [3] * 50)
    ]
    summaries = summarize_agreement(agreement_study, set())
    assert summaries[0] == TaskAgreement("Z", 3, 1, 1, 4950, 1825, 3075)


def test_serialized_summary_contains_only_task_and_integer_counts(agreement_study: Study) -> None:
    agreement_study.grades = [grade("A:Z:r1", "Private grader", 2)]
    summaries = summarize_agreement(agreement_study, set())
    assert asdict(summaries[0]) == {
        "task_id": "Z",
        "eligible_runs": 3,
        "graded_runs": 1,
        "multi_rated_runs": 0,
        "pairs": 0,
        "exact_matches": 0,
        "decision_matches": 0,
    }
