from __future__ import annotations

from pathlib import Path

import pytest

from mblab.comparison_protocol import check_comparable
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition, ManualGrade, ManualScoring
from mblab.study import LineError, Study, StudyError

from .conftest import FIXED_NOW, make_record


@pytest.fixture
def pair(tmp_path: Path, definition: ExperimentDefinition) -> tuple[Study, Study]:
    studies = []
    for side in ("left", "right"):
        current = definition.model_copy(deep=True)
        plan = generate_plan(current, now=FIXED_NOW)
        studies.append(Study(tmp_path / side, current, plan, [], []))
    return studies[0], studies[1]


def _replan(study: Study) -> None:
    study.plan = generate_plan(study.definition, seed=77, now=FIXED_NOW)


def test_comparison_accepts_different_metadata_order_and_sample_sizes(
    pair: tuple[Study, Study],
) -> None:
    # Given
    left, right = pair
    right.definition.platform = "Other platform"
    right.definition.experiment_id = "other-experiment"
    right.definition.experiment_version = "2.0.0"
    right.definition.repetitions = 1
    right.definition.conditions.reverse()
    right.definition.tasks.reverse()
    right.definition.conditions[0].label = "Other heading"
    right.definition.tasks[0].title = "Other title"
    right.definition.selection_options[0].observed_label_aliases = ["new alias"]
    right.definition.confounders = []
    _replan(right)
    plan = right.require_plan()
    for index, run in enumerate(reversed(plan.runs), start=1):
        run.sequence = index
        run.run_id = f"custom-{index}"
    # When
    reports = check_comparable(left, right)
    # Then
    assert all(report.ok for report in reports)
    assert reports[0].n_missing > reports[1].n_missing > 0


@pytest.mark.parametrize(
    "field",
    [
        "service",
        "prompt",
        "scoring",
        "reference_answer",
        "task_id",
        "condition_id",
        "model_selection",
        "message_override",
        "option_label",
    ],
)
def test_comparison_rejects_changed_protocol(pair: tuple[Study, Study], field: str) -> None:
    # Given
    left, right = pair
    definition = right.definition
    changes = {
        "service": (definition, "service", "other service"),
        "prompt": (definition.tasks[0], "prompt", "different input"),
        "scoring": (
            definition.tasks[0],
            "scoring",
            ManualScoring(method="manual", rubric="test", max_points=1, pass_threshold=1),
        ),
        "reference_answer": (definition.tasks[0], "reference_answer", "other answer"),
        "task_id": (definition.tasks[0], "id", "another-task"),
        "condition_id": (definition.conditions[0], "id", "another-condition"),
        "model_selection": (
            definition.conditions[0],
            "model_selection",
            definition.selection_options[-1].id,
        ),
        "message_override": (
            definition.conditions[0],
            "message_override",
            definition.selection_options[-1].id,
        ),
        "option_label": (definition.selection_options[0], "label", "different label"),
    }
    target, name, value = changes[field]
    setattr(target, name, value)
    _replan(right)
    # When / Then
    with pytest.raises(StudyError, match="Left/Right"):
        check_comparable(left, right)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "problem",
    [
        "no_plan",
        "pilot",
        "api",
        "hash",
        "snapshot",
        "identity",
        "version",
        "strategy",
        "missing_slot",
        "duplicate_slot",
        "sequence",
        "run_id",
        "run_identity",
        "run_version",
        "block",
    ],
)
def test_comparison_rejects_invalid_plan(
    pair: tuple[Study, Study], side: int, problem: str
) -> None:
    # Given
    study = pair[side]
    plan = study.require_plan()
    if problem == "no_plan":
        study.plan = None
    elif problem in {"pilot", "api"}:
        setattr(
            study.definition,
            "pilot" if problem == "pilot" else "execution_method",
            True if problem == "pilot" else "official_api",
        )
    elif problem == "hash":
        plan.definition_sha256 = "private-value"
    elif problem == "snapshot":
        plan.definition_snapshot["title"] = "private-value"
    elif problem in {"identity", "version", "strategy"}:
        setattr(
            plan,
            {"identity": "experiment_id", "version": "experiment_version", "strategy": "strategy"}[
                problem
            ],
            "private-value",
        )
    elif problem == "missing_slot":
        plan.runs.pop()
    elif problem == "duplicate_slot":
        plan.runs[1] = plan.runs[0].model_copy(update={"run_id": "unique", "sequence": 2})
    else:
        field = {
            "sequence": "sequence",
            "run_id": "run_id",
            "run_identity": "experiment_id",
            "run_version": "experiment_version",
            "block": "block",
        }[problem]
        value = {
            "sequence": plan.runs[1].sequence,
            "run_id": plan.runs[1].run_id,
            "run_identity": "private-value",
            "run_version": "9.0.0",
            "block": 999,
        }[problem]
        setattr(plan.runs[0], field, value)
    # When / Then
    with pytest.raises(StudyError, match="Left" if side == 0 else "Right") as caught:
        check_comparable(*pair)
    assert "private-value" not in str(caught.value)
    assert str(study.root) not in str(caught.value)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "unknown",
        "identity",
        "prompt",
        "whitespace",
        "hash",
        "version",
        "unreadable_record",
        "unreadable_grade",
        "mixed_origin",
    ],
)
def test_comparison_rejects_invalid_records(pair: tuple[Study, Study], problem: str) -> None:
    # Given
    left, right = pair
    record = make_record(right.definition, right.require_plan(), "A:T1:r1")
    right.records = [record]
    if problem == "duplicate":
        right.records.append(record)
    elif problem.startswith("unreadable"):
        right.load_errors = [LineError("private-path", 1, "private-content")]
    elif problem == "mixed_origin":
        right.records.append(
            make_record(right.definition, right.require_plan(), "B:T1:r1", data_origin="synthetic")
        )
    else:
        field, value = {
            "unknown": ("run_id", "private-content"),
            "identity": ("condition_id", "private-content"),
            "prompt": ("prompt_text", "private-content"),
            "whitespace": ("prompt_text", record.prompt_text + "\n"),
            "hash": ("definition_sha256", "private-content"),
            "version": ("experiment_version", "9.0.0"),
        }[problem]
        setattr(record, field, value)
    # When / Then
    with pytest.raises(StudyError, match="Right") as caught:
        check_comparable(left, right)
    assert "private-content" not in str(caught.value)
    assert "private-path" not in str(caught.value)


def test_comparison_preserves_excluded_deviations_and_warnings(pair: tuple[Study, Study]) -> None:
    # Given
    left, right = pair
    right.records = [
        make_record(
            right.definition,
            right.require_plan(),
            "A:T1:r1",
            prompt_text="different input",
            protocol_deviation="changed prompt",
            definition_sha256="different",
            experiment_version="9.0.0",
        )
    ]
    # When
    _, report = check_comparable(left, right)
    # Then
    assert report.n_records == report.n_excluded == 1
    assert {issue.code for issue in report.warnings} >= {
        "prompt_mismatch",
        "hash_mismatch",
        "version_mismatch",
    }


@pytest.mark.parametrize("problem", ["duplicate", "scale", "unknown"])
def test_comparison_rejects_invalid_manual_grades(pair: tuple[Study, Study], problem: str) -> None:
    # Given
    left, right = pair
    for study in pair:
        study.definition.tasks[0].scoring = ManualScoring(
            method="manual", rubric="test", max_points=2, pass_threshold=2
        )
        _replan(study)
    right.records = [make_record(right.definition, right.require_plan(), "A:T1:r1")]
    grade = ManualGrade(
        run_id="A:T1:r1",
        points=1,
        max_points=2,
        grader_id="private-grader",
        blind=True,
        graded_at=FIXED_NOW,
    )
    right.grades = [grade]
    if problem == "duplicate":
        right.grades.append(grade)
    elif problem == "scale":
        grade.max_points = 3
    else:
        grade.run_id = "private-run"
    # When / Then
    with pytest.raises(StudyError, match="Right") as caught:
        check_comparable(left, right)
    assert "private-grader" not in str(caught.value)
    assert "private-run" not in str(caught.value)


def test_comparison_rejects_real_and_synthetic_studies(pair: tuple[Study, Study]) -> None:
    # Given
    left, right = pair
    right.definition.demo = True
    _replan(right)
    # When / Then
    with pytest.raises(StudyError, match="observed and synthetic"):
        check_comparable(left, right)
