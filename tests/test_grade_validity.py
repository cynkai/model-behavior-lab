from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mblab.cli import app
from mblab.grading import export_blind_sheet, import_blind_sheet
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition, ManualGrade, ManualScoring, Task
from mblab.scoring import score_record, usable_manual_grades
from mblab.study import Study, save_plan, write_jsonl
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record


@pytest.fixture
def manual_study(tmp_path: Path, definition: ExperimentDefinition) -> Study:
    task = Task(
        id="T1",
        title="Explain",
        prompt="Explain X.",
        scoring=ManualScoring(
            method="manual", rubric="2 = complete", max_points=2, pass_threshold=2
        ),
    )
    manual_definition = definition.model_copy(update={"tasks": [task, *definition.tasks[1:]]})
    plan = generate_plan(manual_definition, now=FIXED_NOW)
    record = make_record(manual_definition, plan, "A:T1:r1")
    return Study(tmp_path, manual_definition, plan, [record], [])


@pytest.fixture
def grade() -> ManualGrade:
    return ManualGrade(
        run_id="A:T1:r1", points=2, max_points=2, grader_id="g1", blind=True, graded_at=FIXED_NOW
    )


@pytest.mark.parametrize("corruption", ["duplicate", "scale", "foreign"])
def test_scoring_stays_pending_when_all_grades_are_unusable(
    manual_study: Study, grade: ManualGrade, corruption: str
) -> None:
    grades = {
        "duplicate": [grade, grade.model_copy(update={"points": 0})],
        "scale": [grade.model_copy(update={"points": 100, "max_points": 100})],
        "foreign": [grade.model_copy(update={"run_id": "B:T1:r1"})],
    }[corruption]
    result = score_record(manual_study.definition.task("T1"), manual_study.records[0], grades)
    assert result.status == "pending_manual"
    assert result.score is None and result.correct is None and result.n_graders == 0


def test_scoring_uses_other_valid_graders_when_some_grades_are_corrupt(
    manual_study: Study, grade: ManualGrade
) -> None:
    grades = [
        grade,
        grade.model_copy(update={"points": 0}),
        grade.model_copy(update={"grader_id": "wrong-scale", "points": 100, "max_points": 100}),
        grade.model_copy(update={"grader_id": "valid-1", "points": 1, "blind": False}),
        grade.model_copy(update={"grader_id": "valid-2", "points": 0}),
        grade.model_copy(update={"grader_id": "valid-1", "run_id": "B:T1:r1"}),
    ]
    result = score_record(manual_study.definition.task("T1"), manual_study.records[0], grades)
    assert result.status == "scored" and result.source == "manual"
    assert result.score == pytest.approx(0.25) and result.correct is False
    assert result.n_graders == 2 and result.grader_disagreement


@pytest.mark.parametrize(
    "scenario", [("duplicate", False), ("scale", False), ("duplicate", True), ("scale", True)]
)
def test_validation_pending_matches_usable_grades(
    manual_study: Study, grade: ManualGrade, scenario: tuple[str, bool]
) -> None:
    corruption, has_valid_grade = scenario
    manual_study.grades = {
        "duplicate": [grade, grade.model_copy(update={"points": 0})],
        "scale": [grade.model_copy(update={"points": 100, "max_points": 100})],
    }[corruption]
    if has_valid_grade:
        manual_study.grades.append(grade.model_copy(update={"grader_id": "valid", "blind": False}))
    report = validate_study(manual_study, now=FIXED_NOW)
    codes = {issue.code for issue in report.issues}
    assert ("pending_manual_grades" in codes) is not has_valid_grade
    expected_error = {"duplicate": "duplicate_grade", "scale": "grade_scale_mismatch"}[corruption]
    assert expected_error in {issue.code for issue in report.errors}


@pytest.mark.parametrize("unusable_record", ["excluded", "unplanned", "identity", "duplicate"])
def test_validation_omits_unusable_records_from_pending(
    manual_study: Study, unusable_record: str
) -> None:
    record = manual_study.records[0]
    manual_study.records = {
        "excluded": [record.model_copy(update={"protocol_deviation": "Used a different prompt"})],
        "unplanned": [record.model_copy(update={"run_id": "unplanned"})],
        "identity": [record.model_copy(update={"condition_id": "B"})],
        "duplicate": [record, record],
    }[unusable_record]
    report = validate_study(manual_study, now=FIXED_NOW)
    assert "pending_manual_grades" not in {issue.code for issue in report.issues}


def test_grade_selection_detects_duplicates_before_filtering_scale(grade: ManualGrade) -> None:
    rule = ManualScoring(method="manual", rubric="2 = complete", max_points=2, pass_threshold=2)
    valid = grade.model_copy(update={"grader_id": "valid", "points": 0, "blind": False})
    other_run = valid.model_copy(update={"run_id": "B:T1:r1"})
    grades = (
        grade,
        grade.model_copy(update={"max_points": 100}),
        valid,
        other_run,
        other_run,
    )
    selected = usable_manual_grades(grade.run_id, rule, grades)
    assert selected == [valid]


@pytest.fixture
def saved_study(manual_study: Study) -> Study:
    plan = manual_study.require_plan()
    manual_study.records.append(make_record(manual_study.definition, plan, "B:T1:r1"))
    (manual_study.root / "experiment.yaml").write_text(manual_study.definition.model_dump_json())
    save_plan(plan, manual_study.root / "plan.json")
    write_jsonl(manual_study.records_path, manual_study.records)
    return manual_study


def _fill_sheet(sheet: Path, legacy: bool = False) -> None:
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["points"] = "1"
        if legacy:
            del row["sheet_id"]
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if legacy:
        key = sheet.with_name("blind-key.json")
        data = json.loads(key.read_text())
        del data["sheet_id"]
        key.write_text(json.dumps(data))


def test_grade_schema_strips_surrounding_whitespace_without_changing_case(
    grade: ManualGrade,
) -> None:
    data = {**grade.model_dump(), "grader_id": " \tG1 \n"}
    parsed = ManualGrade.model_validate(data)
    assert parsed.grader_id == "G1"


def test_grade_schema_rejects_whitespace_only_identity(grade: ManualGrade) -> None:
    data = {**grade.model_dump(), "grader_id": " \t\n"}
    # When / Then
    with pytest.raises(ValidationError):
        ManualGrade.model_validate(data)


@pytest.mark.parametrize("identities", [("g1", " g1 "), (" g1 ", "g1"), ("g1", " \t ")])
def test_cli_import_rejects_duplicate_or_blank_fallback_without_partial_writes(
    saved_study: Study, grade: ManualGrade, identities: tuple[str, str]
) -> None:
    stored_id, fallback = identities
    old_line = json.dumps({**grade.model_dump(mode="json"), "grader_id": stored_id}) + "\n"
    saved_study.grades_path.write_text(old_line)
    original = saved_study.grades_path.read_bytes()
    sheet, _, _ = export_blind_sheet(saved_study)
    _fill_sheet(sheet)
    result = CliRunner().invoke(
        app, ["grade", "import", str(sheet), "-s", str(saved_study.root), "--grader", fallback]
    )
    assert result.exit_code == 2, result.output
    assert saved_study.grades_path.read_bytes() == original
    assert Study.load(saved_study.root).grades[0].grader_id == "g1"


def test_import_preserves_case_for_distinct_grader_ids(
    saved_study: Study, grade: ManualGrade
) -> None:
    saved_study.grades = [grade]
    sheet, key, _ = export_blind_sheet(saved_study)
    _fill_sheet(sheet)
    result = import_blind_sheet(saved_study, sheet, key, grader_id=" G1 ")
    assert result.errors == []
    assert len(result.grades) == 2
    assert {item.grader_id for item in result.grades} == {"G1"}


def test_old_grade_file_normalizes_duplicate_ids_without_rewriting(
    saved_study: Study, grade: ManualGrade
) -> None:
    original = "".join(
        json.dumps({**grade.model_dump(mode="json"), "grader_id": identity}) + "\n"
        for identity in ("g1", " g1 ")
    ).encode()
    saved_study.grades_path.write_bytes(original)
    result = score_record(
        saved_study.definition.task("T1"),
        saved_study.records[0],
        Study.load(saved_study.root).grades,
    )
    assert result.status == "pending_manual" and result.n_graders == 0
    assert saved_study.grades_path.read_bytes() == original


@pytest.mark.parametrize("problem", ["unplanned", "identity", "duplicate", "missing_task"])
def test_export_omits_unplaced_records(saved_study: Study, problem: str) -> None:
    valid, changed = saved_study.records
    corrupt = {
        "unplanned": [changed.model_copy(update={"run_id": "unplanned"})],
        "identity": [changed.model_copy(update={"condition_id": "A"})],
        "duplicate": [changed, changed],
        "missing_task": [changed.model_copy(update={"task_id": "missing"})],
    }[problem]
    saved_study.records = [valid, *corrupt]
    _, key, count = export_blind_sheet(saved_study)
    assert count == 1
    assert list(json.loads(key.read_text())["entries"].values()) == [valid.run_id]


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "problem", ["unplanned", "identity", "duplicate", "missing_task", "excluded", "refused"]
)
def test_cli_import_rejects_ineligible_rows_without_partial_writes(
    saved_study: Study, legacy: bool, problem: str
) -> None:
    sheet, key, _ = export_blind_sheet(saved_study)
    _fill_sheet(sheet, legacy)
    valid, changed = saved_study.records
    corrupt = {
        "unplanned": [changed.model_copy(update={"run_id": "unplanned"})],
        "identity": [changed.model_copy(update={"condition_id": "A"})],
        "duplicate": [changed, changed],
        "missing_task": [changed.model_copy(update={"task_id": "missing"})],
        "excluded": [changed.model_copy(update={"protocol_deviation": "Wrong prompt"})],
        "refused": [changed.model_copy(update={"response_status": "refused"})],
    }[problem]
    if problem == "unplanned":
        data = json.loads(key.read_text())
        data["entries"] = {
            gid: "unplanned" if rid == changed.run_id else rid
            for gid, rid in data["entries"].items()
        }
        key.write_text(json.dumps(data))
    write_jsonl(saved_study.records_path, [valid, *corrupt])
    result = CliRunner().invoke(
        app, ["grade", "import", str(sheet), "-s", str(saved_study.root), "--grader", "g1"]
    )
    assert result.exit_code == 2, result.output
    assert not saved_study.grades_path.exists()


def test_scoring_accepts_large_finite_points_without_overflow(
    manual_study: Study, grade: ManualGrade
) -> None:
    task = manual_study.definition.task("T1")
    task.scoring = ManualScoring(
        method="manual", rubric="All points required", max_points=1e308, pass_threshold=1e308
    )
    grades = [
        grade.model_copy(update={"points": 1e308, "max_points": 1e308, "grader_id": g})
        for g in ("one", "two")
    ]
    result = score_record(task, manual_study.records[0], grades)
    assert result.correct is True and result.score == 1.0
    with pytest.raises(ValidationError):
        ManualScoring(
            method="manual", rubric="Invalid scale", max_points=float("inf"), pass_threshold=2
        )


def test_sheet_retains_fractional_threshold_and_detects_small_changes(saved_study: Study) -> None:
    rule = saved_study.definition.task("T1").scoring
    assert isinstance(rule, ManualScoring)
    rule.pass_threshold = 1.0000001
    sheet, key, _ = export_blind_sheet(saved_study)
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        assert next(csv.DictReader(f))["pass_threshold"] == "1.0000001"
    _fill_sheet(sheet)
    assert not import_blind_sheet(saved_study, sheet, key, "g").errors
    rule.pass_threshold = 1.0000002
    changed = import_blind_sheet(saved_study, sheet, key, "g")
    assert not changed.grades
    assert all("pass_threshold" in err for err in changed.errors) and changed.errors
