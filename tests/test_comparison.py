from __future__ import annotations

import json
from datetime import UTC, timedelta
from pathlib import Path

import pytest

from mblab.comparison import compare_studies
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition, RunPlan
from mblab.study import Study, StudyError
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record


def test_keeps_outcome_denominators_separate_when_collection_is_incomplete(
    tmp_path: Path, definition: ExperimentDefinition, plan: RunPlan
) -> None:
    # Given: only the right collection has an incorrect answer for T2.
    left = Study(
        tmp_path / "left",
        definition,
        plan,
        [
            make_record(definition, plan, "A:T1:r1"),
            make_record(definition, plan, "A:T1:r2", response_text="No answer line"),
            make_record(definition, plan, "A:T1:r3", response_status="refused"),
            make_record(definition, plan, "A:T2:r1", protocol_deviation="wrong selection"),
        ],
        [],
    )
    right = Study(
        tmp_path / "right",
        definition,
        plan,
        [
            make_record(definition, plan, "A:T1:r1"),
            make_record(definition, plan, "A:T2:r1", response_text="ANSWER: 2027-02-28"),
        ],
        [],
    )
    # When
    result = compare_studies(left, right)
    # Then
    a = result["conditions"][0]
    assert a["left"] == {
        "planned_runs": 18,
        "recorded_runs": 4,
        "excluded_runs": 1,
        "pending_manual": 0,
        "scored_runs": 2,
        "scored_tasks": 1,
        "task_scored_runs": {"T1": 2, "T2": 0, "T3": 0, "T4": 0, "T5": 0, "T6": 0},
        "correct_runs": 1,
        "accuracy": 0.5,
        "success_denominator": 3,
        "success_rate": pytest.approx(1 / 3),
    }
    assert a["right"]["accuracy"] == 0.5
    assert a["right"]["success_rate"] == 0.5
    assert a["right"]["scored_tasks"] == 2
    assert a["right"]["task_scored_runs"]["T2"] == 1
    assert result["total_tasks"] == 6
    assert result["left"]["recorded_runs"] == 4
    assert result["right"]["recorded_runs"] == 2
    assert [row["condition_id"] for row in result["conditions"]] == ["A", "B", "C", "D"]
    assert a["model_selection"] == "Instant"
    assert a["message_override"] is None
    assert result["conditions"][2]["message_override"] == "Pro"


def test_returns_null_rates_when_neither_study_has_observations(
    tmp_path: Path, definition: ExperimentDefinition, plan: RunPlan
) -> None:
    # Given
    empty = Study(tmp_path, definition, plan, [], [])
    # When
    result = compare_studies(empty, empty)
    # Then
    assert result["synthetic_data"] is False
    assert result["left"]["first_recorded_at"] is None
    assert result["left"]["last_recorded_at"] is None
    assert result["left"]["client_versions"] == {}
    assert result["left"]["planned_runs"] == 72
    for condition in result["conditions"]:
        for side in ("left", "right"):
            assert condition[side]["accuracy"] is None
            assert condition[side]["success_rate"] is None
            assert condition[side]["success_denominator"] == 0
            assert condition[side]["scored_tasks"] == 0
    json.dumps(result, allow_nan=False)


def test_omits_pending_manual_grades_from_both_denominators(manual_study: Path) -> None:
    # Given
    study = Study.load(manual_study)
    # When
    result = compare_studies(study, study)
    # Then
    a = result["conditions"][0]["left"]
    assert a["recorded_runs"] == a["pending_manual"] == 3
    assert a["scored_runs"] == a["success_denominator"] == 0
    assert a["accuracy"] is None and a["success_rate"] is None
    assert a["task_scored_runs"]["T7"] == 0
    assert result["total_tasks"] == 7


def test_metadata_uses_included_sent_times_and_warning_codes(
    tmp_path: Path, definition: ExperimentDefinition, plan: RunPlan
) -> None:
    # Given: recorded_at is identical, but sent timestamps differ in time and offset.
    first = FIXED_NOW.astimezone(UTC)
    last = FIXED_NOW + timedelta(hours=1)
    records = [
        make_record(definition, plan, "A:T1:r1", timestamp=last, client_version="v2"),
        make_record(
            definition,
            plan,
            "A:T1:r2",
            timestamp=first,
            client_version="unknown",
            response_status="refused",
        ),
        make_record(
            definition,
            plan,
            "A:T1:r3",
            timestamp=FIXED_NOW - timedelta(days=1),
            client_version="excluded-version",
            protocol_deviation="wrong selection",
        ),
    ]
    study = Study(tmp_path, definition, plan, records, [])
    warnings = validate_study(study).warnings
    # When
    result = compare_studies(study, study)
    # Then
    summary = result["left"]
    assert summary["first_recorded_at"] == first.isoformat()
    assert summary["last_recorded_at"] == last.astimezone(UTC).isoformat()
    assert summary["client_versions"] == {"unknown": 1, "v2": 1}
    assert summary["warnings"]["unknown_client_version"] == 1
    assert sum(summary["warnings"].values()) == len(warnings)
    assert summary["definition_sha256"] == definition.sha256()
    encoded = json.dumps(result, allow_nan=False)
    for record in records:
        assert record.run_id not in encoded
        assert record.prompt_text not in encoded
        assert record.response_text not in encoded
    assert set(result) == {"synthetic_data", "total_tasks", "left", "right", "conditions"}


def test_marks_comparison_when_records_are_synthetic(
    tmp_path: Path, definition: ExperimentDefinition
) -> None:
    # Given
    definition = definition.model_copy(update={"demo": True})
    plan = generate_plan(definition, now=FIXED_NOW)
    record = make_record(definition, plan, "A:T1:r1", data_origin="synthetic")
    synthetic = Study(tmp_path, definition, plan, [record], [])
    # When
    result = compare_studies(synthetic, synthetic)
    # Then
    assert result["synthetic_data"] is True


def test_refuses_validation_errors_before_returning_counts(
    tmp_path: Path, definition: ExperimentDefinition, plan: RunPlan
) -> None:
    # Given
    record = make_record(definition, plan, "A:T1:r1")
    invalid = Study(tmp_path, definition, plan, [record, record], [])
    valid = Study(tmp_path, definition, plan, [], [])
    # When / Then
    with pytest.raises(StudyError):
        compare_studies(invalid, valid)
