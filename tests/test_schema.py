from __future__ import annotations

import json
from datetime import datetime

import pytest
import yaml
from pydantic import ValidationError

from mblab import __version__
from mblab.schema import (
    SCHEMA_VERSION,
    ApiReported,
    ExperimentDefinition,
    ManualGrade,
    RunPlan,
    RunRecord,
)
from mblab.study import (
    Study,
    StudyError,
    format_validation_error,
    parse_definition,
    template_text,
)
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record


def _template_dict() -> dict:
    return yaml.safe_load(template_text())


def test_template_defines_the_initial_experiment(definition):
    assert definition.experiment_id == "chatgpt-model-selection-override"
    assert [c.id for c in definition.conditions] == ["A", "B", "C", "D"]
    assert definition.planned_run_count == 72
    assert definition.execution_method == "manual_ui"
    assert definition.documented_facts == []  # never pre-filled with unverified claims
    assert definition.not_observable
    roles = {c.id: c.role for c in definition.conditions}
    assert roles == {"A": "control", "B": "control", "C": "test", "D": "test"}


def test_hash_ignores_yaml_formatting_but_tracks_content(definition):
    reformatted = parse_definition(yaml.safe_dump(_template_dict(), sort_keys=True))
    assert reformatted.sha256() == definition.sha256()
    data = _template_dict()
    data["tasks"][0]["scoring"]["expected"] = 266
    data["tasks"][0]["reference_answer"] = "266"
    data["tasks"][0].pop("answer_examples")  # they pin 265 as the correct answer
    assert ExperimentDefinition.model_validate(data).sha256() != definition.sha256()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["conditions"][2].update(message_override="turbo"), "not a selection option"),
        (lambda d: d["conditions"][1].update(id="A"), "duplicate condition id 'A'"),
        (lambda d: d["conditions"][1].update(model_selection="instant"), "identical settings"),
        (
            lambda d: d["tasks"][1].update(scoring={"method": "regex", "pattern": "(unclosed"}),
            "invalid regular expression",
        ),
        (lambda d: d["tasks"][0].update(reference_answer="264"), "not scored correct by its own"),
        (lambda d: d["tasks"][0].pop("reference_answer"), "needs a reference_answer"),
        (lambda d: d["conditions"][2].update(hypothesis_ids=["H9"]), "unknown hypothesis id"),
        (lambda d: d.update(unexpected_field=1), "Extra inputs are not permitted"),
        (lambda d: d.update(repetitions=0), "greater than or equal to 1"),
        (
            lambda d: d["tasks"][0]["scoring"].update(method="vibes"),
            "does not match any of the expected tags",
        ),
    ],
)
def test_invalid_definitions_are_rejected(mutate, message):
    data = _template_dict()
    mutate(data)
    with pytest.raises(StudyError, match=message):
        parse_definition(yaml.safe_dump(data))


def test_yaml_booleans_are_not_silently_accepted_as_labels():
    data = _template_dict()
    text = yaml.safe_dump(data).replace("- enabled\n", "- on\n", 1)
    with pytest.raises(StudyError, match="allowed_values"):
        parse_definition(text)


def test_record_requires_timezone(definition, plan):
    with pytest.raises(ValidationError, match="timezone"):
        make_record(definition, plan, "A:T1:r1", timestamp=datetime(2026, 11, 19, 10, 0))


def test_record_latency_rules(definition, plan):
    rec = make_record(definition, plan, "A:T1:r1")
    assert rec.latency_confidence == "low"  # default for a manual stopwatch
    with pytest.raises(ValidationError, match="not_measured"):
        make_record(definition, plan, "A:T1:r1", latency_method="not_measured")
    with pytest.raises(ValidationError, match="no latency_seconds"):
        make_record(definition, plan, "A:T1:r1", latency_seconds=None)
    with pytest.raises(ValidationError, match="not valid for execution_method"):
        make_record(definition, plan, "A:T1:r1", latency_method="client_timer")
    unmeasured = make_record(
        definition, plan, "A:T1:r1", latency_seconds=None, latency_method="not_measured"
    )
    assert unmeasured.latency_confidence is None


def test_record_thinking_and_response_rules(definition, plan):
    rec = make_record(
        definition, plan, "B:T5:r1", thinking_indicator="shown", thinking_text="Thought for 1m 5s"
    )
    assert rec.thinking_seconds == 65
    with pytest.raises(ValidationError, match="thinking_indicator='shown'"):
        make_record(definition, plan, "B:T5:r1", thinking_text="Thought for 5s")
    with pytest.raises(ValidationError, match="response_text is empty"):
        make_record(definition, plan, "B:T5:r1", response_text="   ")
    errored = make_record(definition, plan, "B:T5:r1", response_status="error", response_text="")
    assert errored.response_status == "error"


def test_blank_strings_become_none(definition, plan):
    rec = make_record(definition, plan, "C:T2:r1", observed_model_label="  ", notes="")
    assert rec.observed_model_label is None and rec.notes is None


def test_record_roundtrip(definition, plan):
    rec = make_record(definition, plan, "D:T3:r2", observed_model_label="Instant")
    assert RunRecord.model_validate_json(rec.model_dump_json()) == rec


def test_manual_grade_bounds():
    ok = ManualGrade(
        run_id="x", points=2, max_points=2, grader_id="g1", blind=True, graded_at=FIXED_NOW
    )
    assert ok.points == 2
    with pytest.raises(ValidationError, match="cannot exceed"):
        ManualGrade(
            run_id="x", points=3, max_points=2, grader_id="g1", blind=True, graded_at=FIXED_NOW
        )


# -- file formats ------------------------------------------------------------------------------

#: Serialized keys of each file as the first format defined them. A field added later must be
#: left out while unset (so these stay the keys of a file that does not use it, and versions
#: before the field can still read it); a change older versions would misread bumps
#: SCHEMA_VERSION instead. If this test fails, do one or the other — do not just add the key.
RECORD_KEYS = {
    "schema_version", "run_id", "experiment_id", "experiment_version", "definition_sha256",
    "platform", "client_version", "condition_id", "model_selection", "message_override",
    "task_id", "repetition", "execution_method", "timestamp", "data_origin", "prompt_text",
    "response_status", "response_text", "observed_model_label", "observed_label_location",
    "thinking_indicator", "thinking_text", "thinking_seconds", "latency_seconds",
    "latency_method", "latency_confidence", "new_conversation", "confounders",
    "protocol_deviation", "notes", "recorded_by", "recorded_at",
}  # fmt: skip
PLAN_KEYS = {
    "schema_version", "experiment_id", "experiment_version", "definition_sha256",
    "definition_snapshot", "seed", "strategy", "generated_at", "tool_version", "runs",
}  # fmt: skip
PLANNED_RUN_KEYS = {
    "run_id", "sequence", "block", "experiment_id", "experiment_version", "platform",
    "execution_method", "condition_id", "model_selection", "message_override", "task_id",
    "repetition",
}  # fmt: skip
GRADE_KEYS = {
    "schema_version", "run_id", "points", "max_points", "grader_id", "blind", "graded_at",
    "notes",
}  # fmt: skip


def _keys(model) -> set[str]:
    return set(json.loads(model.model_dump_json()))


def test_files_that_use_no_newer_field_keep_their_first_format(definition, plan):
    record = make_record(definition, plan, plan.runs[0].run_id)
    assert _keys(record) == RECORD_KEYS
    assert _keys(plan) == PLAN_KEYS
    assert {k for run in plan.model_dump(mode="json")["runs"] for k in run} == PLANNED_RUN_KEYS
    grade = ManualGrade(
        run_id="A:T1:r1", points=1, max_points=2, grader_id="g", blind=True, graded_at=FIXED_NOW
    )
    assert _keys(grade) == GRADE_KEYS


def test_api_records_write_only_the_values_the_api_returned(definition, plan):
    ui = json.loads(make_record(definition, plan, plan.runs[0].run_id).model_dump_json())
    record = RunRecord.model_validate(
        {
            **ui,
            "execution_method": "official_api",
            "latency_method": "client_timer",
            "api_reported": {"model": "gpt-x", "output_tokens": 12},
        }
    )
    data = json.loads(record.model_dump_json())
    assert data["api_reported"] == {"model": "gpt-x", "output_tokens": 12}  # no "refusal": null
    assert RunRecord.model_validate(data) == record
    refused = ApiReported(refusal="I can't help with that.", status="completed")
    assert json.loads(refused.model_dump_json()) == {
        "refusal": "I can't help with that.",
        "status": "completed",
    }


@pytest.mark.parametrize(
    ("model", "kind"),
    [(RunRecord, "record"), (ManualGrade, "grade"), (RunPlan, "plan")],
)
def test_newer_formats_are_refused_with_a_message_that_says_so(model, kind):
    with pytest.raises(ValidationError) as caught:
        model.model_validate_json(json.dumps({"schema_version": 2, "a_field_from_later": 1}))
    message = format_validation_error(caught.value)
    assert message == (
        f"this {kind} has schema_version 2, a newer file format than model-behavior-lab "
        f"{__version__} reads (up to {SCHEMA_VERSION}). Update model-behavior-lab to read it"
    )


def test_a_newer_definition_or_an_unknown_field_is_named_as_such():
    newer = template_text().replace("schema_version: 1", "schema_version: 2", 1)
    with pytest.raises(
        StudyError, match="this definition has schema_version 2, a newer file format"
    ):
        parse_definition(newer)
    typo = template_text().replace("repetitions:", "repetitons:", 1)
    with pytest.raises(StudyError) as caught:
        parse_definition(typo)
    assert "repetitons: Extra inputs are not permitted" in str(caught.value)
    assert "check the spelling, or update model-behavior-lab" in str(caught.value)
    unknown_method = template_text().replace("method: date", "method: weekday", 1)
    with pytest.raises(StudyError, match="update model-behavior-lab if a newer version"):
        parse_definition(unknown_method)
    with pytest.raises(StudyError) as caught:  # other mistakes get no such hint
        parse_definition(template_text().replace("repetitions: 3", "repetitions: 0", 1))
    assert "newer version" not in str(caught.value)


def test_a_record_in_a_newer_format_is_reported_and_kept(study_dir, definition, plan):
    good = make_record(definition, plan, plan.runs[0].run_id)
    newer = json.dumps({**json.loads(good.model_dump_json()), "schema_version": 2})
    (study_dir / "records.jsonl").write_text(good.model_dump_json() + "\n" + newer + "\n")
    study = Study.load(study_dir)
    assert [r.run_id for r in study.records] == [good.run_id]
    (error,) = study.load_errors
    assert error.line == 2 and "a newer file format" in error.message
    issues = [i for i in validate_study(study).issues if i.code == "unreadable_line"]
    assert len(issues) == 1 and "Update model-behavior-lab" in issues[0].message
