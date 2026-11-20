from __future__ import annotations

from datetime import datetime

import pytest
import yaml
from pydantic import ValidationError

from mblab.schema import ExperimentDefinition, ManualGrade, RunRecord
from mblab.study import StudyError, parse_definition, template_text

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
