from __future__ import annotations

from datetime import timedelta

import yaml

from mblab.analysis import analyze
from mblab.schema import ManualGrade, RunRecord
from mblab.study import Study, template_text
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record, write_records


def _codes(report, severity=None):
    return [i.code for i in report.issues if severity is None or i.severity == severity]


def _validate(root):
    return validate_study(Study.load(root), now=FIXED_NOW + timedelta(days=2))


def test_clean_partial_study(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:5]])
    report = _validate(study_dir)
    assert report.ok, report.errors
    assert not report.warnings
    assert "missing_runs" in _codes(report, "info")
    assert (report.n_planned, report.n_records, report.n_missing) == (72, 5, 67)


def test_no_plan(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    (root / "experiment.yaml").write_text(template_text(), encoding="utf-8")
    report = validate_study(Study.load(root))
    assert _codes(report, "info").count("no_plan") == 1


def test_duplicates_unknown_runs_and_condition_mismatch(study_dir, definition, plan):
    good = make_record(definition, plan, "A:T1:r1")
    unknown = good.model_copy(update={"run_id": "Z:T1:r1"})
    mismatch = make_record(definition, plan, "C:T2:r1").model_copy(
        update={"message_override": None}
    )
    write_records(study_dir, [good, good, unknown, mismatch])
    report = _validate(study_dir)
    errors = _codes(report, "error")
    assert "duplicate_record" in errors
    assert "unknown_run" in errors
    assert "condition_mismatch" in errors
    assert not report.ok


def test_prompt_and_protocol_checks(study_dir, definition, plan):
    task_prompt = definition.task("T1").prompt
    records = [
        make_record(definition, plan, "A:T1:r1", prompt_text=task_prompt.replace("train", "bus")),
        make_record(definition, plan, "B:T1:r1", prompt_text=task_prompt.replace("\n", "\r\n")),
        make_record(definition, plan, "C:T1:r1", new_conversation=False),
    ]
    write_records(study_dir, records)
    report = _validate(study_dir)
    assert "prompt_mismatch" in _codes(report, "warning")
    assert "prompt_whitespace" in _codes(report, "info")
    assert "not_new_conversation" in _codes(report, "warning")


def test_confounder_rules(study_dir, definition, plan):
    base = make_record(definition, plan, "A:T1:r1").confounders
    records = [
        make_record(definition, plan, "A:T1:r1", confounders={**base, "memory": "maybe"}),
        make_record(definition, plan, "A:T2:r1", confounders={**base, "mood": "good"}),
        make_record(definition, plan, "A:T3:r1", confounders={"memory": "disabled"}),
    ]
    write_records(study_dir, records)
    report = _validate(study_dir)
    assert "invalid_confounder_value" in _codes(report, "error")
    assert "unknown_confounder" in _codes(report, "error")
    assert "missing_confounder" in _codes(report, "warning")


def test_client_version_change_and_latency_methods(study_dir, definition, plan):
    records = [
        make_record(definition, plan, "A:T1:r1", client_version="1.0"),
        make_record(definition, plan, "A:T2:r1", client_version="1.1"),
        make_record(definition, plan, "A:T3:r1", latency_method="screen_recording"),
        make_record(definition, plan, "A:T4:r1", client_version="unknown"),
    ]
    write_records(study_dir, records)
    report = _validate(study_dir)
    warnings = _codes(report, "warning")
    assert "client_version_changed" in warnings
    assert "mixed_latency_methods" in warnings
    assert "unknown_client_version" in warnings


def test_synthetic_records_never_mix_with_observed(study_dir, definition, plan):
    records = [
        make_record(definition, plan, "A:T1:r1"),
        make_record(definition, plan, "A:T2:r1", data_origin="synthetic"),
    ]
    write_records(study_dir, records)
    assert "mixed_data_origin" in _codes(_validate(study_dir), "error")
    write_records(study_dir, records[1:])
    assert "mixed_data_origin" in _codes(_validate(study_dir), "error")  # not a demo study


def test_privacy_scan_does_not_echo_values(study_dir, definition, plan):
    rec = make_record(
        definition,
        plan,
        "A:T1:r1",
        response_text="Contact me at someone@example.com\nANSWER: 265",
        notes="key " + "sk-" + "x" * 30,  # fake key, assembled at runtime
    )
    write_records(study_dir, [rec])
    report = _validate(study_dir)
    privacy = [i for i in report.issues if i.code == "possible_personal_data"]
    assert len(privacy) == 2
    assert all(
        "someone@example.com" not in i.message and "sk-xxx" not in i.message for i in privacy
    )


def test_definition_change_after_collection_is_flagged(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, "A:T5:r1")])
    data = yaml.safe_load(template_text())
    data["tasks"][4]["scoring"]["expected"] = 634
    data["tasks"][4]["reference_answer"] = "634"
    (study_dir / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    report = _validate(study_dir)
    changed = [i for i in report.issues if i.code == "definition_changed"]
    assert changed and "tasks[T5].scoring.expected" in changed[0].message


def test_exclusions_order_and_duplicate_text(study_dir, definition, plan):
    long_text = "Same long response text that is clearly copied twice by mistake.\nANSWER: 265 " * 2
    first, second, third = (r.run_id for r in plan.runs[:3])
    records = [
        make_record(definition, plan, first, timestamp=FIXED_NOW + timedelta(hours=2)),
        make_record(definition, plan, second, protocol_deviation="forgot to start a new chat"),
        make_record(definition, plan, "A:T1:r1", response_text=long_text),
        make_record(definition, plan, "A:T1:r2", response_text=long_text),
        make_record(definition, plan, third),
    ]
    write_records(study_dir, records)
    report = _validate(study_dir)
    assert report.n_excluded == 1
    assert "excluded_record" in _codes(report, "info")
    assert "order_deviation" in _codes(report, "info")
    assert "duplicate_response_text" in _codes(report, "warning")


def test_unreadable_lines_and_grade_checks(study_dir, definition, plan):
    rec = make_record(definition, plan, "A:T1:r1")
    (study_dir / "records.jsonl").write_text(
        rec.model_dump_json() + "\n{not json}\n", encoding="utf-8"
    )
    grades = [
        ManualGrade(
            run_id="A:T1:r1", points=1, max_points=1, grader_id="g", blind=True, graded_at=FIXED_NOW
        ),
        ManualGrade(
            run_id="nope", points=1, max_points=1, grader_id="g", blind=True, graded_at=FIXED_NOW
        ),
    ]
    (study_dir / "grades.jsonl").write_text(
        "".join(g.model_dump_json() + "\n" for g in grades), encoding="utf-8"
    )
    report = _validate(study_dir)
    assert "unreadable_line" in _codes(report, "error")
    assert "grade_unknown_run" in _codes(report, "error")
    assert "grade_for_automatic_task" in _codes(report, "warning")


def test_record_copy_validation_still_applies(definition, plan):
    rec = make_record(definition, plan, "A:T1:r1")
    assert RunRecord.model_validate_json(rec.model_dump_json()).run_id == "A:T1:r1"


def _edit_definition(root, change):
    data = yaml.safe_load(template_text())
    change(data)
    (root / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return Study.load(root).definition


def test_records_carry_the_definition_they_were_collected_under(study_dir, definition, plan):
    before = make_record(definition, plan, "A:T5:r1")  # recorded under the planned YAML

    def change(data):
        data["tasks"][4]["scoring"]["expected"] = 634
        data["tasks"][4]["reference_answer"] = "634"

    edited = _edit_definition(study_dir, change)
    after = make_record(edited, plan, "B:T5:r1")  # recorded after the edit
    assert before.definition_sha256 == plan.definition_sha256
    assert after.definition_sha256 == edited.sha256() != plan.definition_sha256
    write_records(study_dir, [before, after])
    (issue,) = [i for i in _validate(study_dir).issues if i.code == "hash_mismatch"]
    assert "B:T5:r1" in issue.message and "A:T5:r1" not in issue.message
    assert "tasks[T5].scoring.expected" in issue.message
    (lim,) = [
        lim
        for lim in analyze(Study.load(study_dir)).limitations
        if "after the plan was generated" in lim.text
    ]
    assert "tasks[T5].scoring.expected" in lim.text
    assert "1 record(s) were collected under an experiment.yaml" in lim.text
    assert "scored with the current experiment.yaml" in lim.text


def test_prompts_are_compared_with_the_planned_prompt(study_dir, plan):
    def change(data):
        data["tasks"][0]["prompt"] = data["tasks"][0]["prompt"].replace("train", "bus")

    edited = _edit_definition(study_dir, change)
    # Sent with the edited prompt: it matches the current YAML, not the pre-registered one.
    write_records(study_dir, [make_record(edited, plan, "A:T1:r1")])
    assert "prompt_mismatch" in _codes(_validate(study_dir), "warning")
