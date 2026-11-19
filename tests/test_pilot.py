from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.pilot import PILOT_BANNER, create_pilot_study, displayed_labels
from mblab.report import render_report
from mblab.study import Study, StudyError
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record, write_records
from .test_analysis import BANNED

runner = CliRunner()
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _pilot(study_dir):
    return create_pilot_study(study_dir)


def test_pilot_copies_the_definition_as_a_separate_study(study_dir, definition):
    pilot, root = _pilot(study_dir)
    assert root == study_dir.with_name("study-pilot")
    d = pilot.definition
    assert d.pilot and d.experiment_id == "chatgpt-model-selection-override-pilot"
    assert d.title.startswith("[PILOT]") and d.repetitions == 1
    assert d.tasks == definition.tasks and d.conditions == definition.conditions
    assert len(pilot.require_plan().runs) == 24
    assert "PILOT STUDY" in (root / "experiment.yaml").read_text()
    # The main study is untouched, and its hash ignores the (unset) pilot flag.
    assert not Study.load(study_dir).definition.pilot
    assert "pilot" not in definition.model_dump(mode="json")


def test_pilot_refuses_bad_sources_and_targets(study_dir):
    _, root = _pilot(study_dir)
    with pytest.raises(StudyError, match="pilot or demo"):
        create_pilot_study(root, study_dir.parent / "x")
    with pytest.raises(StudyError, match="not empty"):
        create_pilot_study(study_dir)


def _record_pilot(root, labels):
    pilot = Study.load(root)
    plan = pilot.require_plan()
    records = [
        make_record(
            pilot.definition,
            plan,
            run.run_id,
            observed_model_label=label,
            observed_label_location="below the response" if label else None,
        )
        for run, label in zip(sorted(plan.runs, key=lambda r: r.sequence), labels, strict=False)
    ]
    write_records(root, records)
    return Study.load(root)


def test_displayed_labels_lists_texts_verbatim(study_dir):
    _, root = _pilot(study_dir)
    pilot = _record_pilot(root, ["Pro", "GPT-Pro thinking", "GPT-Pro thinking", None, "instant"])
    items, none = displayed_labels(pilot)
    assert none == 1
    by_text = {i.text: i for i in items}
    assert by_text["GPT-Pro thinking"].runs == 2 and by_text["GPT-Pro thinking"].matches == []
    assert by_text["Pro"].matches == ["pro"]
    assert by_text["instant"].matches == ["instant"]  # case-insensitive, as in analysis
    assert by_text["Pro"].locations == {"below the response": 1}


def test_pilot_records_are_labelled_and_never_fit_the_main_study(study_dir):
    _, root = _pilot(study_dir)
    pilot = _record_pilot(root, ["Pro"] * 6)
    assert validate_study(pilot, now=FIXED_NOW).ok
    a = analyze(pilot)
    assert any("PILOT study" in lim.text for lim in a.limitations)
    text = render_report(a, pilot, now=FIXED_NOW)
    assert PILOT_BANNER in text and not BANNED.search(text)
    # Copying pilot records into the main study is caught by validation.
    write_records(study_dir, pilot.records)
    codes = {i.code for i in validate_study(Study.load(study_dir), now=FIXED_NOW).errors}
    assert "condition_mismatch" in codes


def test_cli_pilot_and_labels(study_dir):
    created = runner.invoke(app, ["pilot", "-s", str(study_dir)], env=ENV)
    assert created.exit_code == 0, created.output
    assert "PILOT DATA" in created.output and "24 runs" in created.output
    root = study_dir.with_name("study-pilot")
    empty = runner.invoke(app, ["labels", "-s", str(root)], env=ENV)
    assert empty.exit_code == 0 and "No displayed labels recorded" in empty.output
    _record_pilot(root, ["Pro", "Mystery label"])
    listed = runner.invoke(app, ["labels", "-s", str(root)], env=ENV)
    assert listed.exit_code == 0, listed.output
    assert "Mystery label" in listed.output and "observed_label_aliases" in listed.output
    assert "post-hoc" not in listed.output
    status = runner.invoke(app, ["status", "-s", str(root)], env=ENV)
    assert "PILOT DATA" in status.output


def test_cli_labels_warns_outside_a_pilot(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, "A:T1:r1", observed_model_label="Pro")])
    result = runner.invoke(app, ["labels", "-s", str(study_dir)], env=ENV)
    assert result.exit_code == 0 and "post-hoc change" in result.output
