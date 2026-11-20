from __future__ import annotations

import csv

import pytest
import yaml
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.exchange import import_file, write_template_csv
from mblab.pilot import PILOT_BANNER, create_pilot_study, displayed_labels
from mblab.plan import PILOT_RUN_PREFIX
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
    # Pilot records copied into the main study are not planned there (their run ids start
    # with "pilot:"): validation reports them and analysis leaves them out.
    write_records(study_dir, pilot.records)
    main = Study.load(study_dir)
    codes = {i.code for i in validate_study(main, now=FIXED_NOW).errors}
    assert codes == {"unknown_run"}
    a = analyze(main)
    assert a.results.empty
    assert a.left_out == {"run not planned or task not defined": [r.run_id for r in pilot.records]}
    assert any("left out of every number" in lim.text for lim in a.limitations)


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


def test_pilot_has_its_own_seed_order_and_run_ids(study_dir, definition, plan):
    pilot, _ = _pilot(study_dir)
    pilot_plan = pilot.require_plan()
    assert pilot_plan.seed == definition.randomization.seed + 1
    assert all(r.run_id.startswith(PILOT_RUN_PREFIX) for r in pilot_plan.runs)
    by_sequence = sorted(plan.runs, key=lambda r: r.sequence)
    main_first_block = [r.run_id for r in by_sequence if r.block == 1]
    pilot_order = [
        r.run_id.removeprefix(PILOT_RUN_PREFIX)
        for r in sorted(pilot_plan.runs, key=lambda r: r.sequence)
    ]
    assert sorted(pilot_order) == sorted(main_first_block)
    assert pilot_order != main_first_block  # rehearsing the pilot is not rehearsing the order


def test_pilot_id_keeps_its_suffix_for_long_ids(study_dir):
    data = yaml.safe_load((study_dir / "experiment.yaml").read_text(encoding="utf-8"))
    data["experiment_id"] = "x" * 64
    (study_dir / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    pilot, _ = _pilot(study_dir)
    assert pilot.definition.experiment_id == "x" * 58 + "-pilot"


def test_filled_pilot_csv_is_refused_by_the_main_study(study_dir):
    pilot, root = _pilot(study_dir)
    path = root / "exports" / "entry-template.csv"
    write_template_csv(pilot, path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows[:3]:
        row.update(timestamp="2026-11-20T10:00:00+09:00", client_version="1.0", response_text="A")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(import_file(pilot, path).records) == 3  # fine in the pilot itself
    result = import_file(Study.load(study_dir), path)
    assert result.records == [] and len(result.errors) == 3
    assert all("is not in the plan" in e for e in result.errors)
