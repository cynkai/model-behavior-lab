from __future__ import annotations

import csv
import json

import pytest
import yaml
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.csvcells import FORMULA_STARTS
from mblab.demo import create_demo_study
from mblab.grading import SHEET_COLUMNS, export_blind_sheet, import_blind_sheet
from mblab.study import Study, StudyError, append_jsonl
from mblab.validate import validate_study

from .conftest import make_record, write_records


def _fill(sheet, points_by_row, grader="g1"):
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row, points in zip(rows, points_by_row, strict=False):
        row["points"] = points
        row["grader_id"] = grader
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_blind_sheet_hides_condition_information(manual_study):
    study = Study.load(manual_study)
    sheet, key, n = export_blind_sheet(study)
    assert n == 6
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")  # BOM: spreadsheets read it as UTF-8
    text = sheet.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0].split(",")
    assert header == SHEET_COLUMNS
    for leaked in ("condition", "run_id", "model_selection", "observed_model_label", "latency"):
        assert leaked not in header
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["grading_id"] for r in rows] == [f"G{i:03d}" for i in range(1, 7)]
    entries = json.loads(key.read_text())["entries"]
    assert sorted(entries.values()) == sorted(r.run_id for r in study.records)


def test_grade_roundtrip_feeds_the_analysis(manual_study):
    study = Study.load(manual_study)
    sheet, key, _ = export_blind_sheet(study)
    _fill(sheet, ["2", "1", "2", "", "0", "2"])
    result = import_blind_sheet(study, sheet, key)
    assert result.errors == [] and result.skipped_blank == 1 and len(result.grades) == 5
    assert all(g.blind for g in result.grades)
    append_jsonl(study.grades_path, result.grades)

    study = Study.load(manual_study)
    assert validate_study(study).ok
    a = analyze(study)
    manual_rows = a.results[a.results["task_id"] == "T7"]
    assert set(manual_rows["score_source"]) == {"manual", "none"}
    assert (manual_rows["score_status"] == "pending_manual").sum() == 1
    assert any("await manual grading" in lim.text for lim in a.limitations)

    again = import_blind_sheet(study, sheet, key)
    assert len(again.errors) == 5 and "already graded" in again.errors[0]


def test_agreement_command_after_two_blind_imports(manual_study, tmp_path):
    runner = CliRunner()
    study_args = ["-s", str(manual_study)]
    exported = runner.invoke(app, ["grade", "export", *study_args])
    assert exported.exit_code == 0, exported.output
    sheet = manual_study / "grading" / "blind-sheet.csv"
    for grader, scores in (("first", ["0", "2"]), ("second", ["1", "2"])):
        _fill(sheet, scores, grader=grader)
        imported = runner.invoke(app, ["grade", "import", str(sheet), *study_args])
        assert imported.exit_code == 0, imported.output

    result = runner.invoke(app, ["grade", "agreement", *study_args])
    assert result.exit_code == 0, result.output
    assert "1/2 (50%)" in result.output and "2/2 (100%)" in result.output

    path = tmp_path / "analysis.json"
    result = runner.invoke(app, ["analyze", *study_args, "--json", str(path)])
    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text())["grading_agreement"] == [
        {
            "task_id": "T7",
            "eligible_runs": 6,
            "graded_runs": 2,
            "multi_rated_runs": 2,
            "pairs": 2,
            "exact_matches": 1,
            "decision_matches": 2,
        }
    ]


def test_agreement_command_for_automatic_study(study_dir):
    result = CliRunner().invoke(app, ["grade", "agreement", "-s", str(study_dir)])
    assert result.exit_code == 0, result.output
    assert "no manually scored tasks" in result.output
    assert "grading_agreement" not in analyze(Study.load(study_dir)).to_dict()


def test_grade_import_rejects_bad_rows(manual_study):
    study = Study.load(manual_study)
    sheet, key, _ = export_blind_sheet(study)
    rows = _fill(sheet, ["3", "x", "1"])
    rows[2]["grader_id"] = ""
    rows[3]["points"] = "1"
    rows[3]["grading_id"] = "G999"
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    result = import_blind_sheet(study, sheet, key)
    messages = " | ".join(result.errors)
    assert "outside 0..2" in messages
    assert "not a number" in messages
    assert "grader_id missing" in messages
    assert "unknown grading_id" in messages


def test_grade_import_rejects_key_from_another_export(manual_study, tmp_path):
    study = Study.load(manual_study)
    sheet, key, _ = export_blind_sheet(study, seed=1)
    _, other_key, _ = export_blind_sheet(study, tmp_path / "second-export", seed=2)
    _fill(sheet, ["2"])
    original = json.loads(key.read_text())
    other = json.loads(other_key.read_text())
    assert original["entries"]["G001"] != other["entries"]["G001"]
    with pytest.raises(StudyError, match="grading sheet and key are from different exports"):
        import_blind_sheet(study, sheet, other_key)
    assert len(import_blind_sheet(study, sheet, key).grades) == 1


def test_grade_import_accepts_legacy_pairs_and_rejects_mixing_formats(manual_study, tmp_path):
    study = Study.load(manual_study)
    sheet, key, _ = export_blind_sheet(study)
    rows = _fill(sheet, ["2"])
    legacy_sheet = tmp_path / "legacy.csv"
    with legacy_sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[name for name in SHEET_COLUMNS if name != "sheet_id"]
        )
        writer.writeheader()
        writer.writerows(
            {name: value for name, value in row.items() if name != "sheet_id"} for row in rows
        )
    legacy_key = tmp_path / "legacy-key.json"
    data = json.loads(key.read_text())
    del data["sheet_id"]
    legacy_key.write_text(json.dumps(data), encoding="utf-8")

    imported = import_blind_sheet(study, legacy_sheet, legacy_key)
    assert imported.errors == [] and len(imported.grades) == 1
    for csv_path, key_path in [(sheet, legacy_key), (legacy_sheet, key)]:
        with pytest.raises(StudyError, match="grading sheet and key are from different exports"):
            import_blind_sheet(study, csv_path, key_path)


@pytest.mark.parametrize("changed", ["response_text", "rubric"])
def test_grade_import_rejects_stale_material_without_writing(manual_study, changed):
    study = Study.load(manual_study)
    runner = CliRunner()
    exported = runner.invoke(app, ["grade", "export", "-s", str(manual_study)])
    assert exported.exit_code == 0, exported.output
    sheet = manual_study / "grading" / "blind-sheet.csv"
    key = sheet.with_name("blind-key.json")
    _fill(sheet, ["2", "1"])
    if changed == "response_text":
        stale_run = json.loads(key.read_text())["entries"]["G001"]
        write_records(
            manual_study,
            [
                record.model_copy(update={"response_text": "Corrected observation"})
                if record.run_id == stale_run
                else record
                for record in study.records
            ],
        )
    else:
        path = manual_study / "experiment.yaml"
        data = yaml.safe_load(path.read_text())
        task = next(task for task in data["tasks"] if task["id"] == "T7")
        task["scoring"]["rubric"] = "2 = correct explanation with units"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["grade", "import", str(sheet), "-s", str(manual_study)])
    assert result.exit_code == 2, result.output
    assert f"{changed} does not match the current study" in result.output
    assert "no grades were imported" in result.output
    assert not study.grades_path.exists()


def test_grade_import_accepts_spreadsheet_newlines_and_number_formats(manual_study):
    study = Study.load(manual_study)
    write_records(
        manual_study,
        [
            record.model_copy(update={"response_text": "Step 1\r\nStep 2\rStep 3"})
            for record in study.records
        ],
    )
    study = Study.load(manual_study)
    sheet, key, _ = export_blind_sheet(study)
    _fill(sheet, ["2"])
    text = sheet.read_text()
    sheet.write_text(text.replace(",2,2,", ",2.0,2.00,"), encoding="utf-8")
    imported = import_blind_sheet(study, sheet, key)
    assert imported.errors == [] and len(imported.grades) == 1


def test_blind_sheet_shows_formula_responses_as_text(manual_study):
    study = Study.load(manual_study)
    plan = study.require_plan()
    texts = ['=HYPERLINK("https://example.invalid","Open")', "- step one\n- step two"]
    records = [
        make_record(study.definition, plan, rec.run_id, response_text=texts[i % 2])
        for i, rec in enumerate(study.records)
    ]
    write_records(manual_study, records)
    study = Study.load(manual_study)
    sheet, key, n = export_blind_sheet(study)
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["response_text"] for r in rows} == {"'" + t for t in texts}
    for row in rows:
        assert not [v for v in row.values() if v and v[0] in FORMULA_STARTS], row
    _fill(sheet, ["2"] * n)
    result = import_blind_sheet(study, sheet, key)
    assert result.errors == [] and len(result.grades) == n


def test_grade_export_needs_manual_tasks(study_dir):
    with pytest.raises(StudyError, match="method: manual"):
        export_blind_sheet(Study.load(study_dir))


def test_demo_study_is_labelled_and_valid(tmp_path):
    study = create_demo_study(tmp_path / "demo")
    assert study.definition.demo is True
    assert study.definition.experiment_id.startswith("demo-")
    assert study.definition.title.startswith("[SYNTHETIC DEMO]")
    assert all(r.data_origin == "synthetic" for r in study.records)
    assert all(r.response_text.startswith("[SYNTHETIC") for r in study.records if r.response_text)
    assert len(study.records) == 68
    report = validate_study(study)
    assert report.ok, report.errors
    assert report.n_excluded == 1
    a = analyze(study)
    assert a.synthetic
    assert a.limitations[0].text.startswith("These are SYNTHETIC demo records")
    assert (tmp_path / "demo" / "experiment.yaml").read_text().startswith("# SYNTHETIC DEMO")


def test_demo_is_deterministic_and_refuses_non_empty_dirs(tmp_path):
    create_demo_study(tmp_path / "one", seed=11)
    create_demo_study(tmp_path / "two", seed=11)
    for name in ("records.jsonl", "plan.json", "experiment.yaml"):
        assert (tmp_path / "one" / name).read_text() == (tmp_path / "two" / name).read_text()
    with pytest.raises(StudyError, match="not empty"):
        create_demo_study(tmp_path / "one")


def test_real_study_rejects_demo_records(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, "A:T1:r1", data_origin="synthetic")])
    assert not validate_study(Study.load(study_dir)).ok
