from __future__ import annotations

import csv
import json

import pytest
from typer.testing import CliRunner

from mblab.cli import app
from mblab.csvcells import FORMULA_STARTS
from mblab.study import Study

from .conftest import make_record, write_records

runner = CliRunner()
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def run(*args, input=None):
    return runner.invoke(app, [str(a) for a in args], input=input, env=ENV)


@pytest.fixture
def planned(tmp_path):
    root = tmp_path / "study"
    assert run("init", root).exit_code == 0
    assert run("plan", "-s", root, "--show", "3").exit_code == 0
    return root


def test_version_and_help():
    assert "mblab" in run("--version").output
    result = run("--help")
    assert result.exit_code == 0
    for command in ("init", "plan", "status", "next", "record", "template", "import", "validate"):
        assert command in result.output


def test_init_refuses_to_overwrite(planned):
    again = run("init", planned)
    assert again.exit_code == 2 and "already exists" in again.output
    assert run("init", planned, "--force").exit_code == 0


def test_init_unknown_template(tmp_path):
    result = run("init", tmp_path / "x", "--template", "nope")
    assert result.exit_code == 2 and "unknown template" in result.output


def test_plan_refuses_to_overwrite(planned):
    again = run("plan", "-s", planned)
    assert again.exit_code == 2 and "already exists" in again.output
    assert run("plan", "-s", planned, "--force", "--show", "0").exit_code == 0


def test_commands_need_a_study(tmp_path):
    result = run("status", "-s", tmp_path)
    assert result.exit_code == 2 and "mblab init" in result.output


def test_next_and_status(planned):
    result = run("next", "-s", planned)
    assert result.exit_code == 0
    assert "Run B:T1:r1" in result.output
    assert "Set the default model selection to: Pro" in result.output
    assert "A train departs at 09:47" in result.output
    status = run("status", "-s", planned)
    assert status.exit_code == 0 and "Recorded 0 of 72" in status.output


def _record_input(response: str, latency: str = "42") -> str:
    answers = [
        "completed",  # status
        *response.splitlines(),
        ".",
        "Pro",  # label
        "below the response",  # where
        "shown",  # thinking indicator
        "Thought for 1m 5s",
        latency,
        "2026-11-20T10:00:00+09:00",  # time sent
        "1.2026.300",  # client version
        "disabled",
        "inactive",
        "none",
        "none",
        "wifi",
        "y",  # new conversation
        "",  # deviation
        "a note",
        "tester",
        "y",  # save
    ]
    return "\n".join(answers) + "\n"


def test_record_interactive(planned):
    result = run(
        "record", "-s", planned, "--no-copy-prompt", input=_record_input("Some work\nANSWER: 265")
    )
    assert result.exit_code == 0, result.output
    assert "Saved to" in result.output
    (rec,) = Study.load(planned).records
    assert rec.run_id == "B:T1:r1"
    assert rec.response_text == "Some work\nANSWER: 265"
    assert rec.observed_model_label == "Pro"
    assert rec.thinking_seconds == 65
    assert rec.latency_seconds == 42 and rec.latency_method == "manual_stopwatch"
    assert rec.confounders["network"] == "wifi"
    assert rec.notes == "a note" and rec.recorded_by == "tester"

    again = run("record", "-s", planned, "--run-id", "B:T1:r1", input="\n")
    assert again.exit_code == 2 and "already has a record" in again.output


def test_record_from_file_with_no_latency(planned, tmp_path):
    response = tmp_path / "resp.txt"
    response.write_text("ANSWER: 633", encoding="utf-8")
    answers = "\n".join(
        ["completed", "", "not_shown", "", "2026-11-20T10:00:00+09:00", "1.0"]
        + ["unknown"] * 5
        + ["y", "", "", "", "y"]
    )
    result = run(
        "record",
        "-s",
        planned,
        "--run-id",
        "B:T5:r1",
        "--response-file",
        response,
        input=answers + "\n",
    )
    assert result.exit_code == 0, result.output
    (rec,) = Study.load(planned).records
    assert rec.latency_method == "not_measured" and rec.observed_model_label is None


def test_template_import_validate_cycle(planned):
    assert run("template", "-s", planned).exit_code == 0
    path = planned / "exports" / "entry-template.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(
        timestamp="2026-11-20T10:00:00", client_version="1.0", response_text="ANSWER: 265"
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failed = run("import", path, "-s", planned)
    assert failed.exit_code == 2 and "nothing was imported" in failed.output

    dry = run("import", path, "-s", planned, "--default-tz", "Asia/Seoul", "--dry-run")
    assert dry.exit_code == 0 and "Would import 1 new" in dry.output
    assert Study.load(planned).records == []

    ok = run("import", path, "-s", planned, "--default-tz", "Asia/Seoul")
    assert ok.exit_code == 0 and "Imported 1 new" in ok.output

    dup = run("import", path, "-s", planned, "--default-tz", "Asia/Seoul")
    assert dup.exit_code == 2 and "already exist" in dup.output

    result = run("validate", "-s", planned)
    assert result.exit_code == 0
    assert "missing_confounder" in result.output  # confounders were left blank
    strict = run("validate", "-s", planned, "--strict")
    assert strict.exit_code == 1


def test_validate_fails_on_errors(planned):
    (planned / "records.jsonl").write_text(json.dumps({"run_id": "x"}) + "\n", encoding="utf-8")
    result = run("validate", "-s", planned)
    assert result.exit_code == 1 and "unreadable_line" in result.output


def test_demo_analyze_and_export(tmp_path):
    root = tmp_path / "demo"
    created = run("demo", root)
    assert created.exit_code == 0 and "SYNTHETIC DEMO DATA" in created.output
    result = run("analyze", "-s", root, "--json", root / "reports" / "analysis.json")
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC DEMO DATA" in result.output
    for heading in ("Conditions", "Displayed model labels", "Pairwise differences"):
        assert heading in result.output
    payload = json.loads((root / "reports" / "analysis.json").read_text())
    assert payload["synthetic_data"] is True

    assert run("export", "-s", root).exit_code == 0
    with (root / "exports" / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 68 and rows[0]["synthetic_data"] == "True"
    assert run("export", "-s", root, "--format", "json").exit_code == 0
    assert len(json.loads((root / "exports" / "results.json").read_text())) == 68
    assert run("export", "-s", root, "--format", "xml").exit_code == 2

    again = run("demo", root)
    assert again.exit_code == 2 and "not empty" in again.output


def _runs_as_formula(value: str) -> bool:
    """True if a spreadsheet would run this CSV cell as a formula (numbers never are)."""
    if not value or value[0] not in FORMULA_STARTS:
        return False
    try:
        float(value.replace(",", "."))
    except ValueError:
        return True
    return False


def test_export_csv_shows_formula_text_as_text(study_dir, definition, plan):
    first, second = sorted(plan.runs, key=lambda r: r.sequence)[:2]
    formula = '=HYPERLINK("https://example.invalid","Open")'
    write_records(
        study_dir,
        [
            make_record(definition, plan, first.run_id, response_text=formula),
            make_record(definition, plan, second.run_id, response_text="- one\n- two"),
        ],
    )
    assert run("export", "-s", study_dir).exit_code == 0
    with (study_dir / "exports" / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {r["run_id"]: r for r in csv.DictReader(handle)}
    assert rows[first.run_id]["response_text"] == "'" + formula
    assert rows[second.run_id]["response_text"] == "'- one\n- two"
    assert rows[first.run_id]["latency_seconds"] == "10.0"  # numbers are not touched
    for row in rows.values():
        assert not [v for v in row.values() if _runs_as_formula(v)], row


def test_analyze_without_records_shows_no_numbers(planned):
    result = run("analyze", "-s", planned)
    assert result.exit_code == 0
    assert "No observations recorded yet" in result.output
    assert "Conditions" not in result.output


def test_grade_export_without_manual_tasks(planned):
    result = run("grade", "export", "-s", planned)
    assert result.exit_code == 2 and "method: manual" in result.output
