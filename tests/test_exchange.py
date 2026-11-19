from __future__ import annotations

import csv
import json
from datetime import timedelta, timezone

import pytest

from mblab.exchange import (
    import_file,
    merge_records,
    parse_timezone,
    template_columns,
    write_template_csv,
)
from mblab.study import Study, StudyError

from .conftest import make_record, write_records


def _template_rows(study_dir):
    study = Study.load(study_dir)
    path = study_dir / "exports" / "t.csv"
    write_template_csv(study, path)
    with path.open(encoding="utf-8", newline="") as handle:
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
