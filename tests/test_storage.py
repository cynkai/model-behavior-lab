"""Data integrity of study files: nothing recorded is lost, nothing is written twice."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mblab import cli as cli_module
from mblab.cli import app
from mblab.exchange import merge_records
from mblab.study import (
    LOCK_FILE,
    RECORDS_FILE,
    Study,
    StudyError,
    append_jsonl,
    write_jsonl,
    write_lock,
)

from .conftest import make_record, write_records

ENV = {"COLUMNS": "200", "NO_COLOR": "1"}
DAMAGED = b'{"run_id": "B:T5:r1", "protocol_deviaton": "typo"}'


def _records_file(root):
    return root / RECORDS_FILE


def _with_damaged_line(root, definition, plan):
    """Two good records with a damaged line between them."""
    good = [make_record(definition, plan, rid) for rid in ("A:T1:r1", "A:T2:r1")]
    path = _records_file(root)
    path.write_bytes(
        good[0].model_dump_json().encode()
        + b"\n"
        + DAMAGED
        + b"\n"
        + good[1].model_dump_json().encode()
        + b"\n"
    )
    return good


def test_a_damaged_utf8_line_is_an_error_not_a_crash(study_dir, definition, plan):
    good = make_record(definition, plan, "A:T1:r1").model_dump_json().encode()
    truncated = '{"notes": "풀'.encode()[:-1]  # cut inside a multi-byte character
    _records_file(study_dir).write_bytes(good + b"\n" + truncated + b"\n")
    study = Study.load(study_dir)
    assert [r.run_id for r in study.records] == ["A:T1:r1"]
    (error,) = study.load_errors
    assert error.line == 2 and "not valid UTF-8" in error.message


def test_append_after_a_missing_final_newline_keeps_both_records(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, "A:T1:r1")])
    path = _records_file(study_dir)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))  # an editor dropped the final newline
    append_jsonl(path, [make_record(definition, plan, "A:T2:r1")])
    study = Study.load(study_dir)
    assert study.load_errors == []
    assert [r.run_id for r in study.records] == ["A:T1:r1", "A:T2:r1"]


def test_adding_records_keeps_unreadable_lines_verbatim(study_dir, definition, plan):
    _with_damaged_line(study_dir, definition, plan)
    result = merge_records(
        Study.load(study_dir), [make_record(definition, plan, "C:T3:r1")], False, False
    )
    assert result.added == ["C:T3:r1"] and result.unreadable_kept == [2]
    assert DAMAGED in _records_file(study_dir).read_bytes()
    assert len(Study.load(study_dir).records) == 3


def test_replacing_keeps_unreadable_lines_and_a_backup(study_dir, definition, plan):
    _with_damaged_line(study_dir, definition, plan)
    before = _records_file(study_dir).read_bytes()
    newer = make_record(definition, plan, "A:T1:r1", notes="corrected")
    result = merge_records(Study.load(study_dir), [newer], True, False)
    assert result.replaced == ["A:T1:r1"] and result.unreadable_kept == [2]
    after = _records_file(study_dir).read_bytes()
    assert after.splitlines()[1] == DAMAGED  # same place, same bytes
    assert (study_dir / f"{RECORDS_FILE}.bak").read_bytes() == before
    stored = {r.run_id: r for r in Study.load(study_dir).records}
    assert stored["A:T1:r1"].notes == "corrected" and "A:T2:r1" in stored


def test_merge_uses_the_file_not_a_stale_copy(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, "A:T1:r1")])
    stale = Study.load(study_dir)
    # Another command saves a record after `stale` was loaded.
    append_jsonl(_records_file(study_dir), [make_record(definition, plan, "B:T1:r1")])
    merge_records(stale, [make_record(definition, plan, "A:T1:r1", notes="new")], True, False)
    stored = {r.run_id: r for r in Study.load(study_dir).records}
    assert set(stored) == {"A:T1:r1", "B:T1:r1"} and stored["A:T1:r1"].notes == "new"
    assert {r.run_id for r in stale.records} == {"A:T1:r1", "B:T1:r1"}  # refreshed

    # A run saved by someone else meanwhile is a conflict, and nothing is written.
    append_jsonl(_records_file(study_dir), [make_record(definition, plan, "C:T1:r1")])
    stale = Study.load(study_dir)
    stale.records = [r for r in stale.records if r.run_id != "C:T1:r1"]
    before = _records_file(study_dir).read_bytes()
    result = merge_records(stale, [make_record(definition, plan, "C:T1:r1")], False, False)
    assert result.conflicts == ["C:T1:r1"]
    assert _records_file(study_dir).read_bytes() == before


def _lock_info(pid: int) -> str:
    return json.dumps({"pid": pid, "host": socket.gethostname(), "purpose": "record"})


def test_write_lock_is_exclusive_reentrant_and_released(study_dir):
    with write_lock(study_dir, "outer"), write_lock(study_dir, "inner"):
        assert (study_dir / LOCK_FILE).exists()  # nesting in one thread is fine
    assert not (study_dir / LOCK_FILE).exists()

    with pytest.raises(RuntimeError), write_lock(study_dir, "boom"):
        raise RuntimeError("fails while holding the lock")
    assert not (study_dir / LOCK_FILE).exists()

    (study_dir / LOCK_FILE).write_text(_lock_info(os.getppid()))  # a live other process
    with pytest.raises(StudyError, match="being written by `record`"), write_lock(study_dir, "x"):
        pass
    assert (study_dir / LOCK_FILE).exists()  # someone else's lock is left alone


@pytest.mark.skipif(sys.platform.startswith("win"), reason="no liveness check on Windows")
def test_a_lock_left_by_a_dead_process_is_cleared(study_dir):
    finished = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    (study_dir / LOCK_FILE).write_text(_lock_info(int(finished.stdout)))
    with write_lock(study_dir, "after a crash"):
        assert json.loads((study_dir / LOCK_FILE).read_text())["purpose"] == "after a crash"


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_times_are_rejected(definition, plan, value):
    with pytest.raises(ValidationError):
        make_record(definition, plan, "A:T1:r1", latency_seconds=value)


def test_cli_import_keeps_a_damaged_line_and_says_so(study_dir, definition, plan, tmp_path):
    _with_damaged_line(study_dir, definition, plan)
    incoming = tmp_path / "more.jsonl"
    write_jsonl(incoming, [make_record(definition, plan, "D:T6:r1")])
    result = CliRunner().invoke(app, ["import", str(incoming), "-s", str(study_dir)], env=ENV)
    assert result.exit_code == 0, result.output
    assert "1 unreadable line(s) (line 2)" in result.output
    assert DAMAGED in _records_file(study_dir).read_bytes()


def test_cli_record_does_not_overwrite_a_run_saved_meanwhile(
    study_dir, definition, plan, monkeypatch
):
    real_confirm = cli_module.typer.confirm

    def confirm(text, *args, **kwargs):
        if text.startswith("Save this record"):
            # Another command records the same run while this one is waiting.
            append_jsonl(
                _records_file(study_dir),
                [make_record(definition, plan, "B:T1:r1", notes="other writer")],
            )
        return real_confirm(text, *args, **kwargs)

    monkeypatch.setattr(cli_module.typer, "confirm", confirm)
    answers = [
        "completed",
        "ANSWER: 265",
        ".",
        "",
        "unknown",
        "",
        "2026-11-20T10:00:00+09:00",
        "1.0",
        *["unknown"] * 5,
        "y",
        "",
        "",
        "",
        "y",
    ]
    result = CliRunner().invoke(
        app,
        ["record", "-s", str(study_dir), "--no-copy-prompt"],
        input="\n".join(answers) + "\n",
        env=ENV,
    )
    assert result.exit_code == 2, result.output
    assert "recorded by another command while you were typing" in result.output
    (kept,) = Study.load(study_dir).records
    assert kept.notes == "other writer"
    unsaved = study_dir / "exports" / "unsaved-B_T1_r1.jsonl"
    assert json.loads(unsaved.read_text())["response_text"] == "ANSWER: 265"
