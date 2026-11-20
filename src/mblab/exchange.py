"""CSV/JSON exchange: entry templates, importing observations, merging records.

The CSV template has one row per planned run, so results can be typed into a
spreadsheet and imported later. Rows that were not filled in are skipped.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from mblab.csvcells import from_row, to_cell
from mblab.records import build_record
from mblab.schema import RunRecord
from mblab.study import (
    RECORDS_FILE,
    Study,
    StudyError,
    append_jsonl,
    format_validation_error,
    load_jsonl_lines,
    rewrite_jsonl_lines,
    write_lock,
)

PLAN_COLUMNS = [
    "run_id",
    "sequence",
    "condition_id",
    "model_selection",
    "message_override",
    "task_id",
    "repetition",
]
ENTRY_COLUMNS = [
    "timestamp",
    "client_version",
    "response_status",
    "response_text",
    "observed_model_label",
    "observed_label_location",
    "thinking_indicator",
    "thinking_text",
    "thinking_seconds",
    "latency_seconds",
    "latency_method",
    "latency_confidence",
    "new_conversation",
]
TRAILING_COLUMNS = ["protocol_deviation", "notes", "recorded_by", "data_origin"]
CONFOUNDER_PREFIX = "confounder."
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def template_columns(study: Study) -> list[str]:
    confounders = [f"{CONFOUNDER_PREFIX}{c.id}" for c in study.definition.confounders]
    return [*PLAN_COLUMNS, "prompt_text", *ENTRY_COLUMNS, *confounders, *TRAILING_COLUMNS]


def write_template_csv(study: Study, path: Path) -> int:
    plan = study.require_plan()
    columns = template_columns(study)
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with a byte-order mark: spreadsheet apps (Excel in particular) then read
    # non-ASCII text such as "—" correctly instead of garbling it. Cells go through
    # to_cell, so a prompt that starts with "=" is shown as text, not run as a formula.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in sorted(plan.runs, key=lambda r: r.sequence):
            row = {col: "" for col in columns}
            row.update(
                run_id=run.run_id,
                sequence=run.sequence,
                condition_id=run.condition_id,
                model_selection=run.model_selection,
                message_override=run.message_override or "",
                task_id=run.task_id,
                repetition=run.repetition,
                prompt_text=study.definition.task(run.task_id).prompt,
            )
            writer.writerow({col: to_cell(value) for col, value in row.items()})
    return len(plan.runs)


# -- parsing helpers --------------------------------------------------------------


def parse_timezone(value: str | None) -> tzinfo | None:
    """Accept '+09:00', '-0500', 'UTC', or an IANA name such as 'Asia/Seoul'."""
    if not value:
        return None
    text = value.strip()
    if text.upper() in {"UTC", "Z"}:
        return UTC
    if text[0] in "+-":
        digits = text[1:].replace(":", "")
        if len(digits) == 4 and digits.isdigit():
            delta = timedelta(hours=int(digits[:2]), minutes=int(digits[2:]))
            return timezone(delta if text[0] == "+" else -delta)
        raise StudyError(f"invalid UTC offset {value!r} (use e.g. +09:00)")
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise StudyError(f"unknown time zone {value!r}") from exc


def parse_timestamp(value: str, default_tz: tzinfo | None) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"timestamp {value!r} is not ISO 8601 (e.g. 2026-11-20T14:03:00+09:00)"
        ) from exc
    if stamp.tzinfo is None:
        if default_tz is None:
            raise ValueError(
                f"timestamp {value!r} has no UTC offset; add one (e.g. +09:00) or pass --default-tz"
            )
        stamp = stamp.replace(tzinfo=default_tz)
    return stamp


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"expected true/false, got {value!r}")


def _squash(text: str) -> str:
    return " ".join(text.split())


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


@dataclass
class ImportResult:
    records: list[RunRecord] = field(default_factory=list)
    skipped_blank: int = 0
    errors: list[str] = field(default_factory=list)


def row_to_record(row: dict[str, str], study: Study, default_tz: tzinfo | None) -> RunRecord | None:
    """Convert one CSV row to a validated record. Returns None for an unfilled row."""
    plan = study.require_plan()
    run_id = (row.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("missing run_id")
    filled = any(not _blank(row.get(col)) for col in ENTRY_COLUMNS)
    if not filled:
        return None
    run = plan.by_run_id().get(run_id)
    if run is None:
        raise ValueError(f"run_id {run_id!r} is not in the plan")

    for col in PLAN_COLUMNS[1:]:
        given = row.get(col)
        if _blank(given):
            continue
        expected = getattr(run, col)
        if str(expected if expected is not None else "") != given.strip():
            raise ValueError(f"{col} is {given!r} but the plan says {expected!r}")

    data: dict[str, Any] = {}
    for col in (
        "client_version",
        "observed_model_label",
        "observed_label_location",
        "thinking_text",
        "protocol_deviation",
        "notes",
        "recorded_by",
    ):
        if not _blank(row.get(col)):
            data[col] = row[col].strip()
    task_prompt = study.definition.task(run.task_id).prompt
    csv_prompt = row.get("prompt_text")
    if not _blank(csv_prompt) and _squash(csv_prompt) != _squash(task_prompt):
        if _blank(row.get("protocol_deviation")):
            raise ValueError(
                "prompt_text differs from the task prompt. If a spreadsheet changed the text "
                "(for example its encoding), clear the prompt_text cell; if a different prompt "
                "was really sent, describe that in protocol_deviation"
            )
        data["prompt_text"] = csv_prompt  # a recorded deviation: keep what was sent
    # Otherwise the planned prompt is stored exactly (whitespace changes are ignored).
    text = row.get("response_text") or ""
    data["response_text"] = text
    data["response_status"] = (row.get("response_status") or "").strip() or (
        "completed" if text.strip() else ""
    )
    if _blank(row.get("timestamp")):
        raise ValueError("timestamp is required")
    data["timestamp"] = parse_timestamp(row["timestamp"], default_tz)

    for col in ("thinking_seconds", "latency_seconds"):
        if not _blank(row.get(col)):
            data[col] = float(row[col].strip().replace(",", "."))
    if not _blank(row.get("latency_method")):
        data["latency_method"] = row["latency_method"].strip()
    elif "latency_seconds" in data:
        data["latency_method"] = (
            "manual_stopwatch" if run.execution_method == "manual_ui" else "client_timer"
        )
    if not _blank(row.get("latency_confidence")):
        data["latency_confidence"] = row["latency_confidence"].strip()
    if not _blank(row.get("thinking_indicator")):
        data["thinking_indicator"] = row["thinking_indicator"].strip()
    elif "thinking_text" in data or "thinking_seconds" in data:
        data["thinking_indicator"] = "shown"
    if not _blank(row.get("new_conversation")):
        data["new_conversation"] = _parse_bool(row["new_conversation"])
    if not _blank(row.get("data_origin")):
        data["data_origin"] = row["data_origin"].strip()
    data["confounders"] = {
        key.removeprefix(CONFOUNDER_PREFIX): value.strip()
        for key, value in row.items()
        if key and key.startswith(CONFOUNDER_PREFIX) and not _blank(value)
    }
    return build_record(run, study.definition, plan, **data)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StudyError(
            f"{path} is not UTF-8 text (byte {exc.start}). Save it from your spreadsheet as "
            '"CSV UTF-8" and import it again'
        ) from None
    # from_row removes the apostrophe that mblab puts before text a spreadsheet would
    # run as a formula, so the template's prompt_text matches the planned prompt again.
    return [from_row(row) for row in csv.DictReader(io.StringIO(text))]


def import_file(study: Study, path: Path, default_tz: tzinfo | None = None) -> ImportResult:
    """Parse a CSV (template format), JSON array, or JSON-lines file of records."""
    result = ImportResult()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for number, row in enumerate(read_csv_rows(path), start=2):
            try:
                record = row_to_record(row, study, default_tz)
            except ValidationError as exc:
                result.errors.append(f"row {number}: {format_validation_error(exc)}")
                continue
            except (ValueError, StudyError) as exc:
                result.errors.append(f"row {number}: {exc}")
                continue
            if record is None:
                result.skipped_blank += 1
            else:
                result.records.append(record)
        return result

    if suffix in {".json", ".jsonl", ".ndjson"}:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise StudyError(f"{path} is not UTF-8 text (byte {exc.start})") from None
        items: list[tuple[int, Any]] = []
        if suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                result.errors.append(f"not valid JSON: {exc}")
                return result
            if not isinstance(data, list):
                result.errors.append("expected a JSON array of records")
                return result
            items = list(enumerate(data, start=1))
        else:
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    try:
                        items.append((number, json.loads(line)))
                    except json.JSONDecodeError as exc:
                        result.errors.append(f"line {number}: not valid JSON: {exc}")
        for number, item in items:
            try:
                result.records.append(RunRecord.model_validate(item))
            except ValidationError as exc:
                result.errors.append(f"item {number}: {format_validation_error(exc)}")
        return result

    raise StudyError(f"unsupported file type {path.suffix!r} (use .csv, .json, or .jsonl)")


@dataclass
class MergeResult:
    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    #: Line numbers of records.jsonl lines that could not be read; always kept as they are.
    unreadable_kept: list[int] = field(default_factory=list)


def merge_records(
    study: Study, incoming: list[RunRecord], replace: bool, dry_run: bool
) -> MergeResult:
    """Add records to the study, refusing duplicates unless ``replace`` is set.

    Works on the file as it is on disk right now, under the study's write lock, so a
    record saved meanwhile by another command is neither lost nor overwritten. Lines
    that cannot be read are written back unchanged; a rewrite keeps the previous file
    as ``records.jsonl.bak``. Nothing is written when there is a conflict.
    """
    batch_ids = [r.run_id for r in incoming]
    dup_in_batch = sorted({rid for rid in batch_ids if batch_ids.count(rid) > 1})
    if dup_in_batch:
        raise StudyError(f"the file contains several records for: {', '.join(dup_in_batch)}")

    path = study.root / RECORDS_FILE
    guard = contextlib.nullcontext() if dry_run else write_lock(study.root, "save records")
    with guard:
        lines = load_jsonl_lines(path, RunRecord.model_validate_json)
        result = MergeResult(unreadable_kept=[line.number for line in lines if line.error])
        existing = {line.item.run_id for line in lines if line.item is not None}
        replacements: dict[str, RunRecord] = {}
        for rec in incoming:
            if rec.run_id not in existing:
                result.added.append(rec.run_id)
            elif replace:
                replacements[rec.run_id] = rec
                result.replaced.append(rec.run_id)
            else:
                result.conflicts.append(rec.run_id)
        if result.conflicts or dry_run or not (result.added or result.replaced):
            return result

        added = [rec for rec in incoming if rec.run_id in set(result.added)]
        if replacements:
            out: list[bytes] = []
            done: set[str] = set()
            for line in lines:
                run_id = line.item.run_id if line.item is not None else None
                if run_id in replacements:
                    if (
                        run_id not in done
                    ):  # one record per replaced run; the old file is kept as .bak
                        out.append(replacements[run_id].model_dump_json().encode("utf-8"))
                        done.add(run_id)
                    continue
                out.append(line.raw)  # unreadable lines are kept exactly as they were
            out += [rec.model_dump_json().encode("utf-8") for rec in added]
            rewrite_jsonl_lines(path, out, backup=True)
        else:
            append_jsonl(path, added)

    fresh = Study.load(study.root)
    study.records, study.load_errors = fresh.records, fresh.load_errors
    return result
