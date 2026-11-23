"""Blind manual grading for tasks with ``method: manual``.

``export_blind_sheet`` writes a CSV for graders that contains only the task,
the rubric, and the response — no condition, run id, timestamp, displayed
label, or timing, all of which could reveal the condition. Rows are shuffled
and get neutral ids (G001, G002, ...). The mapping back to run ids is kept in a
separate key file that graders should not see.

Blinding hides metadata only: the style or length of a response may still hint
at the condition. Reports say so.
"""

from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mblab.csvcells import from_row, to_cell
from mblab.formatting import rubric_points
from mblab.plan import seeded_shuffle
from mblab.records import included_completions, parse_decimal
from mblab.schema import ManualGrade, ManualScoring, RunRecord, Task
from mblab.study import GRADING_DIR, Study, StudyError, atomic_write_bytes, atomic_write_text

SHEET_COLUMNS = [
    "grading_id",
    "sheet_id",
    "task_id",
    "task_title",
    "rubric",
    "max_points",
    "pass_threshold",
    "prompt_text",
    "response_text",
    "points",
    "grader_id",
    "notes",
]


def _grading_material(task: Task, rule: ManualScoring, record: RunRecord) -> dict[str, str]:
    return {
        "task_id": task.id,
        "task_title": task.title,
        "rubric": rule.rubric,
        "max_points": rubric_points(rule.max_points),
        "pass_threshold": rubric_points(rule.pass_threshold),
        "prompt_text": record.prompt_text,
        "response_text": record.response_text or "",
    }


def export_blind_sheet(
    study: Study, out_dir: Path | None = None, seed: int | None = None
) -> tuple[Path, Path, int]:
    plan = study.require_plan()
    manual = {t.id: t for t in study.definition.tasks if isinstance(t.scoring, ManualScoring)}
    if not manual:
        raise StudyError("no task uses `method: manual`; there is nothing to grade by hand")
    candidates = [r for r in included_completions(study) if r.task_id in manual]
    if not candidates:
        raise StudyError("no completed responses for manually graded tasks yet")
    rng = random.Random(plan.seed + 1 if seed is None else seed)
    seeded_shuffle(candidates, rng)

    out_dir = out_dir or study.root / GRADING_DIR
    sheet = out_dir / "blind-sheet.csv"
    key = out_dir / "blind-key.json"
    sheet_id = f"sheet-{uuid4().hex}"
    entries: dict[str, str] = {}
    # UTF-8 with a byte-order mark, so graders' spreadsheet apps read the text correctly.
    # Cells go through to_cell: a response such as "=HYPERLINK(...)" or "- item" is shown
    # as text in the grader's spreadsheet instead of being run as a formula.
    with io.StringIO(newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for number, rec in enumerate(candidates, start=1):
            gid = f"G{number:03d}"
            entries[gid] = rec.run_id
            task = manual[rec.task_id]
            rule = task.scoring
            assert isinstance(rule, ManualScoring)
            row = {
                "grading_id": gid,
                "sheet_id": sheet_id,
                **_grading_material(task, rule, rec),
                "points": "",
                "grader_id": "",
                "notes": "",
            }
            writer.writerow({col: to_cell(value) for col, value in row.items()})
        atomic_write_bytes(sheet, handle.getvalue().encode("utf-8-sig"))
    atomic_write_text(
        key,
        json.dumps(
            {
                "experiment_id": study.definition.experiment_id,
                "created_at": datetime.now().astimezone().isoformat(),
                "sheet": sheet.name,
                "sheet_id": sheet_id,
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
    )
    return sheet, key, len(candidates)


@dataclass
class GradeImport:
    grades: list[ManualGrade] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_blank: int = 0


def _load_blind_key(study: Study, key: Path) -> tuple[str | None, dict[str, str]]:
    try:
        data = json.loads(key.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StudyError(f"key file {key} not found") from None
    except (OSError, ValueError):
        raise StudyError(f"could not read grading key {key} as JSON") from None
    if not isinstance(data, dict):
        raise StudyError(f"invalid grading key {key}")
    entries = data.get("entries")
    sheet_id = data.get("sheet_id")
    if (
        not isinstance(entries, dict)
        or any(not isinstance(value, str) for value in entries.values())
        or ("sheet_id" in data and (not isinstance(sheet_id, str) or not sheet_id))
    ):
        raise StudyError(f"invalid grading key {key}")
    if data.get("experiment_id") != study.definition.experiment_id:
        raise StudyError("grading key belongs to a different experiment")
    return sheet_id, entries


def _material_difference(row: dict[str, str], expected: dict[str, str]) -> str | None:
    """Allow line-ending and numeric formatting changes made by spreadsheet software."""
    for column, value in expected.items():
        actual = row.get(column)
        if actual is None:
            return column
        if column in {"max_points", "pass_threshold"}:
            try:
                if parse_decimal(actual, column) == float(value):
                    continue
            except ValueError:
                pass
            return column
        # The sheet was read in universal-newline mode.
        if actual != value.replace("\r\n", "\n").replace("\r", "\n"):
            return column
    return None


def import_blind_sheet(
    study: Study,
    sheet: Path,
    key: Path,
    grader_id: str | None = None,
    now: datetime | None = None,
) -> GradeImport:
    result = GradeImport()
    sheet_id, entries = _load_blind_key(study, key)
    eligible = {r.run_id: r for r in included_completions(study)}
    existing = {(g.run_id, g.grader_id) for g in study.grades}
    now = now or datetime.now().astimezone()
    try:
        sheet_text = sheet.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StudyError(
            f"{sheet} is not UTF-8 text (byte {exc.start}). Save it from your spreadsheet as "
            '"CSV UTF-8" and import it again'
        ) from None
    with io.StringIO(sheet_text, newline="") as handle:
        reader = csv.DictReader(handle)
        if ("sheet_id" in (reader.fieldnames or [])) != (sheet_id is not None):
            raise StudyError("grading sheet and key are from different exports")
        for number, raw in enumerate(reader, start=2):
            row = from_row(raw)
            if sheet_id is not None and row.get("sheet_id") != sheet_id:
                raise StudyError(f"row {number}: grading sheet and key are from different exports")
            gid = (row.get("grading_id") or "").strip()
            points_text = (row.get("points") or "").strip()
            if not points_text:
                result.skipped_blank += 1
                continue
            run_id = entries.get(gid)
            if run_id is None:
                result.errors.append(f"row {number}: unknown grading_id {gid!r}")
                continue
            record = eligible.get(run_id)
            if record is None:
                result.errors.append(f"row {number}: response is no longer eligible for grading")
                continue
            task = study.definition.task(record.task_id)
            if not isinstance(task.scoring, ManualScoring):
                result.errors.append(f"row {number}: task {task.id} is not manually graded")
                continue
            if sheet_id is not None:
                changed_field = _material_difference(
                    row, _grading_material(task, task.scoring, record)
                )
                if changed_field is not None:
                    result.errors.append(
                        f"row {number}: {changed_field} does not match the current study. "
                        "Export a fresh sheet to grade a changed response or rubric"
                    )
                    continue
            grader = (row.get("grader_id") or "").strip() or (grader_id or "").strip()
            if not grader:
                result.errors.append(f"row {number}: grader_id missing (column or --grader)")
                continue
            try:
                points = parse_decimal(points_text, "points")
            except ValueError as exc:
                result.errors.append(f"row {number}: {exc}")
                continue
            if not 0 <= points <= task.scoring.max_points:
                result.errors.append(
                    f"row {number}: points {points:g} outside 0..{task.scoring.max_points:g}"
                )
                continue
            if (run_id, grader) in existing:
                result.errors.append(f"row {number}: {grader} already graded this response")
                continue
            existing.add((run_id, grader))
            result.grades.append(
                ManualGrade(
                    run_id=run_id,
                    points=points,
                    max_points=task.scoring.max_points,
                    grader_id=grader,
                    blind=True,
                    graded_at=now,
                    notes=(row.get("notes") or "").strip() or None,
                )
            )
    return result
