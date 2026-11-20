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

from mblab.plan import seeded_shuffle
from mblab.schema import ManualGrade, ManualScoring
from mblab.study import GRADING_DIR, Study, StudyError, atomic_write_text

SHEET_COLUMNS = [
    "grading_id",
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


def export_blind_sheet(
    study: Study, out_dir: Path | None = None, seed: int | None = None
) -> tuple[Path, Path, int]:
    plan = study.require_plan()
    manual = {t.id: t for t in study.definition.tasks if isinstance(t.scoring, ManualScoring)}
    if not manual:
        raise StudyError("no task uses `method: manual`; there is nothing to grade by hand")
    candidates = [
        r
        for r in study.records
        if r.task_id in manual and r.response_status == "completed" and not r.excluded
    ]
    if not candidates:
        raise StudyError("no completed responses for manually graded tasks yet")
    rng = random.Random(plan.seed + 1 if seed is None else seed)
    seeded_shuffle(candidates, rng)

    out_dir = out_dir or study.root / GRADING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / "blind-sheet.csv"
    key = out_dir / "blind-key.json"
    entries: dict[str, str] = {}
    # UTF-8 with a byte-order mark, so graders' spreadsheet apps read the text correctly.
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for number, rec in enumerate(candidates, start=1):
            gid = f"G{number:03d}"
            entries[gid] = rec.run_id
            task = manual[rec.task_id]
            rule = task.scoring
            assert isinstance(rule, ManualScoring)
            writer.writerow(
                {
                    "grading_id": gid,
                    "task_id": task.id,
                    "task_title": task.title,
                    "rubric": rule.rubric,
                    "max_points": f"{rule.max_points:g}",
                    "pass_threshold": f"{rule.pass_threshold:g}",
                    "prompt_text": rec.prompt_text,
                    "response_text": rec.response_text,
                    "points": "",
                    "grader_id": "",
                    "notes": "",
                }
            )
    atomic_write_text(
        key,
        json.dumps(
            {
                "experiment_id": study.definition.experiment_id,
                "created_at": datetime.now().astimezone().isoformat(),
                "sheet": sheet.name,
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


def import_blind_sheet(
    study: Study,
    sheet: Path,
    key: Path,
    grader_id: str | None = None,
    now: datetime | None = None,
) -> GradeImport:
    result = GradeImport()
    if not key.exists():
        raise StudyError(f"key file {key} not found")
    entries: dict[str, str] = json.loads(key.read_text(encoding="utf-8"))["entries"]
    records = {r.run_id: r for r in study.records}
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
        for number, row in enumerate(csv.DictReader(handle), start=2):
            gid = (row.get("grading_id") or "").strip()
            points_text = (row.get("points") or "").strip()
            if not points_text:
                result.skipped_blank += 1
                continue
            run_id = entries.get(gid)
            if run_id is None or run_id not in records:
                result.errors.append(f"row {number}: unknown grading_id {gid!r}")
                continue
            task = study.definition.task(records[run_id].task_id)
            if not isinstance(task.scoring, ManualScoring):
                result.errors.append(f"row {number}: task {task.id} is not manually graded")
                continue
            grader = (row.get("grader_id") or "").strip() or (grader_id or "")
            if not grader:
                result.errors.append(f"row {number}: grader_id missing (column or --grader)")
                continue
            try:
                points = float(points_text.replace(",", "."))
            except ValueError:
                result.errors.append(f"row {number}: points {points_text!r} is not a number")
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
