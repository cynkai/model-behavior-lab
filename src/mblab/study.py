"""Study directory layout and file I/O. Everything lives in local files.

A study directory looks like this::

    my-study/
      experiment.yaml     # pre-registered definition (you edit this before collecting)
      plan.json           # randomized run order + hash and snapshot of the definition
      records.jsonl       # one observed run per line (append-only in normal use)
      grades.jsonl        # manual grades, if any task is graded by hand
      grading/            # blind grading sheet and its key
      reports/            # generated Markdown reports and analysis JSON
      exports/            # generated CSV / JSON exports
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from mblab.schema import ExperimentDefinition, ManualGrade, RunPlan, RunRecord

DEFINITION_FILE = "experiment.yaml"
PLAN_FILE = "plan.json"
RECORDS_FILE = "records.jsonl"
GRADES_FILE = "grades.jsonl"
GRADING_DIR = "grading"
REPORTS_DIR = "reports"
EXPORTS_DIR = "exports"

DEFAULT_TEMPLATE = "chatgpt-model-selection-override"

M = TypeVar("M", bound=BaseModel)


class StudyError(Exception):
    """A problem the user can fix (missing file, invalid definition, ...)."""


@dataclass(frozen=True)
class LineError:
    file: str
    line: int
    message: str


def format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# -- templates ---------------------------------------------------------------


def available_templates() -> list[str]:
    folder = resources.files("mblab") / "templates"
    return sorted(
        p.name.removesuffix(".yaml") for p in folder.iterdir() if p.name.endswith(".yaml")
    )


def template_text(name: str = DEFAULT_TEMPLATE) -> str:
    if name not in available_templates():
        raise StudyError(
            f"unknown template '{name}' (available: {', '.join(available_templates())})"
        )
    return (resources.files("mblab") / "templates" / f"{name}.yaml").read_text(encoding="utf-8")


# -- definition ----------------------------------------------------------------


def parse_definition(text: str, source: str = DEFINITION_FILE) -> ExperimentDefinition:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StudyError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError(f"{source}: expected a mapping at the top level")
    try:
        return ExperimentDefinition.model_validate(data)
    except ValidationError as exc:
        raise StudyError(f"{source}: {format_validation_error(exc)}") from exc


def load_definition(path: Path) -> ExperimentDefinition:
    if not path.exists():
        raise StudyError(f"{path} not found — run `mblab init` first or pass --study")
    return parse_definition(path.read_text(encoding="utf-8"), str(path))


# -- plan --------------------------------------------------------------------------


def load_plan(path: Path) -> RunPlan | None:
    if not path.exists():
        return None
    try:
        return RunPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise StudyError(f"{path}: {format_validation_error(exc)}") from exc


def save_plan(plan: RunPlan, path: Path) -> None:
    atomic_write_text(path, plan.model_dump_json(indent=2) + "\n")


# -- JSON lines --------------------------------------------------------------------


def read_jsonl(path: Path, model: type[M]) -> tuple[list[M], list[LineError]]:
    """Read a JSON-lines file. Bad lines are reported, not raised, so every
    problem can be listed at once by ``mblab validate``."""
    items: list[M] = []
    errors: list[LineError] = []
    if not path.exists():
        return items, errors
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(model.model_validate_json(line))
            except ValidationError as exc:
                errors.append(LineError(path.name, number, format_validation_error(exc)))
    return items, errors


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    text = "".join(item.model_dump_json() + "\n" for item in items)
    atomic_write_text(path, text)


def append_jsonl(path: Path, items: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")


# -- study -------------------------------------------------------------------------


@dataclass
class Study:
    root: Path
    definition: ExperimentDefinition
    plan: RunPlan | None
    records: list[RunRecord]
    grades: list[ManualGrade]
    load_errors: list[LineError] = field(default_factory=list)

    @staticmethod
    def paths(root: Path) -> dict[str, Path]:
        return {
            "definition": root / DEFINITION_FILE,
            "plan": root / PLAN_FILE,
            "records": root / RECORDS_FILE,
            "grades": root / GRADES_FILE,
            "grading": root / GRADING_DIR,
            "reports": root / REPORTS_DIR,
            "exports": root / EXPORTS_DIR,
        }

    @classmethod
    def load(cls, root: Path | str) -> Study:
        root = Path(root)
        if not root.is_dir():
            raise StudyError(f"study directory {root} does not exist")
        paths = cls.paths(root)
        definition = load_definition(paths["definition"])
        plan = load_plan(paths["plan"])
        records, record_errors = read_jsonl(paths["records"], RunRecord)
        grades, grade_errors = read_jsonl(paths["grades"], ManualGrade)
        return cls(root, definition, plan, records, grades, record_errors + grade_errors)

    @property
    def records_path(self) -> Path:
        return self.root / RECORDS_FILE

    @property
    def grades_path(self) -> Path:
        return self.root / GRADES_FILE

    @property
    def plan_path(self) -> Path:
        return self.root / PLAN_FILE

    def require_plan(self) -> RunPlan:
        if self.plan is None:
            raise StudyError(f"{self.plan_path} not found — run `mblab plan` first")
        return self.plan

    def recorded_run_ids(self) -> set[str]:
        return {r.run_id for r in self.records}

    def is_synthetic(self) -> bool:
        return self.definition.demo or any(r.data_origin == "synthetic" for r in self.records)


def dump_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
