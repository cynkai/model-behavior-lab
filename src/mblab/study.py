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

Every command that writes to ``records.jsonl``, ``grades.jsonl``, ``plan.json`` or
the API study's unrecorded-response log holds one study-wide lock (``.mblab.lock``)
and reads the files again inside it, so two writers never work from stale copies.
Lines that cannot be read are never deleted by a rewrite; they are kept exactly as
they were and reported by ``mblab validate``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

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
#: Billed API responses that arrived but could not be recorded (official_api studies).
UNRECORDED_FILE = "api-unrecorded.jsonl"
#: Held while any command writes to the study.
LOCK_FILE = ".mblab.lock"

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


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write a file in one step (temporary file + rename), flushed to disk first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.replace("\r\n", "\n").encode("utf-8"))


# -- the study-wide write lock ------------------------------------------------------

_held_locks: dict[tuple[str, int], int] = {}
_held_guard = threading.Lock()


def _pid_alive(pid: int) -> bool:
    if sys.platform.startswith("win"):
        return True  # os.kill(pid, 0) would terminate the process on Windows; never guess
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # exists but belongs to someone else, or cannot be checked
    return True


def _lock_holder(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def write_lock(root: Path, purpose: str) -> Iterator[None]:
    """Hold the study-wide lock while writing; fail at once if another process holds it.

    Re-entrant within one thread. A lock left by a process that no longer exists on
    this computer is removed automatically.
    """
    root = Path(root)
    key = (str(root.resolve()), threading.get_ident())
    with _held_guard:
        if key in _held_locks:
            _held_locks[key] += 1
            nested = True
        else:
            nested = False
    if nested:
        try:
            yield
        finally:
            with _held_guard:
                _held_locks[key] -= 1
        return

    path = root / LOCK_FILE
    info = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "purpose": purpose,
        "since": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    for attempt in (1, 2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            holder = _lock_holder(path)
            pid = holder.get("pid")
            stale = (
                isinstance(pid, int)
                and holder.get("host") == info["host"]
                and pid != os.getpid()
                and not _pid_alive(pid)
            )
            if attempt == 1 and stale:
                path.unlink(missing_ok=True)  # left behind by a process that no longer exists
                continue
            who = (
                f"`{holder.get('purpose', 'another command')}` (pid {pid}, since "
                f"{holder.get('since', '?')})"
                if holder
                else "another command"
            )
            raise StudyError(
                f"the study is being written by {who}. Wait for it to finish and try again; "
                f"if nothing is running, delete {path}."
            ) from None
    try:
        os.write(fd, json.dumps(info).encode("utf-8"))
        os.close(fd)
        with _held_guard:
            _held_locks[key] = 1
        yield
    finally:
        with _held_guard:
            _held_locks.pop(key, None)
        path.unlink(missing_ok=True)


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


@dataclass(frozen=True)
class JsonLine:
    """One non-empty line of a JSON-lines file, kept as raw bytes so it can be
    written back unchanged when it cannot be parsed."""

    number: int
    raw: bytes
    item: Any | None
    error: str | None


def load_jsonl_lines(path: Path, parse: Callable[[str], Any]) -> list[JsonLine]:
    """Parse every line on its own: a damaged line (bad JSON, bad UTF-8, wrong fields)
    becomes an error for that line only, never an exception for the whole file."""
    if not path.exists():
        return []
    lines: list[JsonLine] = []
    for number, raw in enumerate(path.read_bytes().split(b"\n"), start=1):
        raw = raw.rstrip(b"\r")
        if not raw.strip():
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            lines.append(JsonLine(number, raw, None, f"not valid UTF-8 text (byte {exc.start})"))
            continue
        try:
            lines.append(JsonLine(number, raw, parse(text), None))
        except ValidationError as exc:
            lines.append(JsonLine(number, raw, None, format_validation_error(exc)))
        except ValueError as exc:
            lines.append(JsonLine(number, raw, None, str(exc)))
    return lines


def read_jsonl(path: Path, model: type[M]) -> tuple[list[M], list[LineError]]:
    """Read a JSON-lines file. Bad lines are reported, not raised, so every
    problem can be listed at once by ``mblab validate``."""
    lines = load_jsonl_lines(path, model.model_validate_json)
    items = [line.item for line in lines if line.error is None]
    errors = [LineError(path.name, line.number, line.error) for line in lines if line.error]
    return items, errors


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> None:
    text = "".join(item.model_dump_json() + "\n" for item in items)
    atomic_write_text(path, text)


def rewrite_jsonl_lines(path: Path, lines: Sequence[bytes], backup: bool = True) -> None:
    """Replace a JSON-lines file in one step; the previous version is kept as ``<name>.bak``."""
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    atomic_write_bytes(path, b"".join(line + b"\n" for line in lines))


def _append_lines(path: Path, payloads: Sequence[bytes]) -> None:
    """Append lines, first ending a last line that lacks its newline, then flush to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with path.open("ab") as handle:
        if needs_newline:
            handle.write(b"\n")
        for payload in payloads:
            handle.write(payload + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, items: Sequence[BaseModel]) -> None:
    _append_lines(path, [item.model_dump_json().encode("utf-8") for item in items])


def append_json_object(path: Path, data: dict[str, Any]) -> None:
    _append_lines(path, [json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")])


def _parse_unrecorded(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("run_id"), str):
        raise ValueError("expected an object with a run_id")
    return data


def read_unrecorded_log(root: Path) -> tuple[list[dict[str, Any]], list[LineError]]:
    """Entries of the unrecorded-response log, and its unreadable lines."""
    lines = load_jsonl_lines(root / UNRECORDED_FILE, _parse_unrecorded)
    entries = [line.item for line in lines if line.error is None]
    errors = [LineError(UNRECORDED_FILE, line.number, line.error) for line in lines if line.error]
    return entries, errors


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
        _, unrecorded_errors = read_unrecorded_log(root)
        errors = record_errors + grade_errors + unrecorded_errors
        return cls(root, definition, plan, records, grades, errors)

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


def _jsonable(value: object) -> object:
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def dump_json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=_jsonable) + "\n"
