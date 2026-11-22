"""Benchmarks: a fixed task with an independently computed ground truth, and runs
recorded by hand.

A benchmark directory holds

* ``benchmark.yaml`` — the task (its prompt kept verbatim), the conditions (default and
  per-message selections, exactly as in an experiment definition), the answer fields and
  how each is checked, and the reports made from the runs;
* ``runs.jsonl`` — one line per run. Lines are only ever added: a further run of a
  condition gets the next repetition number and its own run id, so earlier runs are
  never overwritten;
* the reports that ``benchmark.yaml`` lists, written by ``mblab benchmark report``.

A benchmark run is a different kind of record from a study record (``mblab init``). A
study keeps every response in full, with the time it was sent, in a pre-registered
randomized order. A benchmark run may hold only the experimenter's structured summary
of a response (``raw_response`` is then null), and every value that was not recorded
stays null. Reports say which.

Answers are checked against the verifier's ground truth, never against a number typed
into the data: the count with the numeric rule of :mod:`mblab.scoring`, the orders with
the verifier. The reference answers in ``benchmark.yaml`` must agree with the verifier,
or the benchmark does not load.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, ClassVar, Literal

import pandas as pd
import yaml
from pydantic import (
    AwareDatetime,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from mblab import task_ordering
from mblab.analysis import _find_reference
from mblab.report import md_table
from mblab.schema import (
    LATENCY_METHODS_BY_EXECUTION,
    Condition,
    Difficulty,
    DocumentedFact,
    Hypothesis,
    Identifier,
    LatencyMethod,
    SelectionOption,
    StrictModel,
    VersionedModel,
    _drop_unset,
    condition_problems,
)
from mblab.scoring import extract_answer, parse_single_number, strip_markdown_edges
from mblab.study import (
    StudyError,
    atomic_write_text,
    format_validation_error,
    read_jsonl,
    write_lock,
)

BENCHMARK_FILE = "benchmark.yaml"
RUNS_FILE = "runs.jsonl"
#: Where the repository keeps its benchmarks (used in the commands a report prints).
BENCHMARKS_DIR = "benchmarks"

#: Verifiers by name. Each computes the ground truth of one kind of task on its own and
#: offers ``ground_truth()``, ``parse_order()``, ``violations()``, ``breakdown()`` and
#: ``describe()``.
VERIFIERS: dict[str, ModuleType] = {"task-ordering-8": task_ordering}

AnswerCheck = Literal["count", "lexicographic_first", "other_valid_order"]

CHECK_TEXT: dict[str, str] = {
    "count": "one number equal to the verified number of valid orders (numeric rule)",
    "lexicographic_first": "the verified lexicographically first valid order",
    "other_valid_order": (
        "an order that satisfies every rule and differs from the response's own answer for "
        "the first order"
    ),
}


# -- definition -----------------------------------------------------------------------------


class AnswerField(StrictModel):
    """One line of the answer format the prompt asks for."""

    id: Identifier
    label: str = Field(
        min_length=1, description="The answer line's label, exactly as the prompt writes it."
    )
    check: AnswerCheck
    reference_answer: str = Field(
        min_length=1,
        description="Pre-registered answer; for other_valid_order, one valid example.",
    )


class BenchmarkTask(StrictModel):
    id: Identifier
    title: str = Field(min_length=1)
    category: str | None = None
    difficulty: Difficulty | None = Field(
        default=None, description="Author-assigned before data collection; not calibrated."
    )
    language: str | None = None
    prompt: str = Field(min_length=1, description="Kept verbatim; every run records its hash.")
    prompt_source: str | None = None
    verifier: str = Field(min_length=1, description="Name of the ground-truth verifier.")
    answers: list[AnswerField] = Field(min_length=1)

    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


class ReportSpec(StrictModel):
    """A report made from the runs of some batches. Earlier reports stay as they are when
    later batches add runs; a new report lists the batches it covers."""

    id: Identifier
    title: str = Field(min_length=1)
    batches: list[Identifier] = Field(min_length=1)
    path: str = Field(min_length=1, description="Where the report is written, in this directory.")
    reproduction_scope: Literal["benchmark", "report"] | None = Field(
        default=None,
        description="Use 'benchmark' to preserve the all-runs commands in a published report.",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _drop_unset(handler(self), "reproduction_scope")


class BenchmarkDefinition(VersionedModel):
    file_kind: ClassVar[str] = "this benchmark"
    schema_version: Literal[1] = 1
    benchmark_id: Identifier
    benchmark_version: str = Field(pattern=r"^\d+\.\d+\.\d+([.-][A-Za-z0-9.-]+)?$")
    title: str = Field(min_length=1)
    service: str = Field(min_length=1)
    platform: str = Field(min_length=1, description="The client, as precisely as it is known.")
    execution_method: Literal["manual_ui"]
    research_questions: list[str] = Field(min_length=1)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    documented_facts: list[DocumentedFact] = Field(default_factory=list)
    not_observable: list[str] = Field(default_factory=list)
    not_recorded: list[str] = Field(default_factory=list)
    notes_batches: list[Identifier] | None = Field(
        default=None, min_length=1, description="Batches described by not_recorded and planned."
    )
    selection_options: list[SelectionOption] = Field(min_length=1)
    conditions: list[Condition] = Field(min_length=1)
    task: BenchmarkTask
    planned: list[str] = Field(
        default_factory=list, description="Further runs that are planned (not results)."
    )
    reports: list[ReportSpec] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _drop_unset(handler(self), "notes_batches")

    @model_validator(mode="after")
    def _check(self) -> BenchmarkDefinition:
        problems = condition_problems(self.selection_options, self.conditions, self.hypotheses)
        for opt in self.selection_options:
            if opt.reasoning_effort is not None:
                problems.append(
                    f"selection option '{opt.id}': reasoning_effort is only valid for "
                    "official-API studies"
                )
        for kind, ids in (
            ("answer", [a.id for a in self.task.answers]),
            ("report", [r.id for r in self.reports]),
            ("report path", [r.path for r in self.reports]),
        ):
            for value, n in Counter(ids).items():
                if n > 1:
                    problems.append(f"duplicate {kind} '{value}'")
        for spec in self.reports:
            parts = PurePosixPath(spec.path).parts
            if spec.path.startswith("/") or ".." in parts or not spec.path.endswith(".md"):
                problems.append(
                    f"report '{spec.id}': path must be a .md file inside the benchmark directory"
                )
        if self.task.verifier not in VERIFIERS:
            known = ", ".join(sorted(VERIFIERS))
            problems.append(f"unknown verifier '{self.task.verifier}' (known: {known})")
        if not problems:
            problems += _reference_problems(self.task)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def condition(self, condition_id: str) -> Condition:
        for cond in self.conditions:
            if cond.id == condition_id:
                return cond
        raise KeyError(condition_id)

    def option_label(self, option_id: str | None) -> str:
        if option_id is None:
            return "none"
        for opt in self.selection_options:
            if opt.id == option_id:
                return opt.label
        return option_id

    def hypothesis_for(self, cond: Condition, pattern: str) -> list[str]:
        """Ids of this condition's hypotheses that predict ``pattern``."""
        wanted = {h.id for h in self.hypotheses if h.predicted_pattern == pattern}
        return [hid for hid in cond.hypothesis_ids if hid in wanted]


def _reference_problems(task: BenchmarkTask) -> list[str]:
    """Pre-registered reference answers must be correct by the verifier."""
    verifier = VERIFIERS[task.verifier]
    truth = verifier.ground_truth()
    results = check_answers(task, verifier, truth, {a.id: a.reference_answer for a in task.answers})
    return [
        f"answer '{r.field.id}': reference_answer {r.field.reference_answer!r} is not correct "
        f"according to the verifier ({r.detail})"
        for r in results
        if not r.correct
    ]


# -- runs -----------------------------------------------------------------------------------


class BenchmarkRun(VersionedModel):
    """One run of one condition. Only what was observed; null where it was not recorded."""

    file_kind: ClassVar[str] = "this run"
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    benchmark_id: str
    batch: Identifier = Field(description="The recording session; reports select runs by batch.")
    condition_id: str
    model_selection: str
    message_override: str | None
    repetition: int = Field(ge=1)
    data_origin: Literal["observed"] = "observed"
    recorded_on: date
    timestamp: AwareDatetime | None = Field(
        default=None, description="When the prompt was sent, with a UTC offset; null if unknown."
    )
    run_order: int | None = Field(
        default=None, ge=1, description="Position of the run within its batch; null if unknown."
    )
    device_type: str | None = None
    device_model: str | None = None
    operating_system: str | None = None
    app_version: str | None = None
    new_conversation: bool | None = None
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_model_label: str | None = Field(
        default=None, description="Model label the app displayed with the response, verbatim."
    )
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_method: LatencyMethod = "not_measured"
    raw_response: str | None = Field(
        default=None, description="The full response text; null if it was not kept."
    )
    answers: dict[str, str] = Field(
        default_factory=dict,
        description="Each answer line's value as the response gave it, by answer id.",
    )
    reported_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Intermediate counts the response reported, by verifier breakdown key.",
    )
    solution_summary: list[str] = Field(
        default_factory=list, description="The experimenter's summary of the solution, verbatim."
    )
    source: str = Field(min_length=1, description="How this record was made.")
    notes: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> BenchmarkRun:
        if self.latency_method == "not_measured":
            if self.latency_seconds is not None:
                raise ValueError("latency_seconds is set but latency_method is 'not_measured'")
        elif self.latency_seconds is None:
            raise ValueError(f"latency_method is '{self.latency_method}' but no latency_seconds")
        if self.latency_method not in LATENCY_METHODS_BY_EXECUTION["manual_ui"]:
            raise ValueError(f"latency_method '{self.latency_method}' is not valid for a UI run")
        if self.raw_response is None and not self.answers:
            raise ValueError("a run needs raw_response or answers")
        return self


def run_problems(definition: BenchmarkDefinition, runs: list[BenchmarkRun]) -> list[str]:
    problems: list[str] = []
    prompt_hash = definition.task.prompt_sha256()
    answer_ids = {a.id for a in definition.task.answers}
    verifier = VERIFIERS[definition.task.verifier]
    run_ids: set[str] = set()
    repetitions: dict[tuple[str, int], str] = {}
    for run in runs:
        where = f"run '{run.run_id}'"
        if run.benchmark_id != definition.benchmark_id:
            problems.append(f"{where}: benchmark_id '{run.benchmark_id}' is not this benchmark")
        if run.run_id in run_ids:
            problems.append(f"duplicate run id '{run.run_id}'")
        run_ids.add(run.run_id)
        try:
            cond = definition.condition(run.condition_id)
        except KeyError:
            problems.append(f"{where}: unknown condition '{run.condition_id}'")
            continue
        if (run.model_selection, run.message_override) != (
            cond.model_selection,
            cond.message_override,
        ):
            problems.append(
                f"{where}: settings {run.model_selection}/{run.message_override} do not match "
                f"condition {cond.id} ({cond.model_selection}/{cond.message_override})"
            )
        key = (run.condition_id, run.repetition)
        if key in repetitions:
            problems.append(
                f"runs '{repetitions[key]}' and '{run.run_id}' are both repetition "
                f"{run.repetition} of condition {run.condition_id}: a further run needs the "
                "next repetition number"
            )
        repetitions[key] = run.run_id
        if run.prompt_sha256 != prompt_hash:
            problems.append(
                f"{where}: prompt_sha256 does not match the prompt in {BENCHMARK_FILE} (the "
                "prompt was edited after this run was recorded, or the run used another prompt)"
            )
        for answer_id in sorted(set(run.answers) - answer_ids):
            problems.append(f"{where}: unknown answer id '{answer_id}'")
        for key_name in run.reported_breakdown:
            try:
                verifier.describe(key_name)
            except KeyError:
                problems.append(f"{where}: unknown breakdown key '{key_name}'")
    batches = {run.batch for run in runs}
    for batch in definition.notes_batches or []:
        if batch not in batches:
            problems.append(f"notes_batches lists batch '{batch}', which has no runs")
    for spec in definition.reports:
        for batch in spec.batches:
            if batch not in batches:
                problems.append(f"report '{spec.id}' lists batch '{batch}', which has no runs")
    return problems


def parse_benchmark(text: str, source: str = BENCHMARK_FILE) -> BenchmarkDefinition:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise StudyError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise StudyError(f"{source}: expected a mapping at the top level")
    try:
        return BenchmarkDefinition.model_validate(data)
    except ValidationError as exc:
        raise StudyError(f"{source}: {format_validation_error(exc)}") from exc


@dataclass
class Benchmark:
    root: Path
    definition: BenchmarkDefinition
    runs: list[BenchmarkRun]

    @classmethod
    def load(cls, root: Path | str) -> Benchmark:
        root = Path(root)
        path = root / BENCHMARK_FILE
        if not path.exists():
            raise StudyError(f"{path} not found — not a benchmark directory")
        definition = parse_benchmark(path.read_text(encoding="utf-8"), str(path))
        runs, errors = read_jsonl(root / RUNS_FILE, BenchmarkRun)
        problems = [f"{e.file} line {e.line}: {e.message}" for e in errors]
        problems += run_problems(definition, runs)
        if problems:
            raise StudyError(f"{root}: " + "; ".join(problems))
        return cls(root, definition, runs)

    @property
    def batches(self) -> list[str]:
        return list(dict.fromkeys(run.batch for run in self.runs))

    def report_spec(self, report_id: str) -> ReportSpec:
        for spec in self.definition.reports:
            if spec.id == report_id:
                return spec
        raise StudyError(f"no report '{report_id}' in {BENCHMARK_FILE}")


def is_benchmark_dir(root: Path) -> bool:
    return (root / BENCHMARK_FILE).is_file()


# -- checking answers -----------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerResult:
    field: AnswerField
    answer: str | None
    value: int | float | str | None
    readable: bool
    correct: bool
    detail: str


_SEPARATORS = re.compile(r"[\s,\-–—→>·/|]+")
_EDGES = " \t\"'“”‘’.,;:!?()[]"


def _read_order(answer: str, verifier: ModuleType) -> tuple[str | None, str]:
    text = _SEPARATORS.sub("", strip_markdown_edges(answer)).strip(_EDGES)
    try:
        return verifier.parse_order(text), "an order of the tasks"
    except ValueError:
        return None, f"{answer!r} is not an order of the tasks (each exactly once)"


def _check_one(
    field: AnswerField,
    text: str,
    verifier: ModuleType,
    truth: Any,
    own_first: str | None,
) -> AnswerResult:
    line = f"{field.label}: {text}"
    answer, detail = extract_answer(line, "answer_line", field.label)
    if answer is None:
        return AnswerResult(field, None, None, False, False, detail)
    if field.check == "count":
        number, why = parse_single_number(answer)
        if number is None:
            return AnswerResult(field, answer, None, False, False, why)
        value = int(number) if number.is_integer() else number
        return AnswerResult(
            field,
            answer,
            value,
            True,
            value == truth.count,
            f"read {value}; expected {truth.count}",
        )
    order, why = _read_order(answer, verifier)
    if order is None:
        return AnswerResult(field, answer, None, False, False, why)
    if field.check == "lexicographic_first":
        ok = order == truth.lexicographic_first
        why = f"read {order}; the verified first order is {truth.lexicographic_first}"
        return AnswerResult(field, answer, order, True, ok, why)
    broken = verifier.violations(order)
    other = own_first or truth.lexicographic_first
    if broken:
        why = f"{order} breaks rule {', '.join(str(n) for n in broken)}"
    elif order == other:
        why = f"{order} is the same order as the answer for the first order"
    else:
        why = f"{order} satisfies every rule and differs from {other}"
    return AnswerResult(field, answer, order, True, not broken and order != other, why)


def check_answers(
    task: BenchmarkTask, verifier: ModuleType, truth: Any, values: dict[str, str]
) -> list[AnswerResult]:
    """Check each answer field's value (as the answer line gives it), in field order."""
    results: dict[str, AnswerResult] = {}
    own_first: str | None = None
    # The other valid order is judged against the response's own first order, so that
    # one is read first.
    ordered = sorted(task.answers, key=lambda f: f.check == "other_valid_order")
    for field in ordered:
        if field.id not in values:
            results[field.id] = AnswerResult(field, None, None, False, False, "no answer available")
            continue
        result = _check_one(field, values[field.id], verifier, truth, own_first)
        if field.check == "lexicographic_first" and isinstance(result.value, str):
            own_first = result.value
        results[field.id] = result
    return [results[f.id] for f in task.answers]


def _answer_values(task: BenchmarkTask, run: BenchmarkRun) -> dict[str, str]:
    """The value of each answer line: read from the full response when it was kept,
    otherwise as recorded."""
    if run.raw_response is None:
        return dict(run.answers)
    values = {}
    for field in task.answers:
        answer, _ = extract_answer(run.raw_response, "answer_line", field.label)
        if answer is not None:
            values[field.id] = answer
    return values


# -- evaluation -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakdownCheck:
    key: str
    description: str
    reported: int
    verified: int

    @property
    def matches(self) -> bool:
        return self.reported == self.verified


@dataclass(frozen=True)
class RunResult:
    run: BenchmarkRun
    condition: Condition
    answers: tuple[AnswerResult, ...]
    breakdown: tuple[BreakdownCheck, ...]

    def answer(self, check: str) -> AnswerResult | None:
        for result in self.answers:
            if result.field.check == check:
                return result
        return None

    @property
    def full_success(self) -> bool:
        return all(a.correct for a in self.answers)

    @property
    def partial(self) -> bool:
        return not self.full_success and any(a.correct for a in self.answers)

    @property
    def outcome(self) -> str:
        if self.full_success:
            return "full success"
        return "partially correct" if self.partial else "no part correct"

    def error_type(self, truth: Any) -> str:
        kinds = []
        for result in self.answers:
            if result.correct:
                continue
            if result.field.check == "count":
                if not isinstance(result.value, int | float):
                    kinds.append("count unreadable")
                elif result.value < truth.count:
                    kinds.append("count too low")
                else:
                    kinds.append("count too high")
            elif not result.readable:
                kinds.append(f"{result.field.id} unreadable")
            else:
                kinds.append(f"{result.field.id} wrong")
        return ", ".join(kinds) or "none"

    def error_detail(self, truth: Any) -> str:
        parts = []
        count = self.answer("count")
        if count is not None and not count.correct and isinstance(count.value, int | float):
            diff = count.value - truth.count
            parts.append(
                f"count {count.value} instead of {truth.count} "
                f"({abs(diff):g} {'more' if diff > 0 else 'fewer'})"
            )
        elif count is not None and not count.correct:
            parts.append(f"count: {count.detail}")
        for result in self.answers:
            if not result.correct and result.field.check != "count":
                parts.append(f"{result.field.id}: {result.detail}")
        for check in self.breakdown:
            if not check.matches:
                diff = check.reported - check.verified
                parts.append(
                    f"{check.description}: {check.reported} reported, {check.verified} verified "
                    f"({abs(diff)} {'too many' if diff > 0 else 'missing'})"
                )
        if not parts:
            if self.breakdown:
                return "every reported intermediate count matches the verifier"
            return "no error"
        return "; ".join(parts)


@dataclass(frozen=True)
class Comparison:
    """A per-message condition next to its reference conditions (descriptive only)."""

    condition_id: str
    stacked: bool
    default_reference: str | None
    per_message_reference: str | None
    pattern: str
    text: str

    @property
    def title(self) -> str:
        ref = self.per_message_reference or self.default_reference or "?"
        return f"{ref} vs {self.condition_id}"


@dataclass(frozen=True)
class Evaluation:
    benchmark: Benchmark
    batches: tuple[str, ...]
    truth: Any
    runs: tuple[RunResult, ...]
    comparisons: tuple[Comparison, ...]

    @property
    def definition(self) -> BenchmarkDefinition:
        return self.benchmark.definition

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def full_successes(self) -> int:
        return sum(r.full_success for r in self.runs)

    @property
    def partials(self) -> int:
        return sum(r.partial for r in self.runs)

    @property
    def valid_order_answers(self) -> int:
        """Count valid permutations in the two order fields, independent of rank or repetition."""
        return sum(
            answer.value in self.truth.orders
            for r in self.runs
            for answer in r.answers
            if answer.field.check in {"lexicographic_first", "other_valid_order"}
        )

    def correct(self, check: str) -> tuple[int, int]:
        """(correct, runs) for one answer check."""
        results = [r.answer(check) for r in self.runs]
        return sum(bool(a and a.correct) for a in results), len(results)

    def of_condition(self, condition_id: str) -> list[RunResult]:
        return [r for r in self.runs if r.condition.id == condition_id]

    def runs_per_condition(self) -> str:
        """Runs per condition: "1", or "1 to 3" when conditions differ."""
        counts = sorted({len(self.of_condition(c.id)) for c in self.definition.conditions})
        if not counts:
            return "0"
        return str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"

    def data_sha256(self) -> str:
        lines = "".join(r.run.model_dump_json() + "\n" for r in self.runs)
        return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def evaluate(
    benchmark: Benchmark, batches: list[str] | tuple[str, ...] | None = None
) -> Evaluation:
    """Score every run of the given batches (all runs when none are given)."""
    definition = benchmark.definition
    if batches:
        unknown = [b for b in batches if b not in benchmark.batches]
        if unknown:
            raise StudyError(f"no runs in batch {', '.join(unknown)}")
        chosen = tuple(batches)
    else:
        chosen = tuple(benchmark.batches)
    verifier = VERIFIERS[definition.task.verifier]
    truth = verifier.ground_truth()
    order = {c.id: i for i, c in enumerate(definition.conditions)}
    results = []
    for run in benchmark.runs:
        if run.batch not in chosen:
            continue
        answers = check_answers(
            definition.task, verifier, truth, _answer_values(definition.task, run)
        )
        breakdown = tuple(
            BreakdownCheck(key, verifier.describe(key), value, verifier.breakdown(key))
            for key, value in run.reported_breakdown.items()
        )
        results.append(
            RunResult(run, definition.condition(run.condition_id), tuple(answers), breakdown)
        )
    results.sort(key=lambda r: (order[r.condition.id], r.run.repetition))
    runs = tuple(results)
    comparisons = tuple(compare_conditions(definition, runs, truth))
    return Evaluation(benchmark, chosen, truth, runs, comparisons)


# -- comparisons ----------------------------------------------------------------------------


def _outcome_text(results: list[RunResult]) -> str:
    n = len(results)
    k = sum(r.full_success for r in results)
    return f"{k} of {n} {'run' if n == 1 else 'runs'} fully correct"


def _counts_text(results: list[RunResult]) -> str:
    values = []
    for r in results:
        count = r.answer("count")
        values.append("—" if count is None or count.value is None else str(count.value))
    return ", ".join(values)


def compare_conditions(
    definition: BenchmarkDefinition, runs: tuple[RunResult, ...], truth: Any
) -> list[Comparison]:
    """Each per-message condition next to its reference conditions, as in a study's override
    assessment but descriptive: compare the observed proportions of fully correct runs."""
    by_condition: dict[str, list[RunResult]] = {}
    for r in runs:
        by_condition.setdefault(r.condition.id, []).append(r)
    success_rates = {
        cid: Fraction(sum(r.full_success for r in group), len(group))
        for cid, group in by_condition.items()
    }

    def describe(cid: str) -> str:
        cond = definition.condition(cid)
        default = definition.option_label(cond.model_selection)
        if cond.message_override is None:
            return f"{cid} (default {default}, no per-message selection)"
        return f"{cid} (default {default}, per-message {definition.option_label(cond.message_override)})"

    def wrong_counts(a: str, b: str) -> str:
        ra, rb = by_condition[a], by_condition[b]
        counts = (r.answer("count") for group in (ra, rb) for r in group)
        if any(count is None or not count.readable or count.correct for count in counts):
            return ""
        counts_a, counts_b = _counts_text(ra), _counts_text(rb)
        if len(ra) != 1 or len(rb) != 1:
            return (
                f" Every reported count was wrong. {a} gave {counts_a} and {b} gave {counts_b}. "
                f"The verified count is {truth.count}."
            )
        return (
            f" Both answered the count wrongly, with different values: {a} gave "
            f"{counts_a} and {b} gave {counts_b} (verified: {truth.count})."
            if counts_a != counts_b
            else f" Both gave the same wrong count ({counts_a})."
        )

    out: list[Comparison] = []
    for cond in definition.conditions:
        if cond.message_override is None:
            continue
        stacked = cond.model_selection == cond.message_override
        dref = _find_reference(definition, cond.model_selection)  # type: ignore[arg-type]
        oref = _find_reference(definition, cond.message_override)  # type: ignore[arg-type]
        involved = [c for c in (cond.id, dref, oref) if c is not None]
        if dref is None or oref is None or any(c not in by_condition for c in involved):
            missing = "a reference condition" if dref is None or oref is None else "runs"
            out.append(
                Comparison(
                    cond.id,
                    stacked,
                    dref,
                    oref,
                    "not_comparable",
                    f"{describe(cond.id)} cannot be compared: {missing} missing.",
                )
            )
            continue
        single = all(len(by_condition[c]) == 1 for c in involved)
        metric = "outcome" if single else "observed full-task success rate"
        n_note = (
            "With one run per condition this describes single responses; it is not a test "
            "and not evidence of which model produced them."
            if single
            else "This describes these runs only; it is not a test."
        )
        if stacked:
            hyp = ", ".join(definition.hypothesis_for(cond, "no_difference"))
            label = definition.option_label(cond.model_selection)
            if success_rates[cond.id] == success_rates[dref]:
                gain = "gain" if single else "gain in full-task success rate"
                text = (
                    f"{describe(cond.id)} and {describe(dref)} have the same {metric} "
                    f"({_outcome_text(by_condition[cond.id])}; {_outcome_text(by_condition[dref])}), "
                    f"with counts {_counts_text(by_condition[cond.id])} and "
                    f"{_counts_text(by_condition[dref])}. No {gain} from choosing {label} twice was "
                    "observed"
                    + (f", which is consistent with {hyp}" if hyp else "")
                    + ". Not seeing a difference does not show that the two settings are the "
                    "same: whether reasoning effort, reasoning tokens, or compute differed cannot "
                    f"be observed here. {n_note}"
                )
                pattern = "no_difference"
            else:
                text = (
                    f"{describe(cond.id)} and {describe(dref)} have different {metric}s "
                    f"({_outcome_text(by_condition[cond.id])}; {_outcome_text(by_condition[dref])})."
                    f" {n_note}"
                )
                pattern = "difference"
            out.append(Comparison(cond.id, True, dref, oref, pattern, text))
            continue

        refs = (
            f"the per-message reference {describe(oref)} had "
            f"{_outcome_text(by_condition[oref])}, the default reference {describe(dref)} had "
            f"{_outcome_text(by_condition[dref])}"
        )
        if success_rates[oref] == success_rates[dref]:
            pattern = "references_alike"
            text = (
                f"{describe(cond.id)} had {_outcome_text(by_condition[cond.id])}; {refs}. The two "
                f"references have the same {metric}, so this comparison cannot tell the patterns "
                f"apart. {n_note}"
            )
        elif success_rates[cond.id] == success_rates[oref]:
            pattern = "per_message"
            hyp = ", ".join(definition.hypothesis_for(cond, "per_message"))
            text = (
                f"{describe(cond.id)} has the same {metric} as {oref}: {_outcome_text(by_condition[cond.id])}"
                f"; {refs}. This is consistent with the per-message selection being applied"
                + (f" ({hyp})" if hyp else "")
                + f", but it is not direct evidence for it.{wrong_counts(cond.id, oref)} {n_note}"
            )
        elif success_rates[cond.id] == success_rates[dref]:
            pattern = "default"
            hyp = ", ".join(definition.hypothesis_for(cond, "default"))
            text = (
                f"{describe(cond.id)} has the same {metric} as {dref}: {_outcome_text(by_condition[cond.id])}"
                f"; {refs}. This is consistent with the default selection being applied"
                + (f" ({hyp})" if hyp else "")
                + f", but it is not direct evidence for it.{wrong_counts(cond.id, dref)} {n_note}"
            )
        else:
            pattern = "neither"
            text = (
                f"{describe(cond.id)} had {_outcome_text(by_condition[cond.id])}; {refs}. Its "
                f"{metric} matches neither reference. {n_note}"
            )
        out.append(Comparison(cond.id, False, dref, oref, pattern, text))
    return out


# -- tables ---------------------------------------------------------------------------------


def _mark(result: AnswerResult | None) -> str:
    if result is None:
        return "—"
    if not result.readable:
        return "Unreadable"
    return "Pass" if result.correct else "Fail"


def runs_frame(ev: Evaluation) -> pd.DataFrame:
    """One row per run: settings, answers given, and each check."""
    d = ev.definition
    rows = []
    for r in ev.runs:
        count = r.answer("count")
        first = r.answer("lexicographic_first")
        other = r.answer("other_valid_order")
        rows.append(
            {
                "run_id": r.run.run_id,
                "condition_id": r.condition.id,
                "default_selection": d.option_label(r.condition.model_selection),
                "per_message_selection": d.option_label(r.condition.message_override),
                "count_given": None if count is None else count.value,
                "first_order_given": None if first is None else first.value,
                "other_order_given": None if other is None else other.value,
                "exact_count": _mark(count),
                "lexicographic_first": _mark(first),
                "other_valid_order": _mark(other),
                "full_task_success": "Pass" if r.full_success else "Fail",
                "outcome": r.outcome,
                "error_type": r.error_type(ev.truth),
            }
        )
    return pd.DataFrame(rows)


def selection_grid(ev: Evaluation) -> pd.DataFrame:
    """Rows: default selection; columns: per-message selection; cells: condition and outcome."""
    d = ev.definition
    columns = ["none"] + [o.label for o in d.selection_options]
    rows = []
    for opt in d.selection_options:
        row: dict[str, str] = {"default selection": opt.label}
        for col in columns:
            row[col] = ""
        for cond in d.conditions:
            if cond.model_selection != opt.id:
                continue
            results = ev.of_condition(cond.id)
            if results:
                cell = f"{cond.id}: {sum(x.full_success for x in results)}/{len(results)} correct"
                cell += f" (count {_counts_text(results)})"
            else:
                cell = f"{cond.id}: not run"
            row[d.option_label(cond.message_override)] = cell
        rows.append(row)
    frame = pd.DataFrame(rows)
    used = [c for c in columns if (frame[c] != "").any()]
    return frame[["default selection", *used]]


def breakdown_frame(ev: Evaluation) -> pd.DataFrame:
    rows = [
        {
            "run_id": r.run.run_id,
            "condition_id": r.condition.id,
            "intermediate_count": c.description,
            "key": c.key,
            "reported": c.reported,
            "verified": c.verified,
            "matches": "yes" if c.matches else "no",
        }
        for r in ev.runs
        for c in r.breakdown
    ]
    return pd.DataFrame(rows)


# -- report ---------------------------------------------------------------------------------


def _share(k: int, n: int) -> str:
    return f"{k / n:.1%}".replace(".0%", "%")


def _pct(k: int, n: int) -> str:
    return f"{k} of {n} ({_share(k, n)})" if n else "—"


def _escape(text: str) -> str:
    """Experimenter text inside Markdown prose: keep it literal."""
    return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", " ".join(text.split()))


def _values(runs: list[BenchmarkRun], name: str) -> str:
    values = sorted({str(getattr(r, name)) for r in runs if getattr(r, name) is not None})
    missing = any(getattr(r, name) is None for r in runs)
    if not values:
        return "not recorded"
    return ", ".join(values) + (" (not recorded for some runs)" if missing else "")


@dataclass(frozen=True, slots=True)
class RecordingCoverage:
    """Availability of measurements in the runs selected for a report or dashboard."""

    full_responses: int
    timestamps: int
    run_orders: int
    latencies: int

    @classmethod
    def from_runs(cls, runs: Sequence[BenchmarkRun]) -> RecordingCoverage:
        return cls(
            sum(r.raw_response is not None for r in runs),
            sum(r.timestamp is not None for r in runs),
            sum(r.run_order is not None for r in runs),
            sum(r.latency_seconds is not None for r in runs),
        )


def render_report(ev: Evaluation, spec: ReportSpec) -> str:
    d = ev.definition
    task = d.task
    truth = ev.truth
    runs = [r.run for r in ev.runs]
    coverage = RecordingCoverage.from_runs(runs)
    has_repetitions = any(r.repetition > 1 for r in runs)
    dates = sorted({r.recorded_on.isoformat() for r in runs})
    n_cond = len({r.condition.id for r in ev.runs})
    count_ok, n = ev.correct("count")
    first_ok, _ = ev.correct("lexicographic_first")
    other_ok, _ = ev.correct("other_valid_order")
    failed = [r for r in ev.runs if not r.full_success]
    out: list[str] = [f"# {spec.title}", ""]

    only_counts = (
        bool(failed)
        and first_ok == other_ok == n
        and all((count := r.answer("count")) is not None and count.readable for r in failed)
    )
    abstract = (
        f"One constraint-reasoning task ({task.title.lower()}) was sent with the same prompt in "
        f"the {d.platform} under {n_cond} combination{'' if n_cond == 1 else 's'} of the default model selection and the "
        f"per-message selection (long-press on send). {_pct(ev.full_successes, n)} responses "
        "answered all three parts correctly."
    )
    if only_counts:
        miscounts = ", ".join(f"{r.condition.id}: {_counts_text([r])}" for r in failed)
        abstract += (
            f" The other {len(failed)} gave a correct first order and a valid second order but "
            f"miscounted the valid orders ({miscounts})."
        )
    abstract += (
        f" The ground truth ({truth.count} valid orders, the first being "
        f"{truth.lexicographic_first}) was computed independently by checking all 40,320 "
        "orders against the eight rules."
    )
    shorts = " ".join(s for s in (_short(c) for c in ev.comparisons) if s)
    if shorts:
        if n > n_cond:
            abstract += " The comparisons use the observed full-task success rate."
        abstract += " " + shorts
    abstract += (
        " The recorded answers cannot identify the model behind a response."
        if has_repetitions
        else " With so few runs, none of this is evidence about which model produced a response."
    )
    per_condition = ev.runs_per_condition()
    run_noun = "run" if n == 1 else "runs"
    condition_count = (
        f"{n_cond} of {len(d.conditions)} conditions represented"
        if n_cond < len(d.conditions)
        else f"{n_cond} {'condition' if n_cond == 1 else 'conditions'}"
    )
    distribution = (
        f"{per_condition} per defined condition"
        if n_cond < len(d.conditions)
        else f"{per_condition} run{'' if per_condition == '1' else 's'} each"
    )
    out += [
        f"> **Exploratory, preliminary results.** {n} {run_noun}: {condition_count}, "
        f"{distribution}, one task, recorded by hand on "
        f"{', '.join(dates)}. The figures describe these runs only. They are not a measure of "
        "any model's or setting's accuracy, and no statistical test was run.",
        "",
        "## Abstract",
        "",
        abstract,
        "",
        "## Research Question",
        "",
    ]
    out += [f"- {q}" for q in d.research_questions]
    out += ["", "## Hypotheses", ""]
    for h in d.hypotheses:
        pattern = (
            f" _(predicts the {h.predicted_pattern.replace('_', '-')} pattern)_"
            if h.predicted_pattern
            else ""
        )
        out.append(f"- **{h.id}**{pattern}: {h.statement}")
        out += [f"  - {p}" for p in h.predictions]
    out += [
        "",
        "The comparisons describe the recorded outcomes. No hypothesis test was performed."
        if has_repetitions
        else "The hypotheses are stated, not tested: with this many runs the report only "
        "describes which pattern the outcomes match.",
        "",
        "## Experimental Setup",
        "",
    ]
    chat_counts = Counter(r.new_conversation for r in runs)
    if chat_counts[True] == n:
        chats = "a new chat for every run"
    elif chat_counts[False] == n:
        chats = "ongoing chat for every run"
    elif chat_counts[None] == n:
        chats = "not recorded"
    else:
        chats = (
            f"new {chat_counts[True]}, ongoing {chat_counts[False]}, unrecorded {chat_counts[None]}"
        )
    setup = [
        ["Service", d.service],
        ["Client", d.platform],
        ["Execution", "by hand: the experimenter operated the app and recorded what it returned"],
        ["Recorded on", ", ".join(dates)],
        ["Batches", ", ".join(ev.batches)],
        ["Conversation", chats],
        ["Prompt", f"the same text for every run (SHA-256 `{task.prompt_sha256()[:16]}…`)"],
        ["Device", _values(runs, "device_type")],
        ["Device model", _values(runs, "device_model")],
        ["Operating system", _values(runs, "operating_system")],
        ["App version", _values(runs, "app_version")],
        [
            "Run order and times",
            "not recorded"
            if not coverage.run_orders and not coverage.timestamps
            else f"run order {coverage.run_orders}/{n}, send time {coverage.timestamps}/{n} (see {RUNS_FILE})",
        ],
        [
            "Response times",
            "not recorded"
            if not coverage.latencies
            else f"recorded for {coverage.latencies} of {n} {run_noun}",
        ],
        ["Model label shown by the app", _values(runs, "observed_model_label")],
        [
            "Full response texts",
            "not kept: the runs hold the experimenter's structured summaries"
            if not coverage.full_responses
            else f"kept for {coverage.full_responses} of {n} {run_noun}",
        ],
    ]
    out += [md_table(["Item", "Value"], setup), "", "Conditions:", ""]
    cond_rows = []
    for cond in d.conditions:
        cond_rows.append(
            [
                cond.id,
                d.option_label(cond.model_selection),
                d.option_label(cond.message_override)
                if cond.message_override
                else "none (normal send)",
                cond.role,
                len(ev.of_condition(cond.id)),
            ]
        )
    out += [
        md_table(
            ["Condition", "Default selection", "Per-message selection", "Role", "Runs"], cond_rows
        ),
        "",
        "Selections are the labels the experimenter chose in the app. They name a setting, not "
        "the model that produced a response.",
        "",
    ]
    selected_batches = set(ev.batches)
    notes_apply = d.notes_batches is not None and selected_batches.issubset(d.notes_batches)
    notes_source = (
        f"the benchmark definition for {', '.join(d.notes_batches)}"
        if d.notes_batches
        else "the benchmark definition (batch unspecified)"
    )
    if d.not_recorded and (
        d.notes_batches is None or not selected_batches.isdisjoint(d.notes_batches)
    ):
        heading = "Not recorded:" if notes_apply else f"Recording gaps listed in {notes_source}"
        out += [heading, ""] + [f"- {item}" for item in d.not_recorded] + [""]

    out += ["## Benchmark Task", ""]
    meta = [f"**{task.title}**"]
    if task.category:
        meta.append(f"category: {task.category}")
    if task.difficulty:
        meta.append(f"difficulty: {task.difficulty} (the author's judgement, not calibrated)")
    if task.language:
        meta.append(f"language: {task.language}")
    out += [
        "; ".join(meta) + ".",
        "",
        "The prompt, verbatim:",
        "",
        "```text",
        task.prompt,
        "```",
        "",
    ]
    if task.prompt_source:
        out += [f"_{task.prompt_source}_", ""]
    out += [
        md_table(
            ["Answer", "Answer line", "Checked as"],
            [[a.id, a.label, CHECK_TEXT[a.check]] for a in task.answers],
        ),
        "",
        "Lines the prompt asks for beyond these (an explanation, for instance) are not scored: a "
        "response is judged by its answers against the verified ground truth, not by how "
        "convincing or long its explanation is.",
        "",
        "## Ground Truth",
        "",
        f"Computed by `mblab.{VERIFIERS[task.verifier].__name__.split('.')[-1]}`, which checks "
        "every one of the 8! = 40,320 orders against the eight rules, one rule at a time. It uses "
        "no response and no value from the run data; the reference answers in "
        f"`{BENCHMARK_FILE}` are checked against it.",
        "",
    ]
    first = truth.prefix_counts(1)
    two = truth.prefix_counts(2)
    out += [
        md_table(
            ["Quantity", "Value"],
            [
                ["Valid orders", truth.count],
                ["Lexicographically first valid order", truth.lexicographic_first],
                ["Valid orders by first task", ", ".join(f"{k}: {v}" for k, v in first.items())],
                ["Valid orders by first two tasks", ", ".join(f"{k}: {v}" for k, v in two.items())],
            ],
        ),
        "",
        "All valid orders: " + ", ".join(f"`{o}`" for o in truth.orders) + ".",
        "",
    ]
    implied = getattr(VERIFIERS[task.verifier], "redundant_rules", lambda: ())()
    if implied:
        numbers = ", ".join(str(n) for n in implied)
        out += [
            f"Implied by the other rules: rule {numbers}. Dropping it leaves the same "
            f"{truth.count} valid orders, so it does not change the answer.",
            "",
        ]
    if coverage.full_responses == 0:
        observations = (
            "What each response gave, as recorded by the experimenter. The full response texts "
            "were not kept"
        )
        observations += (
            "; the notes under the table are the experimenter's summaries, verbatim."
            if any(r.solution_summary for r in runs)
            else "."
        )
    else:
        observations = (
            f"Answers were read from the full response text in `{RUNS_FILE}` for "
            f"{coverage.full_responses} of {n} {run_noun}."
        )
        if coverage.full_responses < n:
            observations += " The remaining answers were read from the recorded answer fields."
        if any(r.solution_summary for r in runs):
            observations += " The notes under the table are the experimenter's summaries, verbatim."
    out += ["## Observations", "", observations, ""]
    frame = runs_frame(ev)
    out += [
        md_table(
            ["Run", "Condition", "Default", "Per-message", "Count", "First order", "Second order"],
            [
                [
                    row.run_id,
                    row.condition_id,
                    row.default_selection,
                    row.per_message_selection,
                    _fmt(row.count_given),
                    row.first_order_given,
                    row.other_order_given,
                ]
                for row in frame.itertuples()
            ],
        ),
        "",
    ]
    for r in ev.runs:
        if r.run.solution_summary:
            out.append(f"**{r.run.run_id}** ({r.condition.id}):")
            out.append("")
            out += [f"- {_escape(line)}" for line in r.run.solution_summary]
            out.append("")

    out += ["## Results", ""]
    out += [
        md_table(
            [
                "Run",
                "Condition",
                "Exact count",
                "Lexicographic first",
                "Second valid order",
                "Full task success",
            ],
            [
                [
                    row.run_id,
                    row.condition_id,
                    f"{row.exact_count} ({_fmt(row.count_given)})",
                    row.lexicographic_first,
                    row.other_valid_order,
                    row.full_task_success,
                ]
                for row in frame.itertuples()
            ],
        ),
        "",
        md_table(
            ["Measure", "Result"],
            [
                ["Runs", n],
                ["Full task success (all three answers correct)", _pct(ev.full_successes, n)],
                ["Partially correct (some answers correct)", _pct(ev.partials, n)],
                ["Exact count", _pct(count_ok, n)],
                ["Lexicographically first order", _pct(first_ok, n)],
                ["Second valid order", _pct(other_ok, n)],
                ["Valid orders given (both order answers)", _pct(ev.valid_order_answers, 2 * n)],
            ],
        ),
        "",
        f"{_share(ev.full_successes, n)} describes {n} {run_noun} of one task under "
        f"{'1 setting' if n_cond == 1 else f'{n_cond} different settings'}. "
        "It is not the accuracy of any model or setting, and the runs are not a "
        "sample from one population.",
        "",
        "### Comparisons",
        "",
    ]
    for i, comp in enumerate(ev.comparisons, start=1):
        involved = [c for c in (comp.per_message_reference, comp.condition_id) if c]
        if comp.default_reference and comp.default_reference not in involved:
            involved.append(comp.default_reference)
        out += [f"**Comparison {i}: {comp.title}**", ""]
        out += [
            md_table(
                ["Condition", "Default", "Per-message", "Count", "Full task success"],
                [
                    [
                        cid,
                        d.option_label(d.condition(cid).model_selection),
                        d.option_label(d.condition(cid).message_override),
                        _counts_text(ev.of_condition(cid)),
                        _outcome_text(ev.of_condition(cid)),
                    ]
                    for cid in involved
                ],
            ),
            "",
            comp.text,
            "",
        ]

    out += [
        "## Error Analysis",
        "",
        "Where each wrong answer came from, traced by comparing the intermediate counts in the "
        "experimenter's summary with the verifier's values."
        if only_counts and all(any(not c.matches for c in r.breakdown) for r in failed)
        else "The table lists each answer that failed its check and any recorded intermediate "
        "counts that differ from the verifier.",
        "",
        md_table(
            ["Run", "Condition", "Error type", "Detail"],
            [
                [r.run.run_id, r.condition.id, r.error_type(truth), r.error_detail(truth)]
                for r in ev.runs
            ],
        ),
        "",
    ]
    checks = breakdown_frame(ev)
    if not checks.empty:
        out += [
            md_table(
                ["Run", "Intermediate count", "Reported", "Verified", "Matches"],
                [
                    [row.run_id, row.intermediate_count, row.reported, row.verified, row.matches]
                    for row in checks.itertuples()
                ],
            ),
            "",
        ]

    out += ["## Limitations", ""]
    condition_basis = "defined condition" if n_cond < len(d.conditions) else "condition"
    if has_repetitions:
        out.append(
            f"- The analysis covers {n} {run_noun} of one task. It reports no intervals or "
            "hypothesis tests and does not assess generalisation to other tasks, prompts, or days."
        )
    else:
        out.append(
            f"- {n} {run_noun}, {per_condition} per {condition_basis}, one task: no interval, test, "
            "or generalisation to other tasks, prompts, or days is possible."
        )
    if not coverage.full_responses:
        out.append(
            "- The answers come from the experimenter's structured summaries; the response texts "
            "were not kept and cannot be checked again."
        )
    elif coverage.full_responses < n:
        out.append(
            f"- Full response texts are missing for {n - coverage.full_responses} of {n} runs. "
            "Only their recorded answer fields can be checked."
        )
    out.append(
        "- Selections are what the experimenter chose in the app. The model or reasoning effort "
        "behind a response is not observable."
    )
    if not coverage.timestamps and not coverage.run_orders:
        out.append(
            "- Run order and times were not recorded, so a change in the service during the "
            "session cannot be ruled out."
        )
    else:
        for measurement, count in (
            ("Send time", coverage.timestamps),
            ("Run order", coverage.run_orders),
        ):
            if count < n:
                out.append(f"- {measurement} is missing for {n - count} of {n} {run_noun}.")
    if not d.documented_facts:
        out.append(
            "- No documented facts are recorded: nothing in this report relies on product "
            "documentation, and no product description is presented as observed here."
        )
    if d.not_observable:
        out += ["", "Not observable in this experiment:", ""]
        out += [f"- {item}" for item in d.not_observable]
    out += ["", "## Next Experiments", ""]
    if d.planned:
        heading = (
            "Planned, not yet run (no results exist for them):"
            if notes_apply
            else f"Plans listed in {notes_source}"
        )
        out += [heading, ""]
        out += [f"- {item}" for item in d.planned]
        out.append("")
    directory = f"{BENCHMARKS_DIR}/{ev.benchmark.root.name}"
    show = ["mblab", "benchmark", "show", directory]
    check = ["mblab", "benchmark", "report", directory, "--check"]
    if spec.reproduction_scope != "benchmark":
        for batch in ev.batches:
            show.extend(("--batch", batch))
        check.extend(("--report", spec.id))
    out += [
        f"Further runs are added as new lines of `{RUNS_FILE}` with the next repetition number "
        "and a new batch name; existing lines are not changed. A new report then lists the "
        f"batches it covers in `{BENCHMARK_FILE}`, and this report stays as it is.",
        "",
        "## Reproduction",
        "",
        md_table(
            ["Item", "Value"],
            [
                ["Runs in this report", f"{n} (batches: {', '.join(ev.batches)})"],
                ["SHA-256 of these runs", f"`{ev.data_sha256()}`"],
                ["SHA-256 of the prompt", f"`{task.prompt_sha256()}`"],
            ],
        ),
        "",
        "```bash",
        shlex.join(show),
        shlex.join(check),
        "```",
        "",
        "The first command recomputes the ground truth and scores every run; the second makes "
        "this report again and compares it with this file without writing anything."
        if spec.reproduction_scope == "benchmark"
        else "The first command scores the selected batches. The second checks this report.",
        "",
    ]
    return "\n".join(out)


def _short(comp: Comparison) -> str:
    return {
        "per_message": f"{comp.condition_id} matched its per-message reference "
        f"{comp.per_message_reference}.",
        "default": f"{comp.condition_id} matched its default reference {comp.default_reference}.",
        "no_difference": f"{comp.condition_id} showed no difference from {comp.default_reference}.",
        "difference": f"{comp.condition_id} differed from {comp.default_reference}.",
    }.get(comp.pattern, "")


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def report_path(benchmark: Benchmark, spec: ReportSpec) -> Path:
    return benchmark.root / spec.path


def make_reports(
    benchmark: Benchmark, report_id: str | None = None
) -> list[tuple[ReportSpec, str]]:
    """Render the named report, or all declared reports, from their respective batches."""
    specs = (
        benchmark.definition.reports if report_id is None else [benchmark.report_spec(report_id)]
    )
    return [(spec, render_report(evaluate(benchmark, spec.batches), spec)) for spec in specs]


def saved_report_text(path: Path) -> str | None:
    """Read an existing UTF-8 report. A decoding failure is a content mismatch."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def write_reports(
    benchmark: Benchmark, report_id: str | None = None, *, force: bool = False
) -> list[Path]:
    """Check every selected archive for conflicts before updating any report."""
    with write_lock(benchmark.root, "benchmark report"):
        reports = make_reports(benchmark, report_id)
        pending: list[tuple[Path, str]] = []
        for spec, text in reports:
            path = report_path(benchmark, spec)
            if path.exists():
                if not path.is_file():
                    raise StudyError(f"{path} is not a report file")
                if saved_report_text(path) == text:
                    continue
                if not force:
                    raise StudyError(
                        f"{path} already holds a different report. Inspect it with "
                        f"--report {spec.id} --check or replace it with --report {spec.id} --force."
                    )
            pending.append((path, text))
        for path, text in pending:
            atomic_write_text(path, text)
    return [report_path(benchmark, spec) for spec, _ in reports]


def dump_run(run: BenchmarkRun) -> str:
    """One runs.jsonl line (null values included, so 'not recorded' stays visible)."""
    return json.dumps(run.model_dump(mode="json"), ensure_ascii=False)
