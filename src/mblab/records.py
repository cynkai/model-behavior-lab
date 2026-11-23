"""Helpers for building observation records from a plan entry."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mblab.schema import ExperimentDefinition, PlannedRun, RunPlan, RunRecord

if TYPE_CHECKING:
    from mblab.study import Study

_DURATION_PART = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(hours?|hrs?|h|시간|minutes?|mins?|m|분|seconds?|secs?|s|초)(?![a-z])",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"h": 3600, "시": 3600, "m": 60, "분": 60, "s": 1, "초": 1}
#: "1,050" could be 1050 (thousands separator) or 1.050 (decimal comma).
_GROUPED_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")


def parse_decimal(text: str, what: str) -> float:
    """A number typed by hand: "12.5" or "12,5" (decimal comma).

    A value such as "1,050" is refused: read as a decimal comma it is 1.05, read as a
    thousands separator 1050, and guessing would silently be off by a factor of 1000.
    """
    value = text.strip()
    if _GROUPED_THOUSANDS.fullmatch(value):
        raise ValueError(
            f"{what} {text!r} is ambiguous (1050 or 1.05?); write it without a thousands "
            "separator, e.g. 1050 or 1.05"
        )
    try:
        return float(value.replace(",", "."))
    except ValueError:
        raise ValueError(f"{what} {text!r} is not a number") from None


def parse_thinking_seconds(text: str | None) -> float | None:
    """Parse a client-displayed duration such as "Thought for 1m 12s" or "1분 12초".

    Returns ``None`` when no explicit duration is present (e.g. "Thought for a
    few seconds"); the verbatim text is always kept alongside the number.
    """
    if not text:
        return None
    total = 0.0
    found = False
    for number, unit in _DURATION_PART.findall(text):
        if _GROUPED_THOUSANDS.fullmatch(number):
            return None  # "1,050 seconds": ambiguous, so no number (the text is kept)
        found = True
        total += float(number.replace(",", ".")) * _UNIT_SECONDS[unit[0].lower()]
    return total if found else None


#: Fields that say which planned run a record belongs to. A record must agree with its
#: plan entry on all of them, or it describes a different run (another study, a pilot).
PLAN_IDENTITY_FIELDS = (
    "experiment_id",
    "platform",
    "execution_method",
    "condition_id",
    "model_selection",
    "message_override",
    "task_id",
    "repetition",
)


def identity_conflicts(record: RunRecord, run: PlannedRun) -> list[str]:
    """The identity fields in which ``record`` differs from its plan entry."""
    return [name for name in PLAN_IDENTITY_FIELDS if getattr(record, name) != getattr(run, name)]


def plan_fields(run: PlannedRun) -> dict[str, Any]:
    """Fields a record copies from its plan entry."""
    return {
        "run_id": run.run_id,
        "experiment_version": run.experiment_version,
        **{name: getattr(run, name) for name in PLAN_IDENTITY_FIELDS},
    }


def build_record(
    run: PlannedRun,
    definition: ExperimentDefinition,
    plan: RunPlan | None,
    **observed: Any,
) -> RunRecord:
    """Combine a plan entry with observed fields and validate the result.

    ``definition_sha256`` is the hash of ``definition``, the experiment.yaml in force
    when the run was recorded, so a change made after planning shows in each record
    collected under it (validation compares it with the hash in plan.json).
    """
    fields = plan_fields(run)
    fields["definition_sha256"] = definition.sha256()
    fields["prompt_text"] = definition.task(run.task_id).prompt
    fields["recorded_at"] = datetime.now().astimezone()
    fields.update(observed)
    if fields.get("thinking_text") and fields.get("thinking_seconds") is None:
        fields["thinking_seconds"] = parse_thinking_seconds(fields["thinking_text"])
    return RunRecord.model_validate(fields)


def next_pending(plan: RunPlan, recorded: set[str]) -> PlannedRun | None:
    for run in sorted(plan.runs, key=lambda r: r.sequence):
        if run.run_id not in recorded:
            return run
    return None


def condition_instructions(definition: ExperimentDefinition, run: PlannedRun) -> list[str]:
    """Plain-language steps for setting up the client for this run."""
    if definition.execution_method == "official_api" and definition.api is not None:
        effort = definition.option(run.model_selection).reasoning_effort
        return [
            f"Sent by `mblab run-api` as one independent request: model {definition.api.model}, "
            f"reasoning.effort={effort}, store=false.",
            "Preview it without sending anything: mblab run-api --dry-run",
        ]
    default_label = definition.option_label(run.model_selection)
    steps = [
        "Start a NEW conversation.",
        f"Set the default model selection to: {default_label}",
    ]
    if run.message_override:
        steps.append(
            f"Choose the per-message selection (long-press send): "
            f"{definition.option_label(run.message_override)}"
        )
    else:
        steps.append("Do NOT use a per-message selection (plain tap on send).")
    steps.append("Paste the prompt exactly as shown, start the stopwatch, and send.")
    return steps


def unplaced_records(study: Study) -> dict[str, list[str]]:
    """Records analysis cannot place, by reason, as run ids in file order.

    A record is left out of every number if its run is not planned (or its task is no
    longer defined), if it disagrees with its plan entry on the experiment, condition,
    selections, task, or repetition (a record from another study or a pilot, or an
    edited condition), or if the run has more than one record (no copy is preferred
    over another). `mblab validate` reports each of these as an error.
    """
    plan_runs = study.plan.by_run_id() if study.plan else {}
    task_ids = {t.id for t in study.definition.tasks}
    copies = Counter(r.run_id for r in study.records)
    out: dict[str, list[str]] = {}
    for rec in study.records:
        run = plan_runs.get(rec.run_id)
        if run is None or rec.task_id not in task_ids:
            reason = "run not planned or task not defined"
        elif copies[rec.run_id] > 1:
            reason = "several records for one run"
        elif identity_conflicts(rec, run):
            reason = "identity differs from the plan"
        else:
            continue
        out.setdefault(reason, []).append(rec.run_id)
    return out


def included_completions(study: Study, left_out: set[str] | None = None) -> Iterator[RunRecord]:
    """Yield completed, protocol-adherent records that can be placed in the study plan."""
    if left_out is None:
        left_out = {run_id for ids in unplaced_records(study).values() for run_id in ids}
    for record in study.records:
        if (
            record.run_id not in left_out
            and not record.excluded
            and record.response_status == "completed"
        ):
            yield record
