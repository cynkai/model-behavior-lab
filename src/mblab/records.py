"""Helpers for building observation records from a plan entry."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from mblab.schema import ExperimentDefinition, PlannedRun, RunPlan, RunRecord

_DURATION_PART = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(hours?|hrs?|h|시간|minutes?|mins?|m|분|seconds?|secs?|s|초)(?![a-z])",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"h": 3600, "시": 3600, "m": 60, "분": 60, "s": 1, "초": 1}


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
        found = True
        total += float(number.replace(",", ".")) * _UNIT_SECONDS[unit[0].lower()]
    return total if found else None


def plan_fields(run: PlannedRun, plan: RunPlan | None) -> dict[str, Any]:
    """Fields a record copies from its plan entry."""
    return {
        "run_id": run.run_id,
        "experiment_id": run.experiment_id,
        "experiment_version": run.experiment_version,
        "definition_sha256": plan.definition_sha256 if plan else None,
        "platform": run.platform,
        "condition_id": run.condition_id,
        "model_selection": run.model_selection,
        "message_override": run.message_override,
        "task_id": run.task_id,
        "repetition": run.repetition,
        "execution_method": run.execution_method,
    }


def build_record(
    run: PlannedRun,
    definition: ExperimentDefinition,
    plan: RunPlan | None,
    **observed: Any,
) -> RunRecord:
    """Combine a plan entry with observed fields and validate the result."""
    fields = plan_fields(run, plan)
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
