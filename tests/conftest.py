from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mblab.plan import generate_plan
from mblab.records import build_record
from mblab.schema import ExperimentDefinition, RunPlan, RunRecord
from mblab.study import parse_definition, save_plan, template_text, write_jsonl

FIXED_NOW = datetime(2026, 11, 19, 10, 0, tzinfo=timezone(timedelta(hours=9)))


@pytest.fixture
def definition() -> ExperimentDefinition:
    return parse_definition(template_text())


@pytest.fixture
def plan(definition: ExperimentDefinition) -> RunPlan:
    return generate_plan(definition, now=FIXED_NOW)


def correct_response(definition: ExperimentDefinition, task_id: str) -> str:
    task = definition.task(task_id)
    return f"Working...\nANSWER: {task.reference_answer}"


def make_record(
    definition: ExperimentDefinition, plan: RunPlan, run_id: str, **overrides
) -> RunRecord:
    run = plan.by_run_id()[run_id]
    fields = {
        "client_version": "1.2026.300",
        "timestamp": FIXED_NOW + timedelta(minutes=run.sequence),
        "response_status": "completed",
        "response_text": correct_response(definition, run.task_id),
        "latency_seconds": 10.0,
        "latency_method": "manual_stopwatch",
        "confounders": {
            "memory": "disabled",
            "custom_instructions": "inactive",
            "tools_used": "none",
            "switch_notice": "none",
            "network": "wifi",
        },
        "recorded_at": FIXED_NOW,
    }
    fields.update(overrides)
    return build_record(run, definition, plan, **fields)


@pytest.fixture
def study_dir(tmp_path: Path, definition: ExperimentDefinition, plan: RunPlan) -> Path:
    """A study directory with a definition and a plan but no records."""
    root = tmp_path / "study"
    root.mkdir()
    (root / "experiment.yaml").write_text(template_text(), encoding="utf-8")
    save_plan(plan, root / "plan.json")
    return root


def write_records(root: Path, records: list[RunRecord]) -> None:
    write_jsonl(root / "records.jsonl", records)
