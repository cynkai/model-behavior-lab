"""Compatibility and integrity gates for descriptive manual-study comparisons."""

from __future__ import annotations

from mblab.plan import generate_plan
from mblab.study import Study, StudyError
from mblab.validate import ValidationReport, validate_study


def _check_study(study: Study, side: str) -> ValidationReport:
    """Validate one study without exposing its contents in comparison errors."""
    definition = study.definition
    if definition.execution_method != "manual_ui":
        raise StudyError(f"{side} study: comparison requires manual_ui execution.")
    if definition.pilot:
        raise StudyError(f"{side} study: pilot studies cannot be compared.")
    plan = study.plan
    if plan is None:
        raise StudyError(f"{side} study: a registered plan is required.")
    if (
        plan.definition_sha256 != definition.sha256()
        or plan.definition_snapshot != definition.model_dump(mode="json")
    ):
        raise StudyError(f"{side} study: the definition differs from its registered plan.")
    if (
        plan.experiment_id != definition.experiment_id
        or plan.experiment_version != definition.experiment_version
        or plan.strategy != definition.randomization.strategy
    ):
        raise StudyError(f"{side} study: plan identity or strategy differs from the definition.")

    expected = generate_plan(definition, seed=plan.seed, now=plan.generated_at)
    expected_slots = {(run.condition_id, run.task_id, run.repetition): run for run in expected.runs}
    slots = [(run.condition_id, run.task_id, run.repetition) for run in plan.runs]
    if len(slots) != len(expected_slots) or set(slots) != set(expected_slots):
        raise StudyError(f"{side} study: plan does not contain the complete registered run matrix.")
    for slot, run in zip(slots, plan.runs, strict=True):
        # A hand-planned order and arbitrary unique run IDs do not change a cell's identity.
        ignored = {"run_id", "sequence"}
        if run.model_dump(exclude=ignored) != expected_slots[slot].model_dump(exclude=ignored):
            raise StudyError(f"{side} study: a planned run differs from its registered condition.")

    report = validate_study(study)
    if report.errors:
        codes = ", ".join(sorted({issue.code for issue in report.errors}))
        raise StudyError(f"{side} study: validation errors ({codes}); run mblab validate locally.")
    for record in study.records:
        if record.excluded:
            continue
        if (
            record.definition_sha256 is not None
            and record.definition_sha256 != plan.definition_sha256
        ) or record.experiment_version != plan.experiment_version:
            raise StudyError(
                f"{side} study: an included record differs from the registered definition."
            )
        if record.prompt_text != definition.task(record.task_id).prompt:
            raise StudyError(
                f"{side} study: an included record differs from the registered prompt."
            )
    return report


def check_comparable(left: Study, right: Study) -> tuple[ValidationReport, ValidationReport]:
    """Reject incompatible protocols; return reports for safe warning/count summaries.

    Identity, order, repetitions, platforms, and descriptive metadata can differ
    between studies. Tasks and selected options must retain the same meaning.
    Raw validation messages can contain private data; callers should expose codes
    and counts rather than copying those messages into comparison artifacts.
    """
    left_report = _check_study(left, "Left")
    right_report = _check_study(right, "Right")
    if left.definition.service != right.definition.service:
        raise StudyError("Left/Right studies: service must match exactly.")
    if left.is_synthetic() != right.is_synthetic():
        raise StudyError("Left/Right studies: observed and synthetic data cannot be compared.")

    left_tasks = {task.id: task for task in left.definition.tasks}
    right_tasks = {task.id: task for task in right.definition.tasks}
    if left_tasks.keys() != right_tasks.keys():
        raise StudyError("Left/Right studies: task IDs must match.")
    for task_id, task in left_tasks.items():
        other = right_tasks[task_id]
        if (
            task.prompt != other.prompt
            or task.scoring != other.scoring
            or task.reference_answer != other.reference_answer
        ):
            raise StudyError(
                "Left/Right studies: task prompts, scoring, and reference answers must match."
            )

    left_conditions = {condition.id: condition for condition in left.definition.conditions}
    right_conditions = {condition.id: condition for condition in right.definition.conditions}
    if left_conditions.keys() != right_conditions.keys():
        raise StudyError("Left/Right studies: condition IDs must match.")
    for condition_id, condition in left_conditions.items():
        other = right_conditions[condition_id]
        if (
            condition.model_selection != other.model_selection
            or condition.message_override != other.message_override
        ):
            raise StudyError(
                "Left/Right studies: condition selections and per-message overrides must match."
            )
        for option_id in (condition.model_selection, condition.message_override):
            if option_id is not None and (
                left.definition.option(option_id).label != right.definition.option(option_id).label
            ):
                raise StudyError("Left/Right studies: selected option labels must match exactly.")
    return left_report, right_report
