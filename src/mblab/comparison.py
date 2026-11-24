"""Separate descriptive summaries for two compatible study collections."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import TypedDict

from mblab.analysis import AnalysisSettings, build_results, summarize_conditions
from mblab.comparison_protocol import check_comparable
from mblab.study import Study
from mblab.validate import ValidationReport


class Counts(TypedDict):
    planned_runs: int
    recorded_runs: int
    excluded_runs: int
    pending_manual: int
    scored_runs: int
    scored_tasks: int
    task_scored_runs: dict[str, int]
    correct_runs: int
    accuracy: float | None
    success_denominator: int
    success_rate: float | None


class StudySummary(TypedDict):
    experiment_id: str
    experiment_version: str
    definition_sha256: str
    service: str
    platform: str
    first_recorded_at: str | None
    last_recorded_at: str | None
    client_versions: dict[str, int]
    warnings: dict[str, int]
    planned_runs: int
    recorded_runs: int


class ConditionComparison(TypedDict):
    condition_id: str
    model_selection: str
    message_override: str | None
    left: Counts
    right: Counts


class StudyComparison(TypedDict):
    synthetic_data: bool
    total_tasks: int
    left: StudySummary
    right: StudySummary
    conditions: list[ConditionComparison]


def _summarize(
    study: Study, validation: ValidationReport
) -> tuple[StudySummary, dict[str, Counts]]:
    """Reuse analysis denominators without running comparisons or bootstraps."""
    definition = study.definition
    results = build_results(study)
    summaries = summarize_conditions(
        definition,
        study.plan,
        results,
        AnalysisSettings.preregistered(definition, study.plan),
        latency_method=None,
    )
    included = results[~results["excluded"]]
    scored = included[included["score_status"].isin(["scored", "format_failure"])]
    counts: dict[str, Counts] = {}
    for row in summaries.itertuples(index=False):
        condition_id = str(row.condition_id)
        task_counts = Counter(scored.loc[scored["condition_id"] == condition_id, "task_id"])
        n_scored = int(row.n_scored)
        n_correct = int(row.n_correct)
        n_pending = int(row.n_pending_manual)
        denominator = int(row.n_included) - n_pending
        counts[condition_id] = {
            "planned_runs": int(row.n_planned),
            "recorded_runs": int(row.n_recorded),
            "excluded_runs": int(row.n_excluded),
            "pending_manual": n_pending,
            "scored_runs": n_scored,
            "scored_tasks": len(task_counts),
            "task_scored_runs": {task.id: task_counts[task.id] for task in definition.tasks},
            "correct_runs": n_correct,
            "accuracy": n_correct / n_scored if n_scored else None,
            "success_denominator": denominator,
            "success_rate": n_correct / denominator if denominator else None,
        }
    timestamps = [datetime.fromisoformat(value).astimezone(UTC) for value in included["timestamp"]]
    summary: StudySummary = {
        "experiment_id": definition.experiment_id,
        "experiment_version": definition.experiment_version,
        "definition_sha256": definition.sha256(),
        "service": definition.service,
        "platform": definition.platform,
        "first_recorded_at": min(timestamps).isoformat() if timestamps else None,
        "last_recorded_at": max(timestamps).isoformat() if timestamps else None,
        "client_versions": dict(sorted(Counter(included["client_version"]).items())),
        "warnings": dict(sorted(Counter(issue.code for issue in validation.warnings).items())),
        "planned_runs": sum(item["planned_runs"] for item in counts.values()),
        "recorded_runs": sum(item["recorded_runs"] for item in counts.values()),
    }
    return summary, counts


def compare_studies(left: Study, right: Study) -> StudyComparison:
    """Validate compatibility and summarize each collection independently.

    Accuracy includes format failures, while task success also includes recorded
    refusals, errors and incomplete responses. Missing runs never become failures;
    excluded records and pending manual grades are absent from both denominators.
    """
    left_validation, right_validation = check_comparable(left, right)
    left_summary, left_counts = _summarize(left, left_validation)
    right_summary, right_counts = _summarize(right, right_validation)
    definition = left.definition
    conditions: list[ConditionComparison] = [
        {
            "condition_id": condition.id,
            "model_selection": definition.option_label(condition.model_selection),
            "message_override": (
                definition.option_label(condition.message_override)
                if condition.message_override is not None
                else None
            ),
            "left": left_counts[condition.id],
            "right": right_counts[condition.id],
        }
        for condition in definition.conditions
    ]
    return {
        "synthetic_data": left.is_synthetic() or right.is_synthetic(),
        "total_tasks": len(definition.tasks),
        "left": left_summary,
        "right": right_summary,
        "conditions": conditions,
    }
