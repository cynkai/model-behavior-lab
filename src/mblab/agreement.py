"""Descriptive within-response agreement for usable manual grades."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from mblab.records import included_completions
from mblab.schema import ManualGrade, ManualScoring, RunRecord
from mblab.scoring import usable_manual_grades
from mblab.study import Study


@dataclass(frozen=True, slots=True)
class TaskAgreement:
    task_id: str
    eligible_runs: int
    graded_runs: int
    multi_rated_runs: int
    pairs: int
    exact_matches: int
    decision_matches: int


def summarize_agreement(study: Study, left_out: set[str] | None = None) -> list[TaskAgreement]:
    """Count unordered grader pairs within eligible responses, separately by task."""
    rules = {
        task.id: task.scoring
        for task in study.definition.tasks
        if isinstance(task.scoring, ManualScoring)
    }
    if not rules:
        return []
    grades_by_run: dict[str, list[ManualGrade]] = defaultdict(list)
    for grade in study.grades:
        grades_by_run[grade.run_id].append(grade)
    records_by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for record in included_completions(study, left_out):
        if record.task_id in rules:
            records_by_task[record.task_id].append(record)

    summaries = []
    for task_id, rule in rules.items():
        records = records_by_task[task_id]
        graded_runs = multi_rated_runs = pairs = exact_matches = decision_matches = 0
        for record in records:
            grades = usable_manual_grades(record.run_id, rule, grades_by_run[record.run_id])
            count = len(grades)
            graded_runs += count > 0
            multi_rated_runs += count > 1
            pairs += count * (count - 1) // 2
            points = Counter(grade.points for grade in grades)
            decisions = Counter(grade.points >= rule.pass_threshold for grade in grades)
            exact_matches += sum(n * (n - 1) // 2 for n in points.values())
            decision_matches += sum(n * (n - 1) // 2 for n in decisions.values())
        summaries.append(
            TaskAgreement(
                task_id,
                len(records),
                graded_runs,
                multi_rated_runs,
                pairs,
                exact_matches,
                decision_matches,
            )
        )
    return summaries
