"""Checks for a study: definition vs. plan, records vs. plan, and data hygiene.

Severity levels:

* ``error``   — the data is wrong or inconsistent; analysis results would be
  misleading until it is fixed (exit code 1 in ``mblab validate``).
* ``warning`` — a protocol problem or a threat to validity worth reviewing.
* ``info``    — bookkeeping (missing runs while collecting, exclusions, ...).
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from mblab.plan import diff_snapshots, planned_prompt
from mblab.privacy import scan_text
from mblab.records import identity_conflicts
from mblab.schema import RunRecord
from mblab.study import Study

Severity = Literal["error", "warning", "info"]
_MAX_LISTED = 8


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    run_id: str | None = None


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    n_planned: int = 0
    n_records: int = 0
    n_missing: int = 0
    n_excluded: int = 0

    def add(self, severity: Severity, code: str, message: str, run_id: str | None = None) -> None:
        self.issues.append(Issue(severity, code, message, run_id))

    def by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def errors(self) -> list[Issue]:
        return self.by_severity("error")

    @property
    def warnings(self) -> list[Issue]:
        return self.by_severity("warning")

    @property
    def ok(self) -> bool:
        return not self.errors


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:_MAX_LISTED])
    return shown + (f", … (+{len(items) - _MAX_LISTED} more)" if len(items) > _MAX_LISTED else "")


def changed_since_planning(study: Study, records: list[RunRecord]) -> str:
    """What differed in experiment.yaml for ``records``, as far as it can be known.

    plan.json keeps a snapshot of the definition it was made from, so the change up to
    the current experiment.yaml can be listed. An intermediate edit is not stored.
    """
    plan = study.plan
    if plan is None:
        return ""
    current = study.definition.sha256()
    parts = []
    if any(r.definition_sha256 == current for r in records):
        changes = diff_snapshots(plan.definition_snapshot, study.definition.model_dump(mode="json"))
        parts.append(
            "Changed between plan.json and the current experiment.yaml: "
            f"{_listed(changes) if changes else 'content'}."
        )
    earlier = [r for r in records if r.definition_sha256 != current]
    if earlier:
        parts.append(
            f"{len(earlier)} of them were recorded under an earlier edit that is no longer "
            "stored, so what differed for those is unknown."
        )
    return " ".join(parts)


def _squash(text: str) -> str:
    return " ".join(text.split())


def validate_study(study: Study, now: datetime | None = None) -> ValidationReport:
    report = ValidationReport(n_records=len(study.records))
    definition = study.definition
    plan = study.plan
    now = now or datetime.now().astimezone()

    for err in study.load_errors:
        report.add("error", "unreadable_line", f"{err.file} line {err.line}: {err.message}")

    if not definition.documented_facts:
        report.add(
            "info",
            "no_documented_facts",
            "No documented facts recorded; reports will list only hypotheses and observations.",
        )

    # -- plan ---------------------------------------------------------------------
    if plan is None:
        if study.records:
            report.add("error", "no_plan", "Records exist but plan.json is missing.")
        else:
            report.add("info", "no_plan", "No plan yet — run `mblab plan`.")
        return report

    report.n_planned = len(plan.runs)
    run_ids = [r.run_id for r in plan.runs]
    if len(set(run_ids)) != len(run_ids):
        report.add("error", "plan_corrupt", "plan.json contains duplicate run ids.")
    if sorted(r.sequence for r in plan.runs) != list(range(1, len(plan.runs) + 1)):
        report.add("error", "plan_corrupt", "plan.json sequence numbers are not 1..N.")

    condition_ids = {c.id for c in definition.conditions}
    task_ids = {t.id for t in definition.tasks}
    orphans = [r.run_id for r in plan.runs if r.condition_id not in condition_ids]
    orphans += [r.run_id for r in plan.runs if r.task_id not in task_ids]
    if orphans:
        report.add(
            "error",
            "plan_definition_mismatch",
            f"Plan runs refer to conditions/tasks missing from the definition: {_listed(orphans)}",
        )

    current_hash = definition.sha256()
    if plan.definition_sha256 != current_hash:
        changes = diff_snapshots(plan.definition_snapshot, definition.model_dump(mode="json"))
        what = _listed(changes) if changes else "content"
        if study.records:
            report.add(
                "warning",
                "definition_changed",
                "experiment.yaml changed after the plan was generated and data collection "
                f"began. Changed: {what}. Reports will flag this as a possible post-hoc change.",
            )
        else:
            report.add(
                "warning",
                "definition_changed",
                f"experiment.yaml changed after the plan was generated (changed: {what}). "
                "No records yet — regenerate with `mblab plan --force`.",
            )

    # -- records ------------------------------------------------------------------
    planned = plan.by_run_id()
    seen = Counter(r.run_id for r in study.records)
    for run_id, count in seen.items():
        if count > 1:
            report.add(
                "error",
                "duplicate_record",
                f"{count} records for run {run_id}; keep exactly one (use `mblab import --replace`). "
                "Analysis leaves out every copy until then.",
                run_id,
            )

    confounder_rules = {c.id: c for c in definition.confounders}
    hash_mismatch: list[str] = []
    prompt_mismatch: list[str] = []
    whitespace_only: list[str] = []
    for rec in study.records:
        run = planned.get(rec.run_id)
        if run is None:
            report.add(
                "error",
                "unknown_run",
                f"Run {rec.run_id} is not in the plan (a record from another study or a pilot?). "
                "Analysis leaves it out.",
                rec.run_id,
            )
            continue
        for name in identity_conflicts(rec, run):
            report.add(
                "error",
                "condition_mismatch",
                f"Run {rec.run_id}: {name} is {getattr(rec, name)!r} but the plan says "
                f"{getattr(run, name)!r}. If the run deviated, set protocol_deviation "
                "instead of editing the condition. Analysis leaves this record out.",
                rec.run_id,
            )
        if rec.experiment_version != definition.experiment_version:
            report.add(
                "warning",
                "version_mismatch",
                f"Run {rec.run_id} was recorded under experiment_version "
                f"{rec.experiment_version}, current is {definition.experiment_version}.",
                rec.run_id,
            )
        if rec.definition_sha256 and rec.definition_sha256 != plan.definition_sha256:
            hash_mismatch.append(rec)

        # The pre-registered prompt is the one in plan.json's snapshot; an edit of
        # experiment.yaml after planning must not move the baseline.
        expected_prompt = planned_prompt(plan, rec.task_id) or (
            definition.task(rec.task_id).prompt if rec.task_id in task_ids else None
        )
        if expected_prompt is not None and rec.prompt_text != expected_prompt:
            if _squash(rec.prompt_text) == _squash(expected_prompt):
                whitespace_only.append(rec.run_id)
            else:
                prompt_mismatch.append(rec.run_id)

        if not rec.new_conversation:
            report.add(
                "warning",
                "not_new_conversation",
                f"Run {rec.run_id} was not sent in a new conversation (protocol requires one).",
                rec.run_id,
            )

        for key, value in rec.confounders.items():
            rule = confounder_rules.get(key)
            if rule is None:
                report.add(
                    "error",
                    "unknown_confounder",
                    f"Run {rec.run_id}: confounder '{key}' is not declared in experiment.yaml.",
                    rec.run_id,
                )
            elif value not in rule.allowed_values:
                report.add(
                    "error",
                    "invalid_confounder_value",
                    f"Run {rec.run_id}: confounder '{key}' = {value!r}; allowed: "
                    f"{', '.join(rule.allowed_values)}.",
                    rec.run_id,
                )
        missing_conf = [c for c in confounder_rules if c not in rec.confounders]
        if missing_conf:
            report.add(
                "warning",
                "missing_confounder",
                f"Run {rec.run_id}: confounders not recorded: {', '.join(missing_conf)}.",
                rec.run_id,
            )

        if rec.client_version.strip().lower() == "unknown":
            report.add(
                "warning", "unknown_client_version", f"Run {rec.run_id}: client version unknown."
            )
        if rec.timestamp > now + timedelta(days=1):
            report.add(
                "warning",
                "future_timestamp",
                f"Run {rec.run_id}: timestamp {rec.timestamp.isoformat()} is in the future.",
                rec.run_id,
            )
        if rec.excluded:
            report.n_excluded += 1
            report.add(
                "info",
                "excluded_record",
                f"Run {rec.run_id} is excluded from the primary analysis (protocol deviation: "
                f"{rec.protocol_deviation}).",
                rec.run_id,
            )
        for field_name in (
            "response_text",
            "notes",
            "protocol_deviation",
            "observed_label_location",
        ):
            for finding in scan_text(getattr(rec, field_name), field_name):
                report.add(
                    "warning",
                    "possible_personal_data",
                    f"Run {rec.run_id}: possible {finding.kind} in {finding.field}. "
                    "Remove personal data before sharing this study.",
                    rec.run_id,
                )

    if hash_mismatch:
        report.add(
            "warning",
            "hash_mismatch",
            f"{len(hash_mismatch)} record(s) were collected under an experiment.yaml that "
            f"differs from the one in plan.json: {_listed([r.run_id for r in hash_mismatch])}. "
            + changed_since_planning(study, hash_mismatch),
        )
    if prompt_mismatch:
        report.add(
            "warning",
            "prompt_mismatch",
            "Prompt text differs from the task prompt (wording must be identical across runs): "
            f"{_listed(prompt_mismatch)}.",
        )
    if whitespace_only:
        report.add(
            "info",
            "prompt_whitespace",
            f"Prompt text differs from the task prompt only in whitespace: {_listed(whitespace_only)}.",
        )

    _check_dataset_level(study, report)
    _check_api_records(study, report)
    _check_grades(study, report)

    missing = [
        r.run_id for r in sorted(plan.runs, key=lambda r: r.sequence) if r.run_id not in seen
    ]
    report.n_missing = len(missing)
    if missing:
        report.add(
            "info",
            "missing_runs",
            f"{len(missing)} of {len(plan.runs)} planned runs have no record yet: {_listed(missing)}.",
        )
    return report


def _check_dataset_level(study: Study, report: ValidationReport) -> None:
    records = study.records
    if not records:
        return
    origins = {r.data_origin for r in records}
    if len(origins) > 1:
        report.add(
            "error",
            "mixed_data_origin",
            "Synthetic and observed records are mixed in one study. Keep demo data separate.",
        )
    elif study.definition.demo and "observed" in origins:
        report.add("error", "mixed_data_origin", "A demo study must not contain observed records.")
    elif not study.definition.demo and "synthetic" in origins:
        report.add(
            "error",
            "mixed_data_origin",
            "Synthetic records found in a study that is not marked demo: true.",
        )

    versions = Counter(r.client_version for r in records)
    if len(versions) > 1:
        ordered = sorted(records, key=lambda r: r.timestamp)
        first_seen: dict[str, datetime] = {}
        for rec in ordered:
            first_seen.setdefault(rec.client_version, rec.timestamp)
        listing = ", ".join(
            f"{v} (from {first_seen[v].date().isoformat()}, {versions[v]} runs)"
            for v in sorted(first_seen, key=first_seen.get)
        )
        report.add(
            "warning",
            "client_version_changed",
            f"Client version changed during the experiment: {listing}.",
        )

    methods = {r.latency_method for r in records if r.latency_method != "not_measured"}
    if len(methods) > 1:
        report.add(
            "warning",
            "mixed_latency_methods",
            f"Latency was measured with different methods ({', '.join(sorted(methods))}); "
            "they are summarised separately and never pooled.",
        )

    by_sequence = sorted(
        (r for r in records if study.plan and r.run_id in study.plan.by_run_id()),
        key=lambda r: study.plan.by_run_id()[r.run_id].sequence,  # type: ignore[union-attr]
    )
    inversions = sum(
        1 for prev, cur in itertools.pairwise(by_sequence) if cur.timestamp < prev.timestamp
    )
    if inversions:
        report.add(
            "info",
            "order_deviation",
            f"{inversions} run(s) were executed earlier than a run planned before them; "
            "the randomized order was not followed exactly.",
        )

    texts: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if len(rec.response_text.strip()) >= 80:
            texts[_squash(rec.response_text)].append(rec.run_id)
    for run_ids in texts.values():
        if len(run_ids) > 1:
            report.add(
                "warning",
                "duplicate_response_text",
                f"Identical response text in runs {_listed(run_ids)} — check for a copy-paste mistake.",
            )


def _check_api_records(study: Study, report: ValidationReport) -> None:
    """official_api studies: API-reported values are present and consistent with the request."""
    definition = study.definition
    if definition.execution_method != "official_api":
        return
    option_ids = {o.id for o in definition.selection_options}
    no_values: list[str] = []
    mismatch: list[str] = []
    models: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    for rec in study.records:
        reported = rec.api_reported
        if reported is None:
            no_values.append(rec.run_id)
            continue
        if reported.model:
            models[reported.model] += 1
        if reported.service_tier:
            tiers[reported.service_tier] += 1
        if rec.model_selection in option_ids and reported.reasoning_effort is not None:
            requested = definition.option(rec.model_selection).reasoning_effort
            if reported.reasoning_effort != requested:
                mismatch.append(f"{rec.run_id} ({requested} → {reported.reasoning_effort})")
    if no_values:
        report.add(
            "warning",
            "missing_api_values",
            "Records without API-reported values in an official_api study (expected from "
            f"`mblab run-api`): {_listed(no_values)}.",
        )
    if mismatch:
        report.add(
            "warning",
            "api_effort_mismatch",
            "The reasoning effort echoed by the API differs from the one requested: "
            f"{_listed(mismatch)}.",
        )
    for code, what, counts in (
        ("api_model_changed", "model id", models),
        ("api_service_tier_changed", "service tier", tiers),
    ):
        if len(counts) > 1:
            listing = ", ".join(f"{k} ({n} runs)" for k, n in counts.most_common())
            report.add(
                "warning", code, f"The API reported more than one {what} in this study: {listing}."
            )


def _check_grades(study: Study, report: ValidationReport) -> None:
    records_by_id: dict[str, RunRecord] = {r.run_id: r for r in study.records}
    tasks = {t.id: t for t in study.definition.tasks}
    seen_pairs: Counter[tuple[str, str]] = Counter()
    for grade in study.grades:
        seen_pairs[(grade.run_id, grade.grader_id)] += 1
        rec = records_by_id.get(grade.run_id)
        if rec is None or rec.task_id not in tasks:
            report.add(
                "error", "grade_unknown_run", f"Grade for unknown run {grade.run_id}.", grade.run_id
            )
            continue
        task = tasks[rec.task_id]
        if task.scoring.method != "manual":
            report.add(
                "warning",
                "grade_for_automatic_task",
                f"Run {grade.run_id}: task {task.id} is scored automatically; the grade is ignored.",
                grade.run_id,
            )
        elif grade.max_points != task.scoring.max_points:
            report.add(
                "error",
                "grade_scale_mismatch",
                f"Run {grade.run_id}: grade uses max_points={grade.max_points:g}, task "
                f"{task.id} uses {task.scoring.max_points:g}.",
                grade.run_id,
            )
    for (run_id, grader), count in seen_pairs.items():
        if count > 1:
            report.add(
                "error",
                "duplicate_grade",
                f"Grader {grader} graded run {run_id} {count} times.",
                run_id,
            )

    graded = {g.run_id for g in study.grades}
    pending = [
        r.run_id
        for r in study.records
        if r.task_id in tasks
        and tasks[r.task_id].scoring.method == "manual"
        and r.response_status == "completed"
        and r.run_id not in graded
    ]
    if pending:
        report.add(
            "info",
            "pending_manual_grades",
            f"{len(pending)} response(s) await manual grading — run `mblab grade export`.",
        )
