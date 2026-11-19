"""Analysis: per-run results, condition summaries, and conservative comparisons.

What this module reports, and how it phrases it, follows one rule: describe
what was observed in this sample and never present a behavioural pattern as
evidence about server-side internals. Every interpretive sentence is a
template defined here, so the wording can be reviewed in one place.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from mblab.schema import ExperimentDefinition, RunPlan
from mblab.scoring import score_record
from mblab.stats import (
    Interval,
    bootstrap_median_diff,
    describe,
    newcombe_interval,
    wilson_interval,
)
from mblab.study import Study
from mblab.validate import ValidationReport, validate_study

LabelClass = Literal[
    "matches_override",
    "matches_default",
    "matches_both",
    "other_option",
    "unrecognized",
    "not_displayed",
]
LABEL_CLASSES: list[str] = [
    "matches_override",
    "matches_default",
    "matches_both",
    "other_option",
    "unrecognized",
    "not_displayed",
]
LABEL_CLASS_TEXT = {
    "matches_override": "matched the per-message selection",
    "matches_default": "matched the default selection",
    "matches_both": "matched both (same option or an ambiguous alias)",
    "other_option": "named a different option",
    "unrecognized": "did not match any pre-registered label",
    "not_displayed": "showed no label",
}
OUTCOMES = [
    "correct",
    "incorrect",
    "format_failure",
    "not_scorable",
    "pending",
    "excluded",
    "missing",
]


@dataclass(frozen=True)
class AnalysisSettings:
    confidence: float = 0.95
    min_n: int = 5
    bootstrap_resamples: int = 2000
    seed: int = 0


@dataclass(frozen=True)
class Limitation:
    kind: Literal["standard", "data"]
    text: str


@dataclass(frozen=True)
class Claim:
    category: Literal["documented_fact", "hypothesis", "observation", "not_verified"]
    text: str
    source: str


@dataclass(frozen=True)
class MetricAssessment:
    metric: str
    unit: str
    condition_value: float | None
    default_ref_value: float | None
    override_ref_value: float | None
    references_diff: Interval | None
    vs_default: Interval | None
    vs_override: Interval | None
    verdict: str
    explanation: str


@dataclass(frozen=True)
class OverrideAssessment:
    condition_id: str
    default_ref: str | None
    override_ref: str | None
    stacked: bool
    label_counts: dict[str, int]
    label_summary: str
    metrics: list[MetricAssessment]
    note: str | None = None


@dataclass
class Analysis:
    definition: ExperimentDefinition
    plan: RunPlan | None
    settings: AnalysisSettings
    synthetic: bool
    results: pd.DataFrame
    conditions: pd.DataFrame
    latency: pd.DataFrame
    tasks: pd.DataFrame
    grid: pd.DataFrame
    difficulty: pd.DataFrame
    labels: pd.DataFrame
    comparisons: pd.DataFrame
    overrides: list[OverrideAssessment]
    environment: dict[str, Any]
    validation: ValidationReport
    limitations: list[Limitation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    primary_latency_method: str | None = None

    @property
    def included(self) -> pd.DataFrame:
        if self.results.empty:
            return self.results
        return self.results[~self.results["excluded"]]

    @property
    def has_data(self) -> bool:
        return not self.included.empty

    @property
    def has_scored_data(self) -> bool:
        return bool(self.has_data and self.conditions["n_scored"].sum() > 0)

    def to_dict(self) -> dict[str, Any]:
        def frame(df: pd.DataFrame) -> list[dict[str, Any]]:
            return [] if df.empty else df.astype(object).where(df.notna(), None).to_dict("records")

        def interval(iv: Interval | None) -> dict[str, float] | None:
            return None if iv is None else {"estimate": iv.estimate, "low": iv.low, "high": iv.high}

        return {
            "experiment_id": self.definition.experiment_id,
            "experiment_version": self.definition.experiment_version,
            "definition_sha256": self.definition.sha256(),
            "synthetic_data": self.synthetic,
            "settings": self.settings.__dict__,
            "environment": self.environment,
            "primary_latency_method": self.primary_latency_method,
            "conditions": frame(self.conditions),
            "latency": frame(self.latency),
            "tasks": frame(self.tasks),
            "difficulty": frame(self.difficulty),
            "labels": frame(self.labels),
            "comparisons": frame(self.comparisons),
            "override_assessments": [
                {
                    "condition_id": o.condition_id,
                    "default_reference": o.default_ref,
                    "per_message_reference": o.override_ref,
                    "stacked": o.stacked,
                    "label_counts": o.label_counts,
                    "label_summary": o.label_summary,
                    "note": o.note,
                    "metrics": [
                        {
                            "metric": m.metric,
                            "unit": m.unit,
                            "condition_value": m.condition_value,
                            "default_reference_value": m.default_ref_value,
                            "per_message_reference_value": m.override_ref_value,
                            "references_difference": interval(m.references_diff),
                            "vs_default_reference": interval(m.vs_default),
                            "vs_per_message_reference": interval(m.vs_override),
                            "verdict": m.verdict,
                            "explanation": m.explanation,
                        }
                        for m in o.metrics
                    ],
                }
                for o in self.overrides
            ],
            "limitations": [lim.__dict__ for lim in self.limitations],
            "claims": [c.__dict__ for c in self.claims],
            "validation": {
                "errors": len(self.validation.errors),
                "warnings": len(self.validation.warnings),
            },
        }


# -- per-run results ------------------------------------------------------------------


def classify_label(
    definition: ExperimentDefinition,
    model_selection: str,
    message_override: str | None,
    observed: str | None,
) -> str:
    """Compare a displayed label with the condition's settings (hypothesis-neutral)."""
    if observed is None or not observed.strip():
        return "not_displayed"
    matches = {o.id for o in definition.selection_options if o.matches_label(observed)}
    if not matches:
        return "unrecognized"
    default_hit = model_selection in matches
    if message_override is None:
        return "matches_default" if default_hit else "other_option"
    override_hit = message_override in matches
    if model_selection == message_override or (override_hit and default_hit):
        return "matches_both" if default_hit else "other_option"
    if override_hit:
        return "matches_override"
    if default_hit:
        return "matches_default"
    return "other_option"


RESULT_COLUMNS = [
    "run_id",
    "sequence",
    "block",
    "condition_id",
    "model_selection",
    "message_override",
    "task_id",
    "task_title",
    "difficulty",
    "repetition",
    "timestamp",
    "client_version",
    "platform",
    "data_origin",
    "excluded",
    "protocol_deviation",
    "response_status",
    "score_status",
    "correct",
    "score",
    "outcome",
    "extracted_answer",
    "score_detail",
    "score_source",
    "grader_disagreement",
    "observed_model_label",
    "label_class",
    "thinking_indicator",
    "thinking_text",
    "thinking_seconds",
    "latency_seconds",
    "latency_method",
    "latency_confidence",
    "new_conversation",
    "prompt_text",
    "response_text",
    "notes",
]


def build_results(study: Study) -> pd.DataFrame:
    definition = study.definition
    plan_runs = study.plan.by_run_id() if study.plan else {}
    tasks = {t.id: t for t in definition.tasks}
    grades_by_run: dict[str, list] = {}
    for grade in study.grades:
        grades_by_run.setdefault(grade.run_id, []).append(grade)
    confounder_ids = [c.id for c in definition.confounders]

    seen: set[str] = set()
    rows = []
    for rec in study.records:
        # Skip records analysis cannot place; `mblab validate` reports them as errors.
        if rec.run_id in seen or rec.run_id not in plan_runs or rec.task_id not in tasks:
            continue
        seen.add(rec.run_id)
        run = plan_runs[rec.run_id]
        task = tasks[rec.task_id]
        result = score_record(task, rec, grades_by_run.get(rec.run_id, []))
        if rec.excluded:
            outcome = "excluded"
        elif result.status == "scored":
            outcome = "correct" if result.correct else "incorrect"
        elif result.status == "format_failure":
            outcome = "format_failure"
        elif result.status == "pending_manual":
            outcome = "pending"
        else:
            outcome = "not_scorable"
        row = {
            "run_id": rec.run_id,
            "sequence": run.sequence,
            "block": run.block,
            "condition_id": rec.condition_id,
            "model_selection": rec.model_selection,
            "message_override": rec.message_override,
            "task_id": rec.task_id,
            "task_title": task.title,
            "difficulty": task.difficulty,
            "repetition": rec.repetition,
            "timestamp": rec.timestamp.isoformat(),
            "client_version": rec.client_version,
            "platform": rec.platform,
            "data_origin": rec.data_origin,
            "excluded": rec.excluded,
            "protocol_deviation": rec.protocol_deviation,
            "response_status": rec.response_status,
            "score_status": result.status,
            "correct": result.correct,
            "score": result.score,
            "outcome": outcome,
            "extracted_answer": result.extracted_answer,
            "score_detail": result.detail,
            "score_source": result.source,
            "grader_disagreement": result.grader_disagreement,
            "observed_model_label": rec.observed_model_label,
            "label_class": classify_label(
                definition, rec.model_selection, rec.message_override, rec.observed_model_label
            ),
            "thinking_indicator": rec.thinking_indicator,
            "thinking_text": rec.thinking_text,
            "thinking_seconds": rec.thinking_seconds,
            "latency_seconds": rec.latency_seconds,
            "latency_method": rec.latency_method,
            "latency_confidence": rec.latency_confidence,
            "new_conversation": rec.new_conversation,
            "prompt_text": rec.prompt_text,
            "response_text": rec.response_text,
            "notes": rec.notes,
        }
        for cid in confounder_ids:
            row[f"conf_{cid}"] = rec.confounders.get(cid)
        rows.append(row)

    columns = RESULT_COLUMNS + [f"conf_{cid}" for cid in confounder_ids]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("sequence").reset_index(drop=True)
    df["excluded"] = df["excluded"].astype(bool)
    return df


# -- summaries ----------------------------------------------------------------------------


def _proportion(
    k: int, n: int, confidence: float
) -> tuple[float | None, float | None, float | None]:
    iv = wilson_interval(k, n, confidence)
    return (None, None, None) if iv is None else (iv.estimate, iv.low, iv.high)


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["score_status"].isin(["scored", "format_failure"])]


def summarize_conditions(
    definition: ExperimentDefinition,
    plan: RunPlan | None,
    results: pd.DataFrame,
    settings: AnalysisSettings,
    latency_method: str | None,
) -> pd.DataFrame:
    planned = Counter(r.condition_id for r in plan.runs) if plan else Counter()
    rows = []
    for cond in definition.conditions:
        all_rows = results[results["condition_id"] == cond.id]
        inc = all_rows[~all_rows["excluded"]]
        scored = _scored(inc)
        n_correct = int(scored["correct"].eq(True).sum())
        acc, acc_lo, acc_hi = _proportion(n_correct, len(scored), settings.confidence)
        n_pending = int((inc["score_status"] == "pending_manual").sum())
        succ, succ_lo, succ_hi = _proportion(n_correct, len(inc) - n_pending, settings.confidence)
        known_thinking = inc[inc["thinking_indicator"].isin(["shown", "not_shown"])]
        n_shown = int((known_thinking["thinking_indicator"] == "shown").sum())
        think_rate, _, _ = _proportion(n_shown, len(known_thinking), settings.confidence)
        lat = (
            inc.loc[inc["latency_method"] == latency_method, "latency_seconds"].dropna()
            if latency_method
            else pd.Series(dtype=float)
        )
        lat_desc = describe(lat.tolist())
        think_secs = describe(inc["thinking_seconds"].dropna().tolist())
        rep_acc = []
        for _, rep_rows in scored.groupby("repetition"):
            if len(rep_rows):
                rep_acc.append(float(rep_rows["correct"].eq(True).mean()))
        rows.append(
            {
                "condition_id": cond.id,
                "label": cond.label or cond.display(),
                "role": cond.role,
                "model_selection": definition.option_label(cond.model_selection),
                "message_override": definition.option_label(cond.message_override),
                "n_planned": planned[cond.id],
                "n_recorded": len(all_rows),
                "n_excluded": int(all_rows["excluded"].sum()),
                "n_included": len(inc),
                "n_not_completed": int((inc["response_status"] != "completed").sum()),
                "n_scored": len(scored),
                "n_correct": n_correct,
                "n_format_failures": int((scored["score_status"] == "format_failure").sum()),
                "n_pending_manual": n_pending,
                "accuracy": acc,
                "accuracy_low": acc_lo,
                "accuracy_high": acc_hi,
                "success_rate": succ,
                "success_low": succ_lo,
                "success_high": succ_hi,
                "mean_score": float(scored["score"].mean()) if len(scored) else None,
                "repetition_accuracy_sd": (
                    float(pd.Series(rep_acc).std(ddof=1)) if len(rep_acc) >= 2 else None
                ),
                "latency_method": latency_method,
                "latency_n": lat_desc["n"],
                "latency_median": lat_desc["median"],
                "latency_q1": lat_desc["q1"],
                "latency_q3": lat_desc["q3"],
                "thinking_known_n": len(known_thinking),
                "thinking_shown_n": n_shown,
                "thinking_shown_rate": think_rate,
                "thinking_seconds_n": think_secs["n"],
                "thinking_seconds_median": think_secs["median"],
            }
        )
    return pd.DataFrame(rows)


def summarize_latency(definition: ExperimentDefinition, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc = results[~results["excluded"]] if not results.empty else results
    measured = inc[inc["latency_method"] != "not_measured"] if not inc.empty else inc
    for cond in definition.conditions:
        sub = measured[measured["condition_id"] == cond.id] if not measured.empty else measured
        for method, grp in sub.groupby("latency_method") if not sub.empty else []:
            stats = describe(grp["latency_seconds"].dropna().tolist())
            rows.append({"condition_id": cond.id, "latency_method": method, **stats})
    return pd.DataFrame(
        rows, columns=["condition_id", "latency_method", "n", "median", "q1", "q3", "min", "max"]
    )


def summarize_tasks(definition: ExperimentDefinition, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc = results[~results["excluded"]] if not results.empty else results
    for cond in definition.conditions:
        for task in definition.tasks:
            sub = inc[(inc["condition_id"] == cond.id) & (inc["task_id"] == task.id)]
            scored = _scored(sub)
            outcomes = scored["correct"].tolist()
            answers = [a for a in scored["extracted_answer"].tolist() if a is not None]
            normalized = Counter(" ".join(str(a).split()).casefold() for a in answers)
            rows.append(
                {
                    "condition_id": cond.id,
                    "task_id": task.id,
                    "task_title": task.title,
                    "difficulty": task.difficulty,
                    "n_included": len(sub),
                    "n_scored": len(scored),
                    "n_correct": sum(1 for o in outcomes if o is True),
                    "accuracy": (
                        sum(1 for o in outcomes if o is True) / len(outcomes) if outcomes else None
                    ),
                    "n_format_failures": int((scored["score_status"] == "format_failure").sum()),
                    "consistent_outcome": (len(set(outcomes)) == 1) if len(outcomes) >= 2 else None,
                    "distinct_answers": len(normalized),
                    "majority_answer_share": (
                        max(normalized.values()) / len(answers) if answers else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def repetition_grid(
    definition: ExperimentDefinition, plan: RunPlan | None, results: pd.DataFrame
) -> pd.DataFrame:
    """One cell per planned run (condition x task x repetition), including missing ones."""
    by_run = {} if results.empty else results.set_index("run_id")["outcome"].to_dict()
    rows = []
    if plan:
        for run in plan.runs:
            rows.append(
                {
                    "condition_id": run.condition_id,
                    "task_id": run.task_id,
                    "repetition": run.repetition,
                    "run_id": run.run_id,
                    "outcome": by_run.get(run.run_id, "missing"),
                }
            )
    return pd.DataFrame(
        rows, columns=["condition_id", "task_id", "repetition", "run_id", "outcome"]
    )


def summarize_difficulty(definition: ExperimentDefinition, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc = results[~results["excluded"]] if not results.empty else results
    levels = [
        d for d in ("easy", "medium", "hard") if any(t.difficulty == d for t in definition.tasks)
    ]
    for cond in definition.conditions:
        for level in levels:
            sub = _scored(inc[(inc["condition_id"] == cond.id) & (inc["difficulty"] == level)])
            k = int(sub["correct"].eq(True).sum())
            rows.append(
                {
                    "condition_id": cond.id,
                    "difficulty": level,
                    "n_scored": len(sub),
                    "n_correct": k,
                    "accuracy": k / len(sub) if len(sub) else None,
                }
            )
    return pd.DataFrame(
        rows, columns=["condition_id", "difficulty", "n_scored", "n_correct", "accuracy"]
    )


def summarize_labels(definition: ExperimentDefinition, results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inc = results[~results["excluded"]] if not results.empty else results
    for cond in definition.conditions:
        sub = inc[inc["condition_id"] == cond.id] if not inc.empty else inc
        counts = Counter(sub["label_class"]) if not sub.empty else Counter()
        rows.append({"condition_id": cond.id, **{c: counts.get(c, 0) for c in LABEL_CLASSES}})
    return pd.DataFrame(rows, columns=["condition_id", *LABEL_CLASSES])


# -- comparisons ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MetricData:
    kind: Literal["proportion", "median"]
    k: int = 0
    n: int = 0
    values: tuple[float, ...] = ()

    @property
    def size(self) -> int:
        return self.n if self.kind == "proportion" else len(self.values)

    @property
    def value(self) -> float | None:
        if self.kind == "proportion":
            return self.k / self.n if self.n else None
        return describe(self.values)["median"]


def _metric_data(
    results: pd.DataFrame, cond_id: str, metric: str, latency_method: str | None
) -> _MetricData:
    inc = results[(~results["excluded"]) & (results["condition_id"] == cond_id)]
    if metric == "accuracy":
        scored = _scored(inc)
        return _MetricData("proportion", int(scored["correct"].eq(True).sum()), len(scored))
    if metric == "thinking_shown_rate":
        known = inc[inc["thinking_indicator"].isin(["shown", "not_shown"])]
        return _MetricData(
            "proportion", int((known["thinking_indicator"] == "shown").sum()), len(known)
        )
    if metric == "latency_median":
        if latency_method is None:
            return _MetricData("median")
        vals = inc.loc[inc["latency_method"] == latency_method, "latency_seconds"].dropna()
        return _MetricData("median", values=tuple(float(v) for v in vals))
    if metric == "thinking_seconds_median":
        return _MetricData(
            "median", values=tuple(float(v) for v in inc["thinking_seconds"].dropna())
        )
    raise ValueError(metric)


def _diff(a: _MetricData, b: _MetricData, settings: AnalysisSettings) -> Interval | None:
    if a.kind == "proportion":
        return newcombe_interval(a.k, a.n, b.k, b.n, settings.confidence)
    return bootstrap_median_diff(
        a.values,
        b.values,
        resamples=settings.bootstrap_resamples,
        confidence=settings.confidence,
        seed=settings.seed,
    )


METRICS: dict[str, tuple[str, str]] = {
    "accuracy": ("accuracy", "proportion"),
    "thinking_shown_rate": ("thinking indicator shown", "proportion"),
    "latency_median": ("median response time", "seconds"),
    "thinking_seconds_median": ("median displayed thinking time", "seconds"),
}


def pairwise_comparisons(
    definition: ExperimentDefinition,
    results: pd.DataFrame,
    settings: AnalysisSettings,
    latency_method: str | None,
) -> pd.DataFrame:
    rows = []
    conds = [c.id for c in definition.conditions]
    if results.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "condition_a",
                "condition_b",
                "n_a",
                "n_b",
                "difference",
                "low",
                "high",
                "status",
            ]
        )
    for metric in ("accuracy", "latency_median"):
        for i, a in enumerate(conds):
            for b in conds[i + 1 :]:
                da = _metric_data(results, a, metric, latency_method)
                db = _metric_data(results, b, metric, latency_method)
                if min(da.size, db.size) < settings.min_n:
                    iv, status = None, "insufficient_data"
                else:
                    iv = _diff(da, db, settings)
                    status = (
                        "difference_observed" if iv and iv.excludes_zero else "no_clear_difference"
                    )
                rows.append(
                    {
                        "metric": metric,
                        "condition_a": a,
                        "condition_b": b,
                        "n_a": da.size,
                        "n_b": db.size,
                        "difference": None if iv is None else iv.estimate,
                        "low": None if iv is None else iv.low,
                        "high": None if iv is None else iv.high,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


VERDICT_TEXT = {
    "insufficient_data": (
        "Fewer than {min_n} usable observations in at least one group; no comparison is made."
    ),
    "references_not_distinguishable": (
        "The two reference conditions ({dref} and {oref}) are not distinguishable on this metric "
        "in this sample, so it cannot indicate which selection was applied."
    ),
    "consistent_with_per_message": (
        "{cond} differs from the default-only reference ({dref}) and is not distinguishable from "
        "the per-message reference ({oref}). In this sample that pattern is consistent with the "
        "per-message selection being applied. It does not identify the model that ran on the server."
    ),
    "consistent_with_default": (
        "{cond} differs from the per-message reference ({oref}) and is not distinguishable from "
        "the default-only reference ({dref}). In this sample that pattern is consistent with the "
        "default selection being applied. It does not identify the model that ran on the server."
    ),
    "differs_from_both": (
        "{cond} differs from both reference conditions. Neither simple explanation fits this "
        "sample; look for confounders (notices, tools, client changes) before interpreting."
    ),
    "inconclusive": (
        "{cond} is not distinguishable from either reference condition at this sample size."
    ),
    "no_difference_detected": (
        "No difference was detected between {cond} (same setting chosen twice) and {dref} (chosen "
        "once). Not detecting a difference is not evidence that the two are equivalent."
    ),
    "difference_detected": (
        "{cond} (same setting chosen twice) differs from {dref} (chosen once) in this sample. "
        "The cause is not identified by this experiment."
    ),
}


def _find_reference(definition: ExperimentDefinition, option_id: str) -> str | None:
    for cond in definition.conditions:
        if cond.model_selection == option_id and cond.message_override is None:
            return cond.id
    return None


def assess_overrides(
    definition: ExperimentDefinition,
    results: pd.DataFrame,
    labels: pd.DataFrame,
    settings: AnalysisSettings,
    latency_method: str | None,
) -> list[OverrideAssessment]:
    out: list[OverrideAssessment] = []
    for cond in definition.conditions:
        if cond.message_override is None:
            continue
        stacked = cond.model_selection == cond.message_override
        dref = _find_reference(definition, cond.model_selection)
        oref = _find_reference(definition, cond.message_override)
        counts: dict[str, int] = {}
        if not labels.empty:
            row = labels[labels["condition_id"] == cond.id]
            if not row.empty:
                counts = {c: int(row.iloc[0][c]) for c in LABEL_CLASSES}
        shown = sum(v for k, v in counts.items() if k != "not_displayed")
        if shown == 0:
            label_summary = "No model label was displayed (or recorded) for this condition."
        else:
            parts = [
                f"{counts[c]} {LABEL_CLASS_TEXT[c]}" for c in LABEL_CLASSES[:-1] if counts.get(c)
            ]
            label_summary = (
                f"Of {shown} runs with a displayed label: " + "; ".join(parts) + ". "
                "Labels are what the client displayed, not a server-side record."
            )

        if dref is None or oref is None:
            missing = definition.option_label(
                cond.model_selection if dref is None else cond.message_override
            )
            out.append(
                OverrideAssessment(
                    cond.id,
                    dref,
                    oref,
                    stacked,
                    counts,
                    label_summary,
                    [],
                    note=(
                        f"No reference condition with default '{missing}' and no per-message "
                        "selection is defined, so behavioural comparisons are not possible."
                    ),
                )
            )
            continue

        metrics: list[MetricAssessment] = []
        for metric, (name, unit) in METRICS.items():
            dc = _metric_data(results, cond.id, metric, latency_method)
            dd = _metric_data(results, dref, metric, latency_method)
            do = _metric_data(results, oref, metric, latency_method)
            fmt = {"cond": cond.id, "dref": dref, "oref": oref, "min_n": settings.min_n}
            if min(dc.size, dd.size, do.size) < settings.min_n:
                metrics.append(
                    MetricAssessment(
                        name,
                        unit,
                        dc.value,
                        dd.value,
                        do.value,
                        None,
                        None,
                        None,
                        "insufficient_data",
                        VERDICT_TEXT["insufficient_data"].format(**fmt),
                    )
                )
                continue
            vs_d = _diff(dc, dd, settings)
            if stacked:
                verdict = (
                    "difference_detected"
                    if vs_d and vs_d.excludes_zero
                    else "no_difference_detected"
                )
                metrics.append(
                    MetricAssessment(
                        name,
                        unit,
                        dc.value,
                        dd.value,
                        do.value,
                        None,
                        vs_d,
                        vs_d,
                        verdict,
                        VERDICT_TEXT[verdict].format(**fmt),
                    )
                )
                continue
            refs = _diff(do, dd, settings)
            vs_o = _diff(dc, do, settings)
            if refs is None or not refs.excludes_zero:
                verdict = "references_not_distinguishable"
            elif vs_d and vs_o and vs_d.excludes_zero and not vs_o.excludes_zero:
                verdict = "consistent_with_per_message"
            elif vs_d and vs_o and vs_o.excludes_zero and not vs_d.excludes_zero:
                verdict = "consistent_with_default"
            elif vs_d and vs_o and vs_d.excludes_zero and vs_o.excludes_zero:
                verdict = "differs_from_both"
            else:
                verdict = "inconclusive"
            metrics.append(
                MetricAssessment(
                    name,
                    unit,
                    dc.value,
                    dd.value,
                    do.value,
                    refs,
                    vs_d,
                    vs_o,
                    verdict,
                    VERDICT_TEXT[verdict].format(**fmt),
                )
            )
        out.append(OverrideAssessment(cond.id, dref, oref, stacked, counts, label_summary, metrics))
    return out


# -- environment, limitations, claims ----------------------------------------------------


def _value_counts(series: pd.Series) -> dict[str, int]:
    """Counts with missing values (None or NaN) reported as 'not recorded'."""
    return {
        str(k): int(v)
        for k, v in Counter(series.astype(object).where(series.notna(), "not recorded")).items()
    }


def describe_environment(study: Study, results: pd.DataFrame) -> dict[str, Any]:
    definition = study.definition
    env: dict[str, Any] = {
        "service": definition.service,
        "platform": definition.platform,
        "execution_method": definition.execution_method,
        "planned_runs": len(study.plan.runs) if study.plan else 0,
        "recorded_runs": len(results),
        "client_versions": {},
        "first_run": None,
        "last_run": None,
        "confounders": {},
        "recorders": sorted({r.recorded_by for r in study.records if r.recorded_by}),
    }
    if results.empty:
        return env
    env["client_versions"] = dict(Counter(results["client_version"]))
    stamps = sorted(results["timestamp"])
    env["first_run"], env["last_run"] = stamps[0], stamps[-1]
    for conf in definition.confounders:
        col = f"conf_{conf.id}"
        env["confounders"][conf.id] = _value_counts(results[col])
    return env


def build_limitations(analysis: Analysis) -> list[Limitation]:
    d = analysis.definition
    out: list[Limitation] = []
    if analysis.synthetic:
        out.append(
            Limitation(
                "data",
                "These are SYNTHETIC demo records generated by `mblab demo`. The values are random "
                "and carry no information about any real model or service.",
            )
        )
    if d.execution_method == "manual_ui":
        out += [
            Limitation(
                "standard",
                "A model label is what the client displayed. It is not checked against any "
                "server-side record of which model produced the response.",
            ),
            Limitation(
                "standard",
                "Not observable in this experiment: " + "; ".join(d.not_observable)
                if d.not_observable
                else "Server-side model identifiers, routing, and hidden reasoning effort are not observable.",
            ),
            Limitation(
                "standard",
                "Behavioural proxies (accuracy, response time, thinking indicator) can change for "
                "reasons unrelated to model selection, such as server load, server-side "
                "experiments, or usage limits.",
            ),
            Limitation(
                "standard",
                "The experimenter sets each condition by hand and cannot be blind to it. Automatic "
                "scoring does not see the condition.",
            ),
        ]
    out += [
        Limitation(
            "standard",
            "Results describe the tested account, client version(s), platform, region and dates. "
            "The service can change without notice.",
        ),
        Limitation(
            "standard",
            "Intervals treat responses as independent and are conditional on this fixed task set; "
            "they do not describe performance on other tasks. Difficulty labels are the task "
            "author's prior judgement.",
        ),
    ]
    if analysis.has_data:
        n_per = analysis.conditions["n_scored"]
        out.append(
            Limitation(
                "standard",
                f"Small sample: {int(n_per.min())}–{int(n_per.max())} scored responses per "
                "condition. Only large differences can be detected; an exploratory study, not a "
                "confirmatory one.",
            )
        )
    lat_methods = set(analysis.latency["latency_method"]) if not analysis.latency.empty else set()
    if lat_methods & {"manual_stopwatch", "screen_recording"}:
        out.append(
            Limitation(
                "standard",
                "Response times were measured by hand. They include human reaction time, network "
                "and rendering time, and are not comparable with latency measured through an API.",
            )
        )

    res = analysis.results
    if analysis.plan is not None:
        missing = len(analysis.plan.runs) - len(res)
        if missing > 0:
            out.append(
                Limitation(
                    "data",
                    f"{missing} of {len(analysis.plan.runs)} planned runs have no record; results "
                    "cover recorded runs only.",
                )
            )
    if res.empty:
        return out
    excluded = int(res["excluded"].sum())
    if excluded:
        out.append(Limitation("data", f"{excluded} run(s) excluded for protocol deviations."))
    inc = analysis.included
    not_completed = int((inc["response_status"] != "completed").sum())
    if not_completed:
        out.append(
            Limitation(
                "data",
                f"{not_completed} response(s) were errors, refusals or incomplete; they count as "
                "failures in task success rate and are left out of accuracy.",
            )
        )
    fmt = int((inc["score_status"] == "format_failure").sum())
    if fmt:
        out.append(
            Limitation(
                "data",
                f"{fmt} response(s) had no usable answer line and were scored as incorrect.",
            )
        )
    pending = int((inc["score_status"] == "pending_manual").sum())
    if pending:
        out.append(Limitation("data", f"{pending} response(s) still await manual grading."))
    if len(set(inc["client_version"])) > 1:
        out.append(
            Limitation(
                "data",
                "The client version changed during the experiment: "
                + ", ".join(sorted(set(inc["client_version"])))
                + ".",
            )
        )
    plan_hash = analysis.plan.definition_sha256 if analysis.plan else None
    current = d.sha256()
    if plan_hash and plan_hash != current:
        out.append(
            Limitation(
                "data",
                "experiment.yaml changed after the plan was generated. Check `mblab validate` for "
                "what changed; scoring rules edited after seeing data would be a post-hoc change.",
            )
        )
    for conf in d.confounders:
        col = f"conf_{conf.id}"
        values = Counter(inc[col].dropna())
        unknown = int(inc[col].isna().sum()) + values.get("unknown", 0)
        if len(values) > 1 or unknown:
            dist = ", ".join(f"{k}: {v}" for k, v in sorted(values.items()))
            extra = f"; not recorded/unknown in {unknown} run(s)" if unknown else ""
            out.append(
                Limitation("data", f"Confounder '{conf.id}' varied across runs ({dist}){extra}.")
            )
    not_measured = int((inc["latency_method"] == "not_measured").sum())
    if not_measured:
        out.append(Limitation("data", f"Response time was not measured in {not_measured} run(s)."))
    no_label = int((inc["label_class"] == "not_displayed").sum())
    if no_label:
        out.append(
            Limitation("data", f"No model label was displayed or recorded in {no_label} run(s).")
        )
    unrecognized = int((inc["label_class"] == "unrecognized").sum())
    if unrecognized:
        out.append(
            Limitation(
                "data",
                f"{unrecognized} displayed label(s) did not match any pre-registered label or "
                "alias; they are listed verbatim and not interpreted.",
            )
        )
    if not inc["new_conversation"].all():
        out.append(
            Limitation(
                "data",
                f"{int((~inc['new_conversation'].astype(bool)).sum())} run(s) were not sent in a new "
                "conversation.",
            )
        )
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def build_claims(analysis: Analysis) -> list[Claim]:
    d = analysis.definition
    claims: list[Claim] = []
    for fact in d.documented_facts:
        claims.append(
            Claim(
                "documented_fact",
                fact.statement,
                f"{fact.source_url} (accessed {fact.accessed_on})",
            )
        )
    for hyp in d.hypotheses:
        claims.append(Claim("hypothesis", f"{hyp.id}: {hyp.statement.strip()}", "experiment.yaml"))
    origin = "synthetic demo data" if analysis.synthetic else "observed in this study"
    if analysis.has_data:
        for row in analysis.conditions.itertuples():
            if row.n_scored:
                claims.append(
                    Claim(
                        "observation",
                        f"Condition {row.condition_id}: {row.n_correct} of {row.n_scored} scored "
                        f"responses correct ({_pct(row.accuracy)}, {int(analysis.settings.confidence * 100)}% "
                        f"CI {_pct(row.accuracy_low)}–{_pct(row.accuracy_high)}).",
                        origin,
                    )
                )
        for assessment in analysis.overrides:
            claims.append(
                Claim(
                    "observation",
                    f"Condition {assessment.condition_id}: {assessment.label_summary}",
                    origin,
                )
            )
    for item in d.not_observable:
        claims.append(Claim("not_verified", item, "not observable in this experiment"))
    claims.append(
        Claim(
            "not_verified",
            "Which model, routing path, or reasoning effort actually produced each response.",
            "behavioural patterns are consistent or inconsistent with hypotheses; they do not "
            "establish internals",
        )
    )
    return claims


# -- entry point ----------------------------------------------------------------------------------


def analyze(study: Study, settings: AnalysisSettings | None = None) -> Analysis:
    if settings is None:
        seed = study.plan.seed if study.plan else 0
        settings = AnalysisSettings(seed=seed)
    definition = study.definition
    results = build_results(study)
    inc = results[~results["excluded"]] if not results.empty else results
    measured = inc[inc["latency_method"] != "not_measured"] if not inc.empty else inc
    primary = measured["latency_method"].mode().iloc[0] if not measured.empty else None

    conditions = summarize_conditions(definition, study.plan, results, settings, primary)
    labels = summarize_labels(definition, results)
    analysis = Analysis(
        definition=definition,
        plan=study.plan,
        settings=settings,
        synthetic=study.is_synthetic(),
        results=results,
        conditions=conditions,
        latency=summarize_latency(definition, results),
        tasks=summarize_tasks(definition, results),
        grid=repetition_grid(definition, study.plan, results),
        difficulty=summarize_difficulty(definition, results),
        labels=labels,
        comparisons=pairwise_comparisons(definition, results, settings, primary),
        overrides=assess_overrides(definition, results, labels, settings, primary),
        environment=describe_environment(study, results),
        validation=validate_study(study),
        primary_latency_method=primary,
    )
    analysis.limitations = build_limitations(analysis)
    analysis.claims = build_claims(analysis)
    return analysis
