"""Markdown report: research question → hypotheses → method → results → limits.

The report never contains a number that is not computed from recorded data,
and when there is no data it says so instead of rendering empty tables.
Interpretive sentences come from the templates in :mod:`mblab.analysis`.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from mblab import __version__
from mblab.analysis import LABEL_CLASS_TEXT, LABEL_CLASSES, Analysis, hypothesis_tallies
from mblab.demo import DEMO_BANNER
from mblab.formatting import (
    COMPARISON_STATUS,
    METRIC_NAMES,
    METRIC_UNITS,
    VERDICT_SHORT,
    interval_text,
    method_text,
    metric_value,
    pct,
    pct_ci,
    position_text,
    seconds,
    tokens,
)
from mblab.pilot import PILOT_BANNER
from mblab.schema import (
    DateScoring,
    ExactMatchScoring,
    ManualScoring,
    NumericScoring,
    RegexScoring,
    SequenceScoring,
)
from mblab.stats import Interval
from mblab.study import GRADES_FILE, RECORDS_FILE, Study

SCOPE_NOTE = (
    "**Scope of claims.** This report describes what the client displayed and how the "
    "responses behaved in this sample. It does not show which model, routing path, or "
    "reasoning effort produced a response — those are not observable in this experiment."
)
API_SCOPE_NOTE = (
    "**Scope of claims.** This report describes the requests sent through the provider's "
    "official API, the values the API returned (model id, status, token counts), and how the "
    "responses behaved in this sample. API-reported values are recorded as returned and not "
    "independently verified; how the provider applies a reasoning-effort setting internally is "
    "not observable in this experiment."
)
NO_DATA = (
    "_No observations have been recorded yet, so this report intentionally contains no "
    "result tables or figures._"
)


def _cell(value: object) -> str:
    text = "—" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(_cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scoring_text(task) -> str:
    rule = task.scoring
    if isinstance(rule, NumericScoring):
        tol = f" ± {rule.abs_tolerance:g}" if rule.abs_tolerance else ""
        if isinstance(rule.expected, list):
            numbers = ", ".join(f"{v:g}" for v in rule.expected)
            return f"numeric: exactly the numbers {numbers}{tol}, in this order"
        return f"numeric: one number equal to {rule.expected:g}{tol}"
    if isinstance(rule, DateScoring):
        return f"date: exactly one YYYY-MM-DD date, equal to {rule.expected.isoformat()}"
    if isinstance(rule, SequenceScoring):
        return "sequence: " + ", ".join(rule.expected) + ", each named once, in this order"
    if isinstance(rule, RegexScoring):
        return f"regex on the answer line: `{rule.pattern}`"
    if isinstance(rule, ExactMatchScoring):
        return "exact match after normalization (" + ", ".join(rule.normalize) + ")"
    if isinstance(rule, ManualScoring):
        return f"manual, {rule.max_points:g} points, pass ≥ {rule.pass_threshold:g}: {rule.rubric}"
    return rule.method


def _is_api(a: Analysis) -> bool:
    return a.definition.execution_method == "official_api"


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} ({v})" for k, v in sorted(counts.items())) or "—"


def _interval(row: object) -> Interval | None:
    diff = getattr(row, "difference", None)
    if diff is None or pd.isna(diff):
        return None
    return Interval(diff, row.low, row.high)  # type: ignore[union-attr]


# -- sections --------------------------------------------------------------------------------


def _header(a: Analysis, study: Study, now: datetime) -> list[str]:
    d = a.definition
    planned = len(a.plan.runs) if a.plan else 0
    origin = "synthetic demo data" if a.synthetic else "observed by the experimenter"
    if d.pilot:
        origin += ", pilot study"
    out = [f"# {d.title}", ""]
    if a.synthetic:
        out += [f"> **{DEMO_BANNER}**", ""]
    if d.pilot:
        out += [f"> **{PILOT_BANNER}**", ""]
    departures = a.settings.departures(d)
    if departures:
        out += [
            "> **NOT THE PRE-REGISTERED ANALYSIS:** "
            + "; ".join(departures)
            + ". The settings fixed in experiment.yaml were overridden on the command line, so "
            "every result below is exploratory.",
            "",
        ]
    if a.validation.errors:
        out += [
            f"> **VALIDATION ERRORS ({len(a.validation.errors)}).** Records that cannot be "
            "matched to the plan are left out of every number below. Fix the errors listed in "
            "Appendix B (`mblab validate`) before relying on or sharing this report.",
            "",
        ]
    out += [
        md_table(
            ["Item", "Value"],
            [
                ["Experiment", f"`{d.experiment_id}` v{d.experiment_version}"],
                ["Definition SHA-256", f"`{d.sha256()}`"],
                [
                    "Generated",
                    f"{now.isoformat(timespec='seconds')} by model-behavior-lab {__version__}",
                ],
                ["Data", f"{len(a.results)} recorded of {planned} planned runs ({origin})"],
            ],
        ),
        "",
        API_SCOPE_NOTE if _is_api(a) else SCOPE_NOTE,
        "",
    ]
    return out


def _research_and_hypotheses(a: Analysis) -> list[str]:
    d = a.definition
    out = ["## 1. Research question", ""]
    out += [f"- {q.strip()}" for q in d.research_questions]
    out += ["", "## 2. Hypotheses", ""]
    out += [
        "Pre-registered in `experiment.yaml` (fixed before data collection; the definition hash "
        "above identifies the exact version).",
        "",
    ]
    if not d.hypotheses:
        out.append("_No hypotheses were declared._")
    for h in d.hypotheses:
        out.append(f"- **{h.id}** — {h.statement.strip()}")
        for p in h.predictions:
            out.append(f"  - Prediction: {p}")
    out += ["", "**Documented facts** (verified in official documentation):", ""]
    if d.documented_facts:
        out.append(
            md_table(
                ["Statement", "Source", "Accessed"],
                [
                    [f.statement, f.source_url, f.accessed_on.isoformat()]
                    for f in d.documented_facts
                ],
            )
        )
    else:
        out.append(
            "_None recorded. Nothing in this report should be read as a documented fact about "
            "the service._"
        )
    out.append("")
    return out


def _environment(a: Analysis) -> list[str]:
    d, env = a.definition, a.environment
    versions = ", ".join(f"{v} ({n} runs)" for v, n in env["client_versions"].items()) or "—"
    dates = f"{env['first_run']} → {env['last_run']}" if env["first_run"] else "—"
    method = {
        "manual_ui": "manual_ui — the experimenter operated the official client by hand and "
        "typed in what it displayed; nothing was automated or intercepted",
        "official_api": "official_api — requests sent by `mblab run-api` through the "
        "provider's official SDK; responses and usage recorded as the API returned them",
    }[d.execution_method]
    rows = [
        ["Service", d.service],
        ["Platform", d.platform],
        ["Execution method", method],
        ["SDK version(s)" if d.api else "Client version(s)", versions],
        ["Runs sent", dates],
    ]
    if d.api:
        reported: dict[str, int] = {}
        for models in a.api["models"]:
            for model, n in models.items():
                reported[model] = reported.get(model, 0) + n
        rows += [
            ["Model id requested", f"`{d.api.model}`"],
            ["Model id(s) reported by the API", _counts(reported)],
            ["max_output_tokens", d.api.max_output_tokens or "not set"],
            ["Stored by the provider", "no (store=false)"],
        ]
    else:
        rows.append(["Recorders", ", ".join(env["recorders"]) or "—"])
    out = ["## 3. Experimental environment", "", md_table(["Item", "Value"], rows), ""]
    if d.confounders:
        rows = []
        for conf in d.confounders:
            dist = env["confounders"].get(conf.id, {})
            rows.append(
                [
                    conf.id,
                    conf.question,
                    ", ".join(f"{k}: {v}" for k, v in sorted(dist.items())) or "—",
                ]
            )
        out += [
            "Recorded confounders:",
            "",
            md_table(["Id", "Question", "Values (runs)"], rows),
            "",
        ]
    if d.latency_protocol:
        out += [f"Response-time protocol: {d.latency_protocol.strip()}", ""]
    return out


def _conditions(a: Analysis) -> list[str]:
    d = a.definition
    rows = []
    for cond, summary in zip(d.conditions, a.conditions.itertuples(), strict=True):
        rows.append(
            [
                cond.id,
                d.option_label(cond.model_selection),
                (
                    f"`{d.option(cond.model_selection).reasoning_effort}`"
                    if d.api
                    else d.option_label(cond.message_override)
                ),
                cond.role,
                ", ".join(cond.hypothesis_ids) or "—",
                summary.n_planned,
                summary.n_recorded,
                summary.n_excluded,
            ]
        )
    seed = a.plan.seed if a.plan else d.randomization.seed
    strategy = a.plan.strategy if a.plan else d.randomization.strategy
    return [
        "## 4. Experimental conditions",
        "",
        md_table(
            [
                "Condition",
                "Setting" if d.api else "Default selection",
                "reasoning.effort sent" if d.api else "Per-message selection",
                "Role",
                "Hypotheses",
                "Planned",
                "Recorded",
                "Excluded",
            ],
            rows,
        ),
        "",
        f"{len(d.conditions)} conditions × {len(d.tasks)} tasks × {d.repetitions} repetitions = "
        f"{d.planned_run_count} runs, "
        + ("one independent API request" if d.api else "one new conversation")
        + " per run, in a randomized order "
        f"(seed {seed}, strategy `{strategy}`)."
        + (
            ""
            if a.order is None
            else " The runs were done in that order."
            if a.order.share == 1
            else f" The runs followed that order for {a.order.share:.0%} of run pairs."
        ),
        "",
    ]


def _method(a: Analysis) -> list[str]:
    d, s = a.definition, a.settings
    rows = [
        [t.id, t.title, t.difficulty or "—", scoring_text(t), t.reference_answer or "—"]
        for t in d.tasks
    ]
    n_manual = sum(1 for t in d.tasks if isinstance(t.scoring, ManualScoring))
    blinding = (
        "All tasks are scored automatically by pre-registered rules, which never see the condition."
        if n_manual == 0
        else f"{n_manual} task(s) are graded by hand from a blind sheet without condition "
        "metadata (response style can still hint at the condition)."
    )
    conf = int(s.confidence * 100)
    return [
        "## 5. Evaluation method",
        "",
        md_table(["Task", "Title", "Difficulty", "Scoring rule", "Reference answer"], rows),
        "",
        "Difficulty is the task author's prior judgement, not a calibrated rating.",
        "",
        "- **Accuracy** — correct answers among scored responses. A response without a usable "
        "answer line is a *format failure* and counts as incorrect; format failures are also "
        "reported separately.",
        "- **Task success rate** — correct answers among all included runs; errors, refusals, and "
        "incomplete responses count as failures.",
        "- **Response time** — summarised per measurement method and never pooled across methods.",
        *(
            [
                "- **Reported reasoning tokens** — `usage.output_tokens_details.reasoning_tokens` "
                "as the API returned it; runs without a reported count are left out, not "
                "counted as zero.",
                "- **Reported model id and effort** — the `model` and `reasoning.effort` values "
                "in each API response, copied as returned.",
            ]
            if _is_api(a)
            else [
                "- **Thinking indicator** — whether the client showed a thinking/reasoning "
                "indicator, and the duration it displayed (copied verbatim).",
                "- **Displayed label** — the model name the client showed, classified against "
                "pre-registered labels and aliases as matching the per-message selection, the "
                "default, both, another option, or none.",
            ]
        ),
        f"- {blinding}",
        f"- Intervals: {conf}% Wilson (proportions), Newcombe hybrid score (differences of "
        f"proportions), percentile bootstrap with {s.bootstrap_resamples} resamples and seed "
        f"{s.seed} (differences of medians). Groups with fewer than {s.min_n} observations are "
        "not compared.",
        f"- Per-message vs. default: a condition counts as consistent with a reference only if "
        f"the interval for their difference lies within ±{s.equivalence_fraction:.0%} of the gap "
        "between the two reference conditions (a pre-registered equivalence margin). An interval "
        "that merely includes 0 is inconclusive. The position column places the condition on a "
        "scale where the default-only reference is 0 and the per-message reference is 1.",
        "- Runs with a recorded protocol deviation are excluded from the primary analysis and "
        "listed in Appendix A.",
        "",
    ]


def _results(a: Analysis) -> list[str]:
    out = ["## 6. Results", ""]
    if not a.has_data:
        return [*out, NO_DATA, ""]
    conf = int(a.settings.confidence * 100)
    planned = len(a.plan.runs) if a.plan else 0
    inc = a.included
    out += [
        "### 6.1 Data completeness",
        "",
        md_table(
            ["Planned", "Recorded", "Excluded", "Missing", "Not completed", "Format failures"],
            [
                [
                    planned,
                    len(a.results),
                    int(a.results["excluded"].sum()),
                    max(planned - len(a.results), 0),
                    int((inc["response_status"] != "completed").sum()),
                    int((inc["score_status"] == "format_failure").sum()),
                ]
            ],
        ),
        "",
        "### 6.2 Condition comparison",
        "",
    ]
    api = _is_api(a)
    api_by = a.api.set_index("condition_id") if api else None
    rows = []
    for r in a.conditions.itertuples():
        if api_by is not None:
            reasoning = api_by.loc[r.condition_id]
            last = f"{tokens(reasoning.reasoning_tokens_median)} (n={reasoning.reasoning_tokens_n})"
        else:
            last = f"{pct(r.thinking_shown_rate)} (n={r.thinking_known_n})"
        rows.append(
            [
                r.condition_id,
                r.model_selection,
                *([] if api else [r.message_override]),
                f"{r.n_correct}/{r.n_scored}",
                pct(r.accuracy),
                pct_ci(r.accuracy_low, r.accuracy_high),
                pct(r.success_rate),
                r.n_format_failures,
                "—"
                if r.repetition_accuracy_sd is None or pd.isna(r.repetition_accuracy_sd)
                else f"{100 * r.repetition_accuracy_sd:.0f} pp",
                f"{seconds(r.latency_median)} (n={r.latency_n})",
                last,
            ]
        )
    out += [
        md_table(
            [
                "Cond.",
                *(["Setting"] if api else ["Default", "Per-message"]),
                "Correct",
                "Accuracy",
                f"{conf}% CI",
                "Success",
                "Format fail.",
                "SD across reps",
                f"Median time ({method_text(a.primary_latency_method)})",
                "Median reasoning tokens" if api else "Thinking shown",
            ],
            rows,
        ),
        "",
        "### 6.3 Response time by measurement method",
        "",
    ]
    if a.latency.empty:
        out += ["_No response times were measured._", ""]
    else:
        out += [
            md_table(
                ["Cond.", "Method", "n", "Median", "Q1", "Q3", "Min", "Max"],
                [
                    [
                        r.condition_id,
                        method_text(r.latency_method),
                        r.n,
                        seconds(r.median),
                        seconds(r.q1),
                        seconds(r.q3),
                        seconds(r.min),
                        seconds(r.max),
                    ]
                    for r in a.latency.itertuples()
                ],
            ),
            "",
        ]
    out += _api_results(a) if api else _label_results(a)
    out += ["### 6.6 Pairwise differences (descriptive)", ""]
    return out + _rest_of_results(a, conf)


def _api_results(a: Analysis) -> list[str]:
    rows = [
        [
            r.condition_id,
            f"`{r.requested_effort}`",
            f"{r.n_api_values}/{r.n_included}",
            _counts(r.models),
            r.n_effort_mismatch,
            _counts(r.incomplete_reasons),
            f"{tokens(r.reasoning_tokens_median)} [{tokens(r.reasoning_tokens_q1)}–"
            f"{tokens(r.reasoning_tokens_q3)}] (n={r.reasoning_tokens_n})",
            tokens(r.output_tokens_median),
            f"{tokens(r.input_tokens_sum)} / {tokens(r.output_tokens_sum)} (n={r.usage_n})",
        ]
        for r in a.api.itertuples()
    ]
    return [
        "### 6.4 Values reported by the API",
        "",
        md_table(
            [
                "Cond.",
                "Effort sent",
                "Runs with API values",
                "Model id(s) reported",
                "Echoed effort ≠ sent",
                "Incomplete (reason)",
                "Reasoning tokens: median [Q1–Q3]",
                "Median output tokens",
                "Input / output tokens, all recorded responses",
            ],
            rows,
        ),
        "",
        "Values are copied from the API responses and not independently verified. Reasoning "
        "tokens are part of the output tokens. Token totals are listed so cost can be worked out "
        "from the provider's price list; this report does not price anything.",
        "",
        "### 6.5 Per-message selection vs. default",
        "",
        "_Not applicable: every API request carries exactly one setting, so there is no "
        "per-message selection._",
        "",
    ]


def _label_results(a: Analysis) -> list[str]:
    conf = int(a.settings.confidence * 100)
    out = [
        "### 6.4 Client-displayed model labels",
        "",
        md_table(
            ["Cond.", *[LABEL_CLASS_TEXT[c] for c in LABEL_CLASSES]],
            [
                [r.condition_id, *[getattr(r, c) for c in LABEL_CLASSES]]
                for r in a.labels.itertuples(index=False)
            ],
        ),
        "",
        "Labels are what the client displayed. They are not checked against a server-side record.",
        "",
        "### 6.5 Per-message selection vs. default",
        "",
    ]
    if not a.overrides:
        out += ["_No condition uses a per-message selection._", ""]
    for o in a.overrides:
        kind = (
            "the same setting chosen twice"
            if o.stacked
            else "per-message selection differs from the default"
        )
        out += [f"**Condition {o.condition_id}** ({kind})", "", f"- Labels: {o.label_summary}"]
        if o.note:
            out += [f"- {o.note}", ""]
            continue
        out += [
            "",
            md_table(
                [
                    "Metric",
                    o.condition_id,
                    f"Default-only ref. ({o.default_ref})",
                    f"Per-message ref. ({o.override_ref})",
                    f"vs. default ref. [{conf}% CI]",
                    f"vs. per-message ref. [{conf}% CI]",
                    f"Position, 0 = default, 1 = per-message [{conf}% CI]",
                    "Assessment",
                ],
                [
                    [
                        m.metric,
                        metric_value(m.condition_value, m.unit),
                        metric_value(m.default_ref_value, m.unit),
                        metric_value(m.override_ref_value, m.unit),
                        interval_text(m.vs_default, m.unit),
                        "—" if o.stacked else interval_text(m.vs_override, m.unit),
                        "—" if o.stacked else position_text(m.position),
                        VERDICT_SHORT[m.verdict],
                    ]
                    for m in o.metrics
                ],
            ),
            "",
        ]
        out += [f"- *{m.metric}*: {m.explanation}" for m in o.metrics]
        out.append("")
    return out + confounder_lines(a)


def confounder_lines(a: Analysis) -> list[str]:
    """Flagged confounder values per condition, and the sensitivity check (Markdown)."""
    out: list[str] = []
    flags = a.confounder_flags
    if not flags.empty and int(flags["n_flagged"].sum()):
        conds = [c.id for c in a.definition.conditions]
        rows = []
        for conf_id, group in flags.groupby("confounder_id", sort=False):
            by_cond = {r.condition_id: f"{r.n_flagged}/{r.n_runs}" for r in group.itertuples()}
            rows.append([conf_id, *(by_cond.get(c, "—") for c in conds)])
        out += [
            "**Confounders by condition** — runs with a value other than the recommended one:",
            "",
            md_table(["Confounder", *conds], rows),
            "",
        ]
    check = a.sensitivity
    if check is not None:
        reasons = ", ".join(f"{k} ({n})" for k, n in check.reasons.items())
        head = (
            f"**Sensitivity check.** Leaving out the {check.left_out} run(s) with a flagged "
            f"per-run value ({reasons}) "
        )
        if check.changes:
            out += [
                head + f"changes {len(check.changes)} of {check.n_assessments} assessments:",
                "",
                *(
                    f"- {cond}, {metric}: {VERDICT_SHORT[before]} → {VERDICT_SHORT[after]}"
                    for cond, metric, before, after in check.changes
                ),
                "",
            ]
        else:
            out += [head + f"changes none of the {check.n_assessments} assessments.", ""]
    return out


def _rest_of_results(a: Analysis, conf: int) -> list[str]:
    out: list[str] = []
    if a.comparisons.empty:
        out += ["_Not enough data._", ""]
    else:
        out += [
            md_table(
                ["Metric", "First", "Second", "n", f"First − second [{conf}% CI]", "Status"],
                [
                    [
                        METRIC_NAMES[r.metric],
                        r.condition_a,
                        r.condition_b,
                        f"{r.n_a}/{r.n_b}",
                        interval_text(_interval(r), METRIC_UNITS[r.metric]),
                        COMPARISON_STATUS[r.status],
                    ]
                    for r in a.comparisons.itertuples()
                ],
            ),
            "",
            "These are descriptions of this sample, not significance tests; with "
            f"{len(a.comparisons)} comparisons some intervals will exclude 0 by chance.",
            "",
        ]
    out += ["### 6.7 Task-level results", ""]
    tasks = a.tasks
    task_rows = []
    for task in a.definition.tasks:
        row = [task.id, task.title, task.difficulty or "—"]
        for cond in a.definition.conditions:
            cell = tasks[(tasks["task_id"] == task.id) & (tasks["condition_id"] == cond.id)].iloc[0]
            row.append(f"{cell.n_correct}/{cell.n_scored}" if cell.n_scored else "—")
        task_rows.append(row)
    out += [
        md_table(
            ["Task", "Title", "Difficulty", *[c.id for c in a.definition.conditions]], task_rows
        ),
        "",
        "Cells show correct/scored responses.",
        "",
        "### 6.8 Repetition consistency",
        "",
    ]
    cons_rows = []
    for r in tasks.itertuples():
        if r.n_scored == 0:
            continue
        consistent = (
            "—"
            if r.consistent_outcome is None or pd.isna(r.consistent_outcome)
            else ("yes" if r.consistent_outcome else "no")
        )
        cons_rows.append(
            [
                r.condition_id,
                r.task_id,
                r.n_scored,
                consistent,
                r.distinct_answers,
                pct(r.majority_answer_share),
            ]
        )
    out += [
        md_table(
            [
                "Cond.",
                "Task",
                "Scored",
                "Same outcome every repetition",
                "Distinct answers",
                "Most common answer share",
            ],
            cons_rows,
        ),
        "",
        "### 6.9 By difficulty",
        "",
    ]
    diff_rows = [
        [r.condition_id, r.difficulty, f"{r.n_correct}/{r.n_scored}", pct(r.accuracy)]
        for r in a.difficulty.itertuples()
    ]
    out += [md_table(["Cond.", "Difficulty", "Correct", "Accuracy"], diff_rows), ""]
    return out


def _interpretation(a: Analysis) -> list[str]:
    out = ["## 7. Interpretation", ""]
    if not a.has_data:
        return [*out, "_Nothing to interpret until observations are recorded._", ""]
    if a.synthetic:
        out += [
            "_The data is synthetic, so the statements below only demonstrate the report "
            "format; they describe random numbers._",
            "",
        ]
    for o in a.overrides:
        out.append(f"- **Condition {o.condition_id}.** {o.label_summary}")
        for m in o.metrics:
            if m.verdict != "insufficient_data":
                out.append(f"  - {m.metric}: {m.explanation}")
        if o.metrics and all(m.verdict == "insufficient_data" for m in o.metrics):
            out.append("  - Every behavioural metric has too little data to compare yet.")
    for tally in hypothesis_tallies(a):
        out.append(
            f"- **{tally.hypothesis_id}** {tally.sentence().removeprefix(tally.hypothesis_id + ' ')}"
        )
    if _is_api(a):
        out += [
            "- Differences between effort levels describe this sample: these six tasks, this "
            "model id, this account, and these dates. Reported token counts are the provider's "
            "figures, recorded as returned.",
            "- Confidence would improve with more repetitions and tasks, other model ids, and "
            "replication on other dates.",
            "",
        ]
        return out
    out += [
        "- A pattern described as *consistent with* a selection is compatible with that "
        "selection being applied in this sample. It does not establish which model, routing "
        "path, or reasoning effort the service used.",
        "- Confidence would improve with more repetitions and tasks, a second experimenter, "
        "other platforms or client versions, and replication on other dates.",
        "",
    ]
    return out


def _limitations(a: Analysis) -> list[str]:
    out = ["## 8. Limitations", ""]
    out += [f"- {lim.text}" for lim in a.limitations if lim.kind == "data"]
    out += [f"- {lim.text}" for lim in a.limitations if lim.kind == "standard"]
    return [*out, ""]


def _claims(a: Analysis) -> list[str]:
    names = {
        "documented_fact": "Documented fact",
        "hypothesis": "Hypothesis (not tested as true)",
        "observation": "Observation (this sample)",
        "not_verified": "Not verified / not observable",
    }
    return [
        "## 9. Claims ledger",
        "",
        "What each statement in this report is based on.",
        "",
        md_table(
            ["Category", "Statement", "Basis"],
            [[names[c.category], c.text, c.source] for c in a.claims],
        ),
        "",
    ]


def _reproduction(a: Analysis, study: Study) -> list[str]:
    plan = a.plan
    rows = [
        ["experiment.yaml (parsed) SHA-256", f"`{a.definition.sha256()}`"],
        ["plan.json definition hash", f"`{plan.definition_sha256}`" if plan else "—"],
        ["records.jsonl SHA-256", f"`{_sha256_file(study.root / RECORDS_FILE) or '—'}`"],
        ["grades.jsonl SHA-256", f"`{_sha256_file(study.root / GRADES_FILE) or '—'}`"],
        ["Plan seed / strategy", f"{plan.seed} / `{plan.strategy}`" if plan else "—"],
        ["Tool version", f"model-behavior-lab {__version__}"],
    ]
    return [
        "## 10. Reproduction",
        "",
        md_table(["Item", "Value"], rows),
        "",
        "With the study directory (`experiment.yaml`, `plan.json`, `records.jsonl`, and "
        "`grades.jsonl` if present):",
        "",
        "```bash",
        'pip install "git+https://github.com/cynkai/model-behavior-lab"',
        "mblab validate -s <study-dir>",
        "mblab analyze -s <study-dir>",
        "mblab report -s <study-dir>",
        "```",
        "",
        (
            "To run the experiment again: `mblab init --template openai-reasoning-effort`, copy "
            "the same `experiment.yaml`, then `mblab plan` (same seed → same order) and "
            "`mblab run-api`. Responses will differ from run to run; compare distributions."
            if _is_api(a)
            else "To run the experiment again: `mblab init`, copy the same `experiment.yaml`, "
            "then `mblab plan` (same seed → same order) and follow docs/experiment-protocol.md."
        ),
        "",
    ]


def _appendix(a: Analysis) -> list[str]:
    out = ["## Appendix A. Excluded runs", ""]
    excluded = a.results[a.results["excluded"]] if not a.results.empty else a.results
    if excluded.empty:
        out += ["_None._", ""]
    else:
        out += [
            md_table(
                ["Run", "Deviation"],
                [[r.run_id, r.protocol_deviation] for r in excluded.itertuples()],
            ),
            "",
        ]
    out += ["## Appendix B. Validation", ""]
    issues = [i for i in a.validation.issues if i.severity in {"error", "warning"}]
    if not issues:
        out += ["No errors or warnings from `mblab validate`.", ""]
    else:
        out += [
            md_table(
                ["Severity", "Code", "Message"], [[i.severity, i.code, i.message] for i in issues]
            ),
            "",
        ]
    return out


def render_report(a: Analysis, study: Study, now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    parts = (
        _header(a, study, now)
        + _research_and_hypotheses(a)
        + _environment(a)
        + _conditions(a)
        + _method(a)
        + _results(a)
        + _interpretation(a)
        + _limitations(a)
        + _claims(a)
        + _reproduction(a, study)
        + _appendix(a)
    )
    if a.synthetic:
        parts += ["---", "", f"**{DEMO_BANNER}**", ""]
    return "\n".join(parts).rstrip() + "\n"
