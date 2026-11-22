"""Model Behavior Lab dashboard (Streamlit).

Run with ``mblab dashboard -s <study-dir>``. Pages:

* Overview — study, hypotheses, progress, validation
* Run plan & recording — the randomized order, the exact prompt, an entry form
* Condition comparison — accuracy, response time, labels, override assessments
* Task-level results — per-task accuracy and every repetition
* Individual responses — filter and read single runs
* Methodology & limitations — what is and is not measured, claims ledger

A benchmark directory (``mblab dashboard -s benchmarks/<id>``) gets one page instead: its
runs, the checks of each answer against the verified ground truth, and the comparisons.

Streamlit renders text, captions, alerts, and widget labels as Markdown. Every value
taken from a study (definition, plan, records, or messages built from them) goes through
``md_text`` first, so a study file from someone else cannot add links, remote images, or
formatting. Full texts (prompts, responses) are shown with ``st.code``, which is literal.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from mblab import api_runner
from mblab.analysis import LABEL_CLASS_TEXT, Analysis, analyze, hypothesis_tallies
from mblab.benchmark import (
    BENCHMARK_FILE,
    RUNS_FILE,
    Benchmark,
    Evaluation,
    RecordingCoverage,
    breakdown_frame,
    evaluate,
    is_benchmark_dir,
    runs_frame,
    selection_grid,
)
from mblab.dashboard import PAGES, charts
from mblab.demo import DEMO_BANNER
from mblab.exchange import merge_records
from mblab.formatting import (
    COMPARISON_STATUS,
    METRIC_NAMES,
    METRIC_UNITS,
    VERDICT_SHORT,
    interval_text,
    md_text,
    method_text,
    metric_value,
    pct,
    pct_ci,
    position_text,
    seconds,
    tokens,
)
from mblab.pilot import PILOT_BANNER
from mblab.provenance import code_version
from mblab.records import build_record, condition_instructions, next_pending, parse_decimal
from mblab.report import scoring_text
from mblab.stats import Interval
from mblab.study import UNRECORDED_FILE, Study, StudyError, format_validation_error

SCOPE = (
    "These pages describe what the client displayed and how responses behaved in this sample. "
    "They are not evidence about server-side model routing — see *Methodology & limitations*."
)
DOCS = "https://github.com/cynkai/model-behavior-lab/blob/main/docs"


def _initial_study_dir() -> str:
    args = sys.argv[1:]
    if "--study" in args and args.index("--study") + 1 < len(args):
        return args[args.index("--study") + 1]
    return os.environ.get("MBLAB_STUDY", ".")


def _stamp(root: Path) -> tuple:
    names = ("experiment.yaml", "plan.json", "records.jsonl", "grades.jsonl", UNRECORDED_FILE)
    return tuple((n, (root / n).stat().st_mtime_ns if (root / n).exists() else 0) for n in names)


@st.cache_resource(show_spinner=False, max_entries=4)
def _load(root: str, stamp: tuple) -> tuple[Study, Analysis]:
    study = Study.load(root)
    return study, analyze(study)


def _mode() -> str | None:
    try:
        return st.context.theme.type
    except Exception:  # pragma: no cover - older Streamlit
        return None


def _table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def _chart(chart, table: pd.DataFrame, caption: str | None = None) -> None:
    if chart is None:
        st.info("Nothing to plot yet for this view.")
        return
    # theme=None keeps the chart's own validated palette and config.
    st.altair_chart(chart, width="stretch", theme=None)
    if caption:
        st.caption(caption)
    with st.expander("Table view"):
        _table(table)


def _banner(a: Analysis) -> None:
    if a.synthetic:
        st.error(f"**{DEMO_BANNER}**", icon="⚠️")
    if a.definition.pilot:
        st.warning(f"**{PILOT_BANNER}**", icon="🧪")


# -- pages ---------------------------------------------------------------------------------


def page_overview(study: Study, a: Analysis) -> None:
    d = a.definition
    st.title(md_text(d.title))
    st.caption(
        f"`{d.experiment_id}` v{md_text(d.experiment_version)} · service: {md_text(d.service)} "
        f"· platform: {md_text(d.platform)} · method: {d.execution_method}"
    )
    _banner(a)
    planned = len(a.plan.runs) if a.plan else 0
    recorded = len(a.results)
    excluded = int(a.results["excluded"].sum()) if recorded else 0
    cols = st.columns(5)
    cols[0].metric("Planned runs", planned)
    cols[1].metric("Recorded", recorded)
    cols[2].metric("Excluded", excluded)
    cols[3].metric("Not yet recorded", max(planned - recorded, 0))
    cols[4].metric(
        "Errors / warnings", f"{len(a.validation.errors)} / {len(a.validation.warnings)}"
    )
    st.subheader("Research question")
    for q in d.research_questions:
        st.markdown(f"- {md_text(q)}")
    st.subheader("Hypotheses (pre-registered)")
    for h in d.hypotheses:
        lines = [f"- **{md_text(h.id)}** — {md_text(h.statement)}"]
        lines += [f"    - Prediction: {md_text(p)}" for p in h.predictions]
        st.markdown("\n".join(lines))
    st.subheader("Documented facts")
    if d.documented_facts:
        _table(pd.DataFrame([f.model_dump(mode="json") for f in d.documented_facts]))
    else:
        st.caption(
            "None recorded in experiment.yaml. Nothing on these pages is a documented fact about "
            "the service."
        )
    st.subheader("Collection progress")
    progress = a.conditions[
        [
            "condition_id",
            "model_selection",
            "message_override",
            "n_planned",
            "n_recorded",
            "n_excluded",
        ]
    ].rename(
        columns={
            "condition_id": "condition",
            "model_selection": "default",
            "message_override": "per-message",
            "n_planned": "planned",
            "n_recorded": "recorded",
            "n_excluded": "excluded",
        }
    )
    _table(progress)
    env = a.environment
    st.subheader("Environment")
    _table(
        pd.DataFrame(
            [
                [
                    "Client version(s)",
                    ", ".join(f"{k} ({v})" for k, v in env["client_versions"].items()) or "—",
                ],
                ["First run", env["first_run"] or "—"],
                ["Last run", env["last_run"] or "—"],
                ["Recorders", ", ".join(env["recorders"]) or "—"],
            ],
            columns=["item", "value"],
        )
    )
    st.subheader("Validation")
    issues = [i for i in a.validation.issues if i.severity != "info"]
    if not issues:
        st.success("No errors or warnings.")
    # Issue codes are mblab's own identifiers; messages quote values from the study.
    for issue in issues:
        (st.error if issue.severity == "error" else st.warning)(
            f"`{issue.code}` {md_text(issue.message)}"
        )
    notes = [i for i in a.validation.issues if i.severity == "info"]
    if notes:
        with st.expander(f"{len(notes)} note(s)"):
            for issue in notes:
                st.markdown(f"- `{issue.code}` {md_text(issue.message)}")


def page_record(study: Study, a: Analysis) -> None:
    d = a.definition
    st.title("Run plan & recording")
    _banner(a)
    if a.plan is None:
        st.warning("No plan yet. Run `mblab plan -s <study>` first.")
        return
    recorded = {r.run_id: r for r in study.records}
    held = api_runner.held_runs(study)  # the runs `mblab run-api` skips
    table = pd.DataFrame(
        [
            {
                "#": run.sequence,
                "run": run.run_id,
                "block": run.block,
                "default": d.option_label(run.model_selection),
                "per-message": d.option_label(run.message_override),
                "task": run.task_id,
                "status": (
                    "excluded"
                    if run.run_id in recorded and recorded[run.run_id].excluded
                    else "recorded"
                    if run.run_id in recorded
                    else "held back"
                    if run.run_id in held.run_ids
                    else "pending"
                ),
            }
            for run in sorted(a.plan.runs, key=lambda r: r.sequence)
        ]
    )
    if held.log_errors:
        st.warning(md_text(held.describe_log_errors()))
    if held.runs:
        st.warning(md_text(held.describe()))
    pending_ids = [r for r in table["run"] if r not in recorded and r not in held.run_ids]
    st.subheader("Next run")
    if not pending_ids:
        st.success(
            "All planned runs are recorded."
            if not held.runs
            else "No run is left to send: every planned run without a record is held back."
        )
    choices = pending_ids + [r for r in table["run"] if r in held.run_ids or r in recorded]
    if not choices:
        return
    # The selection stays put while the page reruns (e.g. when another command saves a
    # run), so a form being filled in never switches to a different run underneath you.
    # After a save it moves on to the next pending run.
    advance = st.session_state.pop("mblab_select_next", None)
    if advance in choices:
        st.session_state["run-select"] = advance
    elif st.session_state.get("run-select") not in choices:
        st.session_state["run-select"] = choices[0]
    run_id = st.selectbox(
        "Run",
        choices,
        key="run-select",
        help="Defaults to the next pending run in the randomized order.",
    )
    run = a.plan.by_run_id()[run_id]
    task = d.task(run.task_id)
    st.markdown(
        f"**#{run.sequence} · {md_text(run.run_id)}** — task {md_text(task.id)}: "
        f"{md_text(task.title)}"
    )
    for number, step in enumerate(condition_instructions(d, run), start=1):
        st.markdown(f"{number}. {md_text(step)}")
    st.caption("Prompt — send exactly this (use the copy button):")
    st.code(task.prompt, language=None, wrap_lines=True)

    if a.synthetic:
        st.info(
            "This is a synthetic demo study; recording is disabled so demo and real data never mix."
        )
    elif d.execution_method == "official_api":
        st.info(
            "This is an official_api study. Its records are written by `mblab run-api` "
            "(preview with `--dry-run`); manual entry is disabled."
        )
    else:
        _record_form(study, a, run_id, run_id in recorded)

    st.subheader("Plan")
    st.caption(
        f"Seed {a.plan.seed}, strategy `{a.plan.strategy}`, generated {a.plan.generated_at:%Y-%m-%d}."
    )
    _table(table)


def _record_form(study: Study, a: Analysis, run_id: str, exists: bool) -> None:
    d = a.definition
    run = a.plan.by_run_id()[run_id]
    prev = max(study.records, key=lambda r: r.recorded_at or r.timestamp, default=None)
    if exists:
        st.warning('This run already has a record. Tick "Replace" in the form to overwrite it.')
    flash = st.session_state.pop("mblab_flash", None)
    if flash:
        st.success(md_text(flash))
    with st.form(f"record-{run_id}", clear_on_submit=False):
        st.markdown("**Record what the client showed**")
        c1, c2 = st.columns(2)
        status = c1.selectbox("Response status", ["completed", "error", "refused", "incomplete"])
        client_version = c2.text_input(
            "Client (app) version", value=prev.client_version if prev else ""
        )
        response = st.text_area("Response text (paste exactly)", height=200)
        c3, c4 = st.columns(2)
        label = c3.text_input("Model label displayed (blank if none)")
        location = c4.text_input("Where was the label displayed?")
        c5, c6 = st.columns(2)
        thinking = c5.selectbox("Thinking indicator", ["unknown", "shown", "not_shown"])
        thinking_text = c6.text_input("Thinking text as shown (e.g. 'Thought for 1m 12s')")
        c7, c8 = st.columns(2)
        latency_text = c7.text_input("Stopwatch seconds (blank = not measured)")
        sent_text = c8.text_input(
            "Time sent (ISO 8601; blank = now minus stopwatch seconds)", value=""
        )
        conf_values: dict[str, str] = {}
        if d.confounders:
            cols = st.columns(min(3, len(d.confounders)))
            for i, conf in enumerate(d.confounders):
                # Per-run observations start empty: what the last run showed is not a
                # default for this one.
                default = (
                    None
                    if conf.per_run
                    else (prev.confounders.get(conf.id) if prev else None)
                    or ("unknown" if "unknown" in conf.allowed_values else conf.allowed_values[0])
                )
                conf_values[conf.id] = cols[i % len(cols)].selectbox(
                    md_text(conf.question),  # widget labels render Markdown, images included
                    conf.allowed_values,
                    index=conf.allowed_values.index(default)
                    if default in conf.allowed_values
                    else None,
                    placeholder="Choose what this run showed",
                )
        new_conversation = st.checkbox("Sent in a new conversation", value=True)
        deviation = st.text_input(
            "Protocol deviation (blank if none — a deviation excludes the run)"
        )
        notes = st.text_input("Notes")
        recorder = st.text_input(
            "Recorder id (pseudonym)", value=(prev.recorded_by if prev and prev.recorded_by else "")
        )
        # Only a box ticked on the form you actually saw can replace a record: if the run
        # gets recorded by another command meanwhile, this box was not there, stays False,
        # and the save is refused instead of silently overwriting that record.
        replace = (
            st.checkbox("Replace the existing record for this run", value=False)
            if exists
            else False
        )
        submitted = st.form_submit_button("Save record", type="primary")
    if not submitted:
        return
    unanswered = [c.question for c in d.confounders if conf_values.get(c.id) is None]
    if unanswered:
        st.error("Not saved: answer " + "; ".join(md_text(q) for q in unanswered))
        return
    try:
        latency = parse_decimal(latency_text, "stopwatch seconds") if latency_text.strip() else None
        if sent_text.strip():
            sent = datetime.fromisoformat(sent_text.strip())
        else:
            sent = datetime.now().astimezone().replace(microsecond=0) - timedelta(
                seconds=round(latency or 0)
            )
        record = build_record(
            run,
            d,
            a.plan,
            client_version=client_version.strip() or "unknown",
            timestamp=sent,
            response_status=status,
            response_text=response,
            observed_model_label=label,
            observed_label_location=location,
            thinking_indicator=thinking,
            thinking_text=thinking_text,
            latency_seconds=latency,
            latency_method="manual_stopwatch" if latency is not None else "not_measured",
            new_conversation=new_conversation,
            confounders=conf_values,
            protocol_deviation=deviation,
            notes=notes,
            recorded_by=recorder,
        )
    except ValidationError as exc:
        st.error(f"Not saved: {md_text(format_validation_error(exc))}")
        return
    except ValueError as exc:
        st.error(f"Not saved: {md_text(exc)}")
        return
    try:
        merged = merge_records(Study.load(study.root), [record], replace=replace, dry_run=False)
    except StudyError as exc:  # e.g. another command is writing to the study right now
        st.error(f"Not saved: {md_text(exc)}")
        st.code(record.model_dump_json(), language="json", wrap_lines=True)
        return
    if merged.conflicts:
        st.error(
            f"Not saved: run {md_text(record.run_id)} already has a record (saved by another command "
            'meanwhile, or "Replace" was not ticked). Your entry is below if you want to '
            "keep it instead."
        )
        st.code(record.model_dump_json(), language="json", wrap_lines=True)
        return
    st.session_state["mblab_flash"] = f"Saved {record.run_id}."
    after = next_pending(a.plan, {r.run_id for r in Study.load(study.root).records})
    st.session_state["mblab_select_next"] = after.run_id if after else None
    st.rerun()


def page_conditions(study: Study, a: Analysis) -> None:
    st.title("Condition comparison")
    _banner(a)
    st.caption(SCOPE)
    if not a.has_scored_data:
        st.info("No scored responses yet. Charts appear once observations are recorded.")
        return
    mode = _mode()
    conf = int(a.settings.confidence * 100)
    api = a.definition.execution_method == "official_api"
    summary = pd.DataFrame(
        [
            {
                "condition": r.condition_id,
                "default": r.model_selection,
                "per-message": r.message_override,
                "correct / scored": f"{r.n_correct}/{r.n_scored}",
                "accuracy": pct(r.accuracy),
                f"{conf}% CI": pct_ci(r.accuracy_low, r.accuracy_high),
                "success rate": pct(r.success_rate),
                "format failures": r.n_format_failures,
                "not completed": r.n_not_completed,
                "SD across repetitions": pct(r.repetition_accuracy_sd),
            }
            for r in a.conditions.itertuples()
        ]
    )
    _chart(
        charts.accuracy_chart(a.conditions, mode, a.settings.confidence),
        summary,
        f"Dots: accuracy; lines: {conf}% Wilson intervals; right: correct/scored. Format failures "
        "count as incorrect. Intervals treat responses as independent and describe this task set only.",
    )
    st.subheader("Response time")
    lat_chart = charts.latency_chart(a.results, a.conditions, a.primary_latency_method, mode)
    lat_table = pd.DataFrame(
        [
            {
                "condition": r.condition_id,
                "method": method_text(r.latency_method),
                "n": r.n,
                "median": seconds(r.median),
                "Q1": seconds(r.q1),
                "Q3": seconds(r.q3),
                "min": seconds(r.min),
                "max": seconds(r.max),
            }
            for r in a.latency.itertuples()
        ]
    )
    _chart(
        lat_chart,
        lat_table,
        "Client-timer latency includes network transfer and provider queueing. It is not "
        "comparable with stopwatch times from UI studies. Methods are never pooled."
        if api
        else "Manual stopwatch timings include reaction, network and rendering time. They are not "
        "comparable with API latency. Methods are never pooled.",
    )
    if api:
        st.subheader("Values reported by the API")
        _table(
            pd.DataFrame(
                [
                    {
                        "condition": r.condition_id,
                        "effort sent": r.requested_effort,
                        "runs with API values": f"{r.n_api_values}/{r.n_included}",
                        "model id(s) reported": ", ".join(
                            f"{k} ({v})" for k, v in sorted(r.models.items())
                        )
                        or "—",
                        "echoed effort ≠ sent": r.n_effort_mismatch,
                        "incomplete": ", ".join(
                            f"{k} ({v})" for k, v in sorted(r.incomplete_reasons.items())
                        )
                        or "—",
                        "reasoning tokens median": tokens(r.reasoning_tokens_median),
                        "Q1–Q3": f"{tokens(r.reasoning_tokens_q1)}–{tokens(r.reasoning_tokens_q3)}",
                        "n": r.reasoning_tokens_n,
                        "input / output tokens, all recorded responses": (
                            f"{tokens(r.input_tokens_sum)} / {tokens(r.output_tokens_sum)}"
                        ),
                    }
                    for r in a.api.itertuples()
                ]
            )
        )
        st.caption(
            "Copied from the API responses as returned, not independently verified. Reasoning "
            "tokens are part of output tokens. No cost is computed."
        )
    else:
        _ui_label_sections(a, mode, conf)
    _comparison_sections(a, conf)


def _ui_label_sections(a: Analysis, mode: str | None, conf: int) -> None:
    st.subheader("Displayed model labels")
    _chart(
        charts.label_chart(a.labels, mode),
        a.labels.rename(columns=LABEL_CLASS_TEXT),
        "Labels are what the client displayed, classified against pre-registered labels and "
        "aliases. They are not a server-side record.",
    )
    st.subheader("Per-message selection vs. default")
    if not a.overrides:
        st.caption("No condition uses a per-message selection.")
    for o in a.overrides:
        kind = "same setting twice" if o.stacked else "per-message differs from default"
        st.markdown(f"**Condition {md_text(o.condition_id)}** ({kind})")
        st.markdown(f"Labels: {md_text(o.label_summary)}")  # displayed labels, as recorded
        if o.note:
            st.caption(md_text(o.note))
            continue
        rows = [
            {
                "metric": m.metric,
                o.condition_id: metric_value(m.condition_value, m.unit),
                f"default-only ref ({o.default_ref})": metric_value(m.default_ref_value, m.unit),
                f"per-message ref ({o.override_ref})": metric_value(m.override_ref_value, m.unit),
                f"vs default ref [{conf}% CI]": interval_text(m.vs_default, m.unit),
                f"vs per-message ref [{conf}% CI]": "—"
                if o.stacked
                else interval_text(m.vs_override, m.unit),
                "position (0 = default, 1 = per-message)": "—"
                if o.stacked
                else position_text(m.position),
                "assessment": VERDICT_SHORT[m.verdict],
            }
            for m in o.metrics
        ]
        _table(pd.DataFrame(rows))
        with st.expander("What the assessments mean"):
            for m in o.metrics:
                st.markdown(f"- *{md_text(m.metric)}*: {md_text(m.explanation)}")
    flags = a.confounder_flags
    if not flags.empty and int(flags["n_flagged"].sum()):
        st.markdown(
            "**Confounders by condition** — runs with a value other than the recommended one"
        )
        _table(
            flags.assign(runs=lambda f: f["n_flagged"].astype(str) + "/" + f["n_runs"].astype(str))
            .pivot(index="confounder_id", columns="condition_id", values="runs")
            .reset_index()
        )
    if a.sensitivity is not None:
        check = a.sensitivity
        reasons = ", ".join(f"{k} ({n})" for k, n in check.reasons.items())
        st.markdown(
            f"**Sensitivity check:** without the {check.left_out} run(s) with a flagged "
            f"per-run value ({md_text(reasons)}), "
            + (
                f"{len(check.changes)} of {check.n_assessments} assessments change:"
                if check.changes
                else f"none of the {check.n_assessments} assessments change."
            )
        )
        for cond, metric, before, after in check.changes:
            st.markdown(
                f"- {md_text(cond)}, {md_text(metric)}: {VERDICT_SHORT[before]} → "
                f"{VERDICT_SHORT[after]}"
            )


def _comparison_sections(a: Analysis, conf: int) -> None:
    tallies = hypothesis_tallies(a)
    if tallies:
        st.subheader("Hypothesis tallies")
        for tally in tallies:
            st.markdown(f"- {md_text(tally.sentence())}")
    st.subheader("Pairwise differences (descriptive)")
    if a.comparisons.empty:
        st.caption("Not enough data.")
    else:
        rows = []
        for r in a.comparisons.itertuples():
            iv = None if pd.isna(r.difference) else Interval(r.difference, r.low, r.high)
            unit = METRIC_UNITS[r.metric]
            rows.append(
                {
                    "metric": METRIC_NAMES[r.metric],
                    "first": r.condition_a,
                    "second": r.condition_b,
                    "n": f"{r.n_a}/{r.n_b}",
                    f"first − second [{conf}% CI]": interval_text(iv, unit),
                    "status": COMPARISON_STATUS[r.status],
                }
            )
        _table(pd.DataFrame(rows))
        st.caption("Descriptive intervals, not significance tests; some will exclude 0 by chance.")


def page_tasks(study: Study, a: Analysis) -> None:
    st.title("Task-level results")
    _banner(a)
    if not a.has_data:
        st.info("No observations recorded yet.")
        return
    mode = _mode()
    task_table = pd.DataFrame(
        [
            {
                "task": r.task_id,
                "title": r.task_title,
                "difficulty": r.difficulty,
                "condition": r.condition_id,
                "correct / scored": f"{r.n_correct}/{r.n_scored}" if r.n_scored else "—",
                "accuracy": pct(r.accuracy),
                "format failures": r.n_format_failures,
                "distinct answers": r.distinct_answers,
                "most common answer share": pct(r.majority_answer_share),
            }
            for r in a.tasks.itertuples()
        ]
    )
    _chart(
        charts.task_heatmap(a.tasks, mode),
        task_table,
        "Empty cells have no scored responses. Difficulty is the task author's prior judgement.",
    )
    st.subheader("Repetitions")
    _chart(
        charts.repetition_chart(a.grid, mode),
        a.grid.pivot_table(
            index="task_id",
            columns=["condition_id", "repetition"],
            values="outcome",
            aggfunc="first",
        ),
        "Every planned run is shown, including runs not recorded yet, so gaps stay visible.",
    )
    st.subheader("By difficulty")
    _table(
        pd.DataFrame(
            [
                {
                    "condition": r.condition_id,
                    "difficulty": r.difficulty,
                    "correct / scored": f"{r.n_correct}/{r.n_scored}",
                    "accuracy": pct(r.accuracy),
                }
                for r in a.difficulty.itertuples()
            ]
        )
    )


def page_responses(study: Study, a: Analysis) -> None:
    st.title("Individual responses")
    _banner(a)
    if a.results.empty:
        st.info("No observations recorded yet.")
        return
    df = a.results
    c1, c2, c3, c4 = st.columns(4)
    conds = c1.multiselect("Condition", sorted(df["condition_id"].unique()))
    tasks = c2.multiselect("Task", sorted(df["task_id"].unique()))
    outcomes = c3.multiselect("Outcome", sorted(df["outcome"].unique()))
    query = c4.text_input("Search response text")
    view = df
    if conds:
        view = view[view["condition_id"].isin(conds)]
    if tasks:
        view = view[view["task_id"].isin(tasks)]
    if outcomes:
        view = view[view["outcome"].isin(outcomes)]
    if query:
        view = view[view["response_text"].str.contains(query, case=False, regex=False, na=False)]
    columns = [
        "run_id",
        "sequence",
        "condition_id",
        "task_id",
        "repetition",
        "outcome",
        "extracted_answer",
        *(
            ["api_model", "api_reasoning_tokens", "api_incomplete_reason"]
            if a.definition.execution_method == "official_api"
            else ["observed_model_label", "label_class", "thinking_text"]
        ),
        "latency_seconds",
        "response_status",
    ]
    st.caption(f"{len(view)} of {len(df)} recorded runs")
    _table(view[columns])
    if view.empty:
        return
    run_id = st.selectbox("Inspect run", list(view["run_id"]))
    row = view[view["run_id"] == run_id].iloc[0]
    left, right = st.columns([3, 2])
    with left:
        st.caption("Prompt as sent")
        st.code(row["prompt_text"], language=None, wrap_lines=True)
        st.caption("Response as recorded")
        st.code(row["response_text"] or "(empty)", language=None, wrap_lines=True)
    with right:
        details = {
            "condition": row["condition_id"],
            "outcome": row["outcome"],
            "score detail": row["score_detail"],
            "extracted answer": row["extracted_answer"],
            "displayed label": row["observed_model_label"],
            "label class": LABEL_CLASS_TEXT.get(row["label_class"], row["label_class"]),
            "thinking": f"{row['thinking_indicator']} {row['thinking_text'] or ''}".strip(),
            **{
                col.removeprefix("api_").replace("_", " ") + " (API)": row[col]
                for col in row.index
                if col.startswith("api_")
            },
            "response time": f"{seconds(row['latency_seconds'])} ({method_text(row['latency_method'])})",
            "sent at": row["timestamp"],
            "client version": row["client_version"],
            "new conversation": row["new_conversation"],
            "protocol deviation": row["protocol_deviation"],
            "notes": row["notes"],
        }
        for key in [c for c in row.index if c.startswith("conf_")]:
            details[key.removeprefix("conf_")] = row[key]
        _table(
            pd.DataFrame(
                [
                    (k, "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
                    for k, v in details.items()
                ],
                columns=["field", "value"],
            )
        )


def page_methodology(study: Study, a: Analysis) -> None:
    d = a.definition
    st.title("Methodology & limitations")
    _banner(a)
    st.markdown(
        "**The question this tool answers:** *what did we actually measure, and how far can "
        "the result be trusted?*"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Observed directly")
        if d.execution_method == "official_api":
            st.markdown(
                "- The request sent (model id, reasoning.effort, prompt) and the response text\n"
                "- Values the API returned: model id, status, echoed effort, token usage "
                "(reasoning tokens included), recorded as returned\n"
                "- Response time from a client-side timer around each API call\n"
                "- SDK version and the time each request was sent"
            )
        else:
            st.markdown(
                "- The prompt as sent and the response text as displayed\n"
                "- The model label the client displayed (verbatim), if any\n"
                "- Whether a thinking indicator appeared, and the duration it displayed\n"
                "- Response time measured by the experimenter (instrument recorded per run)\n"
                "- Client version, platform, date, and declared confounders"
            )
    with c2:
        st.subheader("Not observable here")
        for item in d.not_observable or ["Server-side model identity, routing and effort."]:
            st.markdown(f"- {md_text(item)}")
    st.subheader("Claims ledger")
    names = {
        "documented_fact": "documented fact",
        "hypothesis": "hypothesis",
        "observation": "observation (this sample)",
        "not_verified": "not verified / not observable",
    }
    _table(
        pd.DataFrame(
            [(names[c.category], c.text, c.source) for c in a.claims],
            columns=["category", "statement", "basis"],
        )
    )
    st.subheader("Limitations")
    for lim in a.limitations:
        st.markdown(f"- {md_text(lim.text)}")
    st.subheader("Scoring rules (pre-registered)")
    _table(
        pd.DataFrame(
            [
                (t.id, t.title, t.difficulty or "—", scoring_text(t), t.reference_answer or "—")
                for t in d.tasks
            ],
            columns=["task", "title", "difficulty", "rule", "reference answer"],
        )
    )
    st.subheader("Statistics")
    s = a.settings
    st.markdown(
        f"- Proportions: {int(s.confidence * 100)}% Wilson score intervals. Differences of "
        "proportions: standardized over tasks (each task scored in both conditions counts "
        "equally), stratified Newcombe intervals.\n"
        f"- Differences of medians: percentile bootstrap, {s.bootstrap_resamples} resamples, "
        f"seed {s.seed}.\n"
        f"- Groups with fewer than {s.min_n} observations are not compared.\n"
        f"- A condition counts as consistent with a reference only within ±"
        f"{s.equivalence_fraction:.0%} of the gap between the two references (pre-registered "
        "equivalence margin); an interval that only includes 0 is inconclusive.\n"
        "- Per-message vs. default: a metric can only indicate which selection applied if the two "
        "reference conditions differ on it. The decision table is in "
        f"[docs/methodology.md]({DOCS}/methodology.md)."
    )
    if d.latency_protocol:
        st.subheader("Response-time protocol")
        st.markdown(md_text(d.latency_protocol))
    st.subheader("Reproduction")
    _table(
        pd.DataFrame(
            [
                ["definition SHA-256", d.sha256()],
                ["plan seed / strategy", f"{a.plan.seed} / {a.plan.strategy}" if a.plan else "—"],
                ["code", f"model-behavior-lab {code_version().label()}"],
                ["study directory", str(study.root.resolve())],
            ],
            columns=["item", "value"],
        )
    )
    st.caption(f"Full protocol: [docs/experiment-protocol.md]({DOCS}/experiment-protocol.md)")


# -- benchmark -----------------------------------------------------------------------------


def _benchmark_stamp(root: Path) -> tuple:
    names = (BENCHMARK_FILE, RUNS_FILE)
    return tuple((n, (root / n).stat().st_mtime_ns if (root / n).exists() else 0) for n in names)


@st.cache_resource(show_spinner=False, max_entries=4)
def _load_benchmark(root: str, stamp: tuple) -> Benchmark:
    return Benchmark.load(root)


def _bullets(items: list[str]) -> None:
    for item in items:
        st.markdown("- " + md_text(item))


def page_benchmark(root: Path) -> None:
    try:
        bench = _load_benchmark(str(root.resolve()), _benchmark_stamp(root))
    except StudyError as exc:
        st.error(md_text(exc))
        st.stop()
    batches = bench.batches
    with st.sidebar:
        chosen = batches
        if len(batches) > 1:
            chosen = st.multiselect("Batches", batches, default=batches, key="batches")
        st.caption(f"{len(bench.runs)} observed runs across all batches")
        st.caption(f"model-behavior-lab {code_version().label()}")
    d = bench.definition
    st.title(md_text(d.title))
    if not chosen:
        st.info("Select a batch to view its results." if batches else "No runs recorded yet.")
        return
    ev: Evaluation = evaluate(bench, chosen)
    truth = ev.truth
    n = ev.n_runs

    st.info(
        md_text(
            f"Observed data, recorded by hand: {n} runs, {ev.runs_per_condition()} per condition, "
            "one task. Exploratory: the figures describe these runs only, not any model's "
            "general accuracy, and nothing here is a statistical test."
        ),
        icon="🔎",
    )
    count_ok, _ = ev.correct("count")
    cols = st.columns(4)
    cols[0].metric("Full task success", f"{ev.full_successes} of {n}")
    cols[1].metric("Partially correct", f"{ev.partials} of {n}")
    cols[2].metric("Exact count", f"{count_ok} of {n}")
    cols[3].metric("Valid orders given", f"{ev.valid_order_answers} of {2 * n}")

    frame = runs_frame(ev)
    frame["count_given"] = frame["count_given"].map(
        lambda value: None if value is None else str(value)
    )
    st.subheader("Results by run")
    _table(
        frame[
            [
                "run_id",
                "condition_id",
                "default_selection",
                "per_message_selection",
                "count_given",
                "exact_count",
                "lexicographic_first",
                "other_valid_order",
                "full_task_success",
                "error_type",
            ]
        ].rename(
            columns={
                "run_id": "Run",
                "condition_id": "Condition",
                "default_selection": "Default",
                "per_message_selection": "Per-message",
                "count_given": "Count given",
                "exact_count": "Exact count",
                "lexicographic_first": "Lexicographic first",
                "other_valid_order": "Second valid order",
                "full_task_success": "Full task success",
                "error_type": "Error type",
            }
        )
    )

    st.subheader("Counts given")
    _chart(
        charts.benchmark_count_chart(frame, truth.count, _mode()),
        frame[["run_id", "condition_id", "count_given", "exact_count"]],
        caption=(
            "Each mark shows one run’s numeric answer. The dashed line shows the verified count. "
            "Plot positions are approximate; tables and tooltips preserve exact counts. "
            "Counts outside the finite plotting range appear only in the results table."
        ),
    )

    st.subheader("Default and per-message selection")
    _table(selection_grid(ev))
    st.caption(
        "Rows: the default selection. Columns: the per-message selection (long-press on send). "
        "Each cell: the condition, its fully correct runs, and the count it gave."
    )

    st.subheader("Comparisons")
    for comp in ev.comparisons:
        st.markdown(f"**{md_text(comp.title)}**")
        st.markdown(md_text(comp.text))

    st.subheader("Error analysis")
    _table(
        pd.DataFrame(
            [
                {
                    "Run": r.run.run_id,
                    "Condition": r.condition.id,
                    "Error type": r.error_type(truth),
                    "Detail": r.error_detail(truth),
                }
                for r in ev.runs
            ]
        )
    )
    checks = breakdown_frame(ev)
    if not checks.empty:
        with st.expander("Intermediate counts in the experimenter's notes vs. the verifier"):
            _table(checks)

    st.subheader("Ground truth")
    st.markdown(
        md_text(
            f"{truth.count} valid orders; the lexicographically first is "
            f"{truth.lexicographic_first}. Computed by checking all 40,320 orders against the "
            "eight rules; no response and no run value is used."
        )
    )
    left, right = st.columns(2)
    with left:
        _table(
            pd.DataFrame(
                [{"First task": k, "Valid orders": v} for k, v in truth.prefix_counts(1).items()]
            )
        )
    with right:
        _table(
            pd.DataFrame(
                [
                    {"First two tasks": k, "Valid orders": v}
                    for k, v in truth.prefix_counts(2).items()
                ]
            )
        )
    with st.expander("Task prompt (verbatim)"):
        st.code(d.task.prompt, language=None)
    with st.expander("Experimenter's notes per run (verbatim)"):
        for r in ev.runs:
            st.markdown(f"**{md_text(r.run.run_id)}**")
            st.code("\n".join(r.run.solution_summary) or "(none)", language=None)

    st.subheader("Recording coverage and limitations")
    coverage = RecordingCoverage.from_runs([r.run for r in ev.runs])
    st.caption(
        f"Full response texts kept for {coverage.full_responses} of {n} selected runs, "
        f"response times measured for {coverage.latencies} of {n}. "
        f"Send times recorded for {coverage.timestamps} of {n}, "
        f"run order recorded for {coverage.run_orders} of {n}."
    )
    if d.not_recorded and (not d.notes_batches or set(d.notes_batches).intersection(chosen)):
        scope = f" for {', '.join(d.notes_batches)}" if d.notes_batches else " (batch unspecified)"
        st.caption(md_text(f"Recording omissions listed in the benchmark definition{scope}"))
        _bullets(d.not_recorded)
    if d.not_observable:
        st.caption("Not observable in this experiment")
        _bullets(d.not_observable)
    if not d.documented_facts:
        st.caption(
            "No documented facts are recorded: nothing on this page is a statement from "
            "product documentation."
        )


RENDERERS = {
    "Overview": page_overview,
    "Run plan & recording": page_record,
    "Condition comparison": page_conditions,
    "Task-level results": page_tasks,
    "Individual responses": page_responses,
    "Methodology & limitations": page_methodology,
}


def main() -> None:
    st.set_page_config(page_title="Model Behavior Lab", page_icon="🔬", layout="wide")
    if "study_dir" not in st.session_state:
        st.session_state["study_dir"] = _initial_study_dir()
    root = Path(st.session_state["study_dir"]).expanduser()
    benchmark = is_benchmark_dir(root)
    with st.sidebar:
        st.markdown("### Model Behavior Lab")
        st.caption("Does your AI actually behave as configured?")
        page = None if benchmark else st.radio("Page", PAGES, key="page")
        st.text_input("Study or benchmark directory", key="study_dir")
    if benchmark:
        page_benchmark(root)
        return
    try:
        study, analysis = _load(str(root.resolve()), _stamp(root))
    except StudyError as exc:
        st.error(md_text(exc))
        st.stop()
    with st.sidebar:
        if analysis.synthetic:
            st.error("SYNTHETIC DEMO DATA")
        else:
            st.caption(f"{len(analysis.results)} observed records")
        st.caption(f"model-behavior-lab {code_version().label()}")
    RENDERERS[str(page)](study, analysis)


if __name__ == "__main__":  # Streamlit runs this file as __main__
    main()
