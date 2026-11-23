"""Command-line interface: ``mblab``."""

from __future__ import annotations

import dataclasses
import functools
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypeVar
from urllib.parse import urlsplit

import pandas as pd
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mblab import api_runner, install_hint
from mblab.analysis import LABEL_CLASSES, Analysis, AnalysisSettings, analyze
from mblab.benchmark import (
    BENCHMARK_FILE,
    Benchmark,
    evaluate,
    is_benchmark_dir,
    make_reports,
    report_path,
    runs_frame,
    saved_report_text,
    write_reports,
)
from mblab.clipboard import copy_text, paste_text
from mblab.csvcells import to_cell
from mblab.demo import DEMO_BANNER, create_demo_study
from mblab.exchange import import_file, merge_records, parse_timezone, write_template_csv
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
from mblab.grading import export_blind_sheet, import_blind_sheet
from mblab.pilot import PILOT_BANNER, create_pilot_study, displayed_labels
from mblab.plan import generate_plan
from mblab.provenance import code_version
from mblab.records import build_record, condition_instructions, next_pending, parse_decimal
from mblab.report import compare_reports, render_report
from mblab.schema import PlannedRun, RunPlan, RunRecord
from mblab.stats import Interval
from mblab.study import (
    DEFAULT_TEMPLATE,
    DEFINITION_FILE,
    PLAN_FILE,
    RECORDS_FILE,
    Study,
    StudyError,
    append_jsonl,
    atomic_write_text,
    available_templates,
    dump_json,
    format_validation_error,
    load_definition,
    load_plan,
    parse_definition,
    read_jsonl,
    save_plan,
    template_text,
    write_lock,
)
from mblab.validate import ValidationReport, validate_study

app = typer.Typer(
    help=(
        "Model Behavior Lab — does your AI actually behave as configured?\n\n"
        "Plan, record, validate, and analyse manual experiments on model-selection and "
        "reasoning-effort settings. Every command works on a study directory "
        "(--study / -s, default: current directory)."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
    # Typer before 0.23 prints the local variables of every frame in a traceback, and a
    # local can hold the API key. Keep them out on every supported Typer version.
    pretty_exceptions_show_locals=False,
)
# emoji=False: run ids such as "pilot:B:T6:r1" contain ":B:", which Rich would print as 🅱.
console = Console(highlight=False, soft_wrap=True, emoji=False)
err_console = Console(stderr=True, highlight=False, soft_wrap=True, emoji=False)

F = TypeVar("F", bound=Callable[..., Any])

StudyOpt = Annotated[
    Path,
    typer.Option("--study", "-s", help="Study directory.", envvar="MBLAB_STUDY", show_default=True),
]


def _errors(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except StudyError as exc:
            err_console.print(f"[bold red]error:[/] {escape(str(exc))}")
            raise typer.Exit(2) from None

    return wrapper  # type: ignore[return-value]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mblab {code_version().label()}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Model Behavior Lab."""


# -- helpers ---------------------------------------------------------------------


def _settings(s: Study, **overrides: Any) -> AnalysisSettings:
    """The pre-registered analysis settings, with any options given on the command line."""
    settings = AnalysisSettings.preregistered(s.definition, s.plan)
    given = {k: v for k, v in overrides.items() if v is not None}
    settings = dataclasses.replace(settings, **given)
    departures = settings.departures(s.definition)
    if departures:
        err_console.print(
            "[yellow]warning:[/] not the pre-registered analysis ("
            + "; ".join(departures)
            + "). Results are labelled exploratory."
        )
    return settings


def _print_validation_summary(report: ValidationReport) -> None:
    counts = Counter(i.severity for i in report.issues)
    colour = "red" if counts["error"] else ("yellow" if counts["warning"] else "green")
    console.print(
        f"[{colour}]validation: {counts['error']} error(s), {counts['warning']} warning(s), "
        f"{counts['info']} note(s)[/] — details: mblab validate"
    )


def _show_run(study: Study, run: PlannedRun, total: int) -> str:
    definition = study.definition
    cond = definition.condition(run.condition_id)
    task = definition.task(run.task_id)
    console.print(
        f"[bold]Run {run.run_id}[/]  (#{run.sequence} of {total}, block {run.block})  "
        f"task {task.id}: {task.title}"
    )
    console.print(f"Condition {cond.id}: {cond.label or cond.display()}")
    for number, step in enumerate(condition_instructions(definition, run), start=1):
        console.print(f"  {number}. {step}")
    console.print("----- prompt (send exactly this) -----", style="dim")
    console.print(task.prompt, markup=False)
    console.print("----- end of prompt -----", style="dim")
    return task.prompt


def _warn_unreadable(s: Study) -> None:
    bad = [e for e in s.load_errors if e.file == RECORDS_FILE]
    if bad:
        err_console.print(
            f"[yellow]warning:[/] records.jsonl has {len(bad)} unreadable line(s) (line "
            f"{', '.join(str(e.line) for e in bad)}). Saving keeps them as they are; fix or "
            "remove them (`mblab validate` shows why)."
        )


def _latest(records: list[RunRecord]) -> RunRecord | None:
    if not records:
        return None
    return max(records, key=lambda r: (r.recorded_at or r.timestamp, r.timestamp))


def _read_multiline() -> str:
    console.print("Paste the response text, then enter a line containing only '.'")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines)


def _choose(question: str, choices: list[str], default: str | None) -> str:
    """Ask until the answer is one of ``choices``; with no default, an answer is required."""
    while True:
        value = typer.prompt(f"{question} ({'/'.join(choices)})", default=default).strip()
        if value in choices:
            return value
        err_console.print(f"[red]Choose one of: {', '.join(choices)}[/]")


def _optional_float(text: str, what: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return parse_decimal(text, what)
    except ValueError as exc:
        raise StudyError(str(exc)) from None


# -- commands: setup ---------------------------------------------------------------------


@app.command()
@_errors
def init(
    directory: Annotated[Path, typer.Argument(help="Directory to create the study in.")],
    template: Annotated[
        str, typer.Option(help=f"Experiment template ({', '.join(available_templates())}).")
    ] = DEFAULT_TEMPLATE,
    force: Annotated[bool, typer.Option(help="Overwrite an existing experiment.yaml.")] = False,
) -> None:
    """Create a study directory with an experiment definition to edit."""
    text = template_text(template)
    parsed = parse_definition(text, template)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DEFINITION_FILE
    if target.exists() and not force:
        raise StudyError(f"{target} already exists (use --force to overwrite)")
    target.write_text(text, encoding="utf-8")
    console.print(f"Created {target}")
    console.print("Next steps:")
    if parsed.execution_method == "official_api":
        console.print(
            f"  1. Edit {target}: set `api.model` and the reasoning_effort values it accepts. "
            "Settle the definition before sending any request."
        )
        console.print(f"  2. mblab plan -s {directory}")
        console.print(f"  3. mblab run-api -s {directory} --dry-run   (free preview)")
        return
    console.print(
        f"  1. Edit {target}: set `platform`, and copy the option labels exactly as your "
        "client shows them. Settle the definition before collecting data."
    )
    console.print(f"  2. mblab plan -s {directory}")


@app.command()
@_errors
def pilot(
    study: StudyOpt = Path("."),
    out: Annotated[
        Path | None, typer.Option(help="Pilot study directory (default: <study>-pilot).")
    ] = None,
    repetitions: Annotated[
        int, typer.Option(min=1, max=100, help="Repetitions in the pilot (one block = 1).")
    ] = 1,
    force: Annotated[bool, typer.Option(help="Write into a non-empty directory.")] = False,
) -> None:
    """Create a separate pilot study from a study's definition, to try the protocol first."""
    s, target = create_pilot_study(study, out, repetitions, force)
    console.print(f"[bold yellow]{PILOT_BANNER}[/]")
    console.print(
        f"Created {target}: {len(s.require_plan().runs)} runs "
        f"({len(s.definition.conditions)} conditions x {len(s.definition.tasks)} tasks x "
        f"{repetitions})."
    )
    console.print("Next steps:")
    console.print(f"  1. Run it like the real study: mblab next -s {target} --copy / mblab record")
    console.print(f"  2. mblab labels -s {target}   (every label the client displayed, verbatim)")
    console.print(
        f"  3. In {study / DEFINITION_FILE}, fix option labels and observed_label_aliases, "
        "then plan the main study. Keep the pilot directory as a record of how they were chosen."
    )


@app.command()
@_errors
def labels(study: StudyOpt = Path(".")) -> None:
    """List every model label the client displayed, verbatim, and which option it matches."""
    s = Study.load(study)
    if s.definition.execution_method != "manual_ui":
        raise StudyError("displayed labels exist only in manual_ui studies")
    items, none = displayed_labels(s)
    if not s.definition.pilot and s.records:
        err_console.print(
            "[yellow]warning:[/] this is not a pilot study. Changing labels or aliases after "
            "seeing this study's data is a post-hoc change; `mblab validate` will flag it. "
            "Use `mblab pilot` to choose them before collecting."
        )
    if not items:
        console.print(f"No displayed labels recorded ({none} run(s) without one).")
        return
    table = Table("label (verbatim)", "runs", "conditions", "where shown", "matches option")
    for item in items:
        table.add_row(
            item.text,
            str(item.runs),
            ", ".join(f"{k} ({v})" for k, v in item.conditions.items()),
            ", ".join(f"{k} ({v})" for k, v in item.locations.items()),
            ", ".join(item.matches) or "none (not a label or alias yet)",
        )
    console.print(table)
    console.print(f"{none} run(s) had no displayed label.")
    unmatched = [i.text for i in items if not i.matches]
    if unmatched:
        console.print(
            "Texts that match no option are reported as 'unrecognized'. If one is the name the "
            "client shows for an option, add it to that option's observed_label_aliases in the "
            "main study's experiment.yaml before planning it."
        )


@app.command()
@_errors
def plan(
    study: StudyOpt = Path("."),
    seed: Annotated[int | None, typer.Option(help="Override the seed in experiment.yaml.")] = None,
    force: Annotated[bool, typer.Option(help="Replace an existing plan.json.")] = False,
    show: Annotated[int, typer.Option(help="How many runs of the order to print.")] = 12,
) -> None:
    """Generate the randomized run order (plan.json) from experiment.yaml."""
    definition = load_definition(study / DEFINITION_FILE)
    plan_path = study / PLAN_FILE
    new_plan = generate_plan(definition, seed=seed)
    with write_lock(study, "plan"):
        # Read inside the lock, so a record saved a moment ago counts.
        existing = load_plan(plan_path)
        if existing is not None and not force:
            raise StudyError(
                f"{plan_path} already exists. Regenerating can change the run order; "
                "use --force only if that is intended."
            )
        records, unreadable = read_jsonl(study / RECORDS_FILE, RunRecord)
        if existing is not None and (records or unreadable):
            raise StudyError(
                f"{len(records) + len(unreadable)} run(s) are already recorded under "
                f"{plan_path}, so it is not replaced: it holds the pre-registered order and "
                "the definition snapshot those records are checked against. Keep it; "
                "`mblab validate` and the report show what changed in experiment.yaml since. "
                "To change the design, start a new study directory (and a new "
                "experiment_version)."
            )
        save_plan(new_plan, plan_path)
    console.print(
        f"Wrote {plan_path}: {len(new_plan.runs)} runs = {len(definition.conditions)} conditions "
        f"x {len(definition.tasks)} tasks x {definition.repetitions} repetitions "
        f"(seed {new_plan.seed}, {new_plan.strategy})."
    )
    console.print(f"Definition hash: {new_plan.definition_sha256[:16]}…")
    table = Table("#", "run id", "default", "per-message", "task")
    for run in new_plan.runs[: max(show, 0)]:
        table.add_row(
            str(run.sequence),
            run.run_id,
            definition.option_label(run.model_selection),
            definition.option_label(run.message_override),
            run.task_id,
        )
    if show > 0:
        console.print(table)
    console.print(f"Next: mblab next -s {study} --copy")


# -- commands: collection ---------------------------------------------------------------


@app.command()
@_errors
def status(study: StudyOpt = Path(".")) -> None:
    """Show collection progress and the next run."""
    s = Study.load(study)
    plan_ = s.require_plan()
    definition = s.definition
    console.print(
        f"[bold]{definition.title}[/] — {definition.experiment_id} v{definition.experiment_version}"
        f" ({definition.execution_method}, {definition.platform})"
    )
    if s.is_synthetic():
        console.print("[bold red]SYNTHETIC DEMO DATA — not real observations[/]")
    if definition.pilot:
        console.print(f"[bold yellow]{PILOT_BANNER}[/]")
    api = definition.execution_method == "official_api"
    held = api_runner.held_runs(s)
    recorded = Counter(r.condition_id for r in s.records)
    excluded = Counter(r.condition_id for r in s.records if r.excluded)
    held_by_condition = Counter(r.condition_id for r in held.runs)
    planned = Counter(r.condition_id for r in plan_.runs)
    table = Table(
        "condition", "default", "per-message", "recorded", "excluded", *(["held"] if api else [])
    )
    for cond in definition.conditions:
        table.add_row(
            cond.id,
            definition.option_label(cond.model_selection),
            definition.option_label(cond.message_override),
            f"{recorded[cond.id]}/{planned[cond.id]}",
            str(excluded[cond.id]),
            *([str(held_by_condition[cond.id])] if api else []),
        )
    console.print(table)
    console.print(f"Recorded {len(s.records)} of {len(plan_.runs)} planned runs.")
    _print_held(held)
    nxt = _next_run(s, held)
    if nxt is None:
        console.print(_nothing_next(s, plan_))
    elif api:
        console.print(
            f"Next run: #{nxt.sequence} {nxt.run_id} — send it with mblab run-api -s {study} "
            "(preview: --dry-run)"
        )
    else:
        console.print(f"Next run: #{nxt.sequence} {nxt.run_id} — mblab next -s {study} --copy")
    _print_validation_summary(validate_study(s))


def _print_held(held: api_runner.HeldRuns) -> None:
    """Runs `run-api` holds back, so `status` and `next` point where `run-api` goes."""
    if held.log_errors:
        err_console.print(f"[yellow]warning:[/] {escape(held.describe_log_errors())}")
    if held.runs:
        console.print(f"[yellow]Held back:[/] {escape(held.describe())}")


def _next_run(s: Study, held: api_runner.HeldRuns) -> PlannedRun | None:
    if s.definition.execution_method == "official_api":
        return api_runner.next_to_send(s, held)
    return next_pending(s.require_plan(), s.recorded_run_ids())


def _nothing_next(s: Study, plan_: RunPlan) -> str:
    if len(s.recorded_run_ids() & {r.run_id for r in plan_.runs}) == len(plan_.runs):
        return "All planned runs are recorded."
    return "No run is left to send: every planned run without a record is held back (see above)."


@app.command("next")
@_errors
def next_run(
    study: StudyOpt = Path("."),
    copy: Annotated[bool, typer.Option("--copy", help="Copy the prompt to the clipboard.")] = False,
) -> None:
    """Show the next pending run: what to set in the client and the exact prompt."""
    s = Study.load(study)
    plan_ = s.require_plan()
    held = api_runner.held_runs(s)
    _print_held(held)
    run = _next_run(s, held)
    if run is None:
        console.print(_nothing_next(s, plan_))
        return
    prompt = _show_run(s, run, len(plan_.runs))
    if copy:
        if copy_text(prompt):
            console.print("[green]Prompt copied to the clipboard.[/]")
        else:
            console.print("[yellow]No clipboard tool found; copy the prompt above by hand.[/]")
    if s.definition.execution_method == "official_api":
        console.print(f"Send it: mblab run-api -s {study}   (preview: --dry-run)")
    else:
        console.print(f"After the response: mblab record -s {study} --run-id {run.run_id}")


@app.command()
@_errors
def record(
    study: StudyOpt = Path("."),
    run_id: Annotated[
        str | None, typer.Option(help="Run to record (default: next pending run).")
    ] = None,
    response_file: Annotated[
        Path | None, typer.Option(help="Read the response text from a file.")
    ] = None,
    paste: Annotated[
        bool, typer.Option("--paste", help="Read the response text from the clipboard.")
    ] = False,
    copy_prompt: Annotated[
        bool, typer.Option(help="Copy the prompt to the clipboard first.")
    ] = True,
    replace: Annotated[bool, typer.Option(help="Replace an existing record for this run.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Save without asking.")] = False,
) -> None:
    """Interactively record one observed run (you type what the client showed)."""
    s = Study.load(study)
    if s.definition.execution_method == "official_api":
        raise StudyError(
            "this is an official_api study; its records are written by `mblab run-api`, "
            "not typed in"
        )
    plan_ = s.require_plan()
    _warn_unreadable(s)
    if run_id:
        run = plan_.by_run_id().get(run_id)
        if run is None:
            raise StudyError(f"run {run_id!r} is not in the plan")
    else:
        run = next_pending(plan_, s.recorded_run_ids())
        if run is None:
            console.print("All planned runs are recorded.")
            return
    if run.run_id in s.recorded_run_ids() and not replace:
        raise StudyError(f"run {run.run_id} already has a record (use --replace to overwrite)")

    prompt_text = _show_run(s, run, len(plan_.runs))
    if copy_prompt and not (response_file or paste) and copy_text(prompt_text):
        console.print("[green]Prompt copied to the clipboard.[/]")

    prev = _latest(s.records)
    status_ = _choose(
        "Response status", ["completed", "error", "refused", "incomplete"], "completed"
    )
    if response_file:
        response_text = response_file.read_text(encoding="utf-8")
    elif paste:
        pasted = paste_text()
        if pasted is None:
            raise StudyError("could not read the clipboard; use --response-file instead")
        response_text = pasted
        console.print(f"Read {len(response_text)} characters from the clipboard.")
    else:
        response_text = _read_multiline()

    label = typer.prompt(
        "Model label displayed by the client (blank if none)", default="", show_default=False
    )
    location = ""
    if label.strip():
        location = typer.prompt(
            "Where was the label displayed?",
            default=(prev.observed_label_location if prev else None) or "",
            show_default=bool(prev and prev.observed_label_location),
        )
    thinking = _choose("Thinking indicator", ["shown", "not_shown", "unknown"], "unknown")
    thinking_text = ""
    if thinking == "shown":
        thinking_text = typer.prompt(
            "Thinking text exactly as shown (e.g. 'Thought for 1m 12s'; blank if none)",
            default="",
            show_default=False,
        )
    latency = _optional_float(
        typer.prompt("Stopwatch seconds (blank if not measured)", default="", show_default=False),
        "stopwatch seconds",
    )
    sent_default = datetime.now().astimezone().replace(microsecond=0)
    if latency is not None:
        sent_default -= timedelta(seconds=round(latency))
    sent_text = typer.prompt("Time sent (ISO 8601)", default=sent_default.isoformat())
    try:
        sent_at = datetime.fromisoformat(sent_text.strip())
    except ValueError:
        raise StudyError(f"time sent {sent_text!r} is not ISO 8601") from None
    client_version = typer.prompt(
        "Client (app) version", default=(prev.client_version if prev else "unknown")
    )
    confounders: dict[str, str] = {}
    for conf in s.definition.confounders:
        # A per-run observation (a tool used, a notice shown) is never carried over from
        # the previous run: pressing Enter must not record what happened last time.
        default = (
            None
            if conf.per_run
            else (prev.confounders.get(conf.id) if prev else None)
            or ("unknown" if "unknown" in conf.allowed_values else conf.allowed_values[0])
        )
        confounders[conf.id] = _choose(conf.question, conf.allowed_values, default)
    new_conversation = typer.confirm("Was this a new conversation?", default=True)
    deviation = typer.prompt("Protocol deviation (blank if none)", default="", show_default=False)
    notes = typer.prompt("Notes (blank if none)", default="", show_default=False)
    recorded_by = typer.prompt(
        "Recorder id (pseudonym, blank to skip)",
        default=(prev.recorded_by if prev else None) or "",
        show_default=bool(prev and prev.recorded_by),
    )

    try:
        rec = build_record(
            run,
            s.definition,
            plan_,
            client_version=client_version,
            timestamp=sent_at,
            response_status=status_,
            response_text=response_text,
            observed_model_label=label,
            observed_label_location=location,
            thinking_indicator=thinking,
            thinking_text=thinking_text,
            latency_seconds=latency,
            latency_method="manual_stopwatch" if latency is not None else "not_measured",
            new_conversation=new_conversation,
            confounders=confounders,
            protocol_deviation=deviation,
            notes=notes,
            recorded_by=recorded_by,
        )
    except ValidationError as exc:
        raise StudyError(f"record not saved: {format_validation_error(exc)}") from None

    console.print(
        f"Run {rec.run_id}: status={rec.response_status}, {len(rec.response_text)} chars, "
        f"label={rec.observed_model_label or '—'}, thinking={rec.thinking_indicator}"
        f"{f' ({rec.thinking_seconds:g}s)' if rec.thinking_seconds is not None else ''}, "
        f"latency={'—' if rec.latency_seconds is None else f'{rec.latency_seconds:g}s'}"
    )
    if not yes and not typer.confirm("Save this record?", default=True):
        console.print("Not saved.")
        raise typer.Exit(1)
    merged = merge_records(s, [rec], replace=replace, dry_run=False)
    if merged.conflicts:
        unsaved = s.root / "exports" / f"unsaved-{rec.run_id.replace(':', '_')}.jsonl"
        unsaved.parent.mkdir(parents=True, exist_ok=True)
        unsaved.write_text(rec.model_dump_json() + "\n", encoding="utf-8")
        raise StudyError(
            f"run {rec.run_id} was recorded by another command while you were typing, so "
            f"nothing was saved. Your entry is in {unsaved}; to keep it instead, run "
            f"`mblab import {unsaved} -s {study} --replace`"
        )
    console.print(f"[green]Saved to {s.records_path}.[/]")
    _warn_unreadable(s)
    nxt = next_pending(plan_, s.recorded_run_ids() | {rec.run_id})
    if nxt:
        console.print(f"Next: mblab next -s {study} --copy   (#{nxt.sequence} {nxt.run_id})")


def _api_outcome_line(outcome: api_runner.RunOutcome, total: int) -> str:
    rec, rep = outcome.record, outcome.record.api_reported
    tokens = "—" if rep is None or rep.reasoning_tokens is None else f"{rep.reasoning_tokens:,}"
    retried = f", after {outcome.failed_attempts} retry(ies)" if outcome.failed_attempts else ""
    return (
        f"#{outcome.run.sequence}/{total} {rec.run_id}: {rec.response_status}, "
        f"{seconds(rec.latency_seconds)}, reasoning tokens {tokens}{retried}"
    )


@app.command("run-api")
@_errors
def run_api(
    study: StudyOpt = Path("."),
    limit: Annotated[
        int | None, typer.Option(min=1, help="Send at most N pending runs, in plan order.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print what would be sent. Sends nothing, writes nothing."),
    ] = False,
    min_interval: Annotated[
        float, typer.Option(min=0, help="Minimum seconds between request starts (rate limit).")
    ] = 1.0,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Send without asking first.")] = False,
    resend_unrecorded: Annotated[
        bool,
        typer.Option(
            help="Also send runs whose earlier, billed response could not be recorded "
            "(listed in api-unrecorded.jsonl). They are billed again."
        ),
    ] = False,
) -> None:
    """Send pending runs of an official_api study through the OpenAI Responses API.

    Reads the key from OPENAI_API_KEY only and never stores it. Each request is billed by
    the provider. Runs that already have a record are skipped, so re-running resumes.
    """
    s = Study.load(study)
    plan_ = api_runner.check_api_study(s)
    d = s.definition
    assert d.api is not None
    held = api_runner.unrecorded_run_ids(s.root, s.recorded_run_ids())
    if held:
        err_console.print(
            f"[yellow]warning:[/] {len(held)} run(s) already returned a billed response that could "
            f"not be recorded (see {api_runner.UNRECORDED_FILE}): {', '.join(sorted(held))}. "
            + (
                "They will be sent again (--resend-unrecorded)."
                if resend_unrecorded
                else "They are held back; pass --resend-unrecorded to pay for them again."
            )
        )
    all_pending = api_runner.pending_runs(
        plan_, s.recorded_run_ids(), held=frozenset() if resend_unrecorded else held
    )
    runs = all_pending[:limit] if limit else all_pending
    if not runs:
        console.print("No runs to send: every planned run is recorded or held back.")
        return
    efforts = Counter(d.option(r.model_selection).reasoning_effort for r in runs)
    console.print(
        f"{len(all_pending)} of {len(plan_.runs)} planned runs pending; this would send {len(runs)} request(s) to "
        f"model {d.api.model} (reasoning.effort: "
        + ", ".join(f"{e} ×{n}" for e, n in efforts.items())
        + f"), store=false, max_output_tokens={d.api.max_output_tokens or 'not set'}."
    )
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if base_url:
        host = urlsplit(base_url).hostname or "?"
        err_console.print(
            f"[yellow]warning:[/] OPENAI_BASE_URL is set; the SDK will send requests to {host}. "
            "Make sure that is the provider's official endpoint."
        )
    if dry_run:
        table = Table("#", "run id", "reasoning.effort", "task", "prompt chars")
        for run in runs:
            table.add_row(
                str(run.sequence),
                run.run_id,
                str(d.option(run.model_selection).reasoning_effort),
                run.task_id,
                str(len(d.task(run.task_id).prompt)),
            )
        console.print(table)
        console.print("Request body of the first run (the others differ only in input and effort):")
        console.print(dump_json(api_runner.build_request(d, runs[0])), markup=False)
        key = "is set" if api_runner.api_key_is_set() else "is NOT set (a real run needs it)"
        console.print(
            f"Dry run: nothing was sent and nothing was written. {api_runner.API_KEY_ENV} {key}."
        )
        console.print(
            "Each real request is billed by the provider for input and output tokens, reasoning "
            "tokens included. This tool does not estimate cost."
        )
        return

    sdk = api_runner.load_sdk()
    client = api_runner.make_client(sdk)
    if not yes and not typer.confirm(
        f"Send {len(runs)} request(s), billed to the account of the API key?", default=False
    ):
        console.print("Nothing was sent.")
        raise typer.Exit(1)

    def on_retry(run: PlannedRun, message: str, delay: float) -> None:
        err_console.print(f"[yellow]{run.run_id}: {message} — retrying in {delay:g} s[/]")

    try:
        result = api_runner.send_runs(
            s,
            runs,
            client=client,
            sdk=sdk,
            min_interval=min_interval,
            on_record=lambda o: console.print(_api_outcome_line(o, len(plan_.runs)), markup=False),
            on_retry=on_retry,
            resend_unrecorded=resend_unrecorded,
        )
    except KeyboardInterrupt:
        err_console.print(
            "Interrupted. Every response received so far was written first; run the command "
            "again to resume."
        )
        raise typer.Exit(130) from None
    console.print(f"Recorded {len(result.recorded)} run(s) to {s.records_path}.")
    if result.skipped:
        console.print(
            f"Skipped {len(result.skipped)} run(s) that another command recorded or held back "
            "after this one started: " + ", ".join(r.run_id for r in result.skipped)
        )
    _print_validation_summary(validate_study(Study.load(study)))
    if result.stopped_at is not None:
        raise StudyError(
            f"stopped at run {result.stopped_at.run_id} (no record written for it): "
            f"{result.stop_reason}. Run the command again to resume."
        )


@app.command()
@_errors
def template(
    study: StudyOpt = Path("."),
    out: Annotated[
        Path | None,
        typer.Option(help="Output CSV path (default: <study>/exports/entry-template.csv)."),
    ] = None,
) -> None:
    """Write a CSV with one row per planned run, for entering results in a spreadsheet."""
    s = Study.load(study)
    target = out or study / "exports" / "entry-template.csv"
    count = write_template_csv(s, target)
    console.print(
        f"Wrote {target} ({count} rows). Fill in rows, then: mblab import {target} -s {study}"
    )
    console.print(
        "Timestamps need a UTC offset (2026-11-20T14:03:00+09:00) or pass --default-tz to import."
    )


@app.command("import")
@_errors
def import_(
    file: Annotated[Path, typer.Argument(help="CSV (template format), JSON array, or JSON lines.")],
    study: StudyOpt = Path("."),
    replace: Annotated[bool, typer.Option(help="Replace records that already exist.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Check the file without saving anything.")] = False,
    default_tz: Annotated[
        str | None,
        typer.Option(
            help="Time zone for CSV timestamps without an offset, e.g. +09:00 or Asia/Seoul. "
            "Repeated or skipped local times need to be checked and given an explicit offset."
        ),
    ] = None,
) -> None:
    """Import observations. All rows must be valid, or nothing is imported."""
    s = Study.load(study)
    s.require_plan()
    if not file.exists():
        raise StudyError(f"{file} not found")
    result = import_file(s, file, parse_timezone(default_tz))
    if result.errors:
        for err in result.errors:
            err_console.print(f"[red]{escape(err)}[/]")  # errors quote file values
        raise StudyError(f"{len(result.errors)} invalid row(s); nothing was imported")
    merged = merge_records(s, result.records, replace=replace, dry_run=dry_run)
    if merged.conflicts:
        raise StudyError(
            "records already exist for: "
            + ", ".join(merged.conflicts)
            + " (use --replace to overwrite them); nothing was imported"
        )
    verb = "Would import" if dry_run else "Imported"
    console.print(
        f"{verb} {len(merged.added)} new and {len(merged.replaced)} replaced record(s); "
        f"skipped {result.skipped_blank} unfilled row(s)."
    )
    if merged.unreadable_kept:
        err_console.print(
            f"[yellow]warning:[/] records.jsonl has {len(merged.unreadable_kept)} unreadable "
            f"line(s) (line {', '.join(map(str, merged.unreadable_kept))}). They were kept as "
            "they are; fix or remove them (`mblab validate` shows why)."
        )
    if merged.replaced and not dry_run:
        console.print(f"The previous file was kept as {s.records_path}.bak.")
    if not dry_run:
        _print_validation_summary(validate_study(Study.load(study)))


@app.command()
@_errors
def validate(
    study: StudyOpt = Path("."),
    strict: Annotated[bool, typer.Option(help="Exit with an error on warnings too.")] = False,
    quiet: Annotated[bool, typer.Option(help="Hide informational notes.")] = False,
) -> None:
    """Check the definition, plan, records, and grades for errors and protocol problems."""
    s = Study.load(study)
    report = validate_study(s)
    style = {"error": "bold red", "warning": "yellow", "info": "dim"}
    for issue in report.issues:
        if quiet and issue.severity == "info":
            continue
        console.print(
            f"[{style[issue.severity]}]{issue.severity:7}[/] {issue.code}: {issue.message}",
        )
    console.print(
        f"planned {report.n_planned}, recorded {report.n_records}, missing {report.n_missing}, "
        f"excluded {report.n_excluded}"
    )
    _print_validation_summary(report)
    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)


# -- commands: analysis -----------------------------------------------------------------


def _print_condition_table(a: Analysis) -> None:
    conf = int(a.settings.confidence * 100)
    api = a.definition.execution_method == "official_api"
    table = Table(
        "cond",
        *(["setting"] if api else ["default", "per-message"]),
        "scored",
        "accuracy",
        f"{conf}% CI",
        "success",
        "format fail",
        f"median time ({method_text(a.primary_latency_method)})",
        "median reasoning tokens" if api else "thinking shown",
        title="Conditions",
    )
    api_by = a.api.set_index("condition_id") if api else None
    for row in a.conditions.itertuples():
        if api_by is not None:
            rep = api_by.loc[row.condition_id]
            last = f"{tokens(rep.reasoning_tokens_median)} (n={rep.reasoning_tokens_n})"
        else:
            last = f"{pct(row.thinking_shown_rate)} (n={row.thinking_known_n})"
        table.add_row(
            row.condition_id,
            *([row.model_selection] if api else [row.model_selection, row.message_override]),
            f"{row.n_scored}",
            pct(row.accuracy),
            pct_ci(row.accuracy_low, row.accuracy_high),
            pct(row.success_rate),
            str(row.n_format_failures),
            f"{seconds(row.latency_median)} (n={row.latency_n})",
            last,
        )
    console.print(table)


def _print_api_table(a: Analysis) -> None:
    table = Table(
        "cond",
        "effort sent",
        "API values",
        "model id(s) reported",
        "echoed effort ≠ sent",
        "incomplete",
        "reasoning tokens median [Q1–Q3]",
        "input / output tokens, all recorded responses",
        title="Values reported by the API (as returned, not verified)",
    )
    for r in a.api.itertuples():
        table.add_row(
            r.condition_id,
            str(r.requested_effort),
            f"{r.n_api_values}/{r.n_included}",
            ", ".join(f"{k} ({v})" for k, v in sorted(r.models.items())) or "—",
            str(r.n_effort_mismatch),
            ", ".join(f"{k} ({v})" for k, v in sorted(r.incomplete_reasons.items())) or "—",
            f"{tokens(r.reasoning_tokens_median)} [{tokens(r.reasoning_tokens_q1)}–"
            f"{tokens(r.reasoning_tokens_q3)}]",
            f"{tokens(r.input_tokens_sum)} / {tokens(r.output_tokens_sum)}",
        )
    console.print(table)


def _print_label_table(a: Analysis) -> None:
    headers = {
        "matches_override": "per-message",
        "matches_default": "default",
        "matches_both": "both",
        "other_option": "other option",
        "unrecognized": "unrecognized",
        "not_displayed": "none shown",
    }
    table = Table("cond", *[headers[c] for c in LABEL_CLASSES], title="Displayed model labels")
    for row in a.labels.itertuples(index=False):
        table.add_row(row.condition_id, *[str(getattr(row, c)) for c in LABEL_CLASSES])
    console.print(table)


def _print_overrides(a: Analysis) -> None:
    for item in a.overrides:
        kind = "same setting twice" if item.stacked else "per-message differs from default"
        console.print(f"[bold]Condition {item.condition_id}[/] ({kind})")
        console.print(f"  Labels: {item.label_summary}")
        if item.note:
            console.print(f"  {item.note}")
            continue
        table = Table(
            "metric",
            item.condition_id,
            f"default ref {item.default_ref}",
            f"per-message ref {item.override_ref}",
            "position (0 = default, 1 = per-message)",
            "assessment",
        )
        for m in item.metrics:
            table.add_row(
                m.metric,
                metric_value(m.condition_value, m.unit),
                metric_value(m.default_ref_value, m.unit),
                metric_value(m.override_ref_value, m.unit),
                "—" if item.stacked else position_text(m.position),
                VERDICT_SHORT[m.verdict],
            )
        console.print(table)
    if a.sensitivity is not None:
        check = a.sensitivity
        reasons = ", ".join(f"{k} ({n})" for k, n in check.reasons.items())
        console.print(
            f"Sensitivity check: without the {check.left_out} run(s) with a flagged per-run "
            f"value ({reasons}), "
            + (
                f"{len(check.changes)} of {check.n_assessments} assessments change: "
                + "; ".join(
                    f"{c} {m}: {VERDICT_SHORT[b]} -> {VERDICT_SHORT[x]}"
                    for c, m, b, x in check.changes
                )
                if check.changes
                else f"none of the {check.n_assessments} assessments change."
            ),
            markup=False,
        )


def _print_comparisons(a: Analysis) -> None:
    conf = int(a.settings.confidence * 100)
    table = Table(
        "metric",
        "first",
        "second",
        "n",
        f"first − second [{conf}% CI]",
        "status",
        title="Pairwise differences",
    )
    for row in a.comparisons.itertuples():
        unit = METRIC_UNITS[row.metric]
        iv = None if pd.isna(row.difference) else Interval(row.difference, row.low, row.high)
        table.add_row(
            METRIC_NAMES[row.metric],
            row.condition_a,
            row.condition_b,
            f"{row.n_a}/{row.n_b}",
            interval_text(iv, unit),
            COMPARISON_STATUS[row.status],
        )
    console.print(table)


@app.command("analyze")
@_errors
def analyze_cmd(
    study: StudyOpt = Path("."),
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the full analysis as JSON.")
    ] = None,
    min_n: Annotated[
        int | None,
        typer.Option(
            min=5,
            help="Minimum observations per group to compare (default: experiment.yaml).",
        ),
    ] = None,
    confidence: Annotated[
        float | None,
        typer.Option(
            min=0.8, max=0.995, help="Confidence level for intervals (default: experiment.yaml)."
        ),
    ] = None,
) -> None:
    """Score responses and summarise results per condition, with uncertainty."""
    s = Study.load(study)
    a = analyze(s, _settings(s, confidence=confidence, min_n=min_n))
    d = a.definition
    console.print(f"[bold]{d.title}[/] — {d.experiment_id} v{d.experiment_version}")
    if a.synthetic:
        console.print(f"[bold red]{DEMO_BANNER}[/]")
    if d.pilot:
        console.print(f"[bold yellow]{PILOT_BANNER}[/]")
    planned = len(a.plan.runs) if a.plan else 0
    console.print(
        f"Recorded {len(a.results)} of {planned} planned runs; "
        f"{int(a.results['excluded'].sum()) if not a.results.empty else 0} excluded."
    )
    _print_validation_summary(a.validation)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(dump_json(a.to_dict()), encoding="utf-8")
        console.print(f"Wrote {json_out}")
    if not a.has_data:
        console.print("No observations recorded yet — there is nothing to analyse.")
        return
    _print_condition_table(a)
    if a.definition.execution_method == "official_api":
        _print_api_table(a)
    else:
        _print_label_table(a)
        _print_overrides(a)
    _print_comparisons(a)
    console.print(
        "Accuracy counts format failures as incorrect; success rate also counts errors and "
        "refusals as failures. Intervals are approximate and assume independent responses."
    )
    console.print(
        f"{len(a.limitations)} limitation(s) apply. Full write-up: mblab report -s {study}"
    )


@app.command()
@_errors
def export(
    study: StudyOpt = Path("."),
    fmt: Annotated[str, typer.Option("--format", help="csv or json.")] = "csv",
    out: Annotated[
        Path | None, typer.Option(help="Output path (default: <study>/exports/results.<format>).")
    ] = None,
) -> None:
    """Export one row per recorded run, with its automatic or manual score."""
    if fmt not in {"csv", "json"}:
        raise StudyError("--format must be csv or json")
    s = Study.load(study)
    a = analyze(s)
    target = out or study / "exports" / f"results.{fmt}"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = a.results.copy()
    frame.insert(0, "synthetic_data", a.synthetic)
    if fmt == "csv":
        # Text that a spreadsheet would run as a formula gets a leading apostrophe.
        text_columns = [
            c
            for c in frame.columns
            if frame[c].dtype == object or pd.api.types.is_string_dtype(frame[c].dtype)
        ]
        frame[text_columns] = frame[text_columns].map(to_cell)
        frame.to_csv(target, index=False, encoding="utf-8-sig")  # BOM: spreadsheets read UTF-8
    else:
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
        target.write_text(dump_json(rows), encoding="utf-8")
    console.print(f"Wrote {target} ({len(frame)} rows).")
    if a.synthetic:
        console.print(f"[bold red]{DEMO_BANNER}[/]")


grade_app = typer.Typer(
    help="Blind manual grading for tasks scored by hand.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(grade_app, name="grade")


@grade_app.command("export")
@_errors
def grade_export(
    study: StudyOpt = Path("."),
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Where to write the sheet and key (default: <study>/grading)."),
    ] = None,
) -> None:
    """Write a shuffled grading sheet without condition information, plus a separate key."""
    if not study.is_dir():
        raise StudyError(f"study directory {study} does not exist")
    with write_lock(study, "grade export"):
        s = Study.load(study)
        sheet, key, n = export_blind_sheet(s, out_dir)
    console.print(f"Wrote {sheet} ({n} responses) and {key}.")
    console.print("Give graders the sheet only. Keep the key file away from them.")


@grade_app.command("import")
@_errors
def grade_import(
    sheet: Annotated[Path, typer.Argument(help="The filled-in grading sheet (CSV).")],
    study: StudyOpt = Path("."),
    key: Annotated[
        Path | None, typer.Option(help="Key file (default: blind-key.json next to the sheet).")
    ] = None,
    grader: Annotated[
        str | None, typer.Option(help="Grader id for rows without a grader_id.")
    ] = None,
) -> None:
    """Import points from a grading sheet into grades.jsonl."""
    with write_lock(study, "grade import"):
        s = Study.load(study)  # read inside the lock: duplicate checks see every saved grade
        result = import_blind_sheet(s, sheet, key or sheet.with_name("blind-key.json"), grader)
        if result.errors:
            for err in result.errors:
                err_console.print(f"[red]{escape(err)}[/]")  # errors quote file values
            raise StudyError(f"{len(result.errors)} invalid row(s); no grades were imported")
        if result.grades:
            append_jsonl(s.grades_path, result.grades)
    console.print(
        f"Imported {len(result.grades)} grade(s); {result.skipped_blank} row(s) without points."
    )


# -- commands: report and dashboard ----------------------------------------------------


#: Diff lines shown by ``mblab report --check`` before the rest is summarised.
CHECK_DIFF_LINES = 80


def _check_report(path: Path, text: str) -> None:
    if not path.is_file():
        raise StudyError(f"{path} not found (--check needs the report to compare with)")
    comparison = compare_reports(path.read_text(encoding="utf-8", errors="replace"), text)
    current = code_version().label()
    if comparison.same:
        console.print(
            f"[green]{escape(str(path))} matches the report made now[/] (the generation time "
            "and the code version are not compared)."
        )
    else:
        console.print(
            f"[red]{escape(str(path))} differs from the report made now[/] in "
            f"{comparison.changed_lines} line(s), not counting the generation time and the code "
            "version:"
        )
        for line in comparison.diff[:CHECK_DIFF_LINES]:
            console.print(line, markup=False)
        if len(comparison.diff) > CHECK_DIFF_LINES:
            console.print(f"… {len(comparison.diff) - CHECK_DIFF_LINES} more diff line(s)")
        if comparison.settings_differ:
            console.print(
                "One of the two reports uses analysis settings other than the pre-registered "
                "ones: repeat the options shown in the report's Reproduction section."
            )
    if comparison.made_with and comparison.made_with != current:
        console.print(
            f"The report was made with model-behavior-lab {comparison.made_with}; this "
            f"is {current}."
            + (
                ""
                if comparison.same
                else " Install the code named in its Reproduction section to compare like with like."
            ),
            markup=False,
        )
    if not comparison.same:
        raise typer.Exit(1)


@app.command()
@_errors
def report(
    study: StudyOpt = Path("."),
    out: Annotated[
        Path | None, typer.Option(help="Output path (default: <study>/reports/report.md).")
    ] = None,
    min_n: Annotated[
        int | None,
        typer.Option(
            min=5,
            help="Minimum observations per group to compare (default: experiment.yaml).",
        ),
    ] = None,
    confidence: Annotated[
        float | None,
        typer.Option(
            min=0.8, max=0.995, help="Confidence level for intervals (default: experiment.yaml)."
        ),
    ] = None,
    check: Annotated[
        Path | None,
        typer.Option(
            help="Compare this existing report with one made now and write nothing. Only the "
            "generation time and the code version are ignored; exit code 1 if they differ.",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Replace an existing report whose content differs.")
    ] = False,
) -> None:
    """Write a Markdown report: question, hypotheses, method, results, limitations.

    An existing report is never replaced by one with other content unless --force is
    given; --check compares instead of writing.
    """
    if check is not None and (out is not None or force):
        raise StudyError("--check writes nothing, so it cannot be combined with --out or --force")
    s = Study.load(study)
    a = analyze(s, _settings(s, confidence=confidence, min_n=min_n))
    text = render_report(a, s)
    if check is not None:
        _check_report(check, text)
        return
    target = out or study / "reports" / "report.md"
    if target.exists() and not force:
        comparison = compare_reports(target.read_text(encoding="utf-8", errors="replace"), text)
        if not comparison.same:
            raise StudyError(
                f"{target} already exists and differs from the report made now "
                f"({comparison.changed_lines} line(s), not counting the generation time and the "
                f"code version). See the differences with `mblab report -s {study} --check "
                f"{target}`, write the new report elsewhere with --out, or replace it with --force."
            )
        made = (
            f", made with model-behavior-lab {comparison.made_with}" if comparison.made_with else ""
        )
        console.print(f"{target} already holds this report{made}. Left unchanged.", markup=False)
    else:
        atomic_write_text(target, text)
        console.print(f"Wrote {target}", markup=False)
    if a.synthetic:
        console.print(f"[bold red]{DEMO_BANNER}[/]")
    if a.definition.pilot:
        console.print(f"[bold yellow]{PILOT_BANNER}[/]")
    if a.validation.errors:
        err_console.print(
            f"[yellow]warning:[/] {len(a.validation.errors)} validation error(s); the report "
            "says so at the top. Details: mblab validate"
        )
    if not a.has_data:
        console.print("No observations yet: the report contains the design but no results.")


@app.command()
@_errors
def dashboard(
    study: StudyOpt = Path("."),
    port: Annotated[int | None, typer.Option(help="Port for the local server.")] = None,
    address: Annotated[
        str,
        typer.Option(
            help="Interface to listen on. The default keeps the dashboard on this computer; "
            "study data can contain response text."
        ),
    ] = "localhost",
    headless: Annotated[bool, typer.Option(help="Do not open a browser window.")] = False,
) -> None:
    """Open the local web dashboard for a study or a benchmark (needs the `dashboard` extra)."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise StudyError(install_hint("dashboard", "the dashboard needs Streamlit")) from None
    if is_benchmark_dir(study):  # fail early with a clear message
        Benchmark.load(study)
    else:
        Study.load(study)
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
        "--server.address",
        address,
    ]
    if port:
        cmd += ["--server.port", str(port)]
    if headless:
        cmd += ["--server.headless", "true"]
    root = str(study.resolve())
    cmd += ["--", "--study", root]
    console.print(f"Starting the dashboard for {root} (Ctrl+C to stop)…")
    raise typer.Exit(subprocess.call(cmd, env={**os.environ, "MBLAB_STUDY": root}))


# -- commands: demo ------------------------------------------------------------------------


@app.command()
@_errors
def demo(
    directory: Annotated[Path, typer.Argument(help="New directory for the synthetic study.")],
    seed: Annotated[int, typer.Option(help="Random seed for the synthetic values.")] = 7,
    force: Annotated[bool, typer.Option(help="Write into a non-empty directory.")] = False,
) -> None:
    """Create a SYNTHETIC demo study (random values, clearly labelled) to try the tools."""
    s = create_demo_study(directory, seed=seed, force=force)
    console.print(f"[bold red]{DEMO_BANNER}[/]")
    console.print(f"Created {directory} with {len(s.records)} synthetic records.")
    console.print(f"Try: mblab analyze -s {directory}   |   mblab dashboard -s {directory}")


# -- commands: benchmark -------------------------------------------------------------------

benchmark_app = typer.Typer(
    help="Benchmarks: a fixed task with an independently computed ground truth.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(benchmark_app, name="benchmark")

BenchmarkDir = Annotated[
    Path, typer.Argument(help="Benchmark directory (the one holding benchmark.yaml).")
]


@benchmark_app.command("show")
@_errors
def benchmark_show(
    directory: BenchmarkDir,
    batch: Annotated[
        list[str] | None, typer.Option(help="Only the runs of this batch (repeatable).")
    ] = None,
) -> None:
    """Recompute the ground truth and score every run (nothing is written)."""
    b = Benchmark.load(directory)
    ev = evaluate(b, batch)
    truth = ev.truth
    console.print(f"[bold]{escape(b.definition.title)}[/] ({escape(b.definition.benchmark_id)})")
    console.print(
        f"{ev.n_runs} observed run(s), recorded by hand, in batch(es) "
        f"{escape(', '.join(ev.batches))}."
    )
    by_first = ", ".join(f"{k}: {v}" for k, v in truth.prefix_counts(1).items())
    console.print(
        f"Ground truth, computed by checking every order: {truth.count} valid orders; the first "
        f"is {truth.lexicographic_first}; by first task {by_first}."
    )
    table = Table(show_lines=False)
    for name in ("Run", "Default", "Per-message", "Count", "Exact count", "First order",
                 "Second order", "Full success", "Error"):  # fmt: skip
        table.add_column(name)
    for row in runs_frame(ev).itertuples():
        count = "—" if row.count_given is None else str(row.count_given)
        table.add_row(
            escape(row.run_id),
            escape(row.default_selection),
            escape(row.per_message_selection),
            count,
            row.exact_count,
            row.lexicographic_first,
            row.other_valid_order,
            row.full_task_success,
            escape(row.error_type),
        )
    console.print(table)
    n = ev.n_runs
    console.print(
        f"Full task success: {ev.full_successes} pass, {n - ev.full_successes} fail (of {n}); "
        f"partially correct: {ev.partials}."
    )
    parts = [
        f"{name} {ev.correct(check)[0]}/{n}"
        for name, check in (
            ("exact count", "count"),
            ("first order", "lexicographic_first"),
            ("second valid order", "other_valid_order"),
        )
    ]
    console.print("By answer: " + "; ".join(parts) + ".")
    for r in ev.runs:
        if not r.full_success or any(not c.matches for c in r.breakdown):
            console.print(f"{escape(r.run.run_id)}: {escape(r.error_detail(truth))}")
    for comp in ev.comparisons:
        console.print(f"[bold]{escape(comp.title)}[/]: {escape(comp.text)}")
    console.print(
        "Descriptive only: with this many runs nothing here is a test or a measure of any "
        "model's general accuracy."
    )


@benchmark_app.command("report")
@_errors
def benchmark_report(
    directory: BenchmarkDir,
    check: Annotated[
        bool,
        typer.Option("--check", help="Make the reports again and compare; write nothing."),
    ] = False,
    report_id: Annotated[
        str | None, typer.Option("--report", help="Write or check only this report ID.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Replace selected reports whose content differs.")
    ] = False,
) -> None:
    """Write the reports benchmark.yaml lists; each covers the runs of its batches."""
    if check and force:
        raise StudyError("--check writes nothing, so it cannot be combined with --force")
    b = Benchmark.load(directory)
    if not b.definition.reports and report_id is None:
        raise StudyError(f"{BENCHMARK_FILE} lists no reports")
    if not check:
        for path in write_reports(b, report_id, force=force):
            console.print(f"Report ready: {escape(str(path))}")
        return
    differing = 0
    for spec, text in make_reports(b, report_id):
        path = report_path(b, spec)
        if not path.is_file():
            console.print(f"[red]{escape(str(path))} not found[/]")
            differing += 1
            continue
        existing = saved_report_text(path)
        if existing is None:
            console.print(f"[red]{escape(str(path))} is not valid UTF-8[/]")
            differing += 1
            continue
        comparison = compare_reports(existing, text, ignore_volatile=False)
        if comparison.same:
            console.print(f"[green]{escape(str(path))} matches the report made now.[/]")
            continue
        differing += 1
        console.print(
            f"[red]{escape(str(path))} differs from the report made now[/] in "
            f"{comparison.changed_lines} line(s):"
        )
        for line in comparison.diff[:CHECK_DIFF_LINES]:
            console.print(line, markup=False)
    if differing:
        raise typer.Exit(1)
