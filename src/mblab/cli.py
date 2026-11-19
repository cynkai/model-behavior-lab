"""Command-line interface: ``mblab``."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from mblab import __version__
from mblab.clipboard import copy_text, paste_text
from mblab.exchange import import_file, merge_records, parse_timezone, write_template_csv
from mblab.plan import generate_plan
from mblab.records import build_record, condition_instructions, next_pending
from mblab.schema import PlannedRun, RunRecord
from mblab.study import (
    DEFAULT_TEMPLATE,
    DEFINITION_FILE,
    PLAN_FILE,
    RECORDS_FILE,
    Study,
    StudyError,
    append_jsonl,
    available_templates,
    format_validation_error,
    load_definition,
    load_plan,
    parse_definition,
    read_jsonl,
    save_plan,
    template_text,
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
)
console = Console(highlight=False, soft_wrap=True)
err_console = Console(stderr=True, highlight=False, soft_wrap=True)

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
            err_console.print(f"[bold red]error:[/] {exc}")
            raise typer.Exit(2) from None

    return wrapper  # type: ignore[return-value]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mblab {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Model Behavior Lab."""


# -- helpers ---------------------------------------------------------------------


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


def _choose(question: str, choices: list[str], default: str) -> str:
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
        return float(text.replace(",", "."))
    except ValueError:
        raise StudyError(f"{what}: {text!r} is not a number") from None


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
    parse_definition(text, template)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DEFINITION_FILE
    if target.exists() and not force:
        raise StudyError(f"{target} already exists (use --force to overwrite)")
    target.write_text(text, encoding="utf-8")
    console.print(f"Created {target}")
    console.print("Next steps:")
    console.print(
        f"  1. Edit {target}: set `platform`, and copy the option labels exactly as your "
        "client shows them. Settle the definition before collecting data."
    )
    console.print(f"  2. mblab plan -s {directory}")


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
    existing = load_plan(plan_path)
    records, _ = read_jsonl(study / RECORDS_FILE, RunRecord)
    if existing is not None and not force:
        raise StudyError(
            f"{plan_path} already exists. Regenerating can change the run order; "
            "use --force only if that is intended."
        )
    new_plan = generate_plan(definition, seed=seed)
    if existing is not None and records:
        old_order = [r.run_id for r in existing.runs]
        new_order = [r.run_id for r in new_plan.runs]
        if old_order != new_order:
            err_console.print(
                f"[yellow]warning:[/] {len(records)} record(s) exist and the run order changed. "
                "Recorded runs keep their run ids; report this as a protocol deviation."
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
    recorded = Counter(r.condition_id for r in s.records)
    excluded = Counter(r.condition_id for r in s.records if r.excluded)
    planned = Counter(r.condition_id for r in plan_.runs)
    table = Table("condition", "default", "per-message", "recorded", "excluded")
    for cond in definition.conditions:
        table.add_row(
            cond.id,
            definition.option_label(cond.model_selection),
            definition.option_label(cond.message_override),
            f"{recorded[cond.id]}/{planned[cond.id]}",
            str(excluded[cond.id]),
        )
    console.print(table)
    console.print(f"Recorded {len(s.records)} of {len(plan_.runs)} planned runs.")
    nxt = next_pending(plan_, s.recorded_run_ids())
    if nxt is None:
        console.print("All planned runs are recorded.")
    else:
        console.print(f"Next run: #{nxt.sequence} {nxt.run_id} — mblab next -s {study} --copy")
    _print_validation_summary(validate_study(s))


@app.command("next")
@_errors
def next_run(
    study: StudyOpt = Path("."),
    copy: Annotated[bool, typer.Option("--copy", help="Copy the prompt to the clipboard.")] = False,
) -> None:
    """Show the next pending run: what to set in the client and the exact prompt."""
    s = Study.load(study)
    plan_ = s.require_plan()
    run = next_pending(plan_, s.recorded_run_ids())
    if run is None:
        console.print("All planned runs are recorded.")
        return
    prompt = _show_run(s, run, len(plan_.runs))
    if copy:
        if copy_text(prompt):
            console.print("[green]Prompt copied to the clipboard.[/]")
        else:
            console.print("[yellow]No clipboard tool found; copy the prompt above by hand.[/]")
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
    plan_ = s.require_plan()
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
        default = (prev.confounders.get(conf.id) if prev else None) or (
            "unknown" if "unknown" in conf.allowed_values else conf.allowed_values[0]
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
    if rec.run_id in s.recorded_run_ids():
        merge_records(s, [rec], replace=True, dry_run=False)
    else:
        append_jsonl(s.records_path, [rec])
    console.print(f"[green]Saved to {s.records_path}.[/]")
    nxt = next_pending(plan_, s.recorded_run_ids() | {rec.run_id})
    if nxt:
        console.print(f"Next: mblab next -s {study} --copy   (#{nxt.sequence} {nxt.run_id})")


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
        typer.Option(help="Time zone for timestamps without an offset, e.g. +09:00 or Asia/Seoul."),
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
            err_console.print(f"[red]{err}[/]")
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
