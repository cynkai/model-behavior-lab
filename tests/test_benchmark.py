"""Benchmark 01: the stored runs, their scoring against the verified ground truth, the
report made from them, and the rules that keep further runs from overwriting earlier ones."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mblab import task_ordering
from mblab.benchmark import (
    BENCHMARK_FILE,
    RUNS_FILE,
    Benchmark,
    BenchmarkRun,
    _answer_values,
    check_answers,
    dump_run,
    evaluate,
    make_reports,
    render_report,
    runs_frame,
    selection_grid,
)
from mblab.cli import app
from mblab.study import StudyError

from .test_analysis import BANNED

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "benchmark-001-constraint-reasoning"
PRELIMINARY = "2026-11-22-preliminary"
#: SHA-256 of the six preliminary runs (as the report prints it). Later runs are appended
#: in new batches; these six lines are never edited.
PRELIMINARY_SHA256 = "cab09c4877bf7ffef2aa9fa6bccb0849196484b80c2d625eef9cf1cd6191cbf4"
RUN_KEYS = {
    "schema_version",
    "run_id",
    "benchmark_id",
    "batch",
    "condition_id",
    "model_selection",
    "message_override",
    "repetition",
    "data_origin",
    "recorded_on",
    "timestamp",
    "run_order",
    "device_type",
    "device_model",
    "operating_system",
    "app_version",
    "new_conversation",
    "prompt_sha256",
    "observed_model_label",
    "latency_seconds",
    "latency_method",
    "raw_response",
    "answers",
    "reported_breakdown",
    "solution_summary",
    "source",
    "notes",
}


@pytest.fixture(scope="module")
def bench() -> Benchmark:
    return Benchmark.load(BENCH)


@pytest.fixture(scope="module")
def ev(bench):
    return evaluate(bench)


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / BENCH.name
    shutil.copytree(BENCH, root)
    return root


def _append(root: Path, *runs: BenchmarkRun) -> None:
    with (root / RUNS_FILE).open("a", encoding="utf-8") as handle:
        for run in runs:
            handle.write(dump_run(run) + "\n")


def _repeat(bench: Benchmark, cid: str, repetition: int, **changes) -> BenchmarkRun:
    base = next(r for r in bench.runs if r.condition_id == cid)
    update = {"run_id": f"{cid}:T1:r{repetition}", "repetition": repetition, "batch": "next"}
    return base.model_copy(update={**update, **changes})


# -- the stored data ------------------------------------------------------------------------


def test_six_conditions_with_the_recorded_settings(bench):
    settings = {c.id: (c.model_selection, c.message_override) for c in bench.definition.conditions}
    assert settings == {
        "A": ("instant", None),
        "B": ("pro", None),
        "C": ("instant", "pro"),
        "D": ("pro", "instant"),
        "E": ("extra-high", None),
        "F": ("extra-high", "extra-high"),
    }
    labels = {o.id: o.label for o in bench.definition.selection_options}
    assert labels == {"instant": "Instant", "pro": "Pro", "extra-high": "Extra High"}


def test_the_answers_are_recorded_as_given(bench):
    assert [(r.run_id, r.answers["count"]) for r in bench.runs] == [
        ("A:T1:r1", "20"),
        ("B:T1:r1", "23"),
        ("C:T1:r1", "23"),
        ("D:T1:r1", "25"),
        ("E:T1:r1", "23"),
        ("F:T1:r1", "23"),
    ]
    for run in bench.runs:
        assert run.answers["lexicographic_first"] == "ACBDEFGH"
        assert run.answers["other_valid_order"] == "ACBDEGFH"
        assert run.solution_summary
        assert run.repetition == 1 and run.batch == PRELIMINARY


def test_values_that_were_not_recorded_stay_null(bench):
    for run in bench.runs:
        assert run.raw_response is None  # only the experimenter's summary was kept
        assert run.timestamp is None and run.run_order is None
        assert run.device_model is None and run.operating_system is None
        assert run.app_version is None and run.observed_model_label is None
        assert run.latency_seconds is None and run.latency_method == "not_measured"
        assert run.device_type == "tablet" and run.new_conversation is True
        assert run.recorded_on == date(2026, 11, 22)
        assert run.data_origin == "observed"


def test_the_preliminary_runs_are_unchanged(bench):
    assert evaluate(bench, [PRELIMINARY]).data_sha256() == PRELIMINARY_SHA256


def test_run_lines_have_exactly_these_keys():
    for line in (BENCH / RUNS_FILE).read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == RUN_KEYS


def test_every_run_used_the_prompt_in_the_definition(bench):
    assert {r.prompt_sha256 for r in bench.runs} == {bench.definition.task.prompt_sha256()}


def test_reference_answers_agree_with_the_verifier(bench):
    truth = task_ordering.ground_truth()
    refs = {a.check: a.reference_answer for a in bench.definition.task.answers}
    assert refs["count"] == str(truth.count) == "23"
    assert refs["lexicographic_first"] == truth.lexicographic_first == "ACBDEFGH"
    assert task_ordering.is_valid(refs["other_valid_order"])


# -- scoring --------------------------------------------------------------------------------


def test_four_runs_pass_and_two_fail(ev):
    assert {r.condition.id: r.full_success for r in ev.runs} == {
        "A": False,
        "B": True,
        "C": True,
        "D": False,
        "E": True,
        "F": True,
    }
    assert (ev.n_runs, ev.full_successes, ev.partials) == (6, 4, 2)
    assert ev.correct("count") == (4, 6)
    assert ev.correct("lexicographic_first") == (6, 6)
    assert ev.correct("other_valid_order") == (6, 6)


def test_the_wrong_counts_are_traced_to_one_branch_each(ev):
    by = {r.condition.id: r for r in ev.runs}
    assert by["A"].error_type(ev.truth) == "count too low"
    assert by["D"].error_type(ev.truth) == "count too high"
    wrong = {
        cid: [(c.key, c.reported, c.verified) for c in by[cid].breakdown if not c.matches]
        for cid in "AD"
    }
    assert wrong == {"A": [("first:A", 6, 9)], "D": [("first:C", 10, 8)]}
    for cid in "BCEF":
        assert by[cid].error_type(ev.truth) == "none"
        assert by[cid].breakdown and all(c.matches for c in by[cid].breakdown)


def test_comparisons_name_the_reference_and_stay_descriptive(ev):
    assert [(c.title, c.pattern) for c in ev.comparisons] == [
        ("B vs C", "per_message"),
        ("A vs D", "per_message"),
        ("E vs F", "no_difference"),
    ]
    first, second, third = (c.text for c in ev.comparisons)
    assert "not direct evidence" in first and "(H1)" in first
    assert "D gave 25 and A gave 20" in second
    assert "does not show that the two settings are the same" in third
    for comp in ev.comparisons:
        assert "not a test" in comp.text
        assert not BANNED.search(comp.text)


def test_tables_for_the_dashboard(ev):
    frame = runs_frame(ev)
    assert list(frame["count_given"]) == [20, 23, 23, 25, 23, 23]
    assert list(frame["full_task_success"]) == ["Fail", "Pass", "Pass", "Fail", "Pass", "Pass"]
    grid = selection_grid(ev).set_index("default selection")
    assert grid.loc["Instant", "Pro"].startswith("C: 1/1")
    assert grid.loc["Pro", "Instant"].startswith("D: 0/1")
    assert grid.loc["Extra High", "Extra High"].startswith("F: 1/1")


def _given(count: str, first: str | None = None, other: str | None = None) -> dict[str, str]:
    given = {"count": count, "lexicographic_first": first, "other_valid_order": other}
    return {k: v for k, v in given.items() if v is not None}


@pytest.mark.parametrize(
    ("values", "correct"),
    [
        (_given("23개", "A C B D E F G H", "A-C-B-D-E-G-F-H"), [True, True, True]),
        (_given("23 or 24", "ACBDEFGH", "ACBDEGFH"), [False, True, True]),
        (_given("23", "ACBDEFGH", "ACBDEFGH"), [True, True, False]),  # same order twice
        (_given("23", "ACBDEGFH", "ACBDEFGH"), [True, False, True]),  # differs from its own (2)
        (_given("23", "ACBDEFGH", "ABCDEFGH"), [True, True, False]),  # breaks rule 6
        (_given("23", "ACBDEFG", "ACBDEGFH"), [True, False, True]),  # (2) unreadable
        (_given("23"), [True, False, False]),
    ],
)
def test_answer_checks(bench, values, correct):
    task = bench.definition.task
    results = check_answers(task, task_ordering, task_ordering.ground_truth(), values)
    assert [r.correct for r in results] == correct


def test_an_order_the_rules_forbid_says_which_rule(bench):
    task = bench.definition.task
    values = {"count": "23", "lexicographic_first": "ACBDEFGH", "other_valid_order": "ABCDEFGH"}
    other = check_answers(task, task_ordering, task_ordering.ground_truth(), values)[2]
    assert other.readable and not other.correct
    assert "breaks rule 6" in other.detail


def test_count_is_displayed_with_the_precision_used_for_its_score(bench):
    answers = _given("22.99999", "ACBDEFGH", "ACBDEGFH")
    run = _repeat(bench, "A", 2, answers=answers)
    ev = evaluate(Benchmark(bench.root, bench.definition, [run]))
    result = ev.runs[0]
    assert result.error_type(ev.truth) == "count too low"
    assert "count 22.99999 instead of 23" in result.error_detail(ev.truth)
    assert "A: 22.99999" in render_report(ev, bench.report_spec("preliminary"))


def test_a_kept_response_is_read_by_its_answer_lines(bench):
    raw = (
        "풀이 …\n\n**총 경우의 수:** 23\n사전순 첫 번째 순서: ACBDEFGH\n"
        "다른 유효한 순서: ACFBDEGH\n검증 설명: …"
    )
    run = bench.runs[0].model_copy(update={"raw_response": raw, "answers": {}})
    task = bench.definition.task
    values = _answer_values(task, run)
    assert values["count"] == "23" and values["other_valid_order"] == "ACFBDEGH"
    results = check_answers(task, task_ordering, task_ordering.ground_truth(), values)
    assert all(r.correct for r in results)


# -- report ---------------------------------------------------------------------------------


def test_the_committed_report_is_what_the_data_makes(bench):
    for spec, text in make_reports(bench):
        assert (BENCH / spec.path).read_text(encoding="utf-8") == text, spec.path


@pytest.mark.parametrize(
    ("answers", "detail"),
    [
        ({"count": "25", "lexicographic_first": "ABCDEFGH"}, "lexicographic_first:"),
        ({"count": "unclear"}, "no number"),
        ({"other_valid_order": "ACBDEGFH"}, "count: no answer available"),
    ],
)
def test_report_does_not_attribute_every_failure_to_a_miscount(bench, answers, detail):
    base = bench.runs[0]
    run = _repeat(bench, "A", 2, answers=answers, reported_breakdown={})
    ev = evaluate(Benchmark(bench.root, bench.definition, [base, run]), ["next"])
    text = render_report(ev, bench.report_spec("preliminary"))
    assert "gave a correct first order and a valid second order" not in text
    assert "traced by comparing the intermediate counts" not in text
    assert detail in ev.runs[0].error_detail(ev.truth)
    assert ev.runs[0].error_detail(ev.truth) in text


def test_unreadable_count_with_two_correct_orders_is_not_a_measured_miscount(bench):
    answers = {**bench.runs[0].answers, "count": "unknown"}
    run = _repeat(bench, "A", 2, answers=answers, reported_breakdown={})
    ev = evaluate(Benchmark(bench.root, bench.definition, [run]))
    text = render_report(ev, bench.report_spec("preliminary"))
    assert "miscounted the valid orders" not in text
    assert "count: no number" in text


def test_valid_order_metric_is_independent_of_rank_and_distinctness(bench):
    answers = _given("23", "ACBDEGFH", "ACBDEGFH")
    run = _repeat(bench, "A", 2, answers=answers)
    ev = evaluate(Benchmark(bench.root, bench.definition, [run]))
    assert ev.correct("lexicographic_first") == ev.correct("other_valid_order") == (0, 1)
    assert ev.valid_order_answers == 2
    assert "| Valid orders given (both order answers) | 2 of 2 (100%) |" in render_report(
        ev, bench.report_spec("preliminary")
    )


def test_report_sections_and_wording(bench):
    (spec, text), *_ = make_reports(bench)
    assert spec.title == "Benchmark 01: ChatGPT Model Selection Override — Preliminary Results"
    assert re.findall(r"^## (.+)$", text, flags=re.MULTILINE) == [
        "Abstract",
        "Research Question",
        "Hypotheses",
        "Experimental Setup",
        "Benchmark Task",
        "Ground Truth",
        "Observations",
        "Results",
        "Error Analysis",
        "Limitations",
        "Next Experiments",
        "Reproduction",
    ]
    assert "Exploratory, preliminary results" in text
    assert "| Full task success (all three answers correct) | 4 of 6 (66.7%) |" in text
    assert "66.7% describes 6 runs of one task" in text
    assert "Implied by the other rules: rule 8." in text
    assert bench.definition.task.prompt in text  # verbatim
    assert "latency" not in text.lower()  # no response-time figures: none were recorded
    assert not BANNED.search(text)


# -- adding runs ----------------------------------------------------------------------------


def test_a_further_run_is_added_without_changing_earlier_results(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "A", 2, answers={**bench.runs[0].answers, "count": "23"}))
    more = Benchmark.load(root)
    assert len(more.runs) == 7 and more.batches == [PRELIMINARY, "next"]
    before = evaluate(more, [PRELIMINARY])
    assert before.data_sha256() == PRELIMINARY_SHA256 and before.full_successes == 4
    everything = evaluate(more)
    assert everything.n_runs == 7 and len(everything.of_condition("A")) == 2
    assert everything.runs_per_condition() == "1 to 2"
    assert "describes these runs only" in everything.comparisons[1].text
    # The preliminary report covers its own batch only, so it does not change.
    (spec, text), *_ = make_reports(more)
    assert text == (BENCH / spec.path).read_text(encoding="utf-8")


def test_a_repetition_cannot_be_recorded_twice(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "A", 1, run_id="A:T1:again"))
    with pytest.raises(StudyError, match="both repetition 1 of condition A"):
        Benchmark.load(root)


def test_run_ids_are_unique(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "A", 2, run_id="A:T1:r1"))
    with pytest.raises(StudyError, match="duplicate run id 'A:T1:r1'"):
        Benchmark.load(root)


def test_a_run_must_match_its_condition(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "A", 2, message_override="pro"))
    with pytest.raises(StudyError, match="do not match condition A"):
        Benchmark.load(root)


def test_the_prompt_cannot_change_once_runs_are_recorded(tmp_path):
    root = _copy(tmp_path)
    path = root / BENCHMARK_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("불확실성을 명시하라.", "불확실성을 명시하라!"), encoding="utf-8")
    with pytest.raises(StudyError, match="prompt_sha256 does not match"):
        Benchmark.load(root)


def test_a_wrong_reference_answer_is_refused(tmp_path):
    root = _copy(tmp_path)
    path = root / BENCHMARK_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('reference_answer: "23"', 'reference_answer: "24"'), "utf-8")
    with pytest.raises(StudyError, match="not correct according to the verifier"):
        Benchmark.load(root)


def test_an_unknown_breakdown_key_is_refused(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "B", 2, reported_breakdown={"first:Q": 1}))
    with pytest.raises(StudyError, match="unknown breakdown key 'first:Q'"):
        Benchmark.load(root)


def test_a_report_needs_runs_in_its_batches(tmp_path):
    root = _copy(tmp_path)
    path = root / BENCHMARK_FILE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("batches: [2026-11-22-preliminary]", "batches: [later]"), "utf-8")
    with pytest.raises(StudyError, match="lists batch 'later', which has no runs"):
        Benchmark.load(root)


def test_optional_benchmark_note_scope(tmp_path):
    root = _copy(tmp_path)
    path = root / BENCHMARK_FILE
    original = path.read_text(encoding="utf-8")
    scope = f"notes_batches: [{PRELIMINARY}]\n"
    path.write_text(original.replace(scope, ""), encoding="utf-8")
    assert "notes_batches" not in Benchmark.load(root).definition.model_dump()
    path.write_text(original.replace(scope, "notes_batches: [typo]\n"), encoding="utf-8")
    with pytest.raises(StudyError, match="notes_batches lists batch 'typo'"):
        Benchmark.load(root)


def test_reproduction_commands_also_select_a_batch_of_first_repetitions(bench):
    report = bench.report_spec("preliminary").model_copy(
        update={
            "id": "b-only",
            "batches": ["first-b"],
            "path": "reports/b-only.md",
            "reproduction_scope": None,
        }
    )
    runs = [
        r.model_copy(update={"batch": "first-b"}) if r.condition_id == "B" else r
        for r in bench.runs
    ]
    definition = bench.definition.model_copy(
        update={"reports": [*bench.definition.reports, report]}
    )
    b = Benchmark(bench.root, definition, runs)
    [(spec, text)] = make_reports(b, "b-only")
    assert "reproduction_scope" not in spec.model_dump()
    assert f"mblab benchmark show benchmarks/{b.root.name} --batch first-b" in text
    assert f"mblab benchmark report benchmarks/{b.root.name} --check --report b-only" in text


def test_a_run_needs_an_answer_or_the_response(bench):
    data = bench.runs[0].model_dump(mode="json") | {"answers": {}, "raw_response": None}
    with pytest.raises(ValidationError, match="raw_response or answers"):
        BenchmarkRun.model_validate(data)


def test_only_observed_runs(bench):
    data = bench.runs[0].model_dump(mode="json") | {"data_origin": "synthetic"}
    with pytest.raises(ValidationError):
        BenchmarkRun.model_validate(data)


# -- command line ---------------------------------------------------------------------------


def test_show_prints_the_scores():
    result = CliRunner().invoke(app, ["benchmark", "show", str(BENCH)])
    assert result.exit_code == 0, result.output
    assert "23 valid orders" in result.output
    assert "Full task success: 4 pass, 2 fail (of 6); partially correct: 2." in result.output
    assert "first:A" not in result.output  # descriptions, not keys
    assert "valid orders that start with A: 6 reported, 9 verified" in result.output


def test_report_check_passes_on_the_committed_report():
    result = CliRunner().invoke(app, ["benchmark", "report", str(BENCH), "--check"])
    assert result.exit_code == 0, result.output
    assert "matches the report made now" in result.output


def test_report_check_fails_on_an_edited_report(tmp_path):
    root = _copy(tmp_path)
    report = root / "reports" / "preliminary.md"
    report.write_text(report.read_text(encoding="utf-8").replace("4 of 6", "5 of 6"), "utf-8")
    result = CliRunner().invoke(app, ["benchmark", "report", str(root), "--check"])
    assert result.exit_code == 1
    assert "differs from the report made now" in result.output


def test_report_writes_the_listed_reports(tmp_path):
    root = _copy(tmp_path)
    (root / "reports" / "preliminary.md").unlink()
    result = CliRunner().invoke(app, ["benchmark", "report", str(root)])
    assert result.exit_code == 0, result.output
    expected = (BENCH / "reports" / "preliminary.md").read_text(encoding="utf-8")
    assert (root / "reports" / "preliminary.md").read_text(encoding="utf-8") == expected


def test_report_selection_isolates_the_followup_on_write_and_check(tmp_path, bench):
    root = _copy(tmp_path)
    _append(root, _repeat(bench, "B", 2))
    with (root / BENCHMARK_FILE).open("a", encoding="utf-8") as f:
        f.write(
            "\n  - id: followup\n    title: Follow-up\n    batches: [next]\n"
            "    path: reports/followup.md\n"
        )
    archive = root / "reports" / "preliminary.md"
    archive.write_text("local edits\n", encoding="utf-8")
    cli = CliRunner()
    command = ["benchmark", "report", str(root)]
    selected = [*command, "--report", "followup"]
    followup = root / "reports" / "followup.md"
    assert cli.invoke(app, [*selected, "--check"]).exit_code == 1
    assert not followup.exists()
    assert cli.invoke(app, selected).exit_code == 0
    text = followup.read_text(encoding="utf-8")
    assert "1 (batches: next)" in text
    assert f"mblab benchmark show benchmarks/{root.name} --batch next" in text
    assert f"mblab benchmark report benchmarks/{root.name} --check --report followup" in text
    assert archive.read_text(encoding="utf-8") == "local edits\n"
    assert cli.invoke(app, [*selected, "--check"]).exit_code == 0
    assert cli.invoke(app, [*command, "--check"]).exit_code == 1
    unknown = cli.invoke(app, [*command, "--report", "typo"])
    assert unknown.exit_code == 2 and "no report 'typo'" in unknown.output


def test_show_refuses_a_directory_that_is_not_a_benchmark(tmp_path):
    result = CliRunner().invoke(app, ["benchmark", "show", str(tmp_path)])
    assert result.exit_code == 2
    assert "not a benchmark directory" in result.output
