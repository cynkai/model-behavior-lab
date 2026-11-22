from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab import task_ordering
from mblab.benchmark import Benchmark, check_answers, dump_run, evaluate, render_report, runs_frame
from mblab.cli import app

from .test_benchmark import BENCH


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return Benchmark.load(BENCH)


@pytest.mark.parametrize(
    ("text", "expected", "correct"),
    [
        ("23", 23, True),
        ("**23.000**개", 23, True),
        ("23 orders (23.0)", 23, True),
        ("0,023", 23, True),
        ("1,234", 1234, False),
        ("−3", -3, False),
        ("23.5 orders", Decimal("23.5"), False),
        ("23.000000000000000000000000000001", Decimal("23.000000000000000000000000000001"), False),
        ("22.999999999999999999999999999999", Decimal("22.999999999999999999999999999999"), False),
        ("9,007,199,254,740,993", 9007199254740993, False),
        ("9" * 310, int("9" * 310), False),
        ("9" * 5000, Decimal("9" * 5000), False),
    ],
)
def test_count_preserves_exact_numeric_value(
    benchmark: Benchmark, text: str, expected: int | Decimal, correct: bool
) -> None:
    task = benchmark.definition.task
    result = check_answers(task, task_ordering, task_ordering.ground_truth(), {"count": text})[0]
    assert result.readable and result.value == expected and result.correct is correct
    assert result.detail == f"read {expected}; expected 23"


@pytest.mark.parametrize(
    "text",
    [
        "unknown",
        "23 or 24",
        "23 or 23.00000000000000000001",
        "9007199254740992 or 9007199254740993",
    ],
)
def test_distinct_precise_numbers_are_ambiguous(benchmark: Benchmark, text: str) -> None:
    task = benchmark.definition.task
    result = check_answers(task, task_ordering, task_ordering.ground_truth(), {"count": text})[0]
    assert not result.readable and not result.correct and result.value is None


def test_report_and_frame_preserve_counts_with_missing_values(benchmark: Benchmark) -> None:
    texts = ["23.000000000000000000000000000001", "9007199254740993", "unknown"]
    runs = [
        run.model_copy(update={"answers": {**run.answers, "count": text}, "reported_breakdown": {}})
        for run, text in zip(benchmark.runs, texts, strict=False)
    ]
    ev = evaluate(Benchmark(benchmark.root, benchmark.definition, runs))
    frame = runs_frame(ev)
    report = render_report(ev, benchmark.report_spec("preliminary"))
    assert [str(value) for value in frame["count_given"]] == [*texts[:2], "None"]
    assert list(frame["exact_count"]) == ["Fail", "Fail", "Unreadable"]
    assert ev.runs[0].error_type(ev.truth) == "count too high"
    assert "1e-30 more" in ev.runs[0].error_detail(ev.truth).lower()
    assert all(text in report for text in texts[:2])


def test_cli_reports_precise_wrong_count(benchmark: Benchmark, tmp_path: Path) -> None:
    text = "23.00000000000000000001"
    (tmp_path / "benchmark.yaml").write_bytes((BENCH / "benchmark.yaml").read_bytes())
    runs = [run.model_copy(update={"answers": {"count": text}}) for run in benchmark.runs]
    (tmp_path / "runs.jsonl").write_text("".join(dump_run(run) + "\n" for run in runs))
    result = CliRunner().invoke(app, ["benchmark", "show", str(tmp_path)], terminal_width=240)
    assert result.exit_code == 0, result.output
    assert text in result.output
    assert "Full task success: 0 pass, 6 fail" in result.output


@pytest.mark.parametrize("text", ["9007199254740993", "9" * 310, "9" * 5000])
def test_large_count_survives_frame_with_missing_count(benchmark: Benchmark, text: str) -> None:
    runs = [
        run.model_copy(update={"answers": {"count": count}})
        for run, count in zip(benchmark.runs, [text, "unknown"], strict=False)
    ]
    frame = runs_frame(evaluate(Benchmark(benchmark.root, benchmark.definition, runs)))
    assert frame["count_given"].tolist() == [Decimal(text), None]


def test_error_magnitude_preserves_many_trailing_zeros(benchmark: Benchmark) -> None:
    text = "1" + "0" * 1000
    run = benchmark.runs[0].model_copy(update={"answers": {"count": text}})
    ev = evaluate(Benchmark(benchmark.root, benchmark.definition, [run]))
    assert f"({'9' * 998}77 more)" in ev.runs[0].error_detail(ev.truth)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_chart_preserves_exact_count_tooltips(benchmark: Benchmark, mode: str) -> None:
    import vl_convert as vlc

    from mblab.dashboard.charts import benchmark_count_chart

    texts = ["23.00000000000000000001", "9007199254740993", "9" * 310, "unknown"]
    runs = [
        run.model_copy(update={"answers": {"count": text}})
        for run, text in zip(benchmark.runs, texts, strict=False)
    ]
    ev = evaluate(Benchmark(benchmark.root, benchmark.definition, runs))
    chart = benchmark_count_chart(runs_frame(ev), ev.truth.count, mode)
    assert chart is not None
    spec = chart.to_dict()
    svg = vlc.vegalite_to_svg(spec, vl_version=vlc.get_vegalite_versions()[-1])
    plotted = [row for rows in spec["datasets"].values() for row in rows if "count_given" in row]
    assert [row["count_given"] for row in plotted] == texts[:2]
    assert all(row["result"] == "incorrect" for row in plotted)
    assert spec["layer"][2]["encoding"]["tooltip"][2]["type"] == "nominal"
    assert svg.startswith("<svg")
