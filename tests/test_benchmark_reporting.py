import pytest

from mblab.benchmark import Benchmark, BenchmarkRun, ReportSpec, evaluate, render_report

from .test_benchmark import BENCH, PRELIMINARY


@pytest.fixture(scope="module")
def repeated_benchmark() -> Benchmark:
    original = Benchmark.load(BENCH)
    run = original.runs[1]
    repeat = BenchmarkRun.model_validate(
        run.model_dump()
        | {
            "run_id": "B:T1:r2",
            "repetition": 2,
            "batch": "repeat",
            "timestamp": "2026-11-22T19:00:00+09:00",
            "run_order": 1,
            "new_conversation": False,
            "raw_response": "\n".join(
                f"{answer.label}: {answer.reference_answer}"
                for answer in original.definition.task.answers
            ),
            "answers": {},
            "latency_method": "manual_stopwatch",
            "latency_seconds": 0,
            "device_model": "Example tablet",
            "observed_model_label": "Example label",
            "solution_summary": [],
            "source": "Test fixture with a full response",
        }
    )
    return Benchmark(original.root, original.definition, [*original.runs, repeat])


def _report(bench: Benchmark, *batches: str) -> str:
    spec = ReportSpec(id="repeat", title="Repeat", batches=list(batches), path="reports/repeat.md")
    return render_report(evaluate(bench, spec.batches), spec)


def test_report_describes_fully_recorded_batch(repeated_benchmark: Benchmark) -> None:
    text = _report(repeated_benchmark, "repeat")
    assert "| Full response texts | kept for 1 of 1 run |" in text
    assert "| Response times | recorded for 1 of 1 run |" in text
    assert "| Conversation | ongoing chat for every run |" in text
    assert "| Run order and times | run order 1/1, send time 1/1" in text
    assert "| Model label shown by the app | Example label |" in text
    observations = text.split("## Observations\n", 1)[1].split("## Results\n", 1)[0]
    assert "full response text" in observations
    assert "not kept" not in observations and "notes under the table" not in observations
    limitations = text.split("## Limitations\n", 1)[1].split("## Next Experiments\n", 1)[0]
    assert "Times were not recorded for every run" not in limitations
    assert "is missing for" not in limitations
    assert "summaries" not in limitations
    assert "Not recorded:\n" not in text
    assert "Planned, not yet run" not in text
    assert "Recording gaps listed" not in text
    assert "The tablet model and its operating system." not in text
    assert f"Plans listed in the benchmark definition for {PRELIMINARY}" in text


def test_combined_report_counts_selected_runs(repeated_benchmark: Benchmark) -> None:
    text = _report(repeated_benchmark, PRELIMINARY, "repeat")
    assert "| Full response texts | kept for 1 of 7 runs |" in text
    assert "| Response times | recorded for 1 of 7 runs |" in text
    assert "| Run order and times | run order 1/7, send time 1/7" in text
    assert "| Model label shown by the app | Example label (not recorded for some runs) |" in text
    limitations = text.split("## Limitations\n", 1)[1]
    assert "Full response texts are missing for 6 of 7" in limitations
    assert "Send time is missing for 6 of 7" in limitations
    assert "Run order is missing for 6 of 7" in limitations
    assert "With so few runs" not in text and "is possible" not in limitations
    assert "reports no intervals or hypothesis tests" in limitations
    assert "Not recorded:\n" not in text
    assert f"Recording gaps listed in the benchmark definition for {PRELIMINARY}" in text
    assert "Planned, not yet run" not in text


@pytest.mark.parametrize(
    ("missing_field", "missing_measurement", "present_measurement"),
    [("run_order", "Run order", "Send time"), ("timestamp", "Send time", "Run order")],
)
def test_report_separates_missing_order_from_missing_time(
    repeated_benchmark: Benchmark,
    missing_field: str,
    missing_measurement: str,
    present_measurement: str,
) -> None:
    repeat = repeated_benchmark.runs[-1].model_copy(update={missing_field: None})
    bench = Benchmark(repeated_benchmark.root, repeated_benchmark.definition, [repeat])
    limits = _report(bench, "repeat").split("## Limitations\n", 1)[1]
    assert f"{missing_measurement} is missing for 1 of 1 run" in limits
    assert f"{present_measurement} is missing" not in limits


def test_unscoped_notes_are_attributed_to_the_definition(repeated_benchmark: Benchmark) -> None:
    definition = repeated_benchmark.definition.model_copy(update={"notes_batches": None})
    bench = Benchmark(repeated_benchmark.root, definition, repeated_benchmark.runs)
    text = _report(bench, "repeat")
    assert "Recording gaps listed in the benchmark definition (batch unspecified)" in text
    assert "Plans listed in the benchmark definition (batch unspecified)" in text
    assert "Not recorded:\n" not in text and "Planned, not yet run" not in text


def test_original_report_still_reproduces_with_a_complete_repeat(
    repeated_benchmark: Benchmark,
) -> None:
    spec = repeated_benchmark.report_spec("preliminary")
    assert render_report(evaluate(repeated_benchmark, spec.batches), spec) == (
        BENCH / spec.path
    ).read_text(encoding="utf-8")
