from dataclasses import replace

import pytest

from mblab.benchmark import Benchmark, evaluate, render_report

from .test_benchmark import BENCH, _repeat


@pytest.fixture(scope="module")
def benchmark() -> Benchmark:
    return Benchmark.load(BENCH)


def test_matching_rates_with_unequal_repetitions(benchmark: Benchmark) -> None:
    extra = [_repeat(benchmark, cid, 2) for cid in ("C", "F")]
    ev = evaluate(replace(benchmark, runs=[*benchmark.runs, *extra]))
    comparisons = {c.condition_id: c for c in ev.comparisons}

    assert comparisons["C"].pattern == "per_message"  # 2/2 versus 1/1
    assert comparisons["F"].pattern == "no_difference"  # 2/2 versus 1/1
    for cid in ("C", "F"):
        assert "2 of 2" in comparisons[cid].text and "1 of 1" in comparisons[cid].text
        assert "full-task success rate" in comparisons[cid].text
    abstract = render_report(ev, benchmark.definition.reports[0]).partition("## Abstract")[2]
    assert "full-task success rate" in abstract.partition("## Research Question")[0]


def test_references_with_matching_partial_success_rates(benchmark: Benchmark) -> None:
    correct = next(r.answers for r in benchmark.runs if r.condition_id == "B")
    wrong = next(r.answers for r in benchmark.runs if r.condition_id == "A")
    extra = [
        _repeat(benchmark, "A", 2, answers=correct),
        _repeat(benchmark, "B", 2, answers=wrong),
        _repeat(benchmark, "B", 3, answers=correct),
        _repeat(benchmark, "B", 4, answers=wrong),
    ]
    ev = evaluate(replace(benchmark, runs=[*benchmark.runs, *extra]))
    comparison = next(c for c in ev.comparisons if c.condition_id == "C")

    assert comparison.pattern == "references_alike"  # 1/2 versus 2/4
    assert "1 of 2" in comparison.text and "2 of 4" in comparison.text


def test_repeating_the_same_wrong_count_does_not_change_its_value(benchmark: Benchmark) -> None:
    wrong = next(r.answers for r in benchmark.runs if r.condition_id == "A")
    runs = [
        r.model_copy(update={"answers": wrong}) if r.condition_id == "D" else r
        for r in benchmark.runs
    ]
    runs.append(_repeat(benchmark, "D", 2, answers=wrong))
    comparison = next(
        c for c in evaluate(replace(benchmark, runs=runs)).comparisons if c.condition_id == "D"
    )

    assert comparison.pattern == "per_message"  # 0/2 versus 0/1
    assert "D gave 20, 20 and A gave 20" in comparison.text
    assert "different values" not in comparison.text


@pytest.mark.parametrize("count", ["23", "unknown", None])
def test_failed_order_or_unreadable_count_is_not_a_miscount(
    benchmark: Benchmark, count: str | None
) -> None:
    answers = {"lexicographic_first": "ABCDEFGH"}
    if count is not None:
        answers["count"] = count
    runs = [
        r.model_copy(update={"answers": answers}) if r.condition_id == "D" else r
        for r in benchmark.runs
    ]
    ev = evaluate(replace(benchmark, runs=runs))
    comparison = next(c for c in ev.comparisons if c.condition_id == "D")

    assert comparison.pattern == "per_message"  # neither D nor A was fully correct
    assert "wrong count" not in comparison.text and "count wrongly" not in comparison.text
