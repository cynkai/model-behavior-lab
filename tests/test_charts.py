"""Every chart spec must compile with Vega-Lite (the same engine the browser uses)."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("vl_convert")

import vl_convert as vlc

from mblab.analysis import analyze
from mblab.dashboard import charts
from mblab.demo import create_demo_study
from mblab.study import Study


@pytest.fixture(scope="module")
def demo_analysis(tmp_path_factory):
    root = tmp_path_factory.mktemp("charts") / "demo"
    create_demo_study(root)
    return analyze(Study.load(root))


def _builders(a):
    return {
        "accuracy": lambda m: charts.accuracy_chart(a.conditions, m, a.settings.confidence),
        "latency": lambda m: charts.latency_chart(
            a.results, a.conditions, a.primary_latency_method, m
        ),
        "labels": lambda m: charts.label_chart(a.labels, m),
        "heatmap": lambda m: charts.task_heatmap(a.tasks, m),
        "repetitions": lambda m: charts.repetition_chart(a.grid, m),
    }


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("name", ["accuracy", "latency", "labels", "heatmap", "repetitions"])
def test_chart_compiles(demo_analysis, name, mode):
    spec = _builders(demo_analysis)[name](mode).to_dict()
    svg = vlc.vegalite_to_svg(spec, vl_version=vlc.get_vegalite_versions()[-1])
    assert svg.startswith("<svg")
    assert f'fill="{charts.theme(mode)["surface"]}"' in svg.split("<g ", 1)[0]


def test_charts_are_not_built_without_data(study_dir):
    a = analyze(Study.load(study_dir))
    assert all(build("light") is None for build in _builders(a).values())


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_benchmark_count_chart_compiles(mode):
    from mblab.benchmark import Benchmark, evaluate, runs_frame

    from .test_benchmark import BENCH

    ev = evaluate(Benchmark.load(BENCH))
    chart = charts.benchmark_count_chart(runs_frame(ev), ev.truth.count, mode)
    assert chart is not None
    svg = vlc.vegalite_to_svg(chart.to_dict(), vl_version=vlc.get_vegalite_versions()[-1])
    assert svg.startswith("<svg")


def test_benchmark_count_chart_needs_a_count():
    import pandas as pd

    columns = ["run_id", "condition_id", "default_selection", "per_message_selection"]
    empty = pd.DataFrame(columns=[*columns, "count_given", "exact_count"])
    assert charts.benchmark_count_chart(empty, 23, "light") is None
