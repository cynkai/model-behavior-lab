"""Altair chart builders for the dashboard (pure functions, no Streamlit).

Colour follows a validated reference palette (categorical slots 1-5 in fixed
order, a one-hue blue sequential ramp, reserved status colours that always come
with a shape and a legend label). Every chart in the dashboard is paired with a
table view, so no value is reachable through colour or hover alone.

Charts are only built from rows that have data; callers show a message instead
of an empty or placeholder chart when there is nothing to plot.
"""

from __future__ import annotations

import zlib

import altair as alt
import pandas as pd

from mblab.analysis import LABEL_CLASS_TEXT, LABEL_CLASSES, OUTCOMES

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
THEMES: dict[str, dict] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "on_dark_fill": "#ffffff",
        "on_light_fill": "#0b0b0b",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "sequential": ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
        "on_dark_fill": "#ffffff",
        "on_light_fill": "#0b0b0b",
    },
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
OUTCOME_TEXT = {
    "correct": "correct",
    "incorrect": "incorrect",
    "format_failure": "format failure",
    "not_scorable": "error / refusal",
    "pending": "awaiting grade",
    "excluded": "excluded",
    "missing": "not recorded",
}
OUTCOME_SHAPE = {
    "correct": "circle",
    "incorrect": "cross",
    "format_failure": "triangle-up",
    "not_scorable": "square",
    "pending": "diamond",
    "excluded": "triangle-down",
    "missing": "circle",
}


def theme(mode: str | None) -> dict:
    return THEMES["dark" if mode == "dark" else "light"]


def _style(chart: alt.TopLevelMixin, t: dict) -> alt.TopLevelMixin:
    # fit-x: stretch to the container width but keep the declared plot height, so
    # titles, legends and axes are added around the plot instead of squeezing it.
    return (
        chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
        .configure(background="transparent", font=FONT)
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=t["grid"],
            gridWidth=1,
            domainColor=t["axis"],
            tickColor=t["axis"],
            labelColor=t["ink2"],
            titleColor=t["ink2"],
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelColor=t["ink2"],
            titleColor=t["ink2"],
            orient="top",
            labelFontSize=12,
            labelLimit=360,
            columnPadding=24,
            symbolSize=90,
        )
        .configure_title(color=t["ink"], anchor="start", fontSize=14, fontWeight=600, offset=10)
    )


def condition_axis_labels(conditions: pd.DataFrame) -> dict[str, str]:
    return {
        r.condition_id: f"{r.condition_id} · {r.model_selection} / {r.message_override}"
        for r in conditions.itertuples()
    }


def accuracy_chart(
    conditions: pd.DataFrame, mode: str | None, confidence: float
) -> alt.Chart | None:
    """Dot (point estimate) and rule (Wilson interval) per condition; one hue."""
    t = theme(mode)
    data = conditions[conditions["n_scored"] > 0].copy()
    if data.empty:
        return None
    labels = condition_axis_labels(conditions)
    data["condition"] = data["condition_id"].map(labels)
    data["count"] = data["n_correct"].astype(str) + "/" + data["n_scored"].astype(str)
    order = [labels[c] for c in conditions["condition_id"]]
    conf = int(confidence * 100)
    y = alt.Y("condition:N", sort=order, title=None, axis=alt.Axis(grid=False, labelLimit=260))
    tooltip = [
        alt.Tooltip("condition:N", title="Condition"),
        alt.Tooltip("count:N", title="Correct / scored"),
        alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%"),
        alt.Tooltip("accuracy_low:Q", title=f"{conf}% CI low", format=".0%"),
        alt.Tooltip("accuracy_high:Q", title=f"{conf}% CI high", format=".0%"),
    ]
    base = alt.Chart(data).encode(y=y)
    interval = base.mark_rule(strokeWidth=2, strokeCap="round", color=t["series"][0]).encode(
        x=alt.X(
            "accuracy_low:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5),
            title=f"Accuracy with {conf}% Wilson interval",
        ),
        x2="accuracy_high:Q",
        tooltip=tooltip,
    )
    dots = base.mark_point(
        filled=True, size=120, opacity=1, color=t["series"][0], stroke=t["surface"], strokeWidth=2
    ).encode(x="accuracy:Q", tooltip=tooltip)
    counts = base.mark_text(align="left", dx=8, color=t["muted"], fontSize=11).encode(
        x="accuracy_high:Q", text="count:N"
    )
    chart = (interval + dots + counts).properties(
        height=44 * len(data), title="Accuracy by condition"
    )
    return _style(chart, t)


LOG_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def _jitter(run_id: str) -> float:
    return (zlib.crc32(run_id.encode()) % 1000) / 1000 - 0.5


def latency_chart(
    results: pd.DataFrame, conditions: pd.DataFrame, method: str | None, mode: str | None
) -> alt.Chart | None:
    """Strip plot of individual response times (log scale) with a median tick."""
    t = theme(mode)
    if method is None or results.empty:
        return None
    data = results[
        (~results["excluded"])
        & (results["latency_method"] == method)
        & (results["latency_seconds"] > 0)
    ].copy()
    if data.empty:
        return None
    labels = condition_axis_labels(conditions)
    data["condition"] = data["condition_id"].map(labels)
    data["jitter"] = data["run_id"].map(_jitter)
    order = [labels[c] for c in conditions["condition_id"]]
    y = alt.Y("condition:N", sort=order, title=None, axis=alt.Axis(grid=False, labelLimit=260))
    lo, hi = float(data["latency_seconds"].min()), float(data["latency_seconds"].max())
    ticks = [v for v in LOG_TICKS if lo / 1.5 <= v <= hi * 1.5] or [lo, hi]
    x = alt.X(
        "latency_seconds:Q",
        scale=alt.Scale(type="log"),
        axis=alt.Axis(values=ticks, format="~s"),
        title=f"Seconds (log scale) — {method.replace('_', ' ')}"
        + ("" if method == "client_timer" else ", low precision"),
    )
    dots = (
        alt.Chart(data)
        .mark_circle(
            size=70, opacity=0.85, color=t["series"][0], stroke=t["surface"], strokeWidth=1
        )
        .encode(
            x=x,
            y=y,
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1.4, 1.4])),
            tooltip=[
                alt.Tooltip("run_id:N", title="Run"),
                alt.Tooltip("task_id:N", title="Task"),
                alt.Tooltip("latency_seconds:Q", title="Seconds", format=".1f"),
            ],
        )
    )
    medians = (
        alt.Chart(data)
        .mark_tick(color=t["ink"], thickness=2, size=26)
        .encode(
            x=alt.X("median(latency_seconds):Q"),
            y=y,
            tooltip=[
                alt.Tooltip("condition:N", title="Condition"),
                alt.Tooltip("median(latency_seconds):Q", title="Median seconds", format=".1f"),
                alt.Tooltip("count():Q", title="n"),
            ],
        )
    )
    chart = (dots + medians).properties(
        height=52 * data["condition"].nunique(),
        title="Response time per run (dots) and median (tick)",
    )
    return _style(chart, t)


def label_chart(labels: pd.DataFrame, mode: str | None) -> alt.Chart | None:
    """Stacked horizontal bar of displayed-label classes per condition."""
    t = theme(mode)
    long = labels.melt(id_vars="condition_id", var_name="label_class", value_name="runs")
    long = long[long["runs"] > 0].copy()
    if long.empty:
        return None
    long["class_text"] = long["label_class"].map(LABEL_CLASS_TEXT)
    long["order"] = long["label_class"].map(LABEL_CLASSES.index)
    domain = [LABEL_CLASS_TEXT[c] for c in LABEL_CLASSES]
    colors = [*t["series"], t["muted"]]
    chart = (
        alt.Chart(long)
        .mark_bar(size=20, stroke=t["surface"], strokeWidth=2)
        .encode(
            y=alt.Y("condition_id:N", title=None, sort=list(labels["condition_id"])),
            x=alt.X("sum(runs):Q", title="Runs", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "class_text:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(title=None, columns=2),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("condition_id:N", title="Condition"),
                alt.Tooltip("class_text:N", title="Displayed label"),
                alt.Tooltip("runs:Q", title="Runs"),
            ],
        )
        .properties(
            height=36 * labels["condition_id"].nunique(),
            title="Runs per condition, by how the displayed label compares with the settings",
        )
    )
    return _style(chart, t)


def task_heatmap(tasks: pd.DataFrame, mode: str | None) -> alt.Chart | None:
    """Accuracy per task x condition. Cells without scored responses stay empty."""
    t = theme(mode)
    data = tasks[tasks["n_scored"] > 0].copy()
    if data.empty:
        return None
    data["task"] = data["task_id"] + " · " + data["task_title"]
    data["count"] = data["n_correct"].astype(str) + "/" + data["n_scored"].astype(str)
    # Label ink follows the fill's lightness: dark fills get white, light fills near-black.
    # (A conditional value, not a second colour scale, which Vega-Lite cannot layer.)
    high, low = (
        (t["on_light_fill"], t["on_dark_fill"])
        if mode == "dark"
        else (t["on_dark_fill"], t["on_light_fill"])
    )
    task_order = list(dict.fromkeys(tasks["task_id"] + " · " + tasks["task_title"]))
    base = alt.Chart(data).encode(
        x=alt.X(
            "condition_id:N",
            title="Condition",
            axis=alt.Axis(orient="top", grid=False, labelAngle=0),
        ),
        y=alt.Y("task:N", sort=task_order, title=None, axis=alt.Axis(grid=False, labelLimit=240)),
    )
    cells = base.mark_rect(stroke=t["surface"], strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color(
            "accuracy:Q",
            scale=alt.Scale(domain=[0, 1], range=t["sequential"]),
            legend=alt.Legend(title="Accuracy", format="%", orient="right", gradientLength=120),
        ),
        tooltip=[
            alt.Tooltip("task:N", title="Task"),
            alt.Tooltip("condition_id:N", title="Condition"),
            alt.Tooltip("count:N", title="Correct / scored"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".0%"),
        ],
    )
    text = base.mark_text(fontSize=12).encode(
        text="count:N",
        color=alt.condition(alt.datum.accuracy >= 0.55, alt.value(high), alt.value(low)),
    )
    chart = (cells + text).properties(
        height=38 * data["task"].nunique(), title="Accuracy per task (correct / scored)"
    )
    return _style(chart, t)


def repetition_chart(grid: pd.DataFrame, mode: str | None) -> alt.Chart | None:
    """Every planned run as a mark: task rows x condition columns x repetitions."""
    t = theme(mode)
    if grid.empty or (grid["outcome"] == "missing").all():
        return None
    data = grid.copy()
    data["outcome_text"] = data["outcome"].map(OUTCOME_TEXT)
    colors = {
        "correct": STATUS["good"],
        "incorrect": STATUS["critical"],
        "format_failure": STATUS["serious"],
        "not_scorable": STATUS["warning"],
        "pending": t["muted"],
        "excluded": t["muted"],
        "missing": t["axis"],
    }
    domain = [OUTCOME_TEXT[o] for o in OUTCOMES]
    chart = (
        alt.Chart(data)
        .mark_point(filled=True, size=110, opacity=1)
        .encode(
            x=alt.X(
                "condition_id:N",
                title="Condition",
                axis=alt.Axis(orient="top", grid=False, labelAngle=0),
            ),
            xOffset=alt.XOffset("repetition:O"),
            y=alt.Y("task_id:N", title=None, axis=alt.Axis(grid=False)),
            color=alt.Color(
                "outcome_text:N",
                scale=alt.Scale(domain=domain, range=[colors[o] for o in OUTCOMES]),
                legend=alt.Legend(title=None, columns=4),
            ),
            shape=alt.Shape(
                "outcome_text:N",
                scale=alt.Scale(domain=domain, range=[OUTCOME_SHAPE[o] for o in OUTCOMES]),
                legend=alt.Legend(title=None, columns=4),
            ),
            tooltip=[
                alt.Tooltip("run_id:N", title="Run"),
                alt.Tooltip("repetition:O", title="Repetition"),
                alt.Tooltip("outcome_text:N", title="Outcome"),
            ],
        )
        .properties(
            height=34 * data["task_id"].nunique(),
            title="Each repetition of each task (columns: conditions; marks: repetitions 1..n)",
        )
    )
    return _style(chart, t)
