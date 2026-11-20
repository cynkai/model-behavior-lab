"""Shared number formatting and wording for the CLI, reports, and dashboard."""

from __future__ import annotations

import math
import string

from mblab.stats import Interval

_MARKDOWN_ESCAPES = str.maketrans({c: "\\" + c for c in string.punctuation})


def md_text(value: object) -> str:
    """``value`` as literal text for a Markdown string (Streamlit text, labels, captions).

    Study files and records can come from someone else. Rendered as Markdown, their text
    could add links, images that load from a remote server, headings, or formatting.
    Every ASCII punctuation character is backslash-escaped, which CommonMark always
    reads as the literal character, and line breaks become spaces, so the text shows
    exactly as written and cannot start a new block.
    """
    return " ".join(str(value).split()).translate(_MARKDOWN_ESCAPES)


VERDICT_SHORT = {
    "insufficient_data": "insufficient data",
    "references_not_distinguishable": "references not distinguishable",
    "consistent_with_per_message": "consistent with the per-message selection",
    "consistent_with_default": "consistent with the default selection",
    "differs_from_both": "differs from both references",
    "inconclusive": "inconclusive",
    "no_difference_detected": "no difference detected",
    "difference_detected": "difference detected",
}
COMPARISON_STATUS = {
    "insufficient_data": "insufficient data",
    "no_clear_difference": "CI includes 0",
    "difference_observed": "CI excludes 0",
}
METRIC_NAMES = {
    "accuracy": "accuracy",
    "latency_median": "median response time",
    "reasoning_tokens_median": "median reported reasoning tokens",
}
METRIC_UNITS = {
    "accuracy": "proportion",
    "latency_median": "seconds",
    "reasoning_tokens_median": "tokens",
}


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pct(value: float | None) -> str:
    return "—" if _missing(value) else f"{100 * value:.0f}%"


def pct_ci(low: float | None, high: float | None) -> str:
    if _missing(low) or _missing(high):
        return "—"
    return f"{100 * low:.0f}–{100 * high:.0f}%"


def seconds(value: float | None) -> str:
    if _missing(value):
        return "—"
    if value >= 120:
        return f"{value / 60:.1f} min"
    return f"{value:.1f} s"


def _num(value: float, unit: str) -> str:
    """Signed number with a typographic minus: +12, −3.5, 0."""
    if unit == "proportion":
        text = f"{100 * value:+.0f}"
    elif unit == "tokens":
        text = f"{value:+,.0f}"
    else:
        text = f"{value:+.1f}"
    text = text.replace("-", "−")
    return text.lstrip("+−") if text.strip("+−0.") == "" else text


def signed(value: float, unit: str) -> str:
    suffix = {"proportion": "pp", "tokens": "tokens"}.get(unit, "s")
    return f"{_num(value, unit)} {suffix}"


def interval_text(iv: Interval | None, unit: str) -> str:
    """Difference with its interval, e.g. "−37 pp [−63, −4]"."""
    if iv is None:
        return "—"
    return f"{signed(iv.estimate, unit)} [{_num(iv.low, unit)}, {_num(iv.high, unit)}]"


def tokens(value: float | None) -> str:
    return "—" if _missing(value) else f"{value:,.0f}"


def metric_value(value: float | None, unit: str) -> str:
    if unit == "tokens":
        return tokens(value)
    return pct(value) if unit == "proportion" else seconds(value)


def method_text(method: str | None) -> str:
    return {
        "manual_stopwatch": "manual stopwatch",
        "screen_recording": "screen recording",
        "client_timer": "client timer",
        None: "not measured",
    }.get(method, str(method))
