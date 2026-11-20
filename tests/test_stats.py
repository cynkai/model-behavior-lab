"""Reference values were cross-checked against statsmodels 0.14
(proportion_confint method="wilson"; confint_proportions_2indep method="newcomb")."""

from __future__ import annotations

import math
import random

import pytest

from mblab.formatting import interval_text, pct, pct_ci, seconds
from mblab.stats import (
    Interval,
    bootstrap_median_diff,
    bootstrap_position_by_stratum,
    describe,
    newcombe_interval,
    stratified_newcombe,
    wilson_interval,
    z_value,
)


def test_z_value():
    assert z_value(0.95) == pytest.approx(1.959964, abs=1e-6)
    with pytest.raises(ValueError):
        z_value(1.5)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (0, 10, 0.0, 0.277533),
        (5, 10, 0.236593, 0.763407),
        (10, 10, 0.722467, 1.0),
        (12, 18, 0.437495, 0.837212),
        (1, 3, 0.061492, 0.792340),
    ],
)
def test_wilson_matches_reference(k, n, low, high):
    iv = wilson_interval(k, n)
    assert iv.estimate == pytest.approx(k / n)
    assert (iv.low, iv.high) == (pytest.approx(low, abs=1e-6), pytest.approx(high, abs=1e-6))


def test_wilson_edge_cases():
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


@pytest.mark.parametrize(
    ("k1", "n1", "k2", "n2", "low", "high"),
    [
        (56, 70, 48, 80, 0.052431, 0.333873),  # Newcombe (1998), method 10 worked example
        (12, 18, 7, 18, -0.043328, 0.530010),
        (0, 18, 0, 18, -0.175879, 0.175879),
        (18, 18, 0, 18, 0.751269, 1.0),
    ],
)
def test_newcombe_matches_reference(k1, n1, k2, n2, low, high):
    iv = newcombe_interval(k1, n1, k2, n2)
    assert iv.estimate == pytest.approx(k1 / n1 - k2 / n2)
    assert (iv.low, iv.high) == (pytest.approx(low, abs=1e-6), pytest.approx(high, abs=1e-6))


def test_newcombe_needs_data():
    assert newcombe_interval(1, 0, 1, 5) is None


def test_bootstrap_is_reproducible_and_sensible():
    a = [10, 12, 11, 13, 12, 14, 11]
    b = [3, 4, 5, 4, 3, 5, 4]
    first = bootstrap_median_diff(a, b, seed=1)
    second = bootstrap_median_diff(a, b, seed=1)
    assert first == second
    assert first.estimate == 8.0
    assert first.low <= first.estimate <= first.high
    assert first.excludes_zero
    same = bootstrap_median_diff(a, a, seed=1)
    assert not same.excludes_zero
    assert bootstrap_median_diff([], b) is None


def test_describe():
    stats = describe([1, 2, 3, 4, 100])
    assert stats["n"] == 5 and stats["median"] == 3 and stats["max"] == 100
    assert describe([])["median"] is None


def test_formatting():
    assert pct(0.456) == "46%" and pct(None) == "—"
    assert pct_ci(0.2, 0.71) == "20–71%"
    assert seconds(6.94) == "6.9 s" and seconds(150) == "2.5 min"
    assert interval_text(Interval(-0.37, -0.63, -0.04), "proportion") == "−37 pp [−63, −4]"
    assert interval_text(Interval(0.0, -0.17, 0.17), "proportion") == "0 pp [−17, +17]"
    assert interval_text(None, "seconds") == "—"


def test_stratified_newcombe_equals_newcombe_when_every_task_has_the_same_runs():
    rng = random.Random(3)
    for _ in range(500):
        k, n = rng.randint(1, 8), rng.randint(1, 6)
        a = {f"T{t}": (rng.randint(0, n), n) for t in range(k)}
        b = {f"T{t}": (rng.randint(0, n), n) for t in range(k)}
        strat = stratified_newcombe(a, b)
        pooled = newcombe_interval(
            sum(v[0] for v in a.values()), k * n, sum(v[0] for v in b.values()), k * n
        )
        assert strat.estimate == pytest.approx(pooled.estimate, abs=1e-12)
        assert strat.low == pytest.approx(pooled.low, abs=1e-12)
        assert strat.high == pytest.approx(pooled.high, abs=1e-12)


def test_stratified_newcombe_is_not_moved_by_a_different_task_mix():
    # The same accuracy on each task in both groups, but group b lost most hard-task runs
    # (not completed): pooled, b looks 33 points better; standardized, they are equal.
    a = {"easy": (30, 30), "hard": (0, 30)}
    b = {"easy": (30, 30), "hard": (0, 6)}
    pooled = newcombe_interval(30, 60, 30, 36)
    assert pooled.estimate == pytest.approx(-1 / 3) and pooled.excludes_zero
    strat = stratified_newcombe(a, b)
    assert strat.estimate == 0 and strat.low < 0 < strat.high
    assert stratified_newcombe({"x": (1, 2)}, {"y": (1, 2)}) is None  # no shared task


def test_stratified_comparison_keeps_its_error_rate_under_uneven_incompletion():
    # Six tasks of very different difficulty, 10 runs per task and group, equal accuracy
    # per task; in group b the three hard tasks are mostly not completed.
    difficulty = [0.95, 0.9, 0.85, 0.35, 0.3, 0.25]
    rng = random.Random(11)
    false_pooled = false_strat = 0
    for _ in range(400):
        a, b = {}, {}
        for task, p in enumerate(difficulty):
            a[task] = (sum(rng.random() < p for _ in range(10)), 10)
            n_b = 10 if p > 0.5 else sum(rng.random() > 0.6 for _ in range(10))
            b[task] = (sum(rng.random() < p for _ in range(n_b)), n_b)
        pa, pb = (sum(v[0] for v in g.values()) for g in (a, b))
        na, nb = (sum(v[1] for v in g.values()) for g in (a, b))
        false_pooled += newcombe_interval(pa, na, pb, nb).excludes_zero
        false_strat += stratified_newcombe(a, b).excludes_zero
    assert false_strat / 400 < 0.05  # at most the nominal 5%
    assert false_pooled / 400 > 0.15  # the pooled comparison is biased by the task mix


def test_position_by_stratum():
    d = {"t1": [0.0, 0.0, 1.0], "t2": [0.0, 0.0, 0.0]}
    o = {"t1": [1.0, 1.0, 1.0], "t2": [1.0, 1.0, 0.0]}
    c = {"t1": [1.0, 0.0, 1.0], "t2": [0.0, 1.0, 0.0]}
    pos = bootstrap_position_by_stratum(c, d, o, resamples=500)
    assert pos is not None and pos.estimate == pytest.approx(0.5)
    assert pos.low < 0.5 < pos.high
    # All-or-nothing in every task: every resample is the same, so no interval.
    flat = bootstrap_position_by_stratum({"t": [1.0] * 3}, {"t": [0.0] * 3}, {"t": [1.0] * 3})
    assert flat is not None and flat.estimate == 1 and math.isnan(flat.low)
