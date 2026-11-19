"""Reference values were cross-checked against statsmodels 0.14
(proportion_confint method="wilson"; confint_proportions_2indep method="newcomb")."""

from __future__ import annotations

import pytest

from mblab.formatting import interval_text, pct, pct_ci, seconds
from mblab.stats import (
    Interval,
    bootstrap_median_diff,
    describe,
    newcombe_interval,
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
