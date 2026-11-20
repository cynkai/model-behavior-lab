"""Small, dependency-light statistics with conservative defaults.

* Proportions use the Wilson score interval (sensible at small n and at 0 % or
  100 %), and differences of proportions use Newcombe's hybrid score interval
  (method 10 in Newcombe 1998).
* Differences of medians use a percentile bootstrap driven by
  ``random.Random(seed).random()``, so results are reproducible across Python
  and NumPy versions.

All intervals treat responses as independent observations. Responses to the
same task are correlated, so read them as rough descriptions of uncertainty
for this fixed task set, not as tests of a general claim.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def z_value(confidence: float = 0.95) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval | None:
    """Wilson score interval for a binomial proportion; ``None`` when n == 0."""
    if n <= 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    return Interval(p, *_wilson_bounds(p, n, z_value(confidence)))


def _wilson_bounds(p: float, n: float, z: float) -> tuple[float, float]:
    """Wilson score bounds for a proportion ``p`` from ``n`` (possibly effective) trials."""
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_interval(
    k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95
) -> Interval | None:
    """Newcombe hybrid score interval for p1 - p2; ``None`` if either n is 0."""
    a = wilson_interval(k1, n1, confidence)
    b = wilson_interval(k2, n2, confidence)
    if a is None or b is None:
        return None
    p1, p2 = a.estimate, b.estimate
    diff = p1 - p2
    low = diff - math.sqrt((p1 - a.low) ** 2 + (b.high - p2) ** 2)
    high = diff + math.sqrt((a.high - p1) ** 2 + (p2 - b.low) ** 2)
    return Interval(diff, max(-1.0, low), min(1.0, high))


def stratified_newcombe(
    a: Mapping[str, tuple[int, int]],
    b: Mapping[str, tuple[int, int]],
    confidence: float = 0.95,
) -> Interval | None:
    """Difference of two proportions standardized over strata, with a Newcombe interval.

    ``a`` and ``b`` map a stratum (a task) to ``(successes, n)``. Only strata observed
    in both groups count, each with the same weight, so the estimate is the average of
    the per-stratum differences: what a design that gives every task the same runs sets
    out to measure, and something a different mix of missing or incomplete runs per
    group cannot shift. Each group's standardized proportion gets a Wilson interval with
    the effective sample size ``1 / sum(w**2 / n_k)`` (``w`` = 1 / number of strata), and
    the two are combined as in Newcombe's hybrid score interval — a stratified Newcombe
    interval (cf. Yan & Su, 2010). With the same ``n`` in every stratum it equals
    :func:`newcombe_interval` on the pooled counts. ``None`` without a shared stratum.
    """
    common = [k for k in a if k in b and a[k][1] > 0 and b[k][1] > 0]
    if not common:
        return None
    w = 1 / len(common)
    z = z_value(confidence)
    bounds = []
    for group in (a, b):
        p = sum(group[k][0] / group[k][1] for k in common) * w
        n_effective = 1 / sum(w * w / group[k][1] for k in common)
        bounds.append((p, *_wilson_bounds(p, n_effective, z)))
    (p1, low1, high1), (p2, low2, high2) = bounds
    diff = p1 - p2
    low = diff - math.sqrt((p1 - low1) ** 2 + (high2 - p2) ** 2)
    high = diff + math.sqrt((high1 - p1) ** 2 + (p2 - low2) ** 2)
    return Interval(diff, max(-1.0, low), min(1.0, high))


def bootstrap_median_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """Percentile-bootstrap interval for median(a) - median(b)."""
    xa = np.asarray(list(a), dtype=float)
    xb = np.asarray(list(b), dtype=float)
    if xa.size == 0 or xb.size == 0:
        return None
    rng = random.Random(seed)
    idx_a = np.array(
        [[int(rng.random() * xa.size) for _ in range(xa.size)] for _ in range(resamples)]
    )
    idx_b = np.array(
        [[int(rng.random() * xb.size) for _ in range(xb.size)] for _ in range(resamples)]
    )
    diffs = np.median(xa[idx_a], axis=1) - np.median(xb[idx_b], axis=1)
    alpha = 1 - confidence
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return Interval(float(np.median(xa) - np.median(xb)), float(low), float(high))


def bootstrap_position(
    c: Sequence[float],
    d: Sequence[float],
    o: Sequence[float],
    *,
    statistic: str = "mean",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """Where ``c`` lies between two references: (S(c) - S(d)) / (S(o) - S(d)).

    S is the mean (for 0/1 outcomes, a proportion) or the median. 0 means "at d", 1 "at
    o", 0.5 halfway, and values outside 0..1 lie beyond a reference. The interval is a
    percentile bootstrap over each group's observations; resamples in which the two
    references coincide are skipped, and no interval is given if more than 5% are.
    """
    xs = [np.asarray(list(v), dtype=float) for v in (c, d, o)]
    if any(x.size == 0 for x in xs):
        return None
    stat = np.mean if statistic == "mean" else np.median
    xc, xd, xo = xs
    gap = stat(xo) - stat(xd)
    if gap == 0:
        return None
    rng = random.Random(seed)
    resampled = []
    for x in xs:
        idx = np.array(
            [[int(rng.random() * x.size) for _ in range(x.size)] for _ in range(resamples)]
        )
        resampled.append(stat(x[idx], axis=1))
    rc, rd, ro = resampled
    gaps = ro - rd
    usable = gaps != 0
    if usable.sum() < 0.95 * resamples:
        return Interval(float((stat(xc) - stat(xd)) / gap), math.nan, math.nan)
    positions = (rc[usable] - rd[usable]) / gaps[usable]
    alpha = 1 - confidence
    low, high = np.quantile(positions, [alpha / 2, 1 - alpha / 2])
    return Interval(float((stat(xc) - stat(xd)) / gap), float(low), float(high))


def bootstrap_position_by_stratum(
    c: Mapping[str, Sequence[float]],
    d: Mapping[str, Sequence[float]],
    o: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """:func:`bootstrap_position` for proportions standardized over strata (tasks).

    Each group's value is the mean of its per-stratum means over the strata observed in
    all three groups, and resampling stays within each stratum. When every resample gives
    the same position (all-or-nothing outcomes in every stratum), the bootstrap says
    nothing about uncertainty, so no interval is given.
    """
    common = sorted(k for k in c if k in d and k in o and len(c[k]) and len(d[k]) and len(o[k]))
    if not common:
        return None

    def point(group: Mapping[str, Sequence[float]]) -> float:
        return float(np.mean([np.mean(group[k]) for k in common]))

    gap = point(o) - point(d)
    if gap == 0:
        return None
    estimate = (point(c) - point(d)) / gap
    rng = random.Random(seed)
    resampled = []
    for group in (c, d, o):
        total = np.zeros(resamples)
        for k in common:
            x = np.asarray(list(group[k]), dtype=float)
            idx = np.array(
                [[int(rng.random() * x.size) for _ in range(x.size)] for _ in range(resamples)]
            )
            total += x[idx].mean(axis=1)
        resampled.append(total / len(common))
    rc, rd, ro = resampled
    gaps = ro - rd
    usable = gaps != 0
    if usable.sum() < 0.95 * resamples:
        return Interval(estimate, math.nan, math.nan)
    alpha = 1 - confidence
    low, high = np.quantile((rc[usable] - rd[usable]) / gaps[usable], [alpha / 2, 1 - alpha / 2])
    if low == high:
        return Interval(estimate, math.nan, math.nan)
    return Interval(estimate, float(low), float(high))


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    """n, median, quartiles, min and max (``None`` when empty)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "n": int(arr.size),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
