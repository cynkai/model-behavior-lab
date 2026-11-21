"""Independent ground truth for the task-ordering puzzle of Benchmark 01.

Eight tasks, A to H, are done one at a time, each exactly once. This module checks
every one of the 8! = 40,320 orders against the puzzle's eight rules, written out one by
one below, and keeps the orders that satisfy all of them. Nothing here comes from a
model's answer or from the benchmark data: the reference answers written in a
benchmark definition are checked against this module, never the other way round.

The rules, as the prompt states them:

1. A is done before D and before E.
2. B is done before D.
3. C is done before E and before F.
4. D and E are both done before G.
5. F and G are both done before H.
6. A and B are not next to each other.
7. D and F are not next to each other.
8. C is in one of the first four positions.

Besides the answers themselves, :func:`breakdown` gives the verified value of the
intermediate counts a solution may report — how many valid orders start with A, or how
many orders of the other six tasks leave room for F — so an error can be traced to the
step where it entered.
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations

TASKS = "ABCDEFGH"

Positions = dict[str, int]


def _before(pos: Positions, first: str, second: str) -> bool:
    return pos[first] < pos[second]


def _next_to(pos: Positions, x: str, y: str) -> bool:
    return abs(pos[x] - pos[y]) == 1


#: (rule number, wording, check). Positions are counted from 0.
RULES: tuple[tuple[int, str, Callable[[Positions], bool]], ...] = (
    (1, "A before D and before E", lambda p: _before(p, "A", "D") and _before(p, "A", "E")),
    (2, "B before D", lambda p: _before(p, "B", "D")),
    (3, "C before E and before F", lambda p: _before(p, "C", "E") and _before(p, "C", "F")),
    (4, "D and E before G", lambda p: _before(p, "D", "G") and _before(p, "E", "G")),
    (5, "F and G before H", lambda p: _before(p, "F", "H") and _before(p, "G", "H")),
    (6, "A and B not next to each other", lambda p: not _next_to(p, "A", "B")),
    (7, "D and F not next to each other", lambda p: not _next_to(p, "D", "F")),
    (8, "C in one of the first four positions", lambda p: p["C"] < 4),
)


def parse_order(text: str) -> str:
    """``text`` as eight capital letters; ``ValueError`` unless it names each task once."""
    order = text.strip().upper()
    if len(order) != len(TASKS) or sorted(order) != sorted(TASKS):
        raise ValueError(f"{text!r} is not an order of the tasks {TASKS} (each exactly once)")
    return order


def violations(order: str) -> list[int]:
    """The numbers of the rules that ``order`` breaks (empty for a valid order)."""
    pos = {task: i for i, task in enumerate(parse_order(order))}
    return [number for number, _, holds in RULES if not holds(pos)]


def is_valid(order: str) -> bool:
    return not violations(order)


@functools.cache
def valid_orders() -> tuple[str, ...]:
    """Every order that satisfies all eight rules, in lexicographic order."""
    orders = ("".join(p) for p in permutations(TASKS))
    return tuple(sorted(o for o in orders if is_valid(o)))


@dataclass(frozen=True)
class GroundTruth:
    orders: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.orders)

    @property
    def lexicographic_first(self) -> str:
        return self.orders[0]

    def prefix_counts(self, length: int) -> dict[str, int]:
        """How many valid orders start with each prefix of ``length`` tasks."""
        return dict(sorted(Counter(o[:length] for o in self.orders).items()))


def ground_truth() -> GroundTruth:
    return GroundTruth(valid_orders())


@functools.cache
def redundant_rules() -> tuple[int, ...]:
    """Rules implied by the others: dropping one of them leaves the same valid orders.

    Rule 8 is one: C comes before E and F (rule 3), and so before G and H (rules 4 and 5),
    which leaves C in one of the first four positions anyway.
    """
    positions = [{task: i for i, task in enumerate(p)} for p in permutations(TASKS)]
    implied = []
    for number, _, _ in RULES:
        others = [holds for n, _, holds in RULES if n != number]
        if sum(all(h(pos) for h in others) for pos in positions) == len(valid_orders()):
            implied.append(number)
    return tuple(implied)


# -- intermediate counts ----------------------------------------------------------------
#
# Every valid order ends with H: rules 1-5 put every other task before H. So each valid
# order is an order of A, B, C, D, E, G (a "base order") with F inserted somewhere and H
# last, which is how two of the Benchmark 01 solutions counted. The base orders keep the
# rules among those six tasks: A before D and E, B before D, C before E, D and E before G.

_BASE_TASKS = "ABCDEG"


def _base_order_holds(order: str) -> bool:
    p = {task: i for i, task in enumerate(order)}
    return (
        p["A"] < p["D"]
        and p["A"] < p["E"]
        and p["B"] < p["D"]
        and p["C"] < p["E"]
        and p["D"] < p["G"]
        and p["E"] < p["G"]
    )


@functools.cache
def base_orders() -> tuple[str, ...]:
    orders = ("".join(p) for p in permutations(_BASE_TASKS))
    return tuple(sorted(o for o in orders if _base_order_holds(o)))


@functools.cache
def f_insertions(base: str) -> tuple[str, ...]:
    """The valid orders made by inserting F into ``base`` and putting H last."""
    made = (base[:i] + "F" + base[i:] + "H" for i in range(len(base) + 1))
    return tuple(o for o in made if is_valid(o))


def _ab_apart(base: str) -> bool:
    return abs(base.index("A") - base.index("B")) != 1


def _with_slot(apart: bool | None = None) -> list[str]:
    return [
        b for b in base_orders() if f_insertions(b) and (apart is None or _ab_apart(b) == apart)
    ]


#: Named intermediate counts: (description, how to count).
NAMED_COUNTS: dict[str, tuple[str, Callable[[], int]]] = {
    "base-orders": (
        "orders of A, B, C, D, E, G that keep the precedence rules among those six tasks",
        lambda: len(base_orders()),
    ),
    "base-orders-with-f-slot": (
        "base orders into which F can be inserted (H last) to give a valid order",
        lambda: len(_with_slot()),
    ),
    "base-orders-ab-apart": (
        "base orders with a place for F in which A and B are not next to each other",
        lambda: len(_with_slot(apart=True)),
    ),
    "f-insertions-ab-apart": (
        "valid orders made from base orders in which A and B are not next to each other",
        lambda: sum(len(f_insertions(b)) for b in base_orders() if _ab_apart(b)),
    ),
    "base-orders-ab-adjacent-separable": (
        "base orders in which A and B are next to each other and F can be inserted between them",
        lambda: len(_with_slot(apart=False)),
    ),
    "f-insertions-ab-adjacent": (
        "valid orders made from base orders in which A and B are next to each other",
        lambda: sum(len(f_insertions(b)) for b in base_orders() if not _ab_apart(b)),
    ),
}

_FIRST = re.compile(r"^first:([A-H])$")
_PREFIX = re.compile(r"^prefix:([A-H]{1,8})$")


def describe(key: str) -> str:
    """What a breakdown key counts, in words; ``KeyError`` for an unknown key."""
    if m := _FIRST.match(key):
        return f"valid orders that start with {m.group(1)}"
    if m := _PREFIX.match(key):
        return f"valid orders that start with {', '.join(m.group(1))}"
    if key in NAMED_COUNTS:
        return NAMED_COUNTS[key][0]
    raise KeyError(key)


def breakdown(key: str) -> int:
    """The verified value of one intermediate count.

    ``first:X`` counts valid orders that start with X, ``prefix:XY`` those that start with
    X then Y (any length), and the keys of :data:`NAMED_COUNTS` the steps of counting by
    inserting F into an order of the other tasks. ``KeyError`` for an unknown key.
    """
    if m := _FIRST.match(key) or _PREFIX.match(key):
        prefix = m.group(1)
        return sum(1 for o in valid_orders() if o.startswith(prefix))
    if key in NAMED_COUNTS:
        return NAMED_COUNTS[key][1]()
    raise KeyError(key)
