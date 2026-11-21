"""The independent ground truth of Benchmark 01's task-ordering puzzle.

Every value here is recomputed from the eight rules by enumerating all 8! orders; none is
read from the benchmark data.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from mblab import task_ordering as t

ALL_ORDERS = ["".join(p) for p in permutations("ABCDEFGH")]


def _satisfies_rules(order: str) -> bool:
    """The eight rules written a second time, independently of the module's rule table."""
    i = order.index
    return (
        i("A") < min(i("D"), i("E"))  # 1
        and i("B") < i("D")  # 2
        and i("C") < min(i("E"), i("F"))  # 3
        and max(i("D"), i("E")) < i("G")  # 4
        and max(i("F"), i("G")) < i("H")  # 5
        and abs(i("A") - i("B")) > 1  # 6
        and abs(i("D") - i("F")) > 1  # 7
        and i("C") <= 3  # 8: positions 1 to 4
    )


def test_every_order_is_checked_and_the_two_rule_sets_agree():
    assert len(ALL_ORDERS) == 40_320
    assert tuple(sorted(o for o in ALL_ORDERS if _satisfies_rules(o))) == t.valid_orders()


def test_exactly_23_orders_satisfy_every_rule():
    assert t.ground_truth().count == 23
    assert len(set(t.valid_orders())) == 23


def test_lexicographically_first_order():
    assert t.ground_truth().lexicographic_first == "ACBDEFGH"
    assert min(t.valid_orders()) == "ACBDEFGH"


def test_acbdegfh_is_another_valid_order():
    assert t.is_valid("ACBDEGFH")
    assert t.violations("ACBDEGFH") == []
    assert t.ground_truth().lexicographic_first != "ACBDEGFH"


def test_valid_orders_by_first_task():
    assert t.ground_truth().prefix_counts(1) == {"A": 9, "B": 6, "C": 8}


def test_valid_orders_by_first_two_tasks():
    assert t.ground_truth().prefix_counts(2) == {"AC": 9, "BC": 6, "CA": 5, "CB": 2, "CF": 1}


@pytest.mark.parametrize(
    ("order", "broken"),
    [
        ("BCDAEFGH", [1]),
        ("ACDBEFGH", [2]),
        ("AEBCDGFH", [3]),
        ("ACBDGEFH", [4]),
        ("ACBDEFHG", [5]),
        ("ABCDEFGH", [6]),
        ("ACBDFEGH", [7]),
        # E F G H A B C D: A after E (1), C after E (3), D after G (4), A next to B (6),
        # C in position 7 (8). F and G before H (5) and D away from F (7) still hold.
        ("EFGHABCD", [1, 3, 4, 6, 8]),
    ],
)
def test_each_rule_is_checked_on_its_own(order, broken):
    assert t.violations(order) == broken
    assert not t.is_valid(order)
    assert _satisfies_rules(order) is False


def test_rule_8_is_implied_by_the_other_rules():
    # C precedes E and F, hence G and H: four tasks after it, so it is always in the first
    # four positions. No order breaks rule 8 alone.
    assert t.redundant_rules() == (8,)
    assert not any(t.violations(o) == [8] for o in ALL_ORDERS)


def test_h_is_always_last():
    assert all(o.endswith("H") for o in t.valid_orders())


@pytest.mark.parametrize("bad", ["ACBDEFG", "ACBDEFGHA", "ACBDEFGG", "ACBDEFGX", ""])
def test_anything_but_an_order_of_the_eight_tasks_is_refused(bad):
    with pytest.raises(ValueError, match="not an order"):
        t.violations(bad)


def test_lowercase_and_spaces_around_are_read():
    assert t.parse_order(" acbdefgh ") == "ACBDEFGH"


def test_counting_by_inserting_f_gives_the_same_orders():
    made = sorted(o for base in t.base_orders() for o in t.f_insertions(base))
    assert tuple(made) == t.valid_orders()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("first:A", 9),
        ("first:B", 6),
        ("first:C", 8),
        ("prefix:AC", 9),
        ("prefix:BC", 6),
        ("prefix:CA", 5),
        ("prefix:CB", 2),
        ("prefix:CF", 1),
        ("prefix:ACBDEFGH", 1),
        ("base-orders", 16),
        ("base-orders-with-f-slot", 10),
        ("base-orders-ab-apart", 6),
        ("f-insertions-ab-apart", 19),
        ("base-orders-ab-adjacent-separable", 4),
        ("f-insertions-ab-adjacent", 4),
    ],
)
def test_intermediate_counts(key, value):
    assert t.breakdown(key) == value
    assert t.describe(key)


def test_the_two_decompositions_add_up():
    assert t.breakdown("f-insertions-ab-apart") + t.breakdown("f-insertions-ab-adjacent") == 23


@pytest.mark.parametrize("key", ["first:Z", "prefix:", "orders", "first:a"])
def test_unknown_breakdown_keys_are_refused(key):
    with pytest.raises(KeyError):
        t.breakdown(key)
    with pytest.raises(KeyError):
        t.describe(key)
