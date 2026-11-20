"""Spreadsheet-safe CSV cells (OWASP "CSV Injection")."""

from __future__ import annotations

import random

import pytest

from mblab.csvcells import FORMULA_STARTS, from_cell, from_row, to_cell


@pytest.mark.parametrize(
    ("value", "cell"),
    [
        (
            '=HYPERLINK("https://example.invalid","Open")',
            '\'=HYPERLINK("https://example.invalid","Open")',
        ),
        ("+1+1", "'+1+1"),
        ("- first item\n- second item", "'- first item\n- second item"),
        ("@SUM(A1:A9)", "'@SUM(A1:A9)"),
        ("\t=1+1", "'\t=1+1"),
        ("\r=1+1", "'\r=1+1"),
        ("-", "'-"),
        ("-1+1", "'-1+1"),
        # An apostrophe that import would remove is doubled, so the round trip is exact.
        ("'=1+1", "''=1+1"),
        ("''", "'''"),
        # Left alone: numbers and text that no spreadsheet runs.
        ("-3.5", "-3.5"),
        ("+12", "+12"),
        ("-1,5", "-1,5"),
        ("-2e-3", "-2e-3"),
        ("'quoted'", "'quoted'"),
        ("ANSWER: 265", "ANSWER: 265"),
        ("", ""),
    ],
)
def test_formula_text_gets_an_apostrophe_that_import_removes(value, cell):
    assert to_cell(value) == cell
    assert from_cell(cell) == value


@pytest.mark.parametrize("value", [None, 0, -3, -2.5, True])
def test_non_text_values_are_unchanged(value):
    assert to_cell(value) == value and from_cell(value) == value


def test_every_string_round_trips_and_no_formula_is_left():
    rng = random.Random(20261120)
    alphabet = "'=+-@\t\r a1.,e"
    for _ in range(20_000):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 5)))
        cell = to_cell(value)
        assert from_cell(cell) == value, repr(value)
        if cell and cell[0] in FORMULA_STARTS:
            float(cell.replace(",", "."))  # only a plain number may start like a formula


def test_from_row_keeps_missing_and_extra_cells():
    row = {"a": "'=1", "b": None, None: ["'=x"]}
    assert from_row(row) == {"a": "=1", "b": None, None: ["'=x"]}
