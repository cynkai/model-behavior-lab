"""Spreadsheet-safe CSV cells.

Spreadsheet apps treat a cell that starts with ``=``, ``+``, ``-``, ``@``, a tab, or a
carriage return as a formula. A response such as ``=HYPERLINK("https://…", "Open")`` in
an exported CSV would then run when the file is opened, and an ordinary answer that
starts with "- " shows up as ``#NAME?``. Every text cell mblab writes to a CSV goes
through :func:`to_cell`, which puts an apostrophe in front of such text so the app shows
it as text. Numbers such as ``-3.5`` are left alone.

Importing reverses it: :func:`from_cell` removes that one apostrophe, so a file that
mblab wrote and the user filled in comes back exactly as it was written
(``from_cell(to_cell(text)) == text`` for every string).
"""

from __future__ import annotations

import re
from typing import Any

#: First characters that make a spreadsheet app evaluate a cell (OWASP "CSV Injection").
FORMULA_STARTS = frozenset("=+-@\t\r")
QUOTE = "'"
_NUMBER = re.compile(r"[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?")


def _marked(text: str) -> bool:
    """True if ``text`` starts with the apostrophe that :func:`to_cell` adds."""
    return len(text) >= 2 and text[0] == QUOTE and (text[1] == QUOTE or text[1] in FORMULA_STARTS)


def to_cell(value: Any) -> Any:
    """``value`` for a CSV cell that spreadsheet apps show as text instead of running.

    Text starting with a formula character gets a leading apostrophe. So does text that
    already starts with one that :func:`from_cell` would remove, which keeps the round
    trip exact. Other values, numbers and ``None`` included, are returned unchanged.
    """
    if not isinstance(value, str) or not value:
        return value
    if value[0] in FORMULA_STARTS:
        return value if _NUMBER.fullmatch(value) else QUOTE + value
    return QUOTE + value if _marked(value) else value


def from_cell(value: Any) -> Any:
    """Undo :func:`to_cell` for a value read from a CSV file."""
    if isinstance(value, str) and _marked(value):
        return value[1:]
    return value


def from_row(row: dict[Any, Any]) -> dict[Any, Any]:
    """A ``csv.DictReader`` row with :func:`from_cell` applied to every cell."""
    return {key: from_cell(value) for key, value in row.items()}
