from __future__ import annotations

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from mblab.rates import TokenRates, load_rate_card
from mblab.study import StudyError

_ENTRY: Final = {
    "model": "model-a",
    "service_tier": "default",
    "input_per_million": "2",
    "cached_input_per_million": "0.5",
    "output_per_million": "6",
}


def test_preserves_exact_rates_when_decimal_context_has_low_precision(tmp_path: Path) -> None:
    entry = dict(
        _ENTRY,
        input_per_million="999999999999.999999999999",
        cached_input_per_million="0.000000000001",
        output_per_million="0.000",
    )
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"currency": "USD", "rates": [entry]}), encoding="utf-8")
    with localcontext() as context:
        context.prec = 3
        card = load_rate_card(path)
    assert card.rates[("model-a", "default")] == TokenRates(
        Decimal("999999999999.999999999999"), Decimal("0.000000000001"), Decimal("0.000")
    )
    assert card.rates[("model-a", "default")].output_per_million.as_tuple().exponent == -3


def test_hashes_original_bytes_when_json_has_extra_whitespace(tmp_path: Path) -> None:
    raw = (json.dumps({"currency": "USD", "rates": [_ENTRY]}, indent=3) + "\r\n").encode()
    path = tmp_path / "rates.json"
    path.write_bytes(raw)
    card = load_rate_card(path)
    assert card.sha256 == hashlib.sha256(raw).hexdigest()


def test_keeps_exact_model_and_tier_when_similar_pairs_exist(tmp_path: Path) -> None:
    entries = [_ENTRY, dict(_ENTRY, service_tier="priority"), dict(_ENTRY, model="Model-A")]
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"currency": "USD", "rates": entries}), encoding="utf-8")
    card = load_rate_card(path)
    assert set(card.rates) == {
        ("model-a", "default"),
        ("model-a", "priority"),
        ("Model-A", "default"),
    }


@pytest.mark.parametrize(
    "field", ["input_per_million", "cached_input_per_million", "output_per_million"]
)
@pytest.mark.parametrize(
    "price",
    [
        2,
        0.5,
        True,
        None,
        "",
        "-1",
        "-0",
        "+1",
        "NaN",
        "Infinity",
        "1e2",
        "1E-2",
        ".5",
        "1.",
        " 1",
        "1 ",
        "1\n",
        "１",
        "1000000000000",
        "0.0000000000001",
    ],
)
def test_rejects_rate_when_not_a_bounded_fixed_point_string(
    tmp_path: Path, field: str, price: str | int | float | None
) -> None:
    entry = {**_ENTRY, field: price}
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"currency": "USD", "rates": [entry]}), encoding="utf-8")
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)


@pytest.mark.parametrize("field", ["model", "service_tier"])
@pytest.mark.parametrize("identifier", ["", " ", " model-a", "model-a\n", 3, None])
def test_rejects_identifier_when_empty_padded_or_not_string(
    tmp_path: Path, field: str, identifier: str | int | None
) -> None:
    path = tmp_path / "rates.json"
    path.write_text(
        json.dumps({"currency": "USD", "rates": [{**_ENTRY, field: identifier}]}), encoding="utf-8"
    )
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"currency": "EUR", "rates": [_ENTRY]},
        {"currency": "USD", "rates": []},
        {"currency": "USD", "rates": [_ENTRY], "extra": 1},
        {"rates": [_ENTRY]},
        {"currency": "USD"},
        {"currency": "USD", "rates": {}},
        {"currency": "USD", "rates": [dict(_ENTRY, extra="1")]},
        {"currency": "USD", "rates": [_ENTRY, _ENTRY]},
    ],
)
def test_rejects_document_when_shape_or_pairs_are_invalid(
    tmp_path: Path, payload: JsonValue
) -> None:
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)


@pytest.mark.parametrize("field", list(_ENTRY))
def test_rejects_entry_when_required_field_missing(tmp_path: Path, field: str) -> None:
    entry = {key: value for key, value in _ENTRY.items() if key != field}
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"currency": "USD", "rates": [entry]}), encoding="utf-8")
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b"{",
        b'{"currency":"USD","rates":NaN}',
        b'{"currency":"USD","rates":Infinity}',
        b'{"currency":"USD","rates":-Infinity}',
        b"[" * 2000,
    ],
)
def test_rejects_file_when_encoding_or_json_is_invalid(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "rates.json"
    path.write_bytes(raw)
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)


@pytest.mark.parametrize("key", ["currency", "rates", "model", "input_per_million"])
def test_rejects_duplicate_key_when_document_is_otherwise_valid(tmp_path: Path, key: str) -> None:
    raw = json.dumps({"currency": "USD", "rates": [_ENTRY]})
    value = json.dumps({"currency": "USD", "rates": [_ENTRY], **_ENTRY}[key])
    raw = raw.replace(f'"{key}": {value}', f'"{key}": {value}, "{key}": {value}')
    path = tmp_path / "rates.json"
    path.write_text(raw, encoding="utf-8")
    # When / Then
    with pytest.raises(StudyError, match="duplicate JSON key"):
        load_rate_card(path)


@pytest.mark.parametrize("name", ["missing.json", "."])
def test_reports_study_error_when_file_cannot_be_read(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    # When / Then
    with pytest.raises(StudyError):
        load_rate_card(path)
