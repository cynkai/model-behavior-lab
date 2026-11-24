"""Strict, local USD token rate cards with exact decimal prices."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from mblab.study import StudyError, format_validation_error

_PRICE_PATTERN: Final = re.compile(r"[0-9]{1,12}(?:\.[0-9]{1,12})?")


@dataclass(frozen=True, slots=True)
class TokenRates:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True, slots=True)
class RateCard:
    sha256: str
    rates: Mapping[tuple[str, str], TokenRates]


class _RateEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    service_tier: str
    input_per_million: str
    cached_input_per_million: str
    output_per_million: str

    @field_validator("model", "service_tier")
    @classmethod
    def exact_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "rate card identifiers must be nonempty without surrounding whitespace"
            )
        return value

    @field_validator("input_per_million", "cached_input_per_million", "output_per_million")
    @classmethod
    def fixed_point_price(cls, value: str) -> str:
        if _PRICE_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "rate card prices must be nonnegative fixed-point strings with "
                "at most 12 integer and 12 fractional digits"
            )
        return value


class _RateDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    currency: Literal["USD"]
    rates: Annotated[list[_RateEntry], Field(min_length=1)]


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> Mapping[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise StudyError(f"rate card contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise StudyError(f"rate card contains invalid JSON constant {value!r}")


def load_rate_card(path: Path) -> RateCard:
    """Read a UTF-8 card, rejecting ambiguity before producing exact prices."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StudyError(f"{path}: cannot read rate card: {exc}") from None
    try:
        decoded = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        document = _RateDocument.model_validate(decoded)
    except UnicodeDecodeError:
        raise StudyError(f"{path}: rate card must be UTF-8") from None
    except ValidationError as exc:
        raise StudyError(f"{path}: invalid rate card: {format_validation_error(exc)}") from None
    except (ValueError, RecursionError) as exc:
        raise StudyError(f"{path}: invalid rate card JSON: {exc}") from None
    rates: dict[tuple[str, str], TokenRates] = {}
    for entry in document.rates:
        key = (entry.model, entry.service_tier)
        if key in rates:
            raise StudyError(f"{path}: duplicate model/service_tier pair {key!r}")
        rates[key] = TokenRates(
            input_per_million=Decimal(entry.input_per_million),
            cached_input_per_million=Decimal(entry.cached_input_per_million),
            output_per_million=Decimal(entry.output_per_million),
        )
    return RateCard(sha256=hashlib.sha256(raw).hexdigest(), rates=MappingProxyType(rates))
