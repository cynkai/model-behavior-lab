"""Offline token-cost estimates from placed API records and an explicit rate card."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TypedDict

from pydantic import ValidationError

from mblab.rates import RateCard
from mblab.records import unplaced_records
from mblab.schema import ExperimentDefinition, RunRecord
from mblab.study import RECORDS_FILE, Study, StudyError, read_unrecorded_log


class PricingPayload(TypedDict):
    recorded_runs: int
    priced_runs: int
    unpriced_reasons: dict[str, int]
    estimated_usd: str | None


class ConditionPayload(PricingPayload):
    condition_id: str
    requested_effort: str | None


class UnmatchedRate(TypedDict):
    model: str | None
    service_tier: str | None
    runs: int


class CostEstimate(TypedDict):
    experiment_id: str
    experiment_version: str
    currency: str
    rate_card_sha256: str
    synthetic_data: bool
    pilot: bool
    unmatched_rates: list[UnmatchedRate]
    conditions: list[ConditionPayload]
    total: PricingPayload
    unplaced_records: int
    unreadable_records: int
    unrecorded_attempts: int
    unreadable_unrecorded_attempts: int


def _amount_text(amount: Fraction) -> str:
    """Render terminating decimal fractions without context rounding or exponents."""
    with localcontext() as context:
        context.prec = amount.numerator.bit_length() + amount.denominator.bit_length() + 1
        value = Decimal(amount.numerator) / Decimal(amount.denominator)
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(slots=True)
class _Subtotal:
    """Mutable accumulator for counts and exact priced amounts."""

    recorded: int = 0
    priced: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    amount: Fraction = field(default_factory=Fraction)

    def add(self, result: Fraction | str) -> None:
        self.recorded += 1
        if isinstance(result, str):
            self.reasons[result] += 1
        else:
            self.priced += 1
            self.amount += result

    def to_dict(self) -> PricingPayload:
        return {
            "recorded_runs": self.recorded,
            "priced_runs": self.priced,
            "unpriced_reasons": dict(sorted(self.reasons.items())),
            "estimated_usd": _amount_text(self.amount) if self.priced else None,
        }


def _price(record: RunRecord, card: RateCard) -> Fraction | str:
    usage = record.api_reported
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return "missing_usage"
    if (usage.reasoning_tokens is not None and usage.reasoning_tokens > usage.output_tokens) or (
        usage.total_tokens is not None
        and usage.total_tokens != usage.input_tokens + usage.output_tokens
    ):
        return "inconsistent_usage"
    if usage.model is None or usage.service_tier is None:
        return "unknown_rate"
    rate = card.rates.get((usage.model, usage.service_tier))
    if rate is None:
        return "unknown_rate"
    cached = usage.cached_input_tokens
    if cached is not None and cached > usage.input_tokens:
        return "invalid_cached_tokens"
    if cached is None:
        if usage.input_tokens and rate.input_per_million != rate.cached_input_per_million:
            return "unknown_cached_tokens"
        cached = 0
    return (
        (usage.input_tokens - cached) * Fraction(rate.input_per_million)
        + cached * Fraction(rate.cached_input_per_million)
        + usage.output_tokens * Fraction(rate.output_per_million)
    ) / 1_000_000


def _response_id(record: RunRecord) -> str:
    usage = record.api_reported
    return (usage.response_id or "").strip() if usage is not None else ""


def estimate_costs(study: Study, card: RateCard) -> CostEstimate:
    """Price all placed responses, including deviations and incomplete responses."""
    plan = study.require_plan()
    if study.definition.execution_method != "official_api":
        raise StudyError("cost estimates require an official_api study")
    try:
        original = ExperimentDefinition.model_validate(plan.definition_snapshot)
    except ValidationError as exc:
        raise StudyError("invalid plan definition snapshot; restore the original plan") from exc
    current = study.definition
    original_conditions = {
        c.id: original.option(c.model_selection).reasoning_effort for c in original.conditions
    }
    current_conditions = {
        c.id: current.option(c.model_selection).reasoning_effort for c in current.conditions
    }
    plan_identity = (plan.experiment_id, plan.experiment_version)
    if (
        (current.experiment_id, current.experiment_version) != plan_identity
        or (original.experiment_id, original.experiment_version) != plan_identity
        or original.execution_method != "official_api"
        or original_conditions != current_conditions
    ):
        raise StudyError(
            "study identity or conditions differ from the plan; restore the planned definition "
            "before estimating historical costs"
        )
    left_out = {run_id for ids in unplaced_records(study).values() for run_id in ids}
    placed = [record for record in study.records if record.run_id not in left_out]
    response_ids = Counter(_response_id(record) for record in placed)
    subtotals = {condition_id: _Subtotal() for condition_id in current_conditions}
    total = _Subtotal()
    unmatched: Counter[tuple[str | None, str | None]] = Counter()
    for record in placed:
        if record.condition_id not in subtotals:
            raise StudyError(
                "record condition is absent from the definition; restore the planned definition"
            )
        usage = record.api_reported
        response_id = _response_id(record)
        result = (
            "duplicate_response"
            if response_id and response_ids[response_id] > 1
            else _price(record, card)
        )
        if result == "unknown_rate" and usage is not None:
            unmatched[(usage.model, usage.service_tier)] += 1
        subtotals[record.condition_id].add(result)
        total.add(result)
    conditions: list[ConditionPayload] = [
        {
            **subtotals[condition_id].to_dict(),
            "condition_id": condition_id,
            "requested_effort": effort,
        }
        for condition_id, effort in current_conditions.items()
    ]
    held, held_errors = read_unrecorded_log(study.root)
    return {
        "experiment_id": study.definition.experiment_id,
        "experiment_version": study.definition.experiment_version,
        "currency": "USD",
        "rate_card_sha256": card.sha256,
        "synthetic_data": study.is_synthetic() or original.demo,
        "pilot": study.definition.pilot or original.pilot,
        "unmatched_rates": [
            {"model": model, "service_tier": tier, "runs": count}
            for (model, tier), count in sorted(
                unmatched.items(),
                key=lambda item: tuple((value is not None, value or "") for value in item[0]),
            )
        ],
        "conditions": conditions,
        "total": total.to_dict(),
        "unplaced_records": len(study.records) - len(placed),
        "unreadable_records": sum(error.file == RECORDS_FILE for error in study.load_errors),
        "unrecorded_attempts": len(held),
        "unreadable_unrecorded_attempts": len(held_errors),
    }
