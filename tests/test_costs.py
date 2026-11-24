"""Exact, offline accounting of all placed API usage."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from mblab.costs import estimate_costs
from mblab.plan import generate_plan
from mblab.rates import RateCard, TokenRates
from mblab.schema import ApiReported, ExperimentDefinition, RunRecord
from mblab.study import LineError, Study, StudyError, parse_definition, template_text

from .conftest import FIXED_NOW, make_record


@pytest.fixture
def api_study(tmp_path: Path) -> Study:
    definition = parse_definition(template_text("openai-reasoning-effort"))
    return Study(tmp_path, definition, generate_plan(definition, now=FIXED_NOW), [], [])


@pytest.fixture
def card() -> RateCard:
    return RateCard(
        sha256="a" * 64,
        rates={
            ("returned-model", "default"): TokenRates(Decimal("2"), Decimal("0.5"), Decimal("8")),
            ("equal", "priority"): TokenRates(Decimal("3"), Decimal("3"), Decimal("0")),
        },
    )


def record(study: Study, index: int = 0, **usage: str | int | None) -> RunRecord:
    plan = study.require_plan()
    reported = {
        "model": "returned-model",
        "service_tier": "default",
        "input_tokens": 1000,
        "cached_input_tokens": 200,
        "output_tokens": 100,
        "reasoning_tokens": 80,
    }
    reported.update(usage)
    return make_record(
        study.definition,
        plan,
        plan.runs[index].run_id,
        latency_method="client_timer",
        api_reported=ApiReported.model_validate(reported),
    )


def test_exact_subtotal_includes_deviations_and_incomplete(api_study, card):
    # Given all response outcomes consume the same reported usage.
    api_study.records = [
        record(api_study),
        record(api_study, 1).model_copy(update={"protocol_deviation": "wrong settings"}),
        record(api_study, 2).model_copy(update={"response_status": "incomplete"}),
    ]
    payload = estimate_costs(api_study, card)
    # Then reasoning tokens are already included in output, not billed twice.
    assert payload["total"] == {
        "recorded_runs": 3,
        "priced_runs": 3,
        "unpriced_reasons": {},
        "estimated_usd": "0.0075",
    }
    assert [row["condition_id"] for row in payload["conditions"]] == [
        condition.id for condition in api_study.definition.conditions
    ]
    assert [row["requested_effort"] for row in payload["conditions"]] == ["low", "medium", "high"]
    assert payload["rate_card_sha256"] == card.sha256
    assert payload["experiment_id"] == api_study.definition.experiment_id


@pytest.mark.parametrize(
    ("usage", "reason"),
    [
        ({"input_tokens": None}, "missing_usage"),
        ({"reasoning_tokens": 101}, "inconsistent_usage"),
        ({"total_tokens": 999}, "inconsistent_usage"),
        ({"output_tokens": None}, "missing_usage"),
        ({"model": None}, "unknown_rate"),
        ({"model": "unlisted"}, "unknown_rate"),
        ({"service_tier": None}, "unknown_rate"),
        ({"service_tier": "priority"}, "unknown_rate"),
        ({"cached_input_tokens": 1001}, "invalid_cached_tokens"),
        ({"cached_input_tokens": None}, "unknown_cached_tokens"),
    ],
)
def test_unpriceable_usage_stays_unknown(api_study, card, usage, reason):
    # Given one unpriceable response.
    api_study.records = [record(api_study, **usage)]
    total = estimate_costs(api_study, card)["total"]
    # Then missing evidence cannot become zero spending.
    assert total == {
        "recorded_runs": 1,
        "priced_runs": 0,
        "unpriced_reasons": {reason: 1},
        "estimated_usd": None,
    }


@pytest.mark.parametrize(
    ("usage", "amount"),
    [
        ({"model": "equal", "service_tier": "priority", "cached_input_tokens": None}, "0.003"),
        (
            {
                "input_tokens": 0,
                "cached_input_tokens": None,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
            "0",
        ),
        (
            {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
            "0",
        ),
    ],
)
def test_priceable_unknown_cache_and_zero(api_study, card, usage, amount):
    # Given rates or zero inputs make the cached split immaterial.
    api_study.records = [record(api_study, **usage)]
    total = estimate_costs(api_study, card)["total"]
    # Then the measured amount remains priceable, including zero.
    assert total["priced_runs"] == 1
    assert total["estimated_usd"] == amount


def test_duplicate_response_ids_are_all_excluded_across_conditions(api_study, card):
    # Given distinct planned records carry one response ID; missing IDs remain acceptable.
    api_study.records = [
        record(api_study, 0, response_id="same"),
        record(api_study, 1, response_id=" same "),
    ]
    api_study.records += [record(api_study, 2), record(api_study, 3)]
    total = estimate_costs(api_study, card)["total"]
    # Then both duplicate copies are excluded, not just the second occurrence.
    assert total == {
        "recorded_runs": 4,
        "priced_runs": 2,
        "unpriced_reasons": {"duplicate_response": 2},
        "estimated_usd": "0.005",
    }


def test_large_counts_and_rates_are_exact_under_small_decimal_context(api_study, card):
    # Given more digits than any default decimal context or IEEE float can retain.
    count = 10**100 + 1
    rates = TokenRates(Decimal("1.00000000000000000000000000001"), Decimal("0"), Decimal("0"))
    card = replace(card, rates={("returned-model", "default"): rates})
    api_study.records = [
        record(
            api_study,
            input_tokens=count,
            cached_input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
        )
    ]
    # When callers have a deliberately tiny decimal context.
    with localcontext() as context:
        context.prec = 3
        amount = estimate_costs(api_study, card)["total"]["estimated_usd"]
    # Then every digit survives multiplication and rendering.
    expected_units = count * (10**29 + 1)
    expected = str(expected_units)[:-35] + "." + str(expected_units)[-35:]
    assert amount == expected.rstrip("0").rstrip(".")


def test_missing_records_and_study_level_exclusions(api_study, card):
    # Given duplicate run records, malformed records, and held responses outside subtotals.
    api_study.records = [record(api_study), record(api_study)]
    api_study.load_errors = [
        LineError("records.jsonl", 3, "invalid"),
        LineError("grades.jsonl", 1, "invalid"),
    ]
    (api_study.root / "api-unrecorded.jsonl").write_bytes(
        b'{"run_id":"held"}\n{"run_id":"held"}\ninvalid\n\xff\n'
    )
    payload = estimate_costs(api_study, card)
    # Then counts distinguish each excluded row and do not invent a subtotal.
    assert payload["unplaced_records"] == 2
    assert payload["unreadable_records"] == 1
    assert payload["unrecorded_attempts"] == 2
    assert payload["unreadable_unrecorded_attempts"] == 2
    assert payload["total"] == {
        "recorded_runs": 0,
        "priced_runs": 0,
        "unpriced_reasons": {},
        "estimated_usd": None,
    }


def test_empty_study_is_unknown_and_preserves_flags(api_study, card):
    # Given a pilot synthetic study without a records file.
    api_study.definition = api_study.definition.model_copy(update={"pilot": True, "demo": True})
    payload = estimate_costs(api_study, card)
    # Then absence differs from measured zero.
    assert payload["synthetic_data"] and payload["pilot"]
    assert payload["total"]["estimated_usd"] is None
    assert all(row["estimated_usd"] is None for row in payload["conditions"])


@pytest.mark.parametrize("change", ["effort", "removed", "identity"])
def test_changed_historical_conditions_are_rejected(api_study, card, change):
    # Given a definition edited after planning.
    data = api_study.definition.model_dump()
    if change == "effort":
        data["selection_options"][0]["reasoning_effort"] = "high"
        data["selection_options"][2]["reasoning_effort"] = "low"
    elif change == "removed":
        data["conditions"] = data["conditions"][1:]
    else:
        data["experiment_id"] = "different"
    api_study.definition = ExperimentDefinition.model_validate(data)
    # When estimating, then historical effort cannot be silently relabeled.
    with pytest.raises(StudyError, match="restore the planned definition"):
        estimate_costs(api_study, card)


def test_plan_is_required(api_study, card):
    # Given a study without a plan.
    api_study.plan = None
    # When estimating, then the user receives actionable guidance.
    with pytest.raises(StudyError, match="plan"):
        estimate_costs(api_study, card)


def test_ui_study_is_rejected(study_dir, card):
    # Given a UI study.
    study = Study.load(study_dir)
    # When estimating, then token billing cannot be inferred.
    with pytest.raises(StudyError, match="official_api"):
        estimate_costs(study, card)


@pytest.mark.parametrize("status", ["completed", "incomplete", "error", "refused"])
def test_every_recorded_status_is_priced(api_study, card, status):
    # Given even unsuccessful responses carry measurable usage.
    api_study.records = [record(api_study).model_copy(update={"response_status": status})]
    total = estimate_costs(api_study, card)["total"]
    # Then response success is irrelevant to spend.
    assert total["priced_runs"] == 1
    assert total["estimated_usd"] == "0.0025"


def test_missing_api_reported_and_unplaced_records(api_study, card):
    # Given one placed record without usage and three records outside plan identity.
    api_study.records = [
        record(api_study).model_copy(update={"api_reported": None}),
        record(api_study, 1).model_copy(update={"run_id": "not-planned"}),
        record(api_study, 2).model_copy(update={"task_id": "not-defined"}),
        record(api_study, 3).model_copy(update={"repetition": 99}),
    ]
    payload = estimate_costs(api_study, card)
    # Then unplaced records cannot inflate recorded or priced totals.
    assert payload["unplaced_records"] == 3
    assert payload["total"] == {
        "recorded_runs": 1,
        "priced_runs": 0,
        "unpriced_reasons": {"missing_usage": 1},
        "estimated_usd": None,
    }


def test_unmatched_rates_identify_exact_pairs_in_stable_order(api_study, card):
    # Given repeated missing card entries and absent model/tier evidence.
    api_study.records = [
        record(api_study, 0, model="z"),
        record(api_study, 1, model=None),
        record(api_study, 2, model="z"),
        record(api_study, 3, service_tier=None),
    ]
    payload = estimate_costs(api_study, card)
    # Then only exact returned identities and counts are exposed.
    assert payload["unmatched_rates"] == [
        {"model": None, "service_tier": "default", "runs": 1},
        {"model": "returned-model", "service_tier": None, "runs": 1},
        {"model": "z", "service_tier": "default", "runs": 2},
    ]


def test_original_pilot_and_synthetic_flags_survive_current_clear(api_study, card):
    # Given the original plan designated synthetic pilot data.
    snapshot = dict(api_study.require_plan().definition_snapshot, pilot=True, demo=True)
    api_study.plan = api_study.require_plan().model_copy(update={"definition_snapshot": snapshot})
    # When the current definition no longer carries those flags.
    payload = estimate_costs(api_study, card)
    # Then provenance remains visible.
    assert payload["synthetic_data"] and payload["pilot"]


def test_reordered_conditions_preserve_effort_mapping(api_study, card):
    # Given display order changes without changing any requested effort.
    api_study.definition = api_study.definition.model_copy(
        update={"conditions": list(reversed(api_study.definition.conditions))}
    )
    payload = estimate_costs(api_study, card)
    # Then current definition order is retained without relabeling efforts.
    assert [row["requested_effort"] for row in payload["conditions"]] == ["high", "medium", "low"]


@pytest.mark.parametrize("snapshot", [{}, {"experiment_id": "wrong"}])
def test_invalid_plan_snapshot_is_actionable(api_study, card, snapshot):
    # Given an unreadable definition snapshot in a readable plan.
    api_study.plan = api_study.require_plan().model_copy(update={"definition_snapshot": snapshot})
    # When estimating, then a user-facing error replaces a validation traceback.
    with pytest.raises(StudyError, match="restore the original plan"):
        estimate_costs(api_study, card)


def test_snapshot_identity_must_agree_with_plan(api_study, card):
    # Given a valid snapshot attached to the wrong experiment's plan.
    snapshot = dict(api_study.require_plan().definition_snapshot, experiment_id="different")
    api_study.plan = api_study.require_plan().model_copy(update={"definition_snapshot": snapshot})
    # When estimating, then conflicting historical identity is rejected.
    with pytest.raises(StudyError, match="restore the planned definition"):
        estimate_costs(api_study, card)


def test_blank_response_ids_do_not_link_independent_runs(api_study, card):
    api_study.records = [record(api_study, i, response_id=" ") for i in range(2)]
    assert estimate_costs(api_study, card)["total"]["priced_runs"] == 2
