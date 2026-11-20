from __future__ import annotations

import json
import re
from datetime import timedelta

import pytest
import yaml

from mblab.analysis import VERDICT_TEXT, AnalysisSettings, analyze, classify_label
from mblab.plan import generate_plan
from mblab.report import render_report
from mblab.schema import ExperimentDefinition
from mblab.study import Study, dump_json, save_plan, template_text

from .conftest import FIXED_NOW, make_record, write_records

BANNED = re.compile(
    r"\b(prove[sn]?|proof|confirm(s|ed)?|conclusive(ly)?|definitive(ly)?|definitely|"
    r"guarantee[sd]?)\b",
    re.IGNORECASE,
)


def _populate(root, definition, plan, correct_by_condition, latency_by_condition=None, **extra):
    """One record per planned run; answers right/wrong per condition."""
    records = []
    for run in plan.runs:
        task = definition.task(run.task_id)
        ok = correct_by_condition[run.condition_id]
        answer = task.reference_answer if ok else "0"
        latency = (latency_by_condition or {}).get(run.condition_id, 10.0) + run.repetition
        records.append(
            make_record(
                definition,
                plan,
                run.run_id,
                response_text=f"ANSWER: {answer}",
                latency_seconds=latency,
                **extra,
            )
        )
    write_records(root, records)
    return Study.load(root)


def _verdicts(analysis, condition_id):
    (item,) = [o for o in analysis.overrides if o.condition_id == condition_id]
    return {m.metric: m.verdict for m in item.metrics}


def test_per_message_pattern(study_dir, definition, plan):
    study = _populate(
        study_dir,
        definition,
        plan,
        {"A": False, "B": True, "C": True, "D": False},
        {"A": 5, "B": 100, "C": 100, "D": 5},
    )
    a = analyze(study)
    assert _verdicts(a, "C")["accuracy"] == "consistent_with_per_message"
    assert _verdicts(a, "D")["accuracy"] == "consistent_with_per_message"
    assert _verdicts(a, "C")["median response time"] == "consistent_with_per_message"
    row = a.conditions.set_index("condition_id").loc["C"]
    assert (row.n_scored, row.n_correct, row.accuracy) == (18, 18, 1.0)


def test_default_pattern(study_dir, definition, plan):
    study = _populate(study_dir, definition, plan, {"A": False, "B": True, "C": False, "D": True})
    a = analyze(study)
    assert _verdicts(a, "C")["accuracy"] == "consistent_with_default"
    assert _verdicts(a, "D")["accuracy"] == "consistent_with_default"


def test_references_that_do_not_differ_cannot_discriminate(study_dir, definition, plan):
    study = _populate(study_dir, definition, plan, {"A": True, "B": True, "C": True, "D": False})
    a = analyze(study)
    assert _verdicts(a, "C")["accuracy"] == "references_not_distinguishable"
    # Latency is identical across conditions too, so it cannot discriminate either.
    assert _verdicts(a, "C")["median response time"] == "references_not_distinguishable"


def test_small_samples_are_not_compared(study_dir, definition, plan):
    runs = [r for r in plan.runs if r.repetition == 1 and r.task_id in {"T1", "T2"}]
    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in runs])
    a = analyze(Study.load(study_dir))
    assert set(_verdicts(a, "C").values()) == {"insufficient_data"}
    assert set(a.comparisons["status"]) == {"insufficient_data"}


def test_no_records_produces_no_numbers(study_dir):
    a = analyze(Study.load(study_dir))
    assert not a.has_data and not a.has_scored_data
    assert a.conditions["accuracy"].isna().all()
    assert a.comparisons.empty
    assert (a.grid["outcome"] == "missing").all() and len(a.grid) == 72
    assert all(m.verdict == "insufficient_data" for o in a.overrides for m in o.metrics)
    assert not [c for c in a.claims if c.category == "observation"]
    assert any("72 of 72 planned runs have no record" in lim.text for lim in a.limitations)


def test_outcome_accounting(study_dir, definition, plan):
    records = [
        make_record(definition, plan, "A:T1:r1"),  # correct
        make_record(definition, plan, "A:T1:r2", response_text="It is 265."),  # format failure
        make_record(definition, plan, "A:T1:r3", response_status="error", response_text=""),
        make_record(definition, plan, "A:T2:r1", protocol_deviation="wrong default"),
        make_record(definition, plan, "A:T2:r2", response_text="ANSWER: 2027-02-28"),
    ]
    write_records(study_dir, records)
    a = analyze(Study.load(study_dir))
    row = a.conditions.set_index("condition_id").loc["A"]
    assert (row.n_recorded, row.n_excluded, row.n_included) == (5, 1, 4)
    assert (row.n_scored, row.n_correct, row.n_format_failures, row.n_not_completed) == (3, 1, 1, 1)
    assert row.accuracy == pytest.approx(1 / 3)
    assert row.success_rate == pytest.approx(1 / 4)
    outcomes = a.grid.set_index("run_id")["outcome"]
    assert outcomes["A:T1:r2"] == "format_failure"
    assert outcomes["A:T1:r3"] == "not_scorable"
    assert outcomes["A:T2:r1"] == "excluded"
    assert outcomes["A:T3:r1"] == "missing"


def test_label_classification(definition):
    assert classify_label(definition, "instant", "pro", "Pro") == "matches_override"
    assert classify_label(definition, "instant", "pro", "  instant ") == "matches_default"
    assert classify_label(definition, "instant", None, "Instant") == "matches_default"
    assert classify_label(definition, "instant", None, "Pro") == "other_option"
    assert classify_label(definition, "instant", "pro", "GPT-9") == "unrecognized"
    assert classify_label(definition, "instant", "pro", None) == "not_displayed"


def test_label_aliases_are_pre_registered(definition, plan, study_dir):
    data = yaml.safe_load(template_text())
    data["selection_options"][1]["observed_label_aliases"] = ["ChatGPT Pro"]
    d2 = ExperimentDefinition.model_validate(data)
    assert classify_label(d2, "instant", "pro", "chatgpt pro") == "matches_override"
    assert classify_label(definition, "instant", "pro", "chatgpt pro") == "unrecognized"


def _stacked_definition() -> ExperimentDefinition:
    data = yaml.safe_load(template_text())
    data["selection_options"].append({"id": "xhigh", "label": "Extra High"})
    data["hypotheses"].append({"id": "H2", "statement": "Stacking changes nothing observable."})
    data["conditions"] = [
        {"id": "E", "model_selection": "xhigh", "message_override": None},
        {
            "id": "F",
            "model_selection": "xhigh",
            "message_override": "xhigh",
            "hypothesis_ids": ["H2"],
        },
        {"id": "G", "model_selection": "xhigh", "message_override": "instant"},
    ]
    return ExperimentDefinition.model_validate(data)


def test_stacked_and_missing_reference(tmp_path):
    definition = _stacked_definition()
    plan = generate_plan(definition, now=FIXED_NOW)
    root = tmp_path / "s"
    root.mkdir()
    (root / "experiment.yaml").write_text(yaml.safe_dump(definition.model_dump(mode="json")))
    save_plan(plan, root / "plan.json")
    study = _populate(root, definition, plan, {"E": True, "F": True, "G": True})
    a = analyze(study)
    f = next(o for o in a.overrides if o.condition_id == "F")
    assert f.stacked and f.default_ref == "E" == f.override_ref
    assert {m.metric: m.verdict for m in f.metrics}["accuracy"] == "no_difference_detected"
    g = next(o for o in a.overrides if o.condition_id == "G")
    assert g.metrics == [] and "No reference condition" in g.note


def test_latency_methods_are_not_pooled(study_dir, definition, plan):
    records = [make_record(definition, plan, r.run_id) for r in plan.runs[:10]]
    records += [
        make_record(
            definition, plan, r.run_id, latency_method="screen_recording", latency_seconds=99
        )
        for r in plan.runs[10:13]
    ]
    write_records(study_dir, records)
    a = analyze(Study.load(study_dir))
    assert a.primary_latency_method == "manual_stopwatch"
    assert set(a.latency["latency_method"]) == {"manual_stopwatch", "screen_recording"}
    assert 99 not in set(a.conditions["latency_median"].dropna())


def test_analysis_serializes_and_uses_careful_language(study_dir, definition, plan):
    study = _populate(study_dir, definition, plan, {"A": False, "B": True, "C": True, "D": False})
    a = analyze(study, AnalysisSettings(seed=3))
    payload = json.loads(dump_json(a.to_dict()))
    assert payload["synthetic_data"] is False
    assert len(payload["conditions"]) == 4
    assert payload["override_assessments"][0]["metrics"]
    texts = [lim.text for lim in a.limitations] + [c.text + c.source for c in a.claims]
    texts += list(VERDICT_TEXT.values())
    for text in texts:
        assert not BANNED.search(text), text
    categories = {c.category for c in a.claims}
    assert {"hypothesis", "observation", "not_verified"} <= categories
    assert "documented_fact" not in categories


def test_distinct_answers_count_what_the_rule_read(study_dir, definition, plan):
    # "265" and "265 minutes" are one answer; a format failure is no answer at all
    # (under pandas 3 it used to be counted as a distinct NaN answer).
    runs = [r for r in plan.runs if r.task_id == "T1" and r.condition_id == "A"]
    texts = ["ANSWER: 265", "ANSWER: 265 minutes", "no answer line"]
    write_records(
        study_dir,
        [
            make_record(definition, plan, r.run_id, response_text=t)
            for r, t in zip(runs, texts, strict=True)
        ],
    )
    a = analyze(Study.load(study_dir))
    row = a.tasks.set_index(["condition_id", "task_id"]).loc[("A", "T1")]
    assert (row.n_scored, row.n_format_failures) == (3, 1)
    assert row.distinct_answers == 1 and row.majority_answer_share == 1.0


def test_flagged_confounders_and_the_sensitivity_check(study_dir, definition, plan):
    # Memory on in every run (constant, still flagged), and a usage-limit notice in every
    # run of C, whose responses then look like the default selection's.
    records = []
    for run in plan.runs:
        task = definition.task(run.task_id)
        notice = run.condition_id == "C" and run.repetition == 1
        right = run.condition_id in {"B", "C"} and not notice
        records.append(
            make_record(
                definition,
                plan,
                run.run_id,
                response_text=f"ANSWER: {task.reference_answer if right else '0'}",
                confounders={
                    "memory": "enabled",
                    "custom_instructions": "inactive",
                    "tools_used": "none",
                    "switch_notice": "limit_notice" if notice else "none",
                    "network": "wifi",
                },
            )
        )
    write_records(study_dir, records)
    a = analyze(Study.load(study_dir))
    texts = [lim.text for lim in a.limitations]
    assert any(
        "Confounder 'memory' had a value other than the recommended (disabled) in 72 of 72" in t
        for t in texts
    )
    assert any("'switch_notice'" in t and "C 6/18" in t and "A 0/18" in t for t in texts)
    check = a.sensitivity
    assert (
        check is not None
        and check.left_out == 6
        and check.reasons == {"switch_notice=limit_notice": 6}
    )
    changes = {(c, m): (b, x) for c, m, b, x in check.changes}
    assert changes[("C", "accuracy")][1] == "consistent_with_per_message"
    assert changes[("C", "accuracy")][0] != "consistent_with_per_message"


def test_order_adherence_is_reported(study_dir, definition, plan):
    runs = sorted(plan.runs, key=lambda r: r.sequence)[:6]
    stamps = [FIXED_NOW + timedelta(minutes=m) for m in (6, 5, 4, 3, 2, 1)]  # reversed
    write_records(
        study_dir,
        [
            make_record(definition, plan, r.run_id, timestamp=t)
            for r, t in zip(runs, stamps, strict=True)
        ],
    )
    a = analyze(Study.load(study_dir))
    assert a.order is not None and a.order.share == 0 and a.order.pairs == 15
    assert any("planned random order for 0% of run pairs" in lim.text for lim in a.limitations)
    text = render_report(a, Study.load(study_dir), now=FIXED_NOW)
    assert "The runs followed that order for 0% of run pairs." in text


def test_uneven_incompletion_is_standardized_and_reported(study_dir, definition, plan):
    # B and C answer every task the same way (right on T1-T4, wrong on T5-T6), but C
    # returns errors on T4-T6 in two of three repetitions. Pooled over scored responses C
    # would look 17 points better; standardized by task it is not, and its success rate
    # (errors count as failures) is lower.
    records = []
    for run in plan.runs:
        task = definition.task(run.task_id)
        right = run.condition_id in {"B", "C"} and run.task_id in {"T1", "T2", "T3", "T4"}
        error = run.condition_id == "C" and run.task_id in {"T4", "T5", "T6"} and run.repetition > 1
        records.append(
            make_record(
                definition,
                plan,
                run.run_id,
                response_text="" if error else f"ANSWER: {task.reference_answer if right else '0'}",
                response_status="error" if error else "completed",
            )
        )
    write_records(study_dir, records)
    a = analyze(Study.load(study_dir))
    acc = a.comparisons[(a.comparisons["metric"] == "accuracy")].set_index(
        ["condition_a", "condition_b"]
    )
    row = acc.loc[("B", "C")]
    assert row.difference == pytest.approx(0) and row.low < 0 < row.high
    success = a.comparisons[(a.comparisons["metric"] == "success_rate")].set_index(
        ["condition_a", "condition_b"]
    )
    assert success.loc[("B", "C")].difference == pytest.approx(1 / 9)  # T4: 3/3 vs 1/3
    (lim,) = [lim for lim in a.limitations if "errors, refusals or incomplete" in lim.text]
    assert (
        "Conditions differ in how often this happened (A 0/18, B 0/18, C 6/18, D 0/18)" in lim.text
    )
