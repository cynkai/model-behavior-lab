from __future__ import annotations

import json
import re

import pytest
import yaml

from mblab.analysis import VERDICT_TEXT, AnalysisSettings, analyze, classify_label
from mblab.plan import generate_plan
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
