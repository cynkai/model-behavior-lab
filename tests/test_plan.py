from __future__ import annotations

from collections import Counter

import yaml

from mblab.plan import diff_snapshots, generate_plan, make_run_id
from mblab.schema import ExperimentDefinition
from mblab.study import template_text

from .conftest import FIXED_NOW


def test_plan_covers_every_cell_once(definition, plan):
    assert len(plan.runs) == 72
    assert len({r.run_id for r in plan.runs}) == 72
    assert [r.sequence for r in plan.runs] == list(range(1, 73))
    cells = Counter((r.condition_id, r.task_id) for r in plan.runs)
    assert set(cells.values()) == {3}
    assert plan.definition_sha256 == definition.sha256()
    assert plan.definition_snapshot == definition.model_dump(mode="json")


def test_plan_is_deterministic_for_a_seed(definition):
    a = generate_plan(definition, now=FIXED_NOW)
    b = generate_plan(definition, now=FIXED_NOW)
    assert [r.run_id for r in a.runs] == [r.run_id for r in b.runs]
    c = generate_plan(definition, seed=7, now=FIXED_NOW)
    assert [r.run_id for r in c.runs] != [r.run_id for r in a.runs]


def test_plan_order_is_pinned_for_the_template(definition, plan):
    # Guards cross-version reproducibility: the shuffle only uses Random.random().
    assert [r.run_id for r in plan.runs[:4]] == ["B:T1:r1", "B:T5:r1", "A:T2:r1", "B:T3:r1"]


def test_blocked_randomization_keeps_each_block_complete(plan):
    for block in (1, 2, 3):
        runs = [r for r in plan.runs if r.block == block]
        assert len(runs) == 24
        assert {r.repetition for r in runs} == {block}
        assert len({(r.condition_id, r.task_id) for r in runs}) == 24
    # Within a block the order is shuffled, not definition order.
    first_block = [(r.condition_id, r.task_id) for r in plan.runs[:24]]
    assert first_block != sorted(first_block)


def test_complete_randomization(definition):
    data = definition.model_dump(mode="json")
    data["randomization"]["strategy"] = "complete"
    complete = generate_plan(ExperimentDefinition.model_validate(data), now=FIXED_NOW)
    assert {r.block for r in complete.runs} == {1}
    reps_in_first_24 = {r.repetition for r in complete.runs[:24]}
    assert len(reps_in_first_24) > 1


def test_plan_copies_condition_settings(definition, plan):
    run = plan.by_run_id()["C:T4:r2"]
    assert (run.model_selection, run.message_override) == ("instant", "pro")
    assert run.platform == definition.platform
    assert make_run_id("C", "T4", 2) == "C:T4:r2"


def test_diff_snapshots_reports_changed_paths(definition):
    old = definition.model_dump(mode="json")
    data = yaml.safe_load(template_text())
    data["tasks"][4]["scoring"]["expected"] = 634
    data["tasks"][4]["reference_answer"] = "634"
    data["conditions"].pop()
    new = ExperimentDefinition.model_validate(data).model_dump(mode="json")
    changes = diff_snapshots(old, new)
    assert "tasks[T5].scoring.expected" in changes
    assert "tasks[T5].reference_answer" in changes
    assert "conditions[D] (removed)" in changes
    assert diff_snapshots(old, old) == []
