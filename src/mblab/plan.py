"""Run-plan generation: every condition x task x repetition, in a seeded random order."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from mblab import __version__
from mblab.schema import ExperimentDefinition, PlannedRun, RunPlan


def make_run_id(condition_id: str, task_id: str, repetition: int) -> str:
    # ':' cannot appear in identifiers, so run ids are unambiguous.
    return f"{condition_id}:{task_id}:r{repetition}"


def seeded_shuffle(items: list[Any], rng: random.Random) -> None:
    """Fisher-Yates shuffle driven only by ``rng.random()``.

    ``random.shuffle`` is not guaranteed to give the same order across Python
    versions, but ``Random(seed).random()`` is, so plans stay reproducible.
    """
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]


def generate_plan(
    definition: ExperimentDefinition,
    *,
    seed: int | None = None,
    now: datetime | None = None,
) -> RunPlan:
    seed = definition.randomization.seed if seed is None else seed
    strategy = definition.randomization.strategy
    rng = random.Random(seed)
    cells = [(c.id, t.id) for c in definition.conditions for t in definition.tasks]

    ordered: list[tuple[str, str, int, int]] = []
    if strategy == "blocked_by_repetition":
        for rep in range(1, definition.repetitions + 1):
            block = [(c, t, rep, rep) for c, t in cells]
            seeded_shuffle(block, rng)
            ordered.extend(block)
    else:
        everything = [
            (c, t, rep, 1) for rep in range(1, definition.repetitions + 1) for c, t in cells
        ]
        seeded_shuffle(everything, rng)
        ordered = everything

    runs = []
    for sequence, (cond_id, task_id, rep, block) in enumerate(ordered, start=1):
        cond = definition.condition(cond_id)
        runs.append(
            PlannedRun(
                run_id=make_run_id(cond_id, task_id, rep),
                sequence=sequence,
                block=block,
                experiment_id=definition.experiment_id,
                experiment_version=definition.experiment_version,
                platform=definition.platform,
                execution_method=definition.execution_method,
                condition_id=cond_id,
                model_selection=cond.model_selection,
                message_override=cond.message_override,
                task_id=task_id,
                repetition=rep,
            )
        )

    return RunPlan(
        experiment_id=definition.experiment_id,
        experiment_version=definition.experiment_version,
        definition_sha256=definition.sha256(),
        definition_snapshot=definition.model_dump(mode="json"),
        seed=seed,
        strategy=strategy,
        generated_at=now or datetime.now().astimezone(),
        tool_version=__version__,
        runs=runs,
    )


def diff_snapshots(old: Any, new: Any, path: str = "") -> list[str]:
    """List dotted paths whose values differ between two JSON-like structures."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in old:
                changes.append(f"{sub} (added)")
            elif key not in new:
                changes.append(f"{sub} (removed)")
            else:
                changes.extend(diff_snapshots(old[key], new[key], sub))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        changes = []
        # Lists of objects with ids (conditions, tasks, ...) are compared by id.
        if all(isinstance(x, dict) and "id" in x for x in old + new):
            old_by = {x["id"]: x for x in old}
            new_by = {x["id"]: x for x in new}
            for key in list(old_by) + [k for k in new_by if k not in old_by]:
                sub = f"{path}[{key}]"
                if key not in new_by:
                    changes.append(f"{sub} (removed)")
                elif key not in old_by:
                    changes.append(f"{sub} (added)")
                else:
                    changes.extend(diff_snapshots(old_by[key], new_by[key], sub))
            return changes
        if old != new:
            return [path]
        return []
    return [] if old == new else [path]
