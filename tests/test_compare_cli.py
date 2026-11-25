from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mblab.cli import app
from mblab.plan import generate_plan
from mblab.study import Study, parse_definition, save_plan, template_text

from .conftest import FIXED_NOW, make_record, write_records


def _study(root: Path, platform: str) -> Study:
    root.mkdir()
    text = template_text().replace("platform: ios", f"platform: {platform}", 1)
    (root / "experiment.yaml").write_text(text, encoding="utf-8")
    save_plan(generate_plan(parse_definition(text), now=FIXED_NOW), root / "plan.json")
    return Study.load(root)


def test_compare_cli_uses_separate_denominators_and_leaves_inputs_intact(tmp_path: Path) -> None:
    left = _study(tmp_path / "first", "ios")
    right = _study(tmp_path / "second", "web")
    write_records(
        left.root,
        [
            make_record(left.definition, left.require_plan(), "A:T1:r1"),
            make_record(
                left.definition, left.require_plan(), "A:T1:r2", response_text="No answer given"
            ),
        ],
    )
    write_records(
        right.root,
        [
            make_record(right.definition, right.require_plan(), "A:T1:r1"),
            make_record(
                right.definition,
                right.require_plan(),
                "A:T2:r1",
                response_status="refused",
                response_text="I refuse",
            ),
        ],
    )

    def snapshot():
        return {
            file: (file.read_bytes(), file.stat().st_mtime_ns)
            for root in (left.root, right.root)
            for file in root.iterdir()
        }

    before = snapshot()
    runner = CliRunner()
    command = ["compare", str(left.root), str(right.root)]
    result = runner.invoke(app, [*command, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["left"]["platform"] == "ios" and payload["right"]["platform"] == "web"
    condition = next(row for row in payload["conditions"] if row["condition_id"] == "A")
    assert (condition["left"]["accuracy"], condition["left"]["success_rate"]) == (0.5, 0.5)
    assert (condition["right"]["accuracy"], condition["right"]["success_rate"]) == (1.0, 0.5)
    assert condition["left"]["task_scored_runs"]["T2"] == 0
    assert "No answer given" not in result.output and "I refuse" not in result.output

    table = runner.invoke(app, command)
    assert table.exit_code == 0, table.output
    assert "1/2 (50%)" in table.output and "1/1 (100%)" in table.output
    assert snapshot() == before


def test_compare_load_error_names_the_side(study_dir: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["compare", str(study_dir), str(tmp_path / "missing"), "--json"]
    )
    assert result.exit_code == 2
    assert "Right study could not be loaded" in result.output


def test_compare_rejects_same_directory_via_symlink(study_dir: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(study_dir, target_is_directory=True)
    result = CliRunner().invoke(app, ["compare", str(study_dir), str(alias)])
    assert result.exit_code == 2
    assert "two different study directories" in result.output
