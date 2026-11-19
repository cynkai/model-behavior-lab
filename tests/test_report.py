from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.demo import create_demo_study
from mblab.report import md_table, render_report
from mblab.study import Study

from .conftest import make_record, write_records
from .test_analysis import BANNED

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [
    "## 1. Research question",
    "## 2. Hypotheses",
    "## 3. Experimental environment",
    "## 4. Experimental conditions",
    "## 5. Evaluation method",
    "## 6. Results",
    "## 7. Interpretation",
    "## 8. Limitations",
    "## 9. Claims ledger",
    "## 10. Reproduction",
]


def _render(root: Path) -> str:
    study = Study.load(root)
    return render_report(analyze(study), study, now=NOW)


def test_report_without_data_has_design_but_no_results(study_dir):
    text = _render(study_dir)
    for section in SECTIONS:
        assert section in text
    assert "No observations have been recorded yet" in text
    assert (
        "### 6.2" not in text and "Accuracy" not in text.split("## 6. Results")[1].split("## 7")[0]
    )
    assert "Nothing to interpret until observations are recorded" in text
    assert "_None recorded. Nothing in this report should be read as a documented fact" in text
    assert "SYNTHETIC" not in text


def test_report_with_data(study_dir, definition, plan):
    write_records(
        study_dir,
        [make_record(definition, plan, r.run_id) for r in plan.runs[:30]]
        + [make_record(definition, plan, plan.runs[30].run_id, protocol_deviation="wrong chat")],
    )
    text = _render(study_dir)
    for section in SECTIONS:
        assert section in text
    assert "### 6.2 Condition comparison" in text
    assert "### 6.5 Per-message selection vs. default" in text
    assert "## Appendix A. Excluded runs" in text and "wrong chat" in text
    assert "41 of 72 planned runs have no record" in text  # 31 recorded
    assert "records.jsonl SHA-256" in text
    assert not BANNED.search(text)


def test_demo_report_is_labelled_and_careful(tmp_path):
    create_demo_study(tmp_path / "demo")
    text = _render(tmp_path / "demo")
    assert text.startswith("# [SYNTHETIC DEMO]")
    assert text.count("SYNTHETIC DEMO DATA") >= 2  # top and bottom
    assert "only demonstrate the report format" in text
    assert "A descriptive tally, not a test of H1." in text
    assert not BANNED.search(text)


def test_md_table_escapes_cells():
    table = md_table(["a", "b"], [["x|y", "line\nbreak"], [None, float("nan")]])
    assert "x\\|y" in table and "line break" in table
    assert table.splitlines()[-1] == "| — | — |"


def test_report_command_writes_file(study_dir):
    result = CliRunner().invoke(app, ["report", "-s", str(study_dir)])
    assert result.exit_code == 0, result.output
    assert (study_dir / "reports" / "report.md").read_text().startswith("# ChatGPT Model")
    assert "contains the design but no results" in result.output


@pytest.mark.parametrize("readme", ["README.md", "docs/README.ko.md"])
def test_readme_mentions_only_implemented_commands(readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    commands = set(re.findall(r"\bmblab ([a-z]+(?:-[a-z]+)*(?: (?:export|import))?)\b", text))
    commands.discard("init DIR")
    assert {"init", "plan", "validate", "analyze", "report", "dashboard", "demo"} <= commands
    runner = CliRunner()
    for command in sorted(commands):
        result = runner.invoke(app, [*command.split(), "--help"])
        assert result.exit_code == 0, f"{readme}: `mblab {command}` is not a command"
