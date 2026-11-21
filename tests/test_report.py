from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.demo import create_demo_study
from mblab.formatting import md_text
from mblab.provenance import CodeVersion
from mblab.report import compare_reports, md_table, render_report
from mblab.study import Study

from .conftest import FIXED_NOW, make_record, write_records
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
    assert "A descriptive tally, not a test of H1; the metrics are related" in text
    assert not BANNED.search(text)


def test_md_table_escapes_cells():
    table = md_table(["a", "b"], [["x|y", "line\nbreak"], [None, float("nan")]])
    assert "x\\|y" in table and "line break" in table
    assert table.splitlines()[-1] == "| — | — |"


@pytest.mark.parametrize(
    "text",
    [
        "![pixel](https://example.invalid/p.png)",
        "[open](https://example.invalid) and <https://example.invalid>",
        "# Heading\n\n1. one\n2. two\n- item",
        "*emphasis* __strong__ `code` ~~gone~~ $x^2$ | a | b |",
        '<img src="https://example.invalid/p.png"> &amp; \\[not a link\\](x)',
        "[ref]\n\n[ref]: https://example.invalid",
    ],
)
def test_md_text_renders_as_the_literal_text(text):
    markdown_it = pytest.importorskip("markdown_it")
    rendered = markdown_it.MarkdownIt("commonmark").render(md_text(text)).strip()
    literal = html.escape(" ".join(text.split()), quote=False).replace('"', "&quot;")
    assert rendered == f"<p>{literal}</p>"


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


def test_records_that_contradict_the_plan_are_left_out_and_flagged(study_dir, definition, plan):
    good = make_record(definition, plan, "A:T1:r1")
    twice = make_record(definition, plan, "B:T1:r1")
    relabelled = make_record(definition, plan, "C:T1:r1").model_copy(update={"condition_id": "A"})
    write_records(study_dir, [good, twice, twice, relabelled])
    study = Study.load(study_dir)
    a = analyze(study)
    assert list(a.results["run_id"]) == ["A:T1:r1"]  # neither copy of B, nor the relabelled C
    assert a.left_out == {
        "several records for one run": ["B:T1:r1", "B:T1:r1"],
        "identity differs from the plan": ["C:T1:r1"],
    }
    text = render_report(a, study, now=FIXED_NOW)
    header = text.split("\n## ")[0]
    assert "VALIDATION ERRORS" in header
    assert "3 record(s) are left out of every number" in text
    assert not BANNED.search(text)


# -- reproduction: which code, which data, and checking a report ----------------------------

SHA = "0123456789abcdef0123456789abcdef01234567"
RELEASED = CodeVersion("0.2.0", SHA, False)


def _report(root: Path, code: CodeVersion = RELEASED, now: datetime = NOW) -> str:
    study = Study.load(root)
    return render_report(analyze(study), study, now=now, code=code)


def test_reproduction_names_the_code_and_pins_the_install(study_dir):
    text = _report(study_dir)
    assert (
        f"| Generated | {NOW.isoformat()} by model-behavior-lab 0.2.0 (commit 0123456789ab) |"
        in text
    )
    assert f"| Report made with | model-behavior-lab 0.2.0, commit `{SHA}` |" in text
    assert f'pip install "git+https://github.com/cynkai/model-behavior-lab@{SHA}"' in text
    assert "mblab report -s <study-dir> --check <this-report.md>" in text
    assert "| Plan made with | model-behavior-lab " in text
    assert "api-unrecorded.jsonl" not in text  # a manual UI study has no such file

    unknown = _report(study_dir, CodeVersion("0.2.0"))
    assert 'pip install "git+https://github.com/cynkai/model-behavior-lab@v0.2.0"' in unknown
    assert "| Report made with | model-behavior-lab 0.2.0, commit not recorded |" in unknown
    assert "The commit of the code that made this report is not recorded" in unknown

    dirty = _report(study_dir, CodeVersion("0.2.0", SHA, True))
    assert "with uncommitted changes in `src/mblab`" in dirty
    assert "its exact code cannot be installed" in dirty
    assert not BANNED.search(unknown + dirty)


def test_check_ignores_only_the_generation_time_and_the_code(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:40]])
    original = _report(study_dir)
    later = _report(study_dir, CodeVersion("0.2.1"), now=datetime(2027, 1, 1, tzinfo=UTC))
    same = compare_reports(original, later)
    assert same.same and same.diff == [] and same.made_with == "0.2.0 (commit 0123456789ab)"

    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:41]])
    changed = compare_reports(original, _report(study_dir))
    assert not changed.same and changed.changed_lines > 2
    assert any(line.startswith("-| records.jsonl SHA-256") for line in changed.diff)
    assert any(line.startswith("+| Data | 41 recorded") for line in changed.diff)
    assert not changed.settings_differ


def _cli(*args: object) -> object:
    return CliRunner().invoke(app, [str(a) for a in args], env={"COLUMNS": "200", "NO_COLOR": "1"})


def test_report_never_replaces_a_different_report_without_force(study_dir, definition, plan):
    target = study_dir / "reports" / "report.md"
    assert _cli("report", "-s", study_dir).exit_code == 0
    first = target.read_bytes()
    again = _cli("report", "-s", study_dir)  # same content: the file is left as it was
    assert again.exit_code == 0 and "Left unchanged" in again.output
    assert target.read_bytes() == first

    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:5]])
    refused = _cli("report", "-s", study_dir)
    assert refused.exit_code == 2
    assert "already exists and differs" in refused.output and "--check" in refused.output
    assert target.read_bytes() == first

    elsewhere = _cli("report", "-s", study_dir, "--out", study_dir / "reports" / "new.md")
    assert elsewhere.exit_code == 0 and target.read_bytes() == first
    replaced = _cli("report", "-s", study_dir, "--force")
    assert replaced.exit_code == 0 and target.read_bytes() != first
    assert target.read_text() == (study_dir / "reports" / "new.md").read_text().replace(
        _generated(study_dir / "reports" / "new.md"), _generated(target)
    )


def _generated(path: Path) -> str:
    return next(line for line in path.read_text().splitlines() if line.startswith("| Generated |"))


def test_report_check_compares_and_writes_nothing(study_dir, definition, plan):
    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:20]])
    submitted = study_dir / "submitted.md"
    submitted.write_text(_report(study_dir, CodeVersion("0.1.9", "f" * 40)), encoding="utf-8")
    before = sorted(p.name for p in study_dir.rglob("*"))

    match = _cli("report", "-s", study_dir, "--check", submitted)
    assert match.exit_code == 0, match.output
    assert "matches the report made now" in match.output
    assert "made with model-behavior-lab 0.1.9 (commit ffffffffffff)" in match.output
    assert sorted(p.name for p in study_dir.rglob("*")) == before  # nothing written

    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:21]])
    differ = _cli("report", "-s", study_dir, "--check", submitted)
    assert differ.exit_code == 1
    assert "differs from the report made now" in differ.output
    assert "+| Data | 21 recorded" in differ.output
    assert "Install the code named in its Reproduction section" in differ.output

    explored = _cli("report", "-s", study_dir, "--confidence", "0.9", "--out", submitted, "--force")
    assert explored.exit_code == 0
    plain = _cli("report", "-s", study_dir, "--check", submitted)
    assert plain.exit_code == 1 and "repeat the options" in plain.output
    assert (
        _cli("report", "-s", study_dir, "--check", submitted, "--confidence", "0.9").exit_code == 0
    )

    assert _cli("report", "-s", study_dir, "--check", study_dir / "missing.md").exit_code == 2
    combined = _cli("report", "-s", study_dir, "--check", submitted, "--force")
    assert combined.exit_code == 2 and "cannot be combined" in combined.output
