from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab.analysis import analyze
from mblab.cli import app
from mblab.report import render_report
from mblab.schema import ManualGrade
from mblab.study import Study, write_jsonl

from .conftest import FIXED_NOW, write_records


def _grades(study: Study, ratings: list[list[float]]) -> None:
    write_jsonl(
        study.grades_path,
        [
            ManualGrade(
                run_id=record.run_id,
                points=points,
                max_points=2,
                grader_id=f"private-grader-{grader}",
                blind=True,
                graded_at=FIXED_NOW,
                notes="private-grade-note",
            )
            for record, scores in zip(study.records, ratings, strict=False)
            for grader, points in enumerate(scores)
        ],
    )


def test_report_shows_pair_weighted_agreement_and_coverage(manual_study: Path) -> None:
    # Given: three graders on one response, two on another, one on a third.
    study = Study.load(manual_study)
    _grades(study, [[2, 2, 1], [0, 1], [2]])
    study = Study.load(manual_study)

    # When: the real report is rendered from the recorded grades.
    text = render_report(analyze(study), study, now=FIXED_NOW)

    # Then: within-response pairs determine both denominators, without identities.
    section = text.split("### 6.10 Manual inter-rater agreement\n", 1)[1].split("## 7.", 1)[0]
    assert "| T7 | 6 | 3 | 2 | 4 | 1/4 (25%) | 2/4 (50%) |" in section
    assert "private-grader" not in section
    assert "private-grade-note" not in section
    assert "grading_id" not in section


@pytest.mark.parametrize("ratings", [[], [[2]]])
def test_report_uses_missing_agreement_when_no_response_has_two_graders(
    manual_study: Path, ratings: list[list[float]]
) -> None:
    # Given: a manual study with either no ratings or one rating.
    study = Study.load(manual_study)
    _grades(study, ratings)
    study = Study.load(manual_study)

    # When: the report is rendered.
    text = render_report(analyze(study), study, now=FIXED_NOW)

    # Then: missing comparisons cannot be mistaken for zero agreement.
    assert f"| T7 | 6 | {len(ratings)} | 0 | 0 | — | — |" in text


def test_report_shows_zero_coverage_when_manual_study_has_no_records(manual_study: Path) -> None:
    # Given: manual tasks are defined but no responses have been recorded.
    write_records(manual_study, [])
    study = Study.load(manual_study)

    # When: the report is rendered.
    text = render_report(analyze(study), study, now=FIXED_NOW)

    # Then: the coverage row reports zero eligible responses without dividing by zero.
    assert "| T7 | 0 | 0 | 0 | 0 | — | — |" in text


def test_report_omits_manual_agreement_for_automatic_study(study_dir: Path) -> None:
    # Given: a study whose tasks are all scored automatically.
    study = Study.load(study_dir)

    # When: the report is rendered.
    text = render_report(analyze(study), study, now=FIXED_NOW)

    # Then: automatic studies do not gain a manual agreement section.
    assert "### 6.10" not in text


def test_report_command_preserves_archived_report_when_agreement_changes(
    manual_study: Path,
) -> None:
    # Given: an archived report created before a second grader's rating.
    study = Study.load(manual_study)
    _grades(study, [[2]])
    runner = CliRunner()
    initial = runner.invoke(app, ["report", "-s", str(manual_study)])
    assert initial.exit_code == 0, initial.output
    report = manual_study / "reports" / "report.md"
    archived = report.read_bytes()
    _grades(study, [[2, 1]])

    # When: a report is requested at the existing path without force.
    result = runner.invoke(app, ["report", "-s", str(manual_study)])

    # Then: refusal preserves the archived bytes.
    assert result.exit_code == 2, result.output
    assert report.read_bytes() == archived
