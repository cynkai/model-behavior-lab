"""Per-message vs. default verdicts: equivalence margins, position, tallies, settings."""

from __future__ import annotations

import random
from collections import Counter

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from mblab.analysis import (
    AnalysisSettings,
    HypothesisTally,
    analyze,
    hypothesis_tallies,
    override_verdict,
)
from mblab.cli import app
from mblab.schema import AnalysisPlan, ExperimentDefinition
from mblab.stats import Interval, bootstrap_position, newcombe_interval
from mblab.study import Study, parse_definition, template_text

from .conftest import write_records
from .test_analysis import _populate

REFS = Interval(0.5, 0.3, 0.7)  # per-message reference minus default-only reference


@pytest.mark.parametrize(
    ("vs_d", "vs_o", "verdict"),
    [
        # Within ±0.25 (half the 0.5 gap) of the per-message reference.
        (Interval(0.5, 0.35, 0.65), Interval(0.0, -0.15, 0.15), "consistent_with_per_message"),
        (Interval(0.0, -0.2, 0.2), Interval(-0.5, -0.7, -0.3), "consistent_with_default"),
        # Differs from both, on opposite sides: between the references.
        (Interval(0.25, 0.1, 0.4), Interval(-0.25, -0.4, -0.1), "between_references"),
        # Differs from both on the same side: beyond them.
        (Interval(0.8, 0.6, 1.0), Interval(0.3, 0.1, 0.5), "differs_from_both"),
        # The bug the review found: different from the default, merely "not distinguishable"
        # from the per-message reference. The old rule called this consistent with it.
        (Interval(0.25, 0.05, 0.45), Interval(-0.25, -0.5, 0.01), "inconclusive"),
    ],
)
def test_consistent_needs_the_interval_inside_the_margin(vs_d, vs_o, verdict):
    assert override_verdict(REFS, vs_d, vs_o, 0.5) == verdict


def test_references_that_do_not_differ_decide_nothing():
    same = Interval(0.05, -0.1, 0.2)
    assert override_verdict(same, same, same, 0.5) == "references_not_distinguishable"
    assert override_verdict(None, same, same, 0.5) == "references_not_distinguishable"


def test_a_condition_halfway_is_rarely_called_consistent():
    # The review's simulation: 18 runs per condition, references at 50% and 100%
    # accuracy, the condition halfway (75%). The old rule called 74% of these consistent
    # with one selection; with a margin of half the gap it is about the 2.5% that a 95%
    # interval allows at the margin.
    rng = random.Random(20261120)
    n, verdicts = 18, Counter()
    for _ in range(600):
        kc, kd, ko = (sum(rng.random() < p for _ in range(n)) for p in (0.75, 0.5, 1.0))
        refs = newcombe_interval(ko, n, kd, n)
        vs_d = newcombe_interval(kc, n, kd, n)
        vs_o = newcombe_interval(kc, n, ko, n)
        verdicts[override_verdict(refs, vs_d, vs_o, 0.5)] += 1
    consistent = verdicts["consistent_with_per_message"] + verdicts["consistent_with_default"]
    assert consistent / 600 < 0.06, verdicts


def test_position_places_a_condition_between_the_references():
    d, o = [0.0] * 10 + [1.0] * 10, [1.0] * 20  # 50% and 100%
    at_o = bootstrap_position([1.0] * 20, d, o, resamples=400)
    assert at_o is not None and at_o.estimate == pytest.approx(1.0)
    halfway = bootstrap_position([0.0] * 5 + [1.0] * 15, d, o, resamples=400)
    assert halfway is not None and halfway.estimate == pytest.approx(0.5)
    assert halfway.low < 0.5 < halfway.high
    medians = bootstrap_position([30.0] * 9, [10.0] * 9, [50.0] * 9, statistic="median")
    assert medians is not None and medians.estimate == pytest.approx(0.5)
    assert bootstrap_position([1.0], [1.0], [1.0]) is None  # the references coincide


def test_mixed_condition_is_between_not_consistent(study_dir, definition, plan):
    # Every other repetition of C follows the per-message selection: a mixture.
    study = _populate(study_dir, definition, plan, {"A": False, "B": True, "C": True, "D": False})
    records = []
    for rec in study.records:
        if rec.condition_id == "C" and rec.task_id in {"T1", "T2", "T3"}:
            rec = rec.model_copy(update={"response_text": "ANSWER: 0"})
        records.append(rec)
    write_records(study_dir, records)
    a = analyze(Study.load(study_dir))
    (c,) = [o for o in a.overrides if o.condition_id == "C"]
    accuracy = next(m for m in c.metrics if m.metric == "accuracy")
    assert accuracy.verdict == "between_references"
    assert accuracy.position is not None and accuracy.position.estimate == pytest.approx(0.5)
    assert "lies between them, at 0.50" in accuracy.explanation


def test_tallies_count_only_assessments_that_place_a_condition():
    tally = HypothesisTally(
        "H1",
        ["C"],
        "per_message",
        1,
        2,
        {"consistent_with_per_message": 1, "between_references": 1, "inconclusive": 3},
    )
    text = tally.sentence()
    assert "1 of 2 assessments that place a condition show it" in text
    assert "not independent" in text
    no_diff = HypothesisTally("H2", ["F"], "no_difference", None, 0, {"no_difference_detected": 2})
    assert "No assessment can show it: not detecting a difference is not equivalence" in (
        no_diff.sentence()
    )


def test_inconclusive_verdicts_are_not_in_the_denominator(study_dir, definition, plan):
    study = _populate(study_dir, definition, plan, {"A": False, "B": True, "C": True, "D": False})
    a = analyze(study)
    (h1,) = [t for t in hypothesis_tallies(a) if t.hypothesis_id == "H1"]
    decidable = {
        "consistent_with_per_message",
        "consistent_with_default",
        "between_references",
        "differs_from_both",
    }
    assert h1.usable == sum(n for v, n in h1.counts.items() if v in decidable)
    assert h1.usable < sum(h1.counts.values())  # e.g. references that do not differ


def test_analysis_settings_are_pre_registered():
    definition = parse_definition(template_text())
    assert definition.analysis == AnalysisPlan(confidence=0.95, min_n=5, equivalence_fraction=0.5)
    settings = AnalysisSettings.preregistered(definition, None)
    assert (settings.confidence, settings.min_n, settings.equivalence_fraction) == (0.95, 5, 0.5)
    assert settings.departures(definition) == []
    looser = AnalysisSettings(confidence=0.9, min_n=5, equivalence_fraction=0.5)
    assert looser.departures(definition) == ["confidence 0.9 instead of 0.95"]
    data = yaml.safe_load(template_text())
    for key, bad in (("min_n", 0), ("confidence", 0.5), ("equivalence_fraction", 0.6)):
        broken = {**data, "analysis": {**data["analysis"], key: bad}}
        with pytest.raises(ValidationError):
            ExperimentDefinition.model_validate(broken)


def test_overriding_a_setting_labels_the_analysis_exploratory(study_dir, definition, plan):
    _populate(study_dir, definition, plan, {"A": False, "B": True, "C": True, "D": False})
    runner = CliRunner()
    env = {"COLUMNS": "200", "NO_COLOR": "1"}
    refused = runner.invoke(app, ["analyze", "-s", str(study_dir), "--min-n", "0"], env=env)
    assert refused.exit_code == 2  # below the floor of 5
    result = runner.invoke(app, ["report", "-s", str(study_dir), "--confidence", "0.9"], env=env)
    assert result.exit_code == 0, result.output
    assert "not the pre-registered analysis (confidence 0.9 instead of 0.95)" in result.output
    text = (study_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "**NOT THE PRE-REGISTERED ANALYSIS:** confidence 0.9 instead of 0.95" in text
    assert "so this is an exploratory analysis, not the pre-registered one" in text
    # The reproduction commands repeat the option, so a reviewer makes the same report.
    assert "mblab report -s <study-dir> --confidence 0.9 --check <this-report.md>" in text
    plain = runner.invoke(app, ["report", "-s", str(study_dir), "--force"], env=env)
    assert plain.exit_code == 0 and "pre-registered analysis" not in plain.output
    assert "NOT THE PRE-REGISTERED" not in (study_dir / "reports" / "report.md").read_text()
