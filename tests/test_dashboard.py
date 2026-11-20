"""Smoke tests for the Streamlit dashboard via streamlit.testing (no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mblab.dashboard import PAGES
from mblab.demo import create_demo_study
from mblab.study import Study

APP = str(Path(__file__).resolve().parents[1] / "src" / "mblab" / "dashboard" / "app.py")


def _app(monkeypatch, root) -> AppTest:
    monkeypatch.setenv("MBLAB_STUDY", str(root))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _charts(at: AppTest) -> int:
    return len(at.get("vega_lite_chart"))


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, at.exception
    return at


def test_every_page_renders_demo_data_with_synthetic_banner(monkeypatch, tmp_path):
    create_demo_study(tmp_path / "demo")
    at = _app(monkeypatch, tmp_path / "demo")
    for page in PAGES:
        _goto(at, page)
        assert any("SYNTHETIC DEMO DATA" in e.value for e in at.error), page
    _goto(at, "Condition comparison")
    assert _charts(at) == 3
    _goto(at, "Task-level results")
    assert _charts(at) == 2


def test_no_charts_without_observations(monkeypatch, study_dir):
    at = _app(monkeypatch, study_dir)
    assert not [e for e in at.error if "SYNTHETIC" in e.value]
    for page in PAGES:
        _goto(at, page)
        assert _charts(at) == 0, page
    _goto(at, "Condition comparison")
    assert any("No scored responses yet" in i.value for i in at.info)


def test_recording_form_saves_a_record(monkeypatch, study_dir):
    at = _goto(_app(monkeypatch, study_dir), "Run plan & recording")
    assert at.code[0].value.startswith("A train departs at 09:47")  # B:T1:r1 is first
    by_label = {w.label: w for w in at.text_input}
    by_label["Client (app) version"].input("1.2026.300")
    by_label["Stopwatch seconds (blank = not measured)"].input("12.5")
    by_label["Time sent (ISO 8601; blank = now minus stopwatch seconds)"].input(
        "2026-11-20T10:00:00+09:00"
    )
    at.text_area[0].input("Work.\nANSWER: 265")
    at.button[0].click().run()
    assert not at.exception, at.exception
    (record,) = Study.load(study_dir).records
    assert record.run_id == "B:T1:r1"
    assert record.latency_seconds == 12.5 and record.latency_method == "manual_stopwatch"
    assert record.response_text == "Work.\nANSWER: 265"
    assert any("Saved B:T1:r1" in s.value for s in at.success)


def test_recording_is_disabled_for_demo_studies(monkeypatch, tmp_path):
    create_demo_study(tmp_path / "demo")
    at = _goto(_app(monkeypatch, tmp_path / "demo"), "Run plan & recording")
    assert any("recording is disabled" in i.value for i in at.info)
    assert len(at.text_area) == 0


def test_every_page_renders_an_api_study(monkeypatch, tmp_path):
    from mblab import api_runner
    from mblab.plan import generate_plan
    from mblab.study import parse_definition, save_plan, template_text

    from .test_api_runner import FIXED_NOW, FakeClient, _fake_module

    root = tmp_path / "api"
    root.mkdir()
    text = template_text("openai-reasoning-effort")
    (root / "experiment.yaml").write_text(text, encoding="utf-8")
    save_plan(generate_plan(parse_definition(text), now=FIXED_NOW), root / "plan.json")
    study = Study.load(root)
    client = FakeClient(delay=0.002)  # so each response time is above 0 s
    api_runner.send_runs(
        study,
        api_runner.pending_runs(study.require_plan(), set(), 30),
        client=client,
        sdk=_fake_module(client),
        sleep=lambda s: None,
    )
    at = _app(monkeypatch, root)
    for page in PAGES:
        _goto(at, page)
    _goto(at, "Run plan & recording")
    assert any("manual entry is disabled" in i.value for i in at.info)
    assert len(at.text_area) == 0
    _goto(at, "Condition comparison")
    assert _charts(at) == 2  # accuracy and latency; no displayed-label chart
    assert any("Values reported by the API" in h.value for h in at.subheader)


def test_pilot_study_shows_the_pilot_banner(monkeypatch, study_dir):
    from mblab.pilot import create_pilot_study

    _, root = create_pilot_study(study_dir)
    at = _app(monkeypatch, root)
    for page in PAGES:
        _goto(at, page)
        assert any("PILOT DATA" in w.value for w in at.warning), page


def test_a_run_saved_meanwhile_is_not_overwritten(monkeypatch, study_dir, definition, plan):
    from mblab.study import append_jsonl

    from .conftest import make_record

    at = _goto(_app(monkeypatch, study_dir), "Run plan & recording")
    by_label = {w.label: w for w in at.text_input}
    by_label["Client (app) version"].input("1.0")
    at.text_area[0].input("ANSWER: 265")
    # Another command records the same run while the form is open.
    append_jsonl(
        study_dir / "records.jsonl",
        [make_record(definition, plan, "B:T1:r1", notes="other writer")],
    )
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("already has a record" in e.value for e in at.error)
    (kept,) = Study.load(study_dir).records
    assert kept.notes == "other writer"


def test_saving_moves_on_and_replacing_needs_the_tick(monkeypatch, study_dir, definition, plan):
    from mblab.study import write_jsonl

    from .conftest import make_record

    at = _goto(_app(monkeypatch, study_dir), "Run plan & recording")
    {w.label: w for w in at.text_input}["Client (app) version"].input("1.0")
    at.text_area[0].input("ANSWER: 265")
    at.button[0].click().run()
    assert at.selectbox(key="run-select").value == "B:T5:r1"  # the next pending run

    write_jsonl(
        study_dir / "records.jsonl", [make_record(definition, plan, "B:T1:r1", notes="first")]
    )
    at = _goto(_app(monkeypatch, study_dir), "Run plan & recording")
    at.selectbox(key="run-select").set_value("B:T1:r1").run()
    assert any("already has a record" in w.value for w in at.warning)
    {w.label: w for w in at.text_input}["Notes"].input("second")
    at.text_area[0].input("ANSWER: 265")
    at.checkbox[-1].check()  # "Replace the existing record for this run"
    at.button[0].click().run()
    assert not at.exception, at.exception
    (record,) = [r for r in Study.load(study_dir).records if r.run_id == "B:T1:r1"]
    assert record.notes == "second"
    assert (study_dir / "records.jsonl.bak").exists()
