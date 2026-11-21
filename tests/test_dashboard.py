"""Smoke tests for the Streamlit dashboard via streamlit.testing (no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from mblab.dashboard import PAGES
from mblab.demo import create_demo_study
from mblab.formatting import md_text
from mblab.plan import generate_plan
from mblab.schema import ExperimentDefinition
from mblab.study import Study, save_plan, template_text

from .conftest import FIXED_NOW, make_record, write_records

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


def _answer_per_run_confounders(at: AppTest, value: str = "none") -> None:
    """Per-run confounders (tools, notices) start empty and must be answered."""
    for box in at.selectbox:
        if box.value is None:
            box.set_value(value)


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
    _answer_per_run_confounders(at)
    at.button[0].click().run()
    assert not at.exception, at.exception
    (record,) = Study.load(study_dir).records
    assert record.run_id == "B:T1:r1"
    assert record.latency_seconds == 12.5 and record.latency_method == "manual_stopwatch"
    assert record.response_text == "Work.\nANSWER: 265"
    assert any(md_text("Saved B:T1:r1.") == s.value for s in at.success)


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
    _answer_per_run_confounders(at)
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
    _answer_per_run_confounders(at)
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
    _answer_per_run_confounders(at)
    at.button[0].click().run()
    assert not at.exception, at.exception
    (record,) = [r for r in Study.load(study_dir).records if r.run_id == "B:T1:r1"]
    assert record.notes == "second"
    assert (study_dir / "records.jsonl.bak").exists()


PAYLOAD = "![pixel](https://example.invalid/p.png) [open](https://example.invalid/x)"


def _markdown_strings(at: AppTest) -> list[str]:
    """Everything Streamlit renders as Markdown: text, captions, alerts, widget labels."""
    texts = []
    for elements in (
        at.title,
        at.subheader,
        at.markdown,
        at.caption,
        at.error,
        at.warning,
        at.info,
        at.success,
    ):
        texts += [e.value for e in elements]
    for widgets in (at.selectbox, at.multiselect, at.text_input, at.expander):
        texts += [w.label for w in widgets]
    return texts


def test_text_from_a_study_is_never_rendered_as_markdown(monkeypatch, tmp_path):
    # A study shared by someone else: Markdown in its definition or records must not
    # turn into links or images that load from a remote server.
    data = yaml.safe_load(template_text())
    data["title"] = f"Shared study {PAYLOAD}"
    data["service"] = PAYLOAD
    data["research_questions"] = [PAYLOAD]
    data["hypotheses"][0]["statement"] = PAYLOAD
    data["hypotheses"][0]["predictions"] = [PAYLOAD]
    data["not_observable"] = [PAYLOAD]
    data["latency_protocol"] = PAYLOAD
    data["confounders"][0]["question"] = PAYLOAD
    definition = ExperimentDefinition.model_validate(data)
    plan = generate_plan(definition, now=FIXED_NOW)
    root = tmp_path / "shared"
    root.mkdir()
    (root / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    save_plan(plan, root / "plan.json")
    first, second = sorted(plan.runs, key=lambda r: r.sequence)[:2]
    write_records(
        root,
        [
            make_record(definition, plan, first.run_id, protocol_deviation=PAYLOAD),
            make_record(definition, plan, second.run_id, client_version=PAYLOAD),
        ],
    )
    at = _app(monkeypatch, root)
    shown = []
    for page in PAGES:
        _goto(at, page)
        texts = _markdown_strings(at)
        assert not [t for t in texts if "](https://example.invalid" in t], page
        shown += texts
    # The payload reached the pages, escaped: from the title, the confounder question
    # (a widget label), a validation note (record text), and the methodology page.
    assert any(md_text(f"Shared study {PAYLOAD}") == t for t in shown)
    assert md_text(PAYLOAD) in shown
    assert any(md_text(PAYLOAD) in t for t in shown if t.startswith("- `excluded_record`"))


def test_per_run_confounders_are_never_carried_over(monkeypatch, study_dir):
    at = _goto(_app(monkeypatch, study_dir), "Run plan & recording")
    {w.label: w for w in at.text_input}["Client (app) version"].input("1.0")
    at.text_area[0].input("ANSWER: 265")
    at.button[0].click().run()  # tools_used and switch_notice left unanswered
    assert any("Not saved: answer" in e.value for e in at.error)
    assert Study.load(study_dir).records == []
    _answer_per_run_confounders(at, "none")
    at.button[0].click().run()
    assert len(Study.load(study_dir).records) == 1
    # The next run's form starts empty again for per-run confounders, not with "none".
    assert sum(1 for box in at.selectbox if box.value is None) == 2


def test_held_api_runs_are_shown_where_run_api_skips_them(monkeypatch, tmp_path):
    from mblab import api_runner
    from mblab.study import parse_definition

    from .test_api_runner import FakeClient, _fake_module, make_response

    root = tmp_path / "api"
    root.mkdir()
    text = template_text("openai-reasoning-effort")
    (root / "experiment.yaml").write_text(text, encoding="utf-8")
    save_plan(generate_plan(parse_definition(text), now=FIXED_NOW), root / "plan.json")
    client = FakeClient(script=[make_response(status="in_progress")])
    for limit in (1, 2):  # the first response cannot be recorded; the second run is
        study = Study.load(root)
        api_runner.send_runs(
            study,
            api_runner.pending_runs(
                study.require_plan(),
                study.recorded_run_ids(),
                limit,
                api_runner.held_runs(study).run_ids,
            )[-1:],
            client=client,
            sdk=_fake_module(client),
            sleep=lambda s: None,
        )
    held_runs = api_runner.held_runs(Study.load(root))
    (held,) = held_runs.runs
    at = _goto(_app(monkeypatch, root), "Run plan & recording")
    assert md_text(held_runs.describe()) in [w.value for w in at.warning]
    assert "1 run(s) held back" in held_runs.describe() and held.run_id in held_runs.describe()
    plan_table = at.dataframe[-1].value
    assert plan_table.set_index("run").loc[held.run_id, "status"] == "held back"
    assert at.selectbox(key="run-select").value != held.run_id  # the next run to send


def test_a_benchmark_directory_gets_the_benchmark_page(monkeypatch):
    from .test_benchmark import BENCH

    at = _app(monkeypatch, BENCH)
    assert [t.value for t in at.title] == [
        md_text("ChatGPT Model Selection Override — Constraint Reasoning")
    ]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "Full task success": "4 of 6",
        "Partially correct": "2 of 6",
        "Exact count": "4 of 6",
        "Valid orders given": "12 of 12",
    }
    assert _charts(at) == 1  # counts given; no response-time chart (none were recorded)
    headers = [h.value for h in at.subheader]
    assert "Comparisons" in headers and "Error analysis" in headers
    markdown = " ".join(m.value for m in at.markdown)
    for title in ("B vs C", "A vs D", "E vs F"):
        assert md_text(title) in markdown
    assert not at.sidebar.radio  # one page: the study pages do not apply
    assert any("Exploratory" in i.value for i in at.info)
