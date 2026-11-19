"""The official-API runner, with the OpenAI SDK replaced by a fake module.

Nothing here touches the network or needs a real key: ``fake_sdk`` is put in
``sys.modules["openai"]`` and its client returns scripted responses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from mblab import api_runner
from mblab.analysis import analyze
from mblab.cli import app
from mblab.plan import generate_plan
from mblab.report import render_report
from mblab.schema import ApiReported, ExperimentDefinition
from mblab.study import Study, StudyError, parse_definition, save_plan, template_text
from mblab.validate import validate_study

from .conftest import FIXED_NOW, make_record
from .test_analysis import BANNED

API_TEMPLATE = "openai-reasoning-effort"
FAKE_KEY = "sk-test-" + "k" * 24  # assembled at runtime; not a real key
UI_TEMPLATE_HASH = "ff3c102876cce68e3520c9ce9775c802ccd3b0b813b86038deb0ea44bc88472b"


# -- a fake `openai` module ------------------------------------------------------------


class OpenAIError(Exception):
    pass


class APIStatusError(OpenAIError):
    status_code = 400

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class BadRequestError(APIStatusError):
    pass


class RateLimitError(APIStatusError):
    status_code = 429


class InternalServerError(APIStatusError):
    status_code = 500


class APIConnectionError(OpenAIError):
    pass


def make_response(
    text: str = "Working.\nANSWER: 265",
    *,
    status: str = "completed",
    effort: str | None = "low",
    reasoning_tokens: int | None = 128,
    usage: bool = True,
    incomplete: str | None = None,
    error: tuple[str, str] | None = None,
    refusal: str | None = None,
) -> SimpleNamespace:
    content = [SimpleNamespace(type="output_text", text=text)] if text else []
    if refusal:
        content.append(SimpleNamespace(type="refusal", refusal=refusal))
    details = (
        None if reasoning_tokens is None else SimpleNamespace(reasoning_tokens=reasoning_tokens)
    )
    return SimpleNamespace(
        id="resp_test",
        model="test-model-2026-01-01",
        status=status,
        output=[SimpleNamespace(type="message", content=content)],
        output_text=text,
        reasoning=SimpleNamespace(effort=effort),
        service_tier="default",
        incomplete_details=SimpleNamespace(reason=incomplete) if incomplete else None,
        error=SimpleNamespace(code=error[0], message=error[1]) if error else None,
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=300,
            output_tokens_details=details,
            total_tokens=400,
        )
        if usage
        else None,
    )


class FakeClient:
    """Answers each request from a script; an exception in the script is raised."""

    def __init__(self, script=None, delay: float = 0.0):
        self.delay = delay
        self.calls: list[dict] = []
        self.script = list(script or [])
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        item = self.script.pop(0) if self.script else None
        if isinstance(item, BaseException):
            raise item
        if item is None:
            effort = kwargs["reasoning"]["effort"]
            return make_response(f"ANSWER: {CORRECT[kwargs['input']]}", effort=effort)
        return item


def _fake_module(client: FakeClient) -> ModuleType:
    module = ModuleType("openai")
    module.__version__ = "0.0.0-fake"
    module.OpenAIError = OpenAIError
    module.APIStatusError = APIStatusError
    module.BadRequestError = BadRequestError
    module.RateLimitError = RateLimitError
    module.InternalServerError = InternalServerError
    module.APIConnectionError = APIConnectionError
    module.created_with = []

    def make(**kwargs):
        module.created_with.append(kwargs)
        return client

    module.OpenAI = make
    return module


@pytest.fixture
def api_definition() -> ExperimentDefinition:
    return parse_definition(template_text(API_TEMPLATE))


CORRECT = {
    t.prompt: t.reference_answer for t in parse_definition(template_text(API_TEMPLATE)).tasks
}


@pytest.fixture
def api_study(tmp_path, api_definition) -> Path:
    root = tmp_path / "api-study"
    root.mkdir()
    (root / "experiment.yaml").write_text(template_text(API_TEMPLATE), encoding="utf-8")
    save_plan(generate_plan(api_definition, now=FIXED_NOW), root / "plan.json")
    return root


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    module = _fake_module(client)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return SimpleNamespace(client=client, module=module)


def _send(root: Path, fake, limit=None, **kwargs):
    study = Study.load(root)
    plan = api_runner.check_api_study(study)
    runs = api_runner.pending_runs(plan, study.recorded_run_ids(), limit)
    kwargs.setdefault("sleep", lambda s: None)
    return api_runner.send_runs(study, runs, client=fake.client, sdk=fake.module, **kwargs)


# -- definitions ---------------------------------------------------------------------------


def test_api_template_reuses_the_verified_tasks(api_definition, definition):
    # The six tasks are identical to the UI template, whose answers are recomputed in
    # tests/test_template_answers.py.
    assert api_definition.tasks == definition.tasks
    assert api_definition.execution_method == "official_api"
    assert api_definition.documented_facts == []
    assert [o.reasoning_effort for o in api_definition.selection_options] == [
        "low",
        "medium",
        "high",
    ]
    assert all(c.message_override is None for c in api_definition.conditions)
    assert api_definition.planned_run_count == 54


def test_ui_definition_hash_is_unchanged(definition):
    # Plans made before official_api fields existed must stay valid.
    assert definition.sha256() == UI_TEMPLATE_HASH
    dumped = definition.model_dump(mode="json")
    assert "api" not in dumped
    assert all("reasoning_effort" not in o for o in dumped["selection_options"])


def _api_dict() -> dict:
    return yaml.safe_load(template_text(API_TEMPLATE))


def _ui_dict() -> dict:
    return yaml.safe_load(template_text())


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: {**_api_dict(), "api": None}, "needs an `api` section"),
        (
            lambda: _api_dict() | {"selection_options": [{"id": "low", "label": "low"}]},
            "needs reasoning_effort",
        ),
        (
            lambda: (
                _api_dict()
                | {
                    "selection_options": [
                        {"id": "a", "label": "a", "reasoning_effort": "low"},
                        {"id": "b", "label": "b", "reasoning_effort": "low"},
                    ],
                    "conditions": [{"id": "A", "model_selection": "a"}],
                }
            ),
            "same reasoning_effort",
        ),
        (
            lambda: (
                _api_dict()
                | {
                    "conditions": [
                        {"id": "X", "model_selection": "low", "message_override": "high"}
                    ],
                }
            ),
            "message_override is a client UI setting",
        ),
        (
            lambda: (
                _api_dict()
                | {"selection_options": [{"id": "low", "label": "x", "reasoning_effort": "turbo"}]}
            ),
            "reasoning_effort",
        ),
        (lambda: _ui_dict() | {"api": _api_dict()["api"]}, "only valid for execution_method"),
        (
            lambda: (
                _ui_dict()
                | {
                    "selection_options": [
                        {"id": "instant", "label": "Instant", "reasoning_effort": "low"},
                        {"id": "pro", "label": "Pro"},
                    ]
                }
            ),
            "reasoning_effort is only valid",
        ),
    ],
)
def test_invalid_api_definitions_are_rejected(make, message):
    with pytest.raises(StudyError, match=message):
        parse_definition(yaml.safe_dump(make()))


def test_every_bundled_template_loads_and_plans():
    from mblab.study import available_templates

    assert API_TEMPLATE in available_templates()
    for name in available_templates():
        definition = parse_definition(template_text(name), name)
        assert len(generate_plan(definition, now=FIXED_NOW).runs) == definition.planned_run_count


# -- records ---------------------------------------------------------------------------------


def test_ui_records_reject_api_values(definition, plan):
    with pytest.raises(ValidationError, match="only valid for execution_method 'official_api'"):
        make_record(definition, plan, "A:T1:r1", api_reported=ApiReported(model="x"))
    rec = make_record(definition, plan, "A:T1:r1")
    assert "api_reported" not in rec.model_dump_json()  # UI records serialize as before


def test_api_records_reject_client_display_fields(api_study, fake):
    _send(api_study, fake, limit=1)
    rec = Study.load(api_study).records[0]
    data = json.loads(rec.model_dump_json())
    for field, value in (
        ("observed_model_label", "Pro"),
        ("thinking_indicator", "shown"),
        ("thinking_seconds", 3),
    ):
        with pytest.raises(ValidationError, match="client-display fields"):
            type(rec).model_validate({**data, field: value})


# -- the runner --------------------------------------------------------------------------------


def test_requests_follow_the_plan_and_copy_only_returned_values(api_study, fake, api_definition):
    result = _send(api_study, fake, limit=4)
    assert result.stopped_at is None
    study = Study.load(api_study)
    plan = study.require_plan()
    first_four = sorted(plan.runs, key=lambda r: r.sequence)[:4]
    assert [r.run_id for r in study.records] == [r.run_id for r in first_four]
    for call, run in zip(fake.client.calls, first_four, strict=True):
        assert call == {
            "model": api_definition.api.model,
            "input": api_definition.task(run.task_id).prompt,
            "reasoning": {"effort": api_definition.option(run.model_selection).reasoning_effort},
            "store": False,
            "max_output_tokens": api_definition.api.max_output_tokens,
        }
    rec = study.records[0]
    assert rec.execution_method == "official_api"
    assert rec.latency_method == "client_timer" and rec.latency_seconds is not None
    assert rec.client_version == "openai-python 0.0.0-fake"
    assert rec.response_status == "completed"
    reported = rec.api_reported
    assert reported.model == "test-model-2026-01-01" and reported.response_id == "resp_test"
    assert (reported.input_tokens, reported.output_tokens, reported.reasoning_tokens) == (
        100,
        300,
        128,
    )
    report = validate_study(study)
    assert report.ok and not report.warnings, report.issues


def test_missing_values_stay_missing(api_study, fake):
    fake.client.script = [
        make_response(usage=False, effort=None),
        make_response(reasoning_tokens=None),
    ]
    _send(api_study, fake, limit=2)
    first, second = Study.load(api_study).records
    assert first.api_reported.input_tokens is None and first.api_reported.total_tokens is None
    assert first.api_reported.reasoning_effort is None
    assert second.api_reported.reasoning_tokens is None
    assert second.api_reported.output_tokens == 300  # reported, so kept


def test_runs_resume_and_skip_recorded(api_study, fake):
    _send(api_study, fake, limit=3)
    _send(api_study, fake, limit=2)
    study = Study.load(api_study)
    ids = [r.run_id for r in study.records]
    assert len(ids) == len(set(ids)) == 5
    assert len(fake.client.calls) == 5
    plan_order = [r.run_id for r in sorted(study.plan.runs, key=lambda r: r.sequence)]
    assert ids == plan_order[:5]


def test_rate_limit_spaces_request_starts(api_study, fake):
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    _send(api_study, fake, limit=3, min_interval=2.0, sleep=sleep, clock=lambda: now[0])
    assert sleeps == [2.0, 2.0]


def test_transient_errors_are_retried_and_noted(api_study, fake):
    fake.client.script = [RateLimitError("slow down"), APIConnectionError("reset")]
    waits: list[float] = []
    result = _send(api_study, fake, limit=1, min_interval=0, retry_wait=1.0, sleep=waits.append)
    assert result.stopped_at is None and len(fake.client.calls) == 3
    assert waits == [1.0, 2.0]
    (rec,) = Study.load(api_study).records
    assert "2 earlier attempt(s) failed (RateLimitError, APIConnectionError)" in rec.notes


def test_persistent_transient_error_writes_nothing(api_study, fake):
    fake.client.script = [InternalServerError("boom")] * 3
    result = _send(api_study, fake, limit=2, retry_wait=0)
    assert result.stopped_at is not None and "InternalServerError" in result.stop_reason
    assert len(fake.client.calls) == 3
    assert Study.load(api_study).records == []


@pytest.mark.parametrize(
    "exc",
    [
        BadRequestError("Unsupported value: 'reasoning.effort'"),
        RateLimitError("You exceeded your current quota", code="insufficient_quota"),
    ],
)
def test_permanent_errors_stop_without_retry(api_study, fake, exc):
    fake.client.script = [make_response(), exc]
    result = _send(api_study, fake)
    assert len(fake.client.calls) == 2  # one success, one failure, no retry
    assert len(Study.load(api_study).records) == 1
    assert result.stopped_at is not None and type(exc).__name__ in result.stop_reason


@pytest.mark.parametrize(
    ("response", "status", "text_check"),
    [
        (make_response("", status="incomplete", incomplete="max_output_tokens"), "incomplete", ""),
        (make_response("", status="failed", error=("server_error", "oops")), "error", ""),
        (make_response("", refusal="I can't help with that."), "refused", "can't help"),
        (make_response(""), "completed", ""),  # returned empty: scored as a format failure
    ],
)
def test_response_statuses_are_mapped_without_guessing(
    api_study, fake, response, status, text_check
):
    fake.client.script = [response]
    _send(api_study, fake, limit=1)
    (rec,) = Study.load(api_study).records
    assert rec.response_status == status
    assert text_check in rec.response_text
    assert rec.api_reported.status == response.status
    if status == "incomplete":
        assert rec.api_reported.incomplete_reason == "max_output_tokens"
    if status == "error":
        assert rec.api_reported.error == "server_error: oops"


def test_unexpected_status_is_not_recorded(api_study, fake):
    fake.client.script = [make_response(status="in_progress")]
    result = _send(api_study, fake, limit=1)
    assert result.stopped_at is not None and "unexpected response status" in result.stop_reason
    assert Study.load(api_study).records == []


def test_key_is_never_written_or_echoed(api_study, fake):
    fake.client.script = [
        make_response(),
        BadRequestError(f"Incorrect API key provided: {FAKE_KEY}"),
    ]
    result = _send(api_study, fake)
    assert FAKE_KEY not in result.stop_reason and "[redacted]" in result.stop_reason
    for path in api_study.rglob("*"):
        if path.is_file():
            assert FAKE_KEY not in path.read_text(encoding="utf-8"), path


def test_refuses_ui_studies_and_changed_definitions(study_dir, api_study):
    with pytest.raises(StudyError, match="only sends official_api studies"):
        api_runner.check_api_study(Study.load(study_dir))
    data = _api_dict()
    data["api"]["model"] = "another-model"
    (api_study / "experiment.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(StudyError, match=r"changed after plan\.json"):
        api_runner.check_api_study(Study.load(api_study))


def test_missing_key_and_missing_sdk(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(StudyError, match="OPENAI_API_KEY"):
        api_runner.make_client(_fake_module(FakeClient()))
    monkeypatch.setitem(sys.modules, "openai", None)  # import fails
    with pytest.raises(StudyError, match=r"model-behavior-lab\[api\]"):
        api_runner.load_sdk()


# -- validation, analysis, report -----------------------------------------------------------------


def test_validation_flags_api_inconsistencies(api_study, fake):
    fake.client.script = [
        make_response(effort="high"),  # first run requests medium
        SimpleNamespace(**{**vars(make_response()), "model": "other-model"}),
    ]
    _send(api_study, fake, limit=2)
    codes = {i.code for i in validate_study(Study.load(api_study)).warnings}
    assert {"api_effort_mismatch", "api_model_changed"} <= codes


def test_api_analysis_and_report(api_study, fake):
    _send(api_study, fake)  # all 54 runs, every answer correct
    study = Study.load(api_study)
    a = analyze(study)
    assert set(a.api["n_api_values"]) == {18}
    assert set(a.api["reasoning_tokens_median"]) == {128.0}
    assert "reasoning_tokens_median" in set(a.comparisons["metric"])
    assert "api_reported" in a.to_dict()
    text = render_report(a, study, now=FIXED_NOW)
    assert "### 6.4 Values reported by the API" in text
    assert "Client-displayed model labels" not in text
    assert "_Not applicable: every API request carries exactly one setting" in text
    assert "recorded as returned" in text
    assert not BANNED.search(text)
    # No UI-only limitations (displayed labels, thinking indicator) in an API study.
    assert not any("displayed" in lim.text or "model label" in lim.text for lim in a.limitations)


def test_ui_analysis_has_no_api_output(study_dir, definition, plan):
    from .conftest import write_records

    write_records(study_dir, [make_record(definition, plan, r.run_id) for r in plan.runs[:10]])
    a = analyze(Study.load(study_dir))
    assert a.api.empty and "api_reported" not in a.to_dict()
    assert not any(c.startswith("api_") for c in a.results.columns)


# -- CLI ------------------------------------------------------------------------------------------------

runner = CliRunner()
ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def test_cli_dry_run_sends_nothing_and_needs_no_key_or_sdk(api_study, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)  # the SDK is not even importable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = sorted(p.name for p in api_study.iterdir())
    result = runner.invoke(app, ["run-api", "-s", str(api_study), "--dry-run"], env=ENV)
    assert result.exit_code == 0, result.output
    assert "nothing was sent and nothing was written" in result.output
    assert '"store": false' in result.output and "is NOT set" in result.output
    assert sorted(p.name for p in api_study.iterdir()) == before


def test_cli_run_asks_first_then_sends(api_study, fake):
    declined = runner.invoke(app, ["run-api", "-s", str(api_study)], input="n\n", env=ENV)
    assert declined.exit_code == 1 and "Nothing was sent" in declined.output
    assert fake.client.calls == []
    ok = runner.invoke(
        app, ["run-api", "-s", str(api_study), "--limit", "2", "-y", "--min-interval", "0"], env=ENV
    )
    assert ok.exit_code == 0, ok.output
    assert "Recorded 2 run(s)" in ok.output and len(fake.client.calls) == 2
    # The key goes from the environment straight to the SDK, with SDK retries off.
    assert fake.module.created_with[-1] == {"api_key": FAKE_KEY, "max_retries": 0}
    assert FAKE_KEY not in ok.output


def test_cli_run_reports_a_stop(api_study, fake):
    fake.client.script = [BadRequestError("model not found")]
    result = runner.invoke(app, ["run-api", "-s", str(api_study), "-y"], env=ENV)
    assert result.exit_code == 2
    assert "no record written for it" in result.output and "resume" in result.output


def test_cli_manual_entry_is_refused_for_api_studies(api_study):
    result = runner.invoke(app, ["record", "-s", str(api_study)], env=ENV)
    assert result.exit_code == 2 and "mblab run-api" in result.output


def test_cli_run_api_refuses_ui_studies(study_dir, fake):
    result = runner.invoke(app, ["run-api", "-s", str(study_dir), "--dry-run"], env=ENV)
    assert result.exit_code == 2 and "official_api" in result.output
