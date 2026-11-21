"""The result submission forms ask each execution method only for what is true of it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FORMS = Path(__file__).resolve().parents[1] / ".github" / "ISSUE_TEMPLATE"


def _form(name: str) -> dict:
    return yaml.safe_load((FORMS / name).read_text(encoding="utf-8"))


def _checks(form: dict) -> list[str]:
    (block,) = [b for b in form["body"] if b["type"] == "checkboxes"]
    return [option["label"] for option in block["attributes"]["options"]]


def _field(form: dict, field_id: str) -> dict:
    (block,) = [b for b in form["body"] if b.get("id") == field_id]
    return block


@pytest.mark.parametrize(
    "name", sorted(p.name for p in FORMS.glob("*.yml") if p.name != "config.yml")
)
def test_forms_are_well_formed(name):
    form = _form(name)
    assert form["name"] and form["description"] and form["labels"]
    ids = [b["id"] for b in form["body"] if "id" in b]
    assert len(ids) == len(set(ids))


def test_each_execution_method_has_its_own_pledge():
    app_form, api_form = _form("result_submission.yml"), _form("result_submission_api.yml")
    assert any("by hand in the official client" in c for c in _checks(app_form))
    assert not any("by hand" in c and "collected" in c for c in _checks(api_form))
    assert any("`mblab run-api`" in c for c in _checks(api_form))
    assert any("API key" in c for c in _checks(api_form))
    for form in (app_form, api_form):
        assert "results" in form["labels"]
        assert _field(form, "code")["validations"]["required"]
        assert "reports/report.md" in _field(form, "data")["attributes"]["description"]
    assert "api-unrecorded.jsonl" in _field(api_form, "data")["attributes"]["description"]
    assert "api-unrecorded.jsonl" not in _field(app_form, "data")["attributes"]["description"]
