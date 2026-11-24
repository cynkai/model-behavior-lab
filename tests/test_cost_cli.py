from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mblab.cli import app
from mblab.plan import generate_plan
from mblab.records import build_record
from mblab.schema import ApiReported
from mblab.study import parse_definition, save_plan, template_text, write_jsonl

from .conftest import FIXED_NOW


@pytest.fixture
def priced_study(tmp_path: Path) -> tuple[Path, Path]:
    study = tmp_path / "api-study"
    study.mkdir()
    template = template_text("openai-reasoning-effort")
    (study / "experiment.yaml").write_text(template, encoding="utf-8")
    definition = parse_definition(template)
    plan = generate_plan(definition, now=FIXED_NOW)
    save_plan(plan, study / "plan.json")
    usage = ApiReported(
        model="example-model",
        service_tier="default",
        input_tokens=1000,
        cached_input_tokens=200,
        output_tokens=100,
        reasoning_tokens=60,
    )
    records = [
        build_record(
            run,
            definition,
            plan,
            client_version="test",
            timestamp=FIXED_NOW,
            response_status="completed",
            response_text="private answer",
            data_origin="synthetic",
            api_reported=reported,
        )
        for run, reported in zip(
            plan.runs[:2], [usage, usage.model_copy(update={"service_tier": "other"})], strict=True
        )
    ]
    write_jsonl(study / "records.jsonl", records)
    rates = tmp_path / "rates.json"
    rates.write_text(
        json.dumps(
            {
                "currency": "USD",
                "rates": [
                    {
                        "model": "example-model",
                        "service_tier": "default",
                        "input_per_million": "2",
                        "cached_input_per_million": "0.5",
                        "output_per_million": "6",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return study, rates


def test_cost_json_is_read_only_and_records_coverage(priced_study: tuple[Path, Path]):
    study, rates = priced_study

    def inputs():
        return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in [*study.iterdir(), rates]}

    before = inputs()
    result = CliRunner().invoke(app, ["cost", "-s", str(study), "--rates", str(rates), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["currency"] == "USD" and report["synthetic_data"]
    assert report["rate_card_sha256"] == hashlib.sha256(rates.read_bytes()).hexdigest()
    assert report["total"] == {
        "recorded_runs": 2,
        "priced_runs": 1,
        "estimated_usd": "0.0023",
        "unpriced_reasons": {"unknown_rate": 1},
    }
    assert "private answer" not in result.output
    assert inputs() == before


def test_cost_table_discloses_partial_coverage(priced_study: tuple[Path, Path]):
    study, rates = priced_study
    result = CliRunner().invoke(app, ["cost", "--rates", str(rates), "-s", str(study)])
    assert result.exit_code == 0, result.output
    assert "0.0023" in result.output and "1/2" in result.output
    assert "SYNTHETIC DEMO DATA" in result.output
    assert "no matching model and service tier rate" in result.output


def test_cost_errors_are_actionable(priced_study: tuple[Path, Path], study_dir: Path):
    study, rates = priced_study
    runner = CliRunner()
    invalid_method = runner.invoke(app, ["cost", "--rates", str(rates), "-s", str(study_dir)])
    assert invalid_method.exit_code == 2 and "official_api" in invalid_method.output
    rates.write_text("{", encoding="utf-8")
    invalid_rates = runner.invoke(app, ["cost", "--rates", str(rates), "-s", str(study)])
    assert invalid_rates.exit_code == 2 and "error" in invalid_rates.output.lower()
