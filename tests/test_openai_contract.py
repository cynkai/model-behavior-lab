"""The runner against the real OpenAI SDK's types — no network, no key.

Skipped when the SDK is not installed. It catches SDK interface drift that the
fake module in test_api_runner.py cannot: parameter names, exception classes,
and the shape of a parsed ``Response``.
"""

from __future__ import annotations

import inspect

import pytest

openai = pytest.importorskip("openai")

from openai.resources.responses import Responses  # noqa: E402
from openai.types.responses import Response  # noqa: E402

from mblab.api_runner import build_request, response_fields  # noqa: E402
from mblab.plan import generate_plan  # noqa: E402
from mblab.study import parse_definition, template_text  # noqa: E402

from .conftest import FIXED_NOW  # noqa: E402

PAYLOAD = {
    "id": "resp_contract",
    "object": "response",
    "created_at": 1_790_000_000,
    "status": "incomplete",
    "model": "test-model-2026-01-01",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "incomplete",
            "content": [{"type": "output_text", "text": "ANSWER: 2", "annotations": []}],
        }
    ],
    "error": None,
    "incomplete_details": {"reason": "max_output_tokens"},
    "instructions": None,
    "metadata": {},
    "parallel_tool_calls": True,
    "temperature": None,
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "reasoning": {"effort": "high", "summary": None},
    "service_tier": "default",
    "usage": {
        "input_tokens": 80,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 500,
        "output_tokens_details": {"reasoning_tokens": 490},
        "total_tokens": 580,
    },
}


def test_sdk_accepts_the_request_we_send():
    definition = parse_definition(template_text("openai-reasoning-effort"))
    run = generate_plan(definition, now=FIXED_NOW).runs[0]
    params = inspect.signature(Responses.create).parameters
    assert set(build_request(definition, run)) <= set(params)
    assert {"api_key", "max_retries"} <= set(inspect.signature(openai.OpenAI.__init__).parameters)


def test_sdk_exposes_the_exceptions_we_handle():
    for name in (
        "OpenAIError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "BadRequestError",
    ):
        assert issubclass(getattr(openai, name), openai.OpenAIError)


def test_a_parsed_sdk_response_maps_to_record_fields():
    fields = response_fields(Response.model_validate(PAYLOAD))
    assert fields["response_status"] == "incomplete"
    assert fields["response_text"] == "ANSWER: 2"
    reported = fields["api_reported"]
    assert reported.model == "test-model-2026-01-01"
    assert reported.reasoning_effort == "high"
    assert reported.incomplete_reason == "max_output_tokens"
    assert (reported.input_tokens, reported.reasoning_tokens, reported.total_tokens) == (
        80,
        490,
        580,
    )
