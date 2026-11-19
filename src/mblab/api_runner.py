"""Official-API runner: send planned runs through the OpenAI Responses API.

Rules this module keeps (see docs/methodology.md §12):

* Only the official ``openai`` Python SDK and the Responses API, with
  ``reasoning.effort`` taken from the study's selection options.
* The API key is read from the ``OPENAI_API_KEY`` environment variable when the
  client is created and handed straight to the SDK. It is never printed,
  logged, or written to a study file; error messages are scrubbed of it.
* Runs are sent one at a time in plan order, at most one request start per
  ``min_interval`` seconds. A run that already has a record is skipped, and each
  record is appended as soon as its response arrives, so an interrupted run is
  resumed by running the command again.
* A record holds values copied from the plan or returned by the API, nothing
  else. A request that ends in an exception (rate limit, network, a rejected
  parameter) writes no record: the run stays pending.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mblab.records import build_record
from mblab.schema import ApiReported, ExperimentDefinition, PlannedRun, RunPlan, RunRecord
from mblab.study import Study, StudyError, append_jsonl

API_KEY_ENV = "OPENAI_API_KEY"
INSTALL_HINT = 'the official_api runner needs the OpenAI SDK: pip install "model-behavior-lab[api]"'
#: Retries after a transient error (rate limit, connection problem, 5xx).
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_WAIT = 10.0
_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_*-]{8,}")


# -- requests ----------------------------------------------------------------------


def check_api_study(study: Study) -> RunPlan:
    """Refuse to send anything unless the study is a consistent official_api study."""
    definition = study.definition
    if definition.execution_method != "official_api":
        raise StudyError(
            f"this study uses execution_method '{definition.execution_method}'; "
            "`mblab run-api` only sends official_api studies (record UI runs with `mblab record`)"
        )
    plan = study.require_plan()
    if plan.definition_sha256 != definition.sha256():
        raise StudyError(
            "experiment.yaml changed after plan.json was generated, so requests would not match "
            "the plan. Revert the change, or regenerate with `mblab plan --force` if nothing has "
            "been sent yet (`mblab validate` lists what changed)."
        )
    if study.load_errors:
        raise StudyError(
            "records.jsonl or grades.jsonl has unreadable lines; fix them first (`mblab validate`) "
            "so no run is sent twice"
        )
    return plan


def build_request(definition: ExperimentDefinition, run: PlannedRun) -> dict[str, Any]:
    """Keyword arguments for ``client.responses.create`` — exactly what is sent."""
    assert definition.api is not None  # guaranteed for official_api definitions
    request: dict[str, Any] = {
        "model": definition.api.model,
        "input": definition.task(run.task_id).prompt,
        "reasoning": {"effort": definition.option(run.model_selection).reasoning_effort},
        # The provider keeps no copy of the response for later retrieval.
        "store": False,
    }
    if definition.api.max_output_tokens is not None:
        request["max_output_tokens"] = definition.api.max_output_tokens
    return request


def pending_runs(plan: RunPlan, recorded: set[str], limit: int | None = None) -> list[PlannedRun]:
    runs = [r for r in sorted(plan.runs, key=lambda r: r.sequence) if r.run_id not in recorded]
    return runs if limit is None else runs[:limit]


# -- the SDK ---------------------------------------------------------------------------


def load_sdk() -> Any:
    try:
        import openai
    except ImportError:
        raise StudyError(INSTALL_HINT) from None
    return openai


def api_key_is_set() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def make_client(sdk: Any) -> Any:
    """An SDK client with the key from the environment and SDK retries disabled.

    Retries are done here instead, so the timer covers one attempt only.
    """
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise StudyError(f"set the {API_KEY_ENV} environment variable (the key is never stored)")
    return sdk.OpenAI(api_key=key, max_retries=0)


def redact(text: str) -> str:
    """Remove the API key, and anything shaped like one, from a message."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        text = text.replace(key, "[redacted]")
    return _KEY_PATTERN.sub("[redacted]", text)


def _is_transient(exc: BaseException, sdk: Any) -> bool:
    if getattr(exc, "code", None) == "insufficient_quota":
        return False  # a billing problem; waiting does not help
    return isinstance(exc, sdk.APIConnectionError | sdk.RateLimitError | sdk.InternalServerError)


# -- responses -------------------------------------------------------------------------


def _get(obj: Any, *path: str) -> Any:
    """Follow attributes, returning None as soon as one is missing or None."""
    for name in path:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


def _refusal(response: Any) -> str | None:
    parts = [
        content.refusal
        for item in (_get(response, "output") or [])
        if getattr(item, "type", None) == "message"
        for content in (getattr(item, "content", None) or [])
        if getattr(content, "type", None) == "refusal" and getattr(content, "refusal", None)
    ]
    return "\n".join(parts) if parts else None


def response_fields(response: Any) -> dict[str, Any]:
    """Record fields from an SDK ``Response``: only what it actually contains.

    Raises ``ValueError`` for a status a synchronous request should never end in,
    instead of guessing what it means.
    """
    status = _get(response, "status")
    text = _get(response, "output_text") or ""
    refusal = _refusal(response)
    error = _get(response, "error")
    if status == "completed":
        response_status = "refused" if refusal and not text.strip() else "completed"
    elif status == "incomplete":
        response_status = "incomplete"
    elif status == "failed":
        response_status = "error"
    else:
        raise ValueError(f"the API returned an unexpected response status {status!r}")
    reported = ApiReported(
        response_id=_get(response, "id"),
        model=_get(response, "model"),
        status=status,
        reasoning_effort=_get(response, "reasoning", "effort"),
        service_tier=_get(response, "service_tier"),
        incomplete_reason=_get(response, "incomplete_details", "reason"),
        error=(
            redact(f"{_get(error, 'code') or 'error'}: {_get(error, 'message') or ''}".strip())
            if error is not None
            else None
        ),
        input_tokens=_get(response, "usage", "input_tokens"),
        cached_input_tokens=_get(response, "usage", "input_tokens_details", "cached_tokens"),
        output_tokens=_get(response, "usage", "output_tokens"),
        reasoning_tokens=_get(response, "usage", "output_tokens_details", "reasoning_tokens"),
        total_tokens=_get(response, "usage", "total_tokens"),
    )
    return {
        "response_status": response_status,
        "response_text": refusal if response_status == "refused" else text,
        "api_reported": reported,
    }


# -- the loop ----------------------------------------------------------------------------


@dataclass
class RunOutcome:
    run: PlannedRun
    record: RunRecord
    failed_attempts: int


@dataclass
class ApiRunResult:
    recorded: list[RunOutcome] = field(default_factory=list)
    stopped_at: PlannedRun | None = None
    stop_reason: str | None = None


def send_runs(
    study: Study,
    runs: list[PlannedRun],
    *,
    client: Any,
    sdk: Any,
    min_interval: float = 1.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait: float = DEFAULT_RETRY_WAIT,
    on_record: Callable[[RunOutcome], None] | None = None,
    on_retry: Callable[[PlannedRun, str, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ApiRunResult:
    """Send ``runs`` in order and append a record after each response.

    Stops at the first run that cannot be completed and reports why; earlier
    records are already saved.
    """
    plan = study.require_plan()
    definition = study.definition
    client_version = f"openai-python {getattr(sdk, '__version__', 'unknown')}"
    result = ApiRunResult()
    last_start: float | None = None
    for run in runs:
        request = build_request(definition, run)
        failures: list[str] = []
        while True:
            if last_start is not None:
                wait = min_interval - (clock() - last_start)
                if wait > 0:
                    sleep(wait)
            last_start = clock()
            sent_at = datetime.now().astimezone()
            started = time.perf_counter()
            try:
                response = client.responses.create(**request)
            except sdk.OpenAIError as exc:
                message = redact(f"{type(exc).__name__}: {exc}")
                if _is_transient(exc, sdk) and len(failures) < max_retries:
                    failures.append(type(exc).__name__)
                    delay = retry_wait * 2 ** (len(failures) - 1)
                    if on_retry:
                        on_retry(run, message, delay)
                    sleep(delay)
                    continue
                result.stopped_at, result.stop_reason = run, message
                return result
            elapsed = time.perf_counter() - started
            break
        try:
            fields = response_fields(response)
            notes = None
            if failures:
                notes = (
                    f"{len(failures)} earlier attempt(s) failed ({', '.join(failures)}) and were "
                    "retried; timestamp and latency refer to the final attempt."
                )
            record = build_record(
                run,
                definition,
                plan,
                client_version=client_version,
                timestamp=sent_at,
                latency_seconds=round(elapsed, 3),
                latency_method="client_timer",
                new_conversation=True,
                notes=notes,
                **fields,
            )
        except ValueError as exc:  # includes pydantic's ValidationError
            result.stopped_at = run
            result.stop_reason = redact(f"response not recorded: {exc}")
            return result
        append_jsonl(study.records_path, [record])
        study.records.append(record)
        outcome = RunOutcome(run, record, len(failures))
        result.recorded.append(outcome)
        if on_record:
            on_record(outcome)
    return result
