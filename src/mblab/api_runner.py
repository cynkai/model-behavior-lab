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

import contextlib
import json
import os
import re
import signal
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mblab.records import build_record
from mblab.schema import ApiReported, ExperimentDefinition, PlannedRun, RunPlan, RunRecord
from mblab.study import Study, StudyError, append_jsonl

API_KEY_ENV = "OPENAI_API_KEY"
#: Billed responses that arrived but could not be recorded (one JSON object per line).
UNRECORDED_FILE = "api-unrecorded.jsonl"
LOCK_FILE = ".run-api.lock"
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


def pending_runs(
    plan: RunPlan,
    recorded: set[str],
    limit: int | None = None,
    held: set[str] | frozenset[str] = frozenset(),
) -> list[PlannedRun]:
    """Runs without a record, in plan order. ``held`` runs (a billed response is on file
    in the unrecorded log) are left out so they are not paid for twice by accident."""
    runs = [
        r
        for r in sorted(plan.runs, key=lambda r: r.sequence)
        if r.run_id not in recorded and r.run_id not in held
    ]
    return runs if limit is None else runs[:limit]


# -- billed responses that could not be recorded ------------------------------------


def read_unrecorded(root: Path) -> list[dict[str, Any]]:
    path = root / UNRECORDED_FILE
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                entries.append(json.loads(line))
    return entries


def unrecorded_run_ids(root: Path, recorded: set[str]) -> set[str]:
    """Runs with a billed-but-unrecorded response and still no record."""
    return {e["run_id"] for e in read_unrecorded(root) if e.get("run_id") not in recorded}


def _log_unrecorded(root: Path, run: PlannedRun, response: Any, reason: str) -> None:
    """Keep a trace of a billed response that could not become a record.

    Only identifiers, the status, token counts, and the reason are kept; no text.
    """
    entry = {
        "run_id": run.run_id,
        "received_at": datetime.now().astimezone().isoformat(),
        "response_id": _get(response, "id"),
        "status": _get(response, "status"),
        "input_tokens": _get(response, "usage", "input_tokens"),
        "output_tokens": _get(response, "usage", "output_tokens"),
        "reason": reason,
    }
    with (root / UNRECORDED_FILE).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# -- safety around sending ------------------------------------------------------------


@contextlib.contextmanager
def study_lock(root: Path) -> Iterator[None]:
    """Refuse a second sender on the same study (it would pay for the same runs twice)."""
    path = root / LOCK_FILE
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise StudyError(
            f"another `mblab run-api` seems to be running on this study ({path} exists). "
            "If none is running, delete that file and try again."
        ) from None
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def defer_interrupt() -> Iterator[None]:
    """Hold Ctrl+C until a received (and billed) response has been written."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, frame: pending.append(signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if pending:
        raise KeyboardInterrupt


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
        refusal=redact(refusal) if refusal else None,
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
    resend_unrecorded: bool = False,
) -> ApiRunResult:
    """Send ``runs`` in order and append a record after each response.

    Stops at the first run that cannot be completed and reports why; earlier
    records are already saved.
    """
    plan = study.require_plan()
    definition = study.definition
    client_version = f"openai-python {getattr(sdk, '__version__', 'unknown')}"
    result = ApiRunResult()
    with study_lock(study.root):
        held = (
            set() if resend_unrecorded else unrecorded_run_ids(study.root, study.recorded_run_ids())
        )
        last_start: float | None = None
        for run in runs:
            if run.run_id in held or run.run_id in study.recorded_run_ids():
                continue  # never pay twice for a run that already has a response
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
            # The response is billed from here on: finish writing it before Ctrl+C acts.
            with defer_interrupt():
                try:
                    fields = response_fields(response)
                    notes = None
                    if failures:
                        notes = (
                            f"{len(failures)} earlier attempt(s) failed ({', '.join(failures)}) "
                            "and were retried; timestamp and latency refer to the final attempt."
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
                    reason = redact(f"response not recorded: {exc}")
                    _log_unrecorded(study.root, run, response, reason)
                    result.stopped_at = run
                    result.stop_reason = (
                        f"{reason}. The billed response is listed in {UNRECORDED_FILE} and the "
                        "run is held back from resending (use --resend-unrecorded to send it again)"
                    )
                    return result
                append_jsonl(study.records_path, [record])
                study.records.append(record)
            outcome = RunOutcome(run, record, len(failures))
            result.recorded.append(outcome)
            if on_record:
                on_record(outcome)
    return result
