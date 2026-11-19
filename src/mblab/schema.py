"""Data models: experiment definitions, run plans, observations, and grades.

Design rules
------------
* Every field is something the experimenter decides in advance or can observe
  directly. There are deliberately no fields for things a manual UI experiment
  cannot see (server-side model identifiers, hidden reasoning tokens, routing
  decisions). Do not add them as guesses.
* Scoring rules live in the experiment definition, so they are fixed before data
  collection and covered by the definition hash stored in the run plan.
* UI observations and official-API measurements are different execution
  methods. One experiment uses exactly one method, and records are checked
  against it, so the two kinds of data are never pooled by accident.
* Values an official API reports (model id, token usage) live in one optional
  ``api_reported`` object that is only valid for ``official_api`` records. They
  are copied as returned; a value the API did not return stays ``None``.
* Fields added for ``official_api`` are left out of the serialized form while
  unset, so the definition hash of a ``manual_ui`` study, and its records, are
  byte-for-byte what they were before those fields existed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

SCHEMA_VERSION = 1

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
Identifier = Annotated[str, Field(pattern=ID_PATTERN, max_length=64)]

ExecutionMethod = Literal["manual_ui", "official_api"]
DataOrigin = Literal["observed", "synthetic"]
ResponseStatus = Literal["completed", "error", "refused", "incomplete"]
LatencyMethod = Literal["manual_stopwatch", "screen_recording", "client_timer", "not_measured"]
Confidence = Literal["low", "medium", "high"]
ThinkingIndicator = Literal["shown", "not_shown", "unknown"]
Difficulty = Literal["easy", "medium", "hard"]
RandomizationStrategy = Literal["blocked_by_repetition", "complete"]
ExtractMode = Literal["answer_line", "last_line", "full_response"]
NormalizeStep = Literal[
    "strip_markdown",
    "lowercase",
    "collapse_whitespace",
    "remove_whitespace",
    "strip_edge_punctuation",
]
#: ``reasoning.effort`` values accepted by the OpenAI Responses API, as listed by
#: the official SDK's ``ReasoningEffort`` type. Not every model accepts every value;
#: the API rejects an unsupported one, and `mblab run-api` then stops.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

#: Latency methods that are meaningful for each execution method. A stopwatch
#: reading from a phone and a client-side API timer are different instruments.
LATENCY_METHODS_BY_EXECUTION: dict[str, set[str]] = {
    "manual_ui": {"manual_stopwatch", "screen_recording", "not_measured"},
    "official_api": {"client_timer", "not_measured"},
}

#: Confidence assumed when a latency value is given without one (documented in
#: docs/methodology.md). It only describes the instrument, not the result.
DEFAULT_LATENCY_CONFIDENCE: dict[str, str] = {
    "manual_stopwatch": "low",
    "screen_recording": "medium",
    "client_timer": "high",
}

DEFAULT_NORMALIZE: list[str] = [
    "strip_markdown",
    "collapse_whitespace",
    "lowercase",
    "strip_edge_punctuation",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def normalize_label(text: str) -> str:
    """Case- and whitespace-insensitive form used to compare UI labels."""
    return " ".join(text.split()).casefold()


def _drop_unset(data: dict[str, Any], *names: str) -> dict[str, Any]:
    """Remove official_api-only keys whose value is None (see the module docstring)."""
    for name in names:
        if data.get(name) is None:
            data.pop(name, None)
    return data


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------


class DocumentedFact(StrictModel):
    """A statement verified in official documentation (not a hypothesis)."""

    statement: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    accessed_on: date


class Hypothesis(StrictModel):
    id: Identifier
    statement: str = Field(min_length=1)
    predictions: list[str] = Field(default_factory=list)
    predicted_pattern: Literal["per_message", "default", "no_difference"] | None = Field(
        default=None,
        description=(
            "Which behavioural pattern the hypothesis predicts for its conditions, so reports "
            "can count matching assessments. A count is a description, not a test."
        ),
    )


class SelectionOption(StrictModel):
    """A choice in the client's model / reasoning-effort picker."""

    id: Identifier
    label: str = Field(min_length=1, description="Exactly as shown in the client's picker.")
    observed_label_aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other text the client may display for a response produced with this option. "
            "Fixed in advance so label matching is not decided after seeing the data."
        ),
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="official_api only: the reasoning.effort value sent for this option.",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        return _drop_unset(handler(self), "reasoning_effort")

    def matches_label(self, observed: str) -> bool:
        wanted = {normalize_label(self.label)} | {
            normalize_label(a) for a in self.observed_label_aliases
        }
        return normalize_label(observed) in wanted


class Condition(StrictModel):
    id: Identifier
    label: str | None = None
    model_selection: Identifier = Field(description="Option id selected as the default.")
    message_override: Identifier | None = Field(
        default=None,
        description="Option id chosen for this message only (e.g. long-press on send).",
    )
    role: Literal["control", "test"] | None = None
    hypothesis_ids: list[Identifier] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _default_role(self) -> Condition:
        if self.role is None:
            self.role = "control" if self.message_override is None else "test"
        return self

    @property
    def has_override(self) -> bool:
        return self.message_override is not None

    def display(self) -> str:
        override = self.message_override or "none"
        return f"{self.id}: default={self.model_selection}, per-message={override}"


class ExactMatchScoring(StrictModel):
    method: Literal["exact_match"]
    extract: ExtractMode = "answer_line"
    answer_label: str = "ANSWER"
    accepted: list[str] = Field(min_length=1)
    normalize: list[NormalizeStep] = Field(default_factory=lambda: list(DEFAULT_NORMALIZE))


class RegexScoring(StrictModel):
    method: Literal["regex"]
    extract: ExtractMode = "answer_line"
    answer_label: str = "ANSWER"
    pattern: str = Field(min_length=1)
    ignore_case: bool = True

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:  # pragma: no cover - message is what matters
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return value


class NumericScoring(StrictModel):
    method: Literal["numeric"]
    extract: ExtractMode = "answer_line"
    answer_label: str = "ANSWER"
    expected: float
    abs_tolerance: float = Field(default=0.0, ge=0)


class ManualScoring(StrictModel):
    method: Literal["manual"]
    rubric: str = Field(min_length=1)
    max_points: float = Field(gt=0)
    pass_threshold: float = Field(gt=0, description="Points needed to count as task success.")

    @model_validator(mode="after")
    def _threshold_in_range(self) -> ManualScoring:
        if self.pass_threshold > self.max_points:
            raise ValueError("pass_threshold cannot exceed max_points")
        return self


Scoring = Annotated[
    ExactMatchScoring | RegexScoring | NumericScoring | ManualScoring,
    Field(discriminator="method"),
]


class Task(StrictModel):
    id: Identifier
    title: str = Field(min_length=1)
    category: str | None = None
    difficulty: Difficulty | None = Field(
        default=None, description="Author-assigned before data collection; not calibrated."
    )
    prompt: str = Field(min_length=1, description="Sent verbatim, identical across runs.")
    reference_answer: str | None = None
    scoring: Scoring
    source: str | None = None

    @property
    def is_automatic(self) -> bool:
        return self.scoring.method != "manual"


class Confounder(StrictModel):
    id: Identifier
    question: str = Field(min_length=1)
    allowed_values: list[str] = Field(min_length=2)

    @field_validator("allowed_values")
    @classmethod
    def _unique_values(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("allowed_values contains duplicates")
        return values


class ApiSettings(StrictModel):
    """How an official_api study calls the provider. Pre-registered like everything else."""

    provider: Literal["openai"] = Field(description="Only the OpenAI Responses API is implemented.")
    model: str = Field(min_length=1, description="Model id sent with every request.")
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound on output tokens per request, reasoning tokens included.",
    )


class Randomization(StrictModel):
    seed: int
    strategy: RandomizationStrategy = "blocked_by_repetition"


def _duplicates(items: list[str]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for item in items:
        if item in seen and item not in dups:
            dups.append(item)
        seen.add(item)
    return dups


class ExperimentDefinition(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: Identifier
    experiment_version: str = Field(pattern=r"^\d+\.\d+\.\d+([.-][A-Za-z0-9.-]+)?$")
    title: str = Field(min_length=1)
    service: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    execution_method: ExecutionMethod
    research_questions: list[str] = Field(min_length=1)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    documented_facts: list[DocumentedFact] = Field(default_factory=list)
    not_observable: list[str] = Field(default_factory=list)
    selection_options: list[SelectionOption] = Field(min_length=1)
    conditions: list[Condition] = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    repetitions: int = Field(ge=1, le=100)
    randomization: Randomization
    confounders: list[Confounder] = Field(default_factory=list)
    latency_protocol: str | None = None
    api: ApiSettings | None = Field(
        default=None, description="Required for official_api studies, not allowed otherwise."
    )
    demo: bool = Field(
        default=False, description="True only for synthetic demo studies created by `mblab demo`."
    )
    pilot: bool = Field(
        default=False,
        description=(
            "True for a pilot study made by `mblab pilot`: real observations used to check the "
            "protocol and the labels the client displays, never part of the results."
        ),
    )

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        data = _drop_unset(handler(self), "api")
        if not data.get("pilot"):
            data.pop("pilot", None)  # keeps the hash of non-pilot definitions unchanged
        return data

    def _execution_problems(self) -> list[str]:
        problems: list[str] = []
        if self.execution_method == "official_api":
            if self.api is None:
                problems.append("execution_method 'official_api' needs an `api` section")
            efforts: dict[str, str] = {}
            for opt in self.selection_options:
                if opt.reasoning_effort is None:
                    problems.append(
                        f"selection option '{opt.id}': official_api needs reasoning_effort"
                    )
                elif opt.reasoning_effort in efforts:
                    problems.append(
                        f"selection options '{efforts[opt.reasoning_effort]}' and '{opt.id}' "
                        f"send the same reasoning_effort '{opt.reasoning_effort}'"
                    )
                else:
                    efforts[opt.reasoning_effort] = opt.id
            for cond in self.conditions:
                if cond.message_override is not None:
                    problems.append(
                        f"condition '{cond.id}': message_override is a client UI setting and is "
                        "not valid for official_api"
                    )
        else:
            if self.api is not None:
                problems.append(
                    "the `api` section is only valid for execution_method 'official_api'"
                )
            for opt in self.selection_options:
                if opt.reasoning_effort is not None:
                    problems.append(
                        f"selection option '{opt.id}': reasoning_effort is only valid for "
                        "execution_method 'official_api'"
                    )
        return problems

    @model_validator(mode="after")
    def _check_references(self) -> ExperimentDefinition:
        problems: list[str] = self._execution_problems()
        for kind, ids in (
            ("selection option", [o.id for o in self.selection_options]),
            ("condition", [c.id for c in self.conditions]),
            ("task", [t.id for t in self.tasks]),
            ("hypothesis", [h.id for h in self.hypotheses]),
            ("confounder", [c.id for c in self.confounders]),
        ):
            for dup in _duplicates(ids):
                problems.append(f"duplicate {kind} id '{dup}'")

        option_ids = {o.id for o in self.selection_options}
        hypothesis_ids = {h.id for h in self.hypotheses}
        seen_settings: dict[tuple[str, str | None], str] = {}
        for cond in self.conditions:
            for field_name in ("model_selection", "message_override"):
                value = getattr(cond, field_name)
                if value is not None and value not in option_ids:
                    problems.append(
                        f"condition '{cond.id}': {field_name} '{value}' "
                        "is not a selection option id"
                    )
            for hid in cond.hypothesis_ids:
                if hid not in hypothesis_ids:
                    problems.append(f"condition '{cond.id}': unknown hypothesis id '{hid}'")
            key = (cond.model_selection, cond.message_override)
            if key in seen_settings:
                problems.append(
                    f"conditions '{seen_settings[key]}' and '{cond.id}' use identical settings"
                )
            seen_settings[key] = cond.id

        labels = [normalize_label(o.label) for o in self.selection_options]
        for dup in _duplicates(labels):
            problems.append(f"two selection options share the label '{dup}'")

        if problems:
            raise ValueError("; ".join(problems))

        # Pre-registered answers must satisfy their own pre-registered rule.
        from mblab.scoring import check_reference_answer  # local import: avoids a cycle

        for task in self.tasks:
            if task.is_automatic and not task.reference_answer:
                problems.append(f"task '{task.id}': automatic scoring needs a reference_answer")
            elif task.is_automatic:
                error = check_reference_answer(task)
                if error:
                    problems.append(f"task '{task.id}': {error}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # -- lookups -----------------------------------------------------------
    def option(self, option_id: str) -> SelectionOption:
        for opt in self.selection_options:
            if opt.id == option_id:
                return opt
        raise KeyError(option_id)

    def condition(self, condition_id: str) -> Condition:
        for cond in self.conditions:
            if cond.id == condition_id:
                return cond
        raise KeyError(condition_id)

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def option_label(self, option_id: str | None) -> str:
        if option_id is None:
            return "—"
        try:
            return self.option(option_id).label
        except KeyError:
            return option_id

    @property
    def planned_run_count(self) -> int:
        return len(self.conditions) * len(self.tasks) * self.repetitions

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def sha256(self) -> str:
        """Hash of the parsed definition (YAML comments and formatting do not count)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------


class PlannedRun(StrictModel):
    run_id: str
    sequence: int = Field(ge=1)
    block: int = Field(ge=1)
    experiment_id: str
    experiment_version: str
    platform: str
    execution_method: ExecutionMethod
    condition_id: str
    model_selection: str
    message_override: str | None
    task_id: str
    repetition: int = Field(ge=1)


class RunPlan(StrictModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    experiment_version: str
    definition_sha256: str
    definition_snapshot: dict[str, Any]
    seed: int
    strategy: RandomizationStrategy
    generated_at: AwareDatetime
    tool_version: str
    runs: list[PlannedRun]

    def by_run_id(self) -> dict[str, PlannedRun]:
        return {run.run_id: run for run in self.runs}


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class ApiReported(StrictModel):
    """Values the official API returned for one request, copied as returned.

    ``None`` means the API did not return the value. Nothing here is derived:
    for example ``total_tokens`` is not computed from the other counts.
    """

    response_id: str | None = None
    refusal: str | None = Field(
        default=None, description="Refusal content returned alongside (or instead of) text."
    )
    model: str | None = Field(default=None, description="Model id in the API response.")
    status: str | None = Field(default=None, description="Response status as returned.")
    reasoning_effort: str | None = Field(
        default=None, description="reasoning.effort echoed in the API response."
    )
    service_tier: str | None = None
    incomplete_reason: str | None = None
    error: str | None = Field(
        default=None, description="Error code and message of a failed response."
    )
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


#: Fields that describe what a client UI displayed; an API record has none of them.
CLIENT_DISPLAY_FIELDS = (
    "observed_model_label",
    "observed_label_location",
    "thinking_text",
    "thinking_seconds",
)


class RunRecord(StrictModel):
    """One observed run. Only what the experimenter saw or measured, or the API returned."""

    schema_version: Literal[1] = 1

    # Identity and condition — copied from the plan and checked against it.
    run_id: str = Field(min_length=1)
    experiment_id: str
    experiment_version: str
    definition_sha256: str | None = None
    platform: str
    client_version: str = Field(min_length=1, description="App/client version, or 'unknown'.")
    condition_id: str
    model_selection: str
    message_override: str | None = None
    task_id: str
    repetition: int = Field(ge=1)
    execution_method: ExecutionMethod
    timestamp: AwareDatetime = Field(description="When the prompt was sent, with a UTC offset.")
    data_origin: DataOrigin = "observed"

    # What was sent and what came back.
    prompt_text: str
    response_status: ResponseStatus
    response_text: str = ""

    # Information the client displayed, copied verbatim.
    observed_model_label: str | None = None
    observed_label_location: str | None = None
    thinking_indicator: ThinkingIndicator = "unknown"
    thinking_text: str | None = None
    thinking_seconds: float | None = Field(default=None, ge=0)

    # Timing, with the instrument that produced it.
    latency_seconds: float | None = Field(default=None, ge=0)
    latency_method: LatencyMethod = "not_measured"
    latency_confidence: Confidence | None = None

    # Protocol bookkeeping.
    new_conversation: bool = True
    confounders: dict[str, str] = Field(default_factory=dict)
    protocol_deviation: str | None = None
    notes: str | None = None
    recorded_by: str | None = None
    recorded_at: AwareDatetime | None = None

    # Reported by the official API (official_api records only).
    api_reported: ApiReported | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        return _drop_unset(handler(self), "api_reported")

    @field_validator(
        "observed_model_label",
        "observed_label_location",
        "thinking_text",
        "protocol_deviation",
        "notes",
        "recorded_by",
        "message_override",
        "definition_sha256",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _consistency(self) -> RunRecord:
        if self.latency_method == "not_measured":
            if self.latency_seconds is not None:
                raise ValueError("latency_seconds is set but latency_method is 'not_measured'")
            if self.latency_confidence is not None:
                raise ValueError("latency_confidence is set but latency was not measured")
        else:
            if self.latency_seconds is None:
                raise ValueError(
                    f"latency_method is '{self.latency_method}' but no latency_seconds"
                )
            if self.latency_confidence is None:
                self.latency_confidence = DEFAULT_LATENCY_CONFIDENCE[self.latency_method]  # type: ignore[assignment]
        allowed = LATENCY_METHODS_BY_EXECUTION[self.execution_method]
        if self.latency_method not in allowed:
            raise ValueError(
                f"latency_method '{self.latency_method}' is not valid for "
                f"execution_method '{self.execution_method}' (allowed: {sorted(allowed)})"
            )
        if self.execution_method == "official_api":
            shown = [f for f in CLIENT_DISPLAY_FIELDS if getattr(self, f) is not None]
            if self.thinking_indicator != "unknown":
                shown.append("thinking_indicator")
            if shown:
                raise ValueError(
                    f"{', '.join(shown)}: client-display fields are not valid for "
                    "execution_method 'official_api'"
                )
        elif self.api_reported is not None:
            raise ValueError("api_reported is only valid for execution_method 'official_api'")
        if self.thinking_indicator != "shown" and (
            self.thinking_text is not None or self.thinking_seconds is not None
        ):
            raise ValueError("thinking_text/thinking_seconds need thinking_indicator='shown'")
        # An empty "completed" response is a typing mistake in a manual record, but a
        # returned value in an API record (it is then scored as a format failure).
        if (
            self.response_status == "completed"
            and not self.response_text.strip()
            and self.api_reported is None
        ):
            raise ValueError("response_status is 'completed' but response_text is empty")
        return self

    @property
    def excluded(self) -> bool:
        """Records with a protocol deviation are excluded from the primary analysis."""
        return self.protocol_deviation is not None


class ManualGrade(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    points: float = Field(ge=0)
    max_points: float = Field(gt=0)
    grader_id: str = Field(min_length=1)
    blind: bool
    graded_at: AwareDatetime
    notes: str | None = None

    @model_validator(mode="after")
    def _points_in_range(self) -> ManualGrade:
        if self.points > self.max_points:
            raise ValueError("points cannot exceed max_points")
        return self
