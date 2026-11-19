# Methodology

The question this project is built around is not "which setting is best?" but:

> **What did we actually measure, and how far does the result let us go?**

This document defines what Model Behavior Lab records, how it scores and summarises the
data, which conclusions the reports are allowed to draw, and where the method stops.
The step-by-step procedure for running the first experiment is in
[experiment-protocol.md](experiment-protocol.md).

## 1. Two experimental domains

| | Manual UI experiment (`manual_ui`) | Official API measurement (`official_api`) |
|---|---|---|
| Who sends the prompt | A person, in the official client | A program, through the provider's official API |
| What the setting means | The options the client offers (labels as displayed) | Documented API parameters |
| Response time | Stopwatch or screen recording, low precision | Client-side timer, high precision |
| What is recorded besides the response | What the client displayed (label, thinking indicator) | What the API returned (model id, status, token usage) |
| Status | Implemented | Implemented for the OpenAI Responses API (`mblab run-api`, §12) |

An experiment uses exactly one execution method. Records whose method or latency
instrument does not match are rejected, and response times from different instruments
are never pooled. **A UI result and an API result are never presented as if they came
from the same environment.**

## 2. What a manual UI experiment can and cannot observe

| Quantity | How it is recorded | Notes |
|---|---|---|
| Prompt sent | Copied from the definition; checked against it | Wording must be identical across runs |
| Response text | Pasted by the experimenter | The only input to automatic scoring |
| Displayed model label | Verbatim text + where it was shown | What the client displayed, not a server record |
| Thinking indicator | shown / not shown / unknown, plus verbatim text ("Thought for 1m 12s") | The parsed duration is the client's own display |
| Response time | Seconds + instrument (`manual_stopwatch`, `screen_recording`) + confidence | Includes reaction, network, rendering time |
| Client version, platform, time sent | Typed in | Version changes during a study are flagged |
| Confounders | Values from a declared list (memory, custom instructions, tools, usage-limit notices, network) | Undeclared keys and values are rejected |

**Not observable**, and therefore never recorded or inferred: the server-side model
identifier, routing or fallback decisions, hidden reasoning tokens or compute, and
whether the account is in a server-side experiment. Each experiment lists these under
`not_observable`, and every report repeats them. The record schema has no fields for
them, on purpose.

## 3. Four kinds of statements

| Category | Source | Example |
|---|---|---|
| Documented fact | `documented_facts` in the definition, each with a URL and an access date | (none in the template — added only after checking official documentation) |
| Hypothesis | `hypotheses` in the definition, written before data collection | "H1: the response reflects the per-message selection" |
| Observation | Computed from records | "Condition C: *k* of *n* scored responses correct (*p*%, 95% CI *low*–*high*%)" |
| Not verified | `not_observable`, plus the standing statement below | "Which model actually produced each response" |

Every report ends with a *claims ledger* that lists each statement with its category and
basis. Generated text follows fixed wording rules:

- Allowed: *consistent with*, *not distinguishable*, *differs from*, *inconclusive*,
  *insufficient data*, *not observable*, *does not establish*.
- Not allowed: *proves*, *confirms*, *conclusive*, *definitely*, *guarantees*. A test
  (`tests/test_analysis.py`, `tests/test_report.py`) fails if generated text uses them.
- A behavioural pattern is never described as evidence about internal routing.

## 4. Pre-registration

- The experiment definition (`experiment.yaml`) holds the conditions, tasks, the exact
  prompts, scoring rules, reference answers, label aliases, confounders, and hypotheses.
- `mblab plan` stores the SHA-256 of the *parsed* definition (comments and formatting do
  not count) and a full snapshot in `plan.json`. Each record carries the hash it was
  collected under.
- `mblab validate` diffs the current definition against the snapshot and names every
  changed path (for example `tasks[T5].scoring.expected`). Reports flag a definition
  that changed after data collection began as a possible post-hoc change.
- A definition is rejected unless every automatically scored task has a reference answer
  that passes its own scoring rule.
- Label aliases (`observed_label_aliases`) are fixed in advance, so deciding which
  displayed text "counts as Pro" cannot be done after seeing the data.

## 5. Randomization

All `condition × task × repetition` cells are run once. The default strategy,
`blocked_by_repetition`, shuffles each repetition block separately, so slow drifts (time
of day, server load, an app update) affect all conditions alike. `complete` shuffles all
runs together. The shuffle is a Fisher–Yates driven only by `random.Random(seed).random()`,
whose output Python guarantees across versions, so the same seed gives the same order
everywhere.

## 6. Scoring

**Answer extraction.** Prompts ask for a final line `ANSWER: <answer>`. The last such
line is used (markdown decoration, a full-width colon, and a value on the following line
are tolerated). If no usable line exists, the response is a **format failure**: it counts
as incorrect in accuracy and is also counted separately, because a formatting habit is
not the same as a wrong answer.

| Method | Rule |
|---|---|
| `exact_match` | Normalized answer equals one of `accepted` (normalization steps are declared per task) |
| `regex` | `pattern` matches the answer (anchors in the pattern decide how strict) |
| `numeric` | Exactly one distinct number in the answer, within `abs_tolerance` of `expected` |
| `manual` | Points from human graders against a rubric; pass if the mean ≥ `pass_threshold` |

**Non-completed responses** (error, refusal, incomplete) are not scorable: they are left
out of accuracy and count as failures in task success rate.

**Blind grading.** `mblab grade export` writes a shuffled sheet with neutral ids
(G001…) containing only the task, rubric, prompt, and response — no condition, run id,
timestamp, label, or timing. The mapping is in a separate key file. Several graders can
grade the same response; the mean is used and disagreements are counted. Blinding hides
metadata, not style: response length or tone may still hint at the condition.

## 7. Metrics

| Metric | Definition |
|---|---|
| Accuracy | correct ÷ scored responses (scored = completed and judged, including format failures) |
| Task success rate | correct ÷ included runs (errors and refusals count as failures; runs awaiting a manual grade are left out) |
| Format failures | completed responses without a usable answer |
| SD across repetitions | standard deviation of per-repetition accuracy within a condition |
| Repetition consistency | per condition × task: same outcome in every repetition? distinct answers, share of the most common answer |
| Response time | n, median, quartiles, min, max — per condition **and per instrument** |
| Thinking indicator | share of runs with a known indicator state where it was shown; median displayed duration |
| Displayed-label class | per run: matches the per-message selection / matches the default / matches both / names another option / not a pre-registered label / no label |

Runs with a recorded `protocol_deviation` are excluded from all of these and listed
separately. Missing runs are reported as missing, never imputed.

## 8. Statistics

- **Proportions:** Wilson score interval (behaves sensibly at small n and at 0% / 100%).
- **Differences of proportions:** Newcombe's hybrid score interval (method 10). Both are
  checked against statsmodels and against Newcombe's (1998) worked example in the tests.
- **Differences of medians:** percentile bootstrap, 2,000 resamples, seeded with the plan
  seed; reproducible across Python and NumPy versions.
- **Minimum size:** groups with fewer than 5 observations are not compared
  (`--min-n` changes this).
- **Assumptions, stated in every report:** responses are treated as independent, and
  intervals are conditional on the fixed task set — they say nothing about other tasks.
  Pairwise differences are descriptive, not significance tests; with many comparisons
  some intervals exclude 0 by chance.

## 9. Per-message selection vs. default: the decision table

For a test condition with default *X* and per-message selection *Y*, the reference
conditions are *X with no per-message selection* (default-only reference) and *Y with no
per-message selection* (per-message reference). For each metric (accuracy, thinking
indicator rate, median response time, median displayed thinking time):

| Step | Check | Result |
|---|---|---|
| 1 | Any group below the minimum size | **insufficient data** |
| 2 | The two references are not distinguishable (interval of their difference includes 0) | **references not distinguishable** — the metric cannot say which selection applied |
| 3 | Differs from the default-only reference, not from the per-message reference | **consistent with the per-message selection** |
| 4 | Differs from the per-message reference, not from the default-only reference | **consistent with the default selection** |
| 5 | Differs from both | **differs from both references** — look for confounders |
| 6 | Differs from neither | **inconclusive** |

When *X* and *Y* are the same (the "stacked" case, e.g. Extra High twice), there is one
reference, and the result is **no difference detected** (which is not evidence of
equivalence) or **difference detected** (cause not identified).

Displayed labels are summarised separately, as counts, and are not mixed into this
table. If a hypothesis declares `predicted_pattern`, the report counts how many usable
assessments show that pattern — a description of the sample, not a test of the
hypothesis.

## 10. Validation

`mblab validate` reports errors (the data is wrong), warnings (protocol problems or
threats to validity), and notes. It checks: plan integrity; records against the plan
(unknown runs, duplicates, changed conditions); prompt wording; new-conversation flag;
confounder keys and values; client-version changes; latency instruments; synthetic and
observed records in one study; definition changes after collection; order deviations;
identical long responses in different runs (copy-paste mistakes); grade consistency; and
possible personal data in free text (the matched value is never printed).

## 11. Threats to validity

- **Construct:** a displayed label is not ground truth; accuracy and time are proxies
  that a model change *could* move, not measurements of the model itself.
- **Internal:** usage limits and automatic fallback, server load and time of day, app
  updates, memory or custom instructions, tool use, and the experimenter's expectations
  when copying labels (mitigated by verbatim recording and pre-registered aliases).
- **External:** one account, device, region, and period; results may not transfer to
  other plans, platforms, or next month's service.
- **Statistical:** small samples; six tasks; correlated responses within a task.

## 12. Official-API studies (`official_api`)

An official-API study sends the same kind of pre-registered tasks through a provider's
official API instead of a client app. The implemented case is the OpenAI Responses API
with `reasoning.effort`, through the official `openai` Python SDK (`pip install
"model-behavior-lab[api]"`). The bundled template `openai-reasoning-effort` compares
low / medium / high effort of one model on the six tasks of the UI experiment.

**Definition.** `execution_method: official_api` requires an `api` section (`provider`,
`model`, optional `max_output_tokens`) and a `reasoning_effort` on every selection
option; one condition per option, and no `message_override` (every request carries
exactly one setting). Two options may not send the same effort. These fields are
rejected in `manual_ui` definitions, and they are left out of the definition hash when
unset, so plans of UI studies made before they existed stay valid.

**What is sent.** `model`, the task prompt as `input`, `reasoning: {effort}`,
`max_output_tokens` if set, and `store: false` — nothing else (no instructions, tools,
or conversation state). `mblab run-api --dry-run` prints these requests without sending
anything or needing a key. `run-api` refuses to send if `experiment.yaml` changed after
the plan, if the study is not `official_api`, or if `records.jsonl` has unreadable lines.

**What is recorded.** Each record copies its identity from the plan, like a UI record,
with `latency_method: client_timer`, `client_version` set to the SDK version, and an
`api_reported` object holding what the response contained: response id, model id,
status, echoed reasoning effort, service tier, incomplete reason, error, and token usage
(input, cached input, output, reasoning, total). A value the API did not return stays
`null`; nothing is computed from other values. Status mapping: `completed` → completed
(or *refused* if the only content is a refusal), `incomplete` → incomplete,
`failed` → error; any other status stops the run without a record. A completed response
with empty text is recorded as returned and scored as a format failure.
`api_reported` is rejected in UI records, and client-display fields (displayed label,
thinking indicator) are rejected in API records.

| Quantity | Source | Notes |
|---|---|---|
| Model id | `response.model` | Often a dated snapshot of the requested name; several ids in one study are flagged |
| Echoed effort | `response.reasoning.effort` | A mismatch with the requested effort is flagged |
| Reasoning tokens | `usage.output_tokens_details.reasoning_tokens` | Part of output tokens; missing counts are not treated as zero |
| Response time | Client-side timer around the SDK call | Includes network and provider queueing; final attempt only |

**Execution.** Runs are sent one at a time in plan order, at least `--min-interval`
seconds apart (default 1 s), with the SDK's own retries switched off. Rate-limit,
connection, and server errors are retried twice with exponential backoff; the record
then notes the failed attempts, and its timestamp and latency refer to the final attempt.
Other errors (rejected parameters, authentication, `insufficient_quota`) stop the run at
once. A failed request writes no record, each successful one is appended immediately,
and runs with a record are skipped — so running the command again resumes. The key is
read from `OPENAI_API_KEY` only, passed to the SDK, never printed or written, and
scrubbed from error messages. If `OPENAI_BASE_URL` is set, the CLI warns that requests
go to that host.

**Analysis.** The same metrics as §7 (accuracy, success rate, response time), plus per
condition: runs with API values, reported model ids, echoed-effort mismatches,
incomplete reasons, reasoning tokens (median and quartiles), and input/output token
totals. Pairwise differences (§8) include median reasoning tokens. There is no
per-message decision table (§9) and no displayed-label analysis.

**Cost.** Every request is billed to the key's account for input and output tokens,
with reasoning tokens billed as output. Higher effort tends to use more tokens; a
`max_output_tokens` cap bounds each request but responses that reach it are recorded as
incomplete. A request that times out on the client may still be billed and is retried.
The tool does not price anything — prices change — but reports list token totals per
condition so cost can be computed from the provider's price list.

**What it cannot show.** API-reported values are the provider's figures, recorded as
returned, not independently verified. How the provider applies an effort setting
internally, the hidden reasoning itself, and whether the weights behind a model id
changed during the study are not observable. API results describe API access with one
account and model id; they are not evidence about a consumer app such as ChatGPT.

**UI vs. API.** A study uses one execution method, and records of the other method are
rejected, so the two are never pooled. To relate them, run a UI study and an API study
on the same tasks and compare their reports side by side, with each report's own
limitations; response times from different instruments are never compared directly.
