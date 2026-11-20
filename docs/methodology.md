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
  not count) and a full snapshot in `plan.json`. Each record carries the hash of the
  `experiment.yaml` in force when it was recorded, so a record collected after an edit
  shows it. Once a run is recorded, `mblab plan --force` refuses to replace `plan.json`:
  the plan holds the pre-registered order and snapshot that records are checked against.
- `mblab validate` diffs the current definition against the snapshot and names every
  changed path (for example `tasks[T5].scoring.expected`), and lists the records
  collected under a changed definition. Prompts are compared with the prompt in the
  snapshot, not with the current file. Reports name the changed paths among the
  limitations as a possible post-hoc change; every run is scored with the current file.
- Records must belong to the plan. Imports refuse a record whose run is not planned or
  whose experiment, condition, selections, task, or repetition differ from its plan
  entry, and analysis leaves out any such record already in `records.jsonl`, as well as
  every copy of a run recorded more than once. Reports open with a banner while
  `mblab validate` finds errors.
- A definition is rejected unless every automatically scored task has a reference answer
  that passes its own scoring rule, and its `answer_examples` (if any) score as declared.
- Label aliases (`observed_label_aliases`) are fixed in advance, so deciding which
  displayed text "counts as Pro" cannot be done after seeing the data. They are chosen
  from a **pilot study** (`mblab pilot`): a separate directory with its own
  `experiment_id`, `pilot: true`, run ids that start with `pilot:`, and the main seed + 1
  (so it does not rehearse the main order). Its records are labelled PILOT everywhere and
  cannot be imported into the main study. `mblab labels` lists the texts
  the pilot displayed, verbatim; on a non-pilot study it warns that changing aliases now
  would be post-hoc.

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
are tolerated). If no usable line exists, or the line cannot be read by the task's rule
(no number, two different dates, a name missing from a sequence), the response is a
**format failure**: it counts as incorrect in accuracy and is also counted separately,
because a formatting habit is not the same as a wrong answer.

| Method | Rule |
|---|---|
| `exact_match` | Normalized answer equals one of `accepted` (normalization steps are declared per task) |
| `regex` | `pattern` matches the answer (anchors in the pattern decide how strict) |
| `numeric` | Exactly one distinct number in the answer, within `abs_tolerance` of `expected`; with a list `expected: [37, 13]`, exactly those numbers in that order (a different count is unreadable) |
| `date` | Exactly one `YYYY-MM-DD` date in the answer, equal to `expected`; other text such as a weekday is ignored, a date in another format or two different dates are unreadable |
| `sequence` | Each item of `expected` named exactly once (whole words, any case, any separators such as commas, "and", or "1."), in that order; the same items in another order are wrong, a missing or repeated item is unreadable |
| `manual` | Points from human graders against a rubric; pass if the mean ≥ `pass_threshold` |

A rule must not turn a model's wording habit into an accuracy difference: a correct
answer written another way ("Ben, Cho, Eun, Dev, and Ana", "It prints (37, 13)",
"2027-02-27 (Saturday, 27 February 2027)") has to score correct. Each task can fix
`answer_examples` in advance — answers that must score `correct`, `incorrect`, and
`unreadable` — and the definition is rejected unless the rule scores every one of them
that way (the bundled tasks list examples for all three). Distinct answers per task
are counted from what the rule read (so "265" and "265 minutes" are one answer), and
format failures are not counted as answers.

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

Let *gap* be the difference between the two references and *f* the pre-registered
`analysis.equivalence_fraction` (default 0.5). The **equivalence margin** is ±*f* × |gap|.

| Step | Check | Result |
|---|---|---|
| 1 | Any group below the minimum size | **insufficient data** |
| 2 | The two references are not distinguishable (interval of their difference includes 0) | **references not distinguishable** — the metric cannot say which selection applied |
| 3 | The interval for (condition − per-message reference) lies inside the margin | **consistent with the per-message selection** |
| 4 | The interval for (condition − default-only reference) lies inside the margin | **consistent with the default selection** |
| 5 | Differs from both references, on opposite sides | **between the two references** — e.g. the per-message selection applied in some runs only |
| 6 | Differs from both references, on the same side | **differs from both references** — look for confounders |
| 7 | Anything else | **inconclusive** — the intervals are too wide to place the condition |

"Not distinguishable from a reference" is never read as "consistent with it": the
interval has to fit inside the margin. With *f* = 0.5, "consistent with the per-message
selection" means the condition is, with the stated confidence, closer to the per-message
reference than to the default-only one; a stricter *f* (e.g. 0.25) asks for "behaves
like it" and needs more runs.

Every assessment also reports the condition's **position** *λ* = (condition − default-only
reference) / (per-message reference − default-only reference), with a percentile
bootstrap interval: 0 is the default-only reference, 1 the per-message reference, 0.5
halfway, and values outside 0..1 lie beyond a reference.

How the rule behaves, from 2,000 simulated studies per row (accuracy, 18 runs per
condition, references at 50% and 100%):

| Condition truly at | Previous rule: called "consistent" | *f* = 0.5 | *f* = 0.25 |
|---|---|---|---|
| the per-message reference | 100% (per-message) | 88% (per-message) | 4% (per-message) |
| halfway | 75% (either reference) | 3% | 0% |
| 30% of the way (a partial override) | 87% (either reference) | 1% | 0% |
| the default-only reference | 94% (default) | 2% (default) | 0% |

With references at 50% and 90%, or with fewer clean differences, most results at this
sample size are *inconclusive*; that is the honest reading of 18 runs per condition, and
the position column still shows where each condition lies.

When *X* and *Y* are the same (the "stacked" case, e.g. Extra High twice), there is one
reference, and the result is **no difference detected** (which is not evidence of
equivalence) or **difference detected** (cause not identified).

Displayed labels are summarised separately, as counts, and are not mixed into this
table. If a hypothesis declares `predicted_pattern`, the report counts how many
assessments that place a condition (steps 3–6) show that pattern — a description of the
sample, not a test of the hypothesis, and the metrics are related, so the assessments
are not independent. A `no_difference` prediction is never counted as shown, since not
detecting a difference is not equivalence.

The confidence level, the minimum group size (at least 5), and *f* are fixed in
`experiment.yaml` under `analysis:` before data collection. `--confidence` and `--min-n`
on `mblab analyze` / `mblab report` can override them, but the output then says it is
not the pre-registered analysis, and the report carries a banner and a limitation.

## 10. Validation

`mblab validate` reports errors (the data is wrong), warnings (protocol problems or
threats to validity), and notes. It checks: plan integrity; records against the plan
(unknown runs, duplicates, changed conditions); prompt wording; new-conversation flag;
confounder keys and values; client-version changes; latency instruments; synthetic and
observed records in one study; definition changes after collection; order deviations;
identical long responses in different runs (copy-paste mistakes); grade consistency; and
possible personal data in free text (the matched value is never printed).

Unreadable lines in `records.jsonl`, `grades.jsonl` or `api-unrecorded.jsonl` (bad JSON,
bad UTF-8, wrong fields) are errors for that line only. Commands that rewrite a file keep
such lines byte for byte, and keep the previous file as `.bak`, instead of dropping what
they cannot parse. All writes happen under one study-wide lock, after reading the files
again, so concurrent commands cannot lose or duplicate a run.

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
with `reasoning.effort`, through the official `openai` Python SDK (the `api` extra:
`pip install -e ".[api]"` in a clone). The bundled template `openai-reasoning-effort` compares
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
incomplete. Rate limits, server errors, and failures to connect at all are retried. A
read timeout or a dropped connection can come after the API accepted (and billed) the
request, so such a run is not retried automatically: it is logged in
`api-unrecorded.jsonl` and held back like a response that could not be recorded.
The tool does not price anything — prices change — but reports list token totals per
condition so cost can be computed from the provider's price list. Totals cover every recorded
response, excluded runs included, because all of them were billed. A billed response
that cannot be turned into a record (for example an unexpected status) is listed in
`api-unrecorded.jsonl` with its id and token counts, reported as a limitation, and its
run is held back from resending unless `--resend-unrecorded` is given. Ctrl+C waits
until a received response is written. Only one command writes to a study at a time,
and `run-api` reads the study again under that lock, so a run recorded meanwhile is
skipped instead of being sent (and billed) twice.

**What it cannot show.** API-reported values are the provider's figures, recorded as
returned, not independently verified. How the provider applies an effort setting
internally, the hidden reasoning itself, and whether the weights behind a model id
changed during the study are not observable. API results describe API access with one
account and model id; they are not evidence about a consumer app such as ChatGPT.

**UI vs. API.** A study uses one execution method, and records of the other method are
rejected, so the two are never pooled. To relate them, run a UI study and an API study
on the same tasks and compare their reports side by side, with each report's own
limitations; response times from different instruments are never compared directly.
