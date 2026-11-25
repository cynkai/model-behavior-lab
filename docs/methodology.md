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

Randomization only protects against time trends if the runs are done in that order.
`mblab validate` measures it over every pair of recorded runs: the share done in the
planned order (pairs with the same timestamp are not counted). Anything below 100% is
reported, below 90% as a warning, and the report states the share next to "randomized
order" and lists it among the limitations.

**Confounders.** Each confounder in `experiment.yaml` can declare `recommended` values
(memory `disabled`, tools `none`, notices `none`) and `per_run: true` for what can change
with every response. A value outside the recommended ones is listed among the
limitations with its count per condition — also when every run has it, since a constant
setting such as memory being on still shapes every response. Per-run confounders are
never pre-filled from the previous run (in `mblab record` and the dashboard form), so
pressing Enter cannot record what the last response showed. A **sensitivity check**
leaves out the runs with a flagged per-run value (a usage-limit notice, a visible tool)
and runs the per-message assessments again; the report lists every verdict that
changes, since a notice concentrated in one condition is the main alternative
explanation for a pattern.

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

### Manual inter-rater agreement

`mblab grade agreement -s STUDY`, the Markdown report, and the dashboard's task page
show descriptive agreement separately for each manual task in definition order.
`mblab analyze -s STUDY --json FILE` includes a `grading_agreement` list with `task_id`,
`eligible_runs`, `graded_runs`, `multi_rated_runs`, `pairs`, `exact_matches`, and
`decision_matches`. The six counts are integers. Grader identities and raw text are
absent from this summary.

Eligible runs are completed responses included in the analysis. Unplaced records,
duplicate run records, protocol deviations, errors, refusals, and incomplete responses
are excluded. A usable grade must parse successfully, belong to the response, and have
the task's `max_points`. Grader IDs are case-sensitive and ignore surrounding whitespace.
All entries for a repeated run and grader are discarded, even
when their points agree or one uses the wrong scale. Both blind and unblinded grades
are eligible. The scoring mean uses this same grade filter.

`graded_runs` counts eligible responses with at least one usable grade and
`multi_rated_runs` counts those with at least two. Every unordered pair of usable
graders on the same response contributes once, giving `n * (n - 1) / 2` pairs for
`n` graders. Counts are summed within each task, so responses with more graders carry
more weight. Comparisons never cross responses or tasks. Exact agreement requires
equal points. Decision agreement requires both scores to be at least `pass_threshold`
or both to be below it.

For one response scored 1, 2, and 3 with a threshold of 2, there are three pairs.
None has equal points, and the pair scoring 2 and 3 agrees on passing. Exact agreement
is 0/3 (0%) and decision agreement is 1/3 (shown as 33%). Each percentage uses all pairs for
that task as its denominator. Zero or one usable grader contributes no pairs. Tasks
with no pairs retain their coverage counts and show an em dash for agreement.
Automatic-only studies omit this summary. These metrics are descriptive and cannot
establish grading correctness or reliability.

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
- **Differences of proportions:** standardized over tasks, with a stratified Newcombe
  interval. Every condition is planned with the same runs per task, so a comparison
  averages the per-task differences over the tasks scored in both conditions (each task
  counts equally). Pooling the scored responses instead would compare different mixes of
  tasks whenever runs are missing, excluded, or not completed unevenly — a condition
  that fails hard tasks more often would look more accurate. Each condition's
  standardized proportion gets a Wilson interval with the effective sample size
  1 / Σ (w² / nₖ) (w = 1 / number of tasks), combined as in Newcombe's hybrid score
  interval (method 10; cf. Yan & Su, 2010, for the stratified form). With the same
  number of runs on every task this is exactly the unstratified Newcombe interval, which
  the tests check against statsmodels and Newcombe's (1998) worked example. In 4,000
  simulated studies with equal per-task accuracy but one condition losing most
  hard-task runs, the pooled interval reported a difference 7.8% of the time at 3 runs
  per task and 23% at 10; the standardized one 1.8% and 1.9%. Accuracy and the task
  success rate (errors and refusals count as failures) are both compared, and the
  limitations list each condition's not-completed runs when they differ.
- **Differences of medians:** percentile bootstrap, 2,000 resamples, seeded with the plan
  seed; reproducible across Python and NumPy versions.
- **Minimum size:** groups with fewer than `analysis.min_n` observations (5 by default,
  never less) are not compared; `--min-n` overrides it and marks the analysis exploratory.
- **Assumptions, stated in every report:** responses are treated as independent, and
  intervals are conditional on the fixed task set — they say nothing about other tasks.
  Pairwise differences are descriptive, not significance tests; with many comparisons
  some intervals exclude 0 by chance.

### Cross-study manual UI comparison

`mblab compare LEFT RIGHT [--json]` reads two different study directories and displays separate
condition summaries. `--json` prints structured JSON to stdout. The command writes no
files and makes no service requests.

Both definitions must use `manual_ui` for the same service. The platforms may match or
differ. Both studies need valid plans, definitions unchanged since planning, and records
that pass validation. Pilot studies are rejected. Mixed synthetic and observed records
within a study are rejected, as is a synthetic study paired with an observed study.
Two synthetic studies can be compared with a synthetic-data warning.

Compatibility requires the exact same task ID set and, for each task, the same prompt,
scoring rule, and reference answer. It also requires the exact same condition ID set
and, for each condition, the same default and per-message selection option IDs and
their labels. A missing per-message selection must match a missing selection.
Task and condition ordering, repetition counts, experiment metadata, and collection
periods may differ. Compatibility does not establish that the service's hidden routing
or deployed model stayed constant.

Each condition retains two independent summaries, one per study. Accuracy is correct
responses divided by scored completed responses, including format failures as incorrect.
Task success is correct responses divided by included recorded runs with an outcome,
including incomplete, refused, and error responses as failures. Completed responses
awaiting manual grading are excluded from both denominators. Protocol deviations are
also excluded from both rates. Unrecorded runs contribute to planned counts only.
Scored task coverage shows how many tasks contribute to accuracy, alongside the planned
and recorded counts. JSON also includes scored run counts for each task. Client-version
counts and collection timestamps in UTC use included records. Validation warnings are shown as codes
with counts. These details provide context for differences in coverage and collection
conditions.

These are descriptive summaries of separate studies, with records and rates kept
separate for each sample. Task coverage, collection period, client version, account,
usage limits, and other confounders can vary and affect the rates even with matching
definitions. Those factors prevent attributing a difference to the platform alone.
Server-side model identity remains unobservable in a manual UI study.

To collect another platform, create a new directory and copy only `experiment.yaml`.
Change `platform` before running `mblab plan`, then collect fresh observations there.
Keep the original study's plan, records, and grades in their original directory.

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

**File formats.** `experiment.yaml`, `plan.json`, and each record and grade state their
format in `schema_version` (format 1 so far). Optional fields added by later versions are
left out of a file while unset, so a definition that does not use them keeps its hash and
a study that does not use them stays readable by earlier versions; earlier versions
refuse a field they do not know rather than ignore it. A version that reads a newer
`schema_version` refuses the file (or, in `records.jsonl` and `grades.jsonl`, reports the
line and keeps it untouched) with a message that says to update; an unknown field or
scoring method is reported with the same advice, or to check the spelling.

**Checking a shared report.** The report's *Reproduction* section gives the SHA-256 of
every file it was made from (including `api-unrecorded.jsonl` in official-API studies,
whose billed-but-unrecorded responses are counted among the limitations), the version
and git commit of the code that made it, and the commands that repeat it, with any
analysis options that differ from the pre-registered ones. The commit comes from the git
checkout the code runs from, or from pip's record of a `git+https://…` install; when
neither is available the report says the commit is not recorded, and when the checkout
has uncommitted changes it says the exact code cannot be installed.
`mblab report --check FILE` makes the report again and compares it with `FILE`, ignoring
only the generation time and the lines naming the code; it writes nothing and exits with
status 1 when anything else differs. Without `--check`, `mblab report` does not replace an
existing report whose content differs unless `--force` is given, so following a
reproduction recipe cannot overwrite the report being checked.

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
Reports list token totals per condition, including responses excluded from outcome
analysis. `mblab cost` estimates token costs with a supplied rate card under the
coverage rules below. A billed response
that cannot be turned into a record (for example an unexpected status) is listed in
`api-unrecorded.jsonl` with its id and token counts, reported as a limitation, and its
run is held back from resending unless `--resend-unrecorded` is given. Ctrl+C waits
until a received response is written. Only one command writes to a study at a time,
and `run-api` reads the study again under that lock, so a run recorded meanwhile is
skipped instead of being sent (and billed) twice.

### Token cost estimates

`mblab cost -s STUDY --rates rates.json [--json]` is read-only and accepts official-API
studies. It makes no provider API calls and writes no files. `--json` prints structured
JSON to stdout. Prices are supplied as nonnegative decimal strings per million tokens
in USD, with up to 12 digits on each side of the decimal point. The format is illustrated in the
[README](../README.md#token-cost-estimates).

A rate row matches the exact reported model and service tier. Requested settings do
not fill missing reported values. One rate schedule applies across the study's
collection dates. Results include the SHA-256 of the raw rate-card bytes so the supplied
schedule can be identified. No current provider price is fetched or assumed.

For input tokens I, cached input tokens C, output tokens O, and their rates rI, rC, rO,
the estimate is `((I - C) × rI + C × rC + O × rO) / 1,000,000`, calculated with exact
arithmetic. Reasoning tokens are already part of O. For the fictitious USD rates 2,
0.5, and 6, input 1,000 including 200 cached and output 100 yields 0.0023 USD.

Pricing requires usable input and output counts and a matching rate row. Cached input
must be between zero and input tokens. A missing cached-input count permits calculation
when the two input rates are equal or input tokens are zero. Other missing or invalid
usage remains unknown, as do missing reported model/tier values and unmatched rates.
A reported total must equal input plus output, and reasoning tokens must fit within output.
The output lists missing model/tier rate pairs and the number of affected records.
A subtotal is null if none of its runs could be priced and is zero if measured usage
at the supplied rates costs zero.

The requested effort for each condition comes from the study definition and is checked
against the frozen plan. Restore the planned settings if an edit has reassigned a
condition to a different effort. Coverage follows records that can be placed in the plan.
Eligible records include protocol deviations and error responses even when outcome analysis excludes
them. Repeating a nonblank response ID across placed records excludes each copy. Held
responses in `api-unrecorded.jsonl`, unreadable lines, unplaced records, and records
outside the official-API scope are counted and never priced. Their costs cannot be
recovered from the provisional subtotal.

The result reports priced/recorded counts and provisional subtotals by condition and
for the study. Coverage gaps and their reasons remain visible. Underlying charges
beyond these token rates, including tool fees, taxes, discounts, and subscription
charges, are not estimated.

**What it cannot show.** API-reported values are the provider's figures, recorded as
returned, not independently verified. How the provider applies an effort setting
internally, the hidden reasoning itself, and whether the weights behind a model id
changed during the study are not observable. API results describe API access with one
account and model id; they are not evidence about a consumer app such as ChatGPT.

**UI vs. API.** A study uses one execution method, and records of the other method are
rejected, so the two are never pooled. To relate them, run a UI study and an API study
on the same tasks and compare their reports side by side, with each report's own
limitations; response times from different instruments are never compared directly.
