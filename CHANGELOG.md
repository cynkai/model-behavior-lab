# Changelog

## Unreleased

### Reports you can check

- Reports name the code that made them: the version and the git commit, read from the
  checkout the code runs from or from pip's record of a `git+https://…` install (a
  working copy with uncommitted changes in `src/mblab` is called that; a copy installed
  any other way says the commit is not recorded). `mblab --version` and the dashboard
  show the same. Before, every commit called itself 0.1.0.
- The reproduction commands install that commit (or the version's release tag when the
  commit is unknown) instead of whatever the default branch holds, and repeat the
  `--confidence` / `--min-n` options a report was made with.
- `mblab report --check FILE` makes the report again and compares it with `FILE` without
  writing anything; only the generation time and the lines naming the code are ignored,
  and it exits with status 1 if anything else differs. `mblab report` no longer replaces
  an existing report whose content differs unless `--force` is given (an identical one is
  left as it is), so following the reproduction steps cannot overwrite the report being
  checked.
- Official-API reports list the SHA-256 of `api-unrecorded.jsonl`, which changes their
  limitations, and the sharing instructions ask for it. The reproduction table also says
  which version made the plan.

### Task-standardized comparisons

- Differences of proportions (accuracy, the thinking-indicator rate, and the new task
  success rate comparison) are standardized over the tasks scored in both conditions,
  each task counting equally, with a stratified Newcombe interval. Before, scored
  responses were pooled, so runs missing or not completed unevenly across tasks changed
  the task mix: in simulated studies with equal per-task accuracy the pooled comparison
  reported a difference up to 23% of the time, the standardized one under 2%. With the
  same runs on every task the numbers are unchanged.
- The pairwise comparisons include the task success rate (errors and refusals count as
  failures), and the limitations list each condition's not-completed runs when they
  differ.
- The position λ for proportions is computed the same way (per-task means, resampled
  within tasks); when every task is all-or-nothing it is shown without an interval.

### Confounders and run order

- Confounders can declare `per_run: true` (never pre-filled from the previous run in
  `mblab record` or the dashboard form; the form refuses to save until they are
  answered) and `recommended` values. The bundled template marks tool use and usage-limit
  notices per run and recommends memory off, no custom instructions, no tools, no notices.
- Values outside the recommended ones are listed among the limitations with counts per
  condition, also when every run has them, and shown next to the per-message
  assessments in the report, the dashboard, and `mblab analyze`.
- A sensitivity check repeats the per-message assessments without the runs that had a
  flagged per-run value (a notice, a visible tool) and lists every verdict that changes.
- The share of run pairs done in the planned random order is measured (all pairs, not
  just neighbours), reported by `mblab validate` (a warning below 90%), stated in the
  report next to "randomized order", and listed among the limitations when below 100%.

### Scoring rules

- New scoring methods `date` (exactly one YYYY-MM-DD date, other text ignored) and
  `sequence` (each name once, in order, any separators), and `numeric` takes a list of
  numbers in order. The bundled T2, T4 and T6 use them, so correct answers in other
  wordings — "2027-02-27 (Saturday, 27 February 2027)", "Ben, Cho, Eun, Dev, and Ana",
  "1. Ben, 2. Cho, …", "It prints (37, 13)" — score correct instead of wrong, and an
  answer the rule cannot read (two dates, a missing name, a third number) is a format
  failure rather than a wrong answer. Both templates' definition hashes change; studies
  made from the 0.1.0 template keep theirs.
- Tasks can pre-register `answer_examples` (`correct`, `incorrect`, `unreadable`), checked
  whenever the definition loads; every bundled task has them.
- Distinct answers and the most common answer's share count what each rule read ("265"
  and "265 minutes" are one answer) and no longer count a format failure as an answer
  (under pandas 3 it was counted as a NaN answer).
- A hand-typed number like `1,050` (latency, thinking seconds, grading points) is refused
  as ambiguous instead of being read as 1.05; a displayed duration such as "Thought for
  1,050 seconds" keeps its text and gets no number.

### Per-message vs. default verdicts

- "Consistent with the per-message selection" (or the default) now needs the interval for
  the difference from that reference to lie inside a pre-registered equivalence margin
  (± `equivalence_fraction` × the gap between the references, 0.5 by default). Before, a
  difference that was merely not distinguishable counted as consistent: in 2,000
  simulated studies a condition halfway between the references was called consistent
  with one of them 75% of the time; now 3%.
- New verdict *between the two references* (differs from both, on opposite sides, such as
  a per-message selection applied in some runs only), and every assessment reports the
  condition's position λ (0 = default-only reference, 1 = per-message reference) with a
  bootstrap interval in the CLI, the report, and the dashboard.
- Hypothesis tallies count only assessments that place a condition, never count "no
  difference detected" as support for a no-difference prediction, and say the metrics
  are not independent.
- The confidence level, the minimum group size (at least 5), and the margin are
  pre-registered in `experiment.yaml` under `analysis:` (both bundled templates now state
  them; a definition without the section keeps its hash and gets the defaults).
  `--confidence` / `--min-n` still override them, within range, but label the analysis
  exploratory with a warning, a report banner, and a limitation.

### Bad data kept out of the analysis

- Imports refuse records that do not belong to the plan: a run that is not planned, or
  one whose experiment, condition, selections, task, or repetition differ from its plan
  entry (JSON and JSON lines were not checked before). The CSV template names the
  `experiment_id`, and a row for another experiment is refused.
- Analysis leaves out records it cannot match to the plan, including every copy of a run
  recorded more than once (before, the record's own condition was trusted and the first
  copy was used), and the limitations say how many. Reports open with a banner while
  `mblab validate` finds errors.
- Pilot studies use run ids that start with `pilot:` and the main seed + 1, so a pilot
  record or a filled-in pilot CSV is refused by the main study and the pilot does not
  rehearse the main order. A 64-character `experiment_id` keeps its `-pilot` suffix.
- `mblab plan --force` refuses to replace `plan.json` once a run is recorded: the plan
  holds the pre-registered order and snapshot that records are checked against.
- Each record stores the hash of the `experiment.yaml` in force when it was recorded
  (before, the plan's hash), validation and reports list what changed since planning and
  how many records were collected under a changed definition, and prompts are compared
  with the pre-registered prompt in the plan's snapshot.
- `official_api` studies take no hand-entered data: `mblab template` and CSV import are
  refused, and imported JSON records must carry the values the API returned. A CSV row
  that gives a time without a method is recorded as a manual stopwatch time.

### Security

- The API key can no longer appear in error output. Tracebacks never show local
  variables (Typer before 0.23 did by default, and a local can hold the key), and a
  failure to create the OpenAI client is a one-line error with the key removed.
- CSV files written by mblab (entry template, blind grading sheet, results export) put an
  apostrophe before text that a spreadsheet would run as a formula (starting with `=`,
  `+`, `-`, `@`, a tab, or a carriage return); numbers are left alone. `mblab import` and
  `mblab grade import` remove it again, so files round-trip unchanged.
- The dashboard shows text from a study (definition, records, and messages quoting them)
  literally instead of rendering it as Markdown, so it cannot add links or load remote
  images. Prompts and responses were already shown literally.
- Install hints no longer point at `pip install "model-behavior-lab[...]"`: the project
  is not on PyPI, so that would install whatever package someone published under the
  name. They now name the clone (`pip install -e ".[api]"`) or the GitHub URL, and the
  README says to list every extra in `uv sync`, which removes the extras it is not given.

### Data integrity

- Every command that writes to a study (`record`, `import`, the dashboard form,
  `grade import`, `plan`, `run-api`) takes one study-wide lock (`.mblab.lock`) and reads
  the files again inside it. A run saved by one command is never overwritten, lost, or
  sent again by another; a second writer is told to wait. Locks left by a process that no
  longer exists are cleared automatically.
- Rewriting `records.jsonl` (import with `--replace`, replacing a record) keeps lines it
  cannot read byte for byte and keeps the previous file as `records.jsonl.bak`. Before,
  such lines were silently dropped.
- Appending no longer glues a record onto a last line without a newline, and each line is
  decoded on its own, so one damaged character no longer makes a study unloadable.
- `run-api` re-reads the study under the lock and skips runs recorded meanwhile; read
  timeouts and dropped connections (possibly billed) are logged and held back instead of
  retried; a damaged `api-unrecorded.jsonl` stops sending instead of being skipped.
- The dashboard form no longer switches runs or overwrites a run recorded meanwhile;
  replacing a record needs an explicit tick. `mblab record` saves an entry it cannot store
  to `exports/unsaved-<run>.jsonl` instead of losing it.
- CSV files written by mblab are UTF-8 with a byte-order mark; non-UTF-8 CSV files get a
  clear error; a CSV `prompt_text` that differs from the planned prompt needs a
  `protocol_deviation`. Infinite or NaN numbers are rejected.

### Official-API studies and pilots

- Official-API execution method: `mblab run-api` sends the planned runs of an
  `official_api` study through the OpenAI Responses API (`reasoning.effort`) with the
  official SDK — plan order, pacing, retries for transient errors, resumable, `--dry-run`.
  The key is read from `OPENAI_API_KEY` only and never stored.
- New template `openai-reasoning-effort` (low / medium / high of one model, the same six
  verified tasks).
- Records of API studies carry `api_reported` (model id, status, echoed effort, token
  usage incl. reasoning tokens), copied as returned; rejected in UI records. Validation,
  analysis, reports, and the dashboard show these values for API studies.
- Optional extra `api` (`openai>=1.66`). UI definitions keep their hashes.
- Pilot studies: `mblab pilot` makes a separate one-repetition copy of a study
  (`pilot: true`, labelled PILOT in reports, CLI, and dashboard) to rehearse the protocol,
  and `mblab labels` lists displayed labels verbatim so `observed_label_aliases` can be
  settled before the main study is planned.

- `mblab run-api` no longer loses or re-buys billed responses: responses that cannot be
  recorded are logged in `api-unrecorded.jsonl` and held back (`--resend-unrecorded`),
  Ctrl+C waits until a received response is written, and a lock file refuses a second
  sender on the same study.
- Refusal content returned next to text is kept in `api_reported.refusal`.
- API token totals include excluded runs, and unrecorded billed responses are listed as a
  limitation.

## 0.1.0 — 2026-11-19

First MVP.

- Experiment definitions in YAML with pre-registered scoring rules, label aliases,
  hypotheses (with optional predicted patterns), and confounders; definition hash and
  snapshot stored in the plan.
- Bundled template: ChatGPT Model Selection Override Experiment (4 conditions × 6 tasks
  × 3 repetitions), with independently verified reference answers.
- Seeded, version-stable randomization (`blocked_by_repetition`, `complete`).
- Manual recording via `mblab record`, a CSV template + `mblab import`, or the dashboard
  form; validation with privacy scan.
- Automatic scoring (exact match, regex, numeric) and blind manual grading.
- Analysis with Wilson / Newcombe intervals and a seeded bootstrap; per-message vs.
  default decision table; limitations and claims ledger.
- Markdown reports, CSV/JSON exports, and a Streamlit dashboard.
- Synthetic demo studies (`mblab demo`), labelled everywhere.
