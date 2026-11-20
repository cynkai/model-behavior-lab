# Changelog

## Unreleased

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
