# Changelog

## Unreleased

- `mblab run-api` no longer loses or re-buys billed responses: responses that cannot be
  recorded are logged in `api-unrecorded.jsonl` and held back (`--resend-unrecorded`),
  Ctrl+C waits until a received response is written, and a lock file refuses a second
  sender on the same study.
- Refusal content returned next to text is kept in `api_reported.refusal`.
- API token totals include excluded runs, and unrecorded billed responses are listed as a
  limitation.

## Unreleased

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
