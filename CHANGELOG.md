# Changelog

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
