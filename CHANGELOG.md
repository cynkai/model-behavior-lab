# Changelog

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
