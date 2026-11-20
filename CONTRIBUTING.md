# Contributing to Model Behavior Lab

Thanks for helping. This project's value depends on one thing above all: **every number
it shows must come from recorded data, and every sentence must say no more than that
data supports.** Contributions are reviewed with that in mind.

Be respectful and assume good faith in issues and reviews. Disagreements about
methodology are welcome; make them concrete (what would change, and why).

## Ways to contribute

- **Bug reports** — use the *Bug report* issue template; include `mblab --version` and
  the command you ran.
- **Replications** — run the experiment and share the study directory (see below).
- **Tasks** — propose a task with an unambiguous, independently checkable answer.
- **Code and docs** — fixes, clearer wording, new scoring methods, new templates.

## Development setup

```bash
git clone https://github.com/cynkai/model-behavior-lab.git
cd model-behavior-lab
uv sync --extra dev            # or: python -m venv .venv && pip install -e ".[dev]"
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

`uv.lock` pins the versions CI uses. If you change dependencies in `pyproject.toml`, run
`uv lock` and commit the updated lock file.

### Layout

```text
src/mblab/
  schema.py      # pydantic models: definitions, plans, records, grades
  scoring.py     # answer extraction and scoring rules
  plan.py        # seeded run order, definition diffs
  validate.py    # study checks
  exchange.py    # CSV/JSON import and templates
  api_runner.py  # official-API runner (OpenAI Responses API)
  stats.py       # Wilson, Newcombe, bootstrap
  analysis.py    # summaries, override assessment, limitations, claims
  report.py      # Markdown report
  grading.py     # blind grading sheets
  demo.py        # synthetic demo study
  cli.py         # the `mblab` command
  dashboard/     # Streamlit app and chart builders
  templates/     # bundled experiment definitions
tests/           # pytest suite (no network, no real accounts, no API keys; the SDK is faked)
docs/            # methodology, protocol, Korean README
```

## Ground rules for changes

1. **No fabricated results.** Never commit observed-looking data. Demo data must come
   from `mblab demo` (random values, `data_origin: synthetic`, `demo: true`). Charts and
   tables must not render placeholder numbers when there is no data.
2. **No automation of AI clients.** Do not add code that drives, scrapes, or intercepts
   ChatGPT or any other client, or that calls unofficial APIs. Official-API support is
   welcome as a separate execution method using official SDKs and environment-variable
   credentials.
3. **Careful wording.** Generated text uses "consistent with", "not distinguishable",
   "not observable". Words such as "proves" or "confirms" make the tests fail. Do not
   describe behavioural patterns as evidence about internal routing.
4. **Verified answers.** A new or changed task needs its reference answer recomputed
   independently in a test (see `tests/test_template_answers.py`), and the answer must
   pass its own scoring rule.
5. **Tests for behaviour.** Scoring rules, statistics, and validation checks need tests
   with concrete inputs. Statistical routines should be checked against a reference
   implementation or published values.
6. **Writes go through the storage helpers.** Take `write_lock`, read the file again
   inside it, and add or replace records with `merge_records` / `append_jsonl`. Never
   rewrite a JSON-lines file from a parsed copy: lines that do not parse must survive.
7. **Small, focused pull requests**, with a description of what changed and how you
   checked it.

## Adding things

- **A task:** add it to a template under `src/mblab/templates/`, with `reference_answer`
  and a scoring rule; recompute the answer in a test; keep the shared instruction lines
  identical across tasks.
- **A scoring method:** add a model in `schema.py` to the `Scoring` union, implement it in
  `scoring.py` (return `format_failure` when no usable answer exists), and test edge cases.
- **A template:** a new YAML file in `src/mblab/templates/`; it must load, plan, and pass
  `check_reference_answer` for every task.
- **A chart:** a pure function in `dashboard/charts.py` that returns `None` without data;
  add it to `tests/test_charts.py` so its Vega-Lite spec is compiled in both themes, and
  pair it with a table view in the app.

## Sharing experiment results

1. `mblab validate -s <study>` — resolve errors and every `possible_personal_data`
   warning, and read the responses yourself.
2. Share the study directory: `experiment.yaml`, `plan.json`, `records.jsonl`,
   `grades.jsonl` (if any), `reports/report.md`.
3. Open a *Result submission* issue with the definition hash, platform, client
   version(s), and dates. Do not include account details or personal data.

Reviewers rerun `mblab validate` and `mblab report` on the shared directory; the numbers
should match the submitted report exactly.

## Pull request checklist

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] Docs updated (README commands match what is implemented)
- [ ] No observed data, secrets, or personal data in the diff
