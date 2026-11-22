# Benchmarks

A benchmark is a fixed task with a ground truth computed independently of any response,
sent under several settings, with runs recorded by hand. Each one lives in its own
directory:

| Benchmark | Runs | Report |
|---|---|---|
| [`benchmark-001-constraint-reasoning`](benchmark-001-constraint-reasoning/) | 6 (one per condition) | [Preliminary Results](benchmark-001-constraint-reasoning/reports/preliminary.md) |

These are **observed data**. They are kept apart from studies (`mblab init`) and never
pooled with them: a study records every response in full, with the time it was sent, in a
pre-registered random order; a benchmark run may hold only the experimenter's summary of
a response, and the report says so. There is no synthetic data here.

## Files

```text
benchmark-001-constraint-reasoning/
  benchmark.yaml        # the task, the conditions, the answer lines, the reports
  runs.jsonl            # one line per run, only ever appended to
  reports/              # made by `mblab benchmark report`; never edited by hand
```

**`benchmark.yaml`** holds:

- The research questions, hypotheses (with the pattern each predicts), and what is not
  observable or was not recorded. `documented_facts` lists only statements checked in
  official documentation, with URL and date; it is empty unless someone did that.
- `selection_options` and `conditions`, as in an experiment definition. A condition is a
  default selection plus an optional per-message selection (long-press on send). Labels
  are what the app showed; they name a setting, not the model that produced a response.
- `task`: the prompt, verbatim, and the answer lines it asks for. Every run stores the
  SHA-256 of the prompt, so the prompt cannot be edited after runs exist. `verifier` names
  the code that computes the ground truth; each answer's `reference_answer` must agree with
  it, or the benchmark does not load.
- `reports`: each report covers the runs of the batches it lists. Its reproduction
  commands select those batches and that report ID. The archived preliminary report sets
  `reproduction_scope: benchmark` to reproduce its original commands for the whole benchmark.
- `notes_batches` optionally names the batches that the collection notes in
  `not_recorded` and the follow-up plans in `planned` describe. Recording coverage in a
  report or the dashboard is calculated from the selected runs.

**`runs.jsonl`** — one JSON object per line, every key present, `null` where nothing was
recorded:

| Field | Meaning |
|---|---|
| `run_id` | Unique; `<condition>:<task>:r<repetition>` by convention |
| `condition_id`, `model_selection`, `message_override` | The condition and its settings (checked against `benchmark.yaml`) |
| `repetition` | 1, 2, 3, … per condition; the same repetition of a condition cannot be recorded twice |
| `batch` | The recording session (for example `2026-11-22-preliminary`); reports select runs by batch |
| `recorded_on`, `timestamp`, `run_order` | The day; the send time with a UTC offset and the position in the session, or `null` |
| `device_type`, `device_model`, `operating_system`, `app_version` | The client, as far as it was recorded |
| `new_conversation` | Whether the run was the first message of a new chat |
| `prompt_sha256` | SHA-256 of the prompt the run used |
| `observed_model_label` | The model label the app showed with the response, verbatim, or `null` |
| `latency_seconds`, `latency_method` | Response time and how it was measured (`not_measured` if it was not) |
| `raw_response` | The full response text, or `null` if it was not kept |
| `answers` | Each answer line's value as the response gave it (used when `raw_response` is `null`) |
| `reported_breakdown` | Intermediate counts the response reported, by verifier key (for example `first:A`) |
| `solution_summary` | The experimenter's summary of the solution, verbatim |
| `source` | How the record was made |

## Checking a benchmark

```bash
mblab benchmark show benchmarks/benchmark-001-constraint-reasoning
mblab benchmark report benchmarks/benchmark-001-constraint-reasoning --check
mblab benchmark report benchmarks/benchmark-001-constraint-reasoning --report preliminary --check
mblab dashboard -s benchmarks/benchmark-001-constraint-reasoning
```

`show` recomputes the ground truth and scores every run. The count is read with the
same numeric rule as study answers; the orders are checked by the verifier, so any valid
second order counts, not only the example in `benchmark.yaml`. `report --check` makes the
reports again and compares them with the files; the test suite does the same, and also
pins the SHA-256 of the six preliminary runs.

## Adding runs

1. Run the same prompt (unchanged) in a new chat for each condition.
2. Append one line per run to `runs.jsonl`, with the next `repetition` for that condition,
   a new `run_id`, and a new `batch` name (for example `2026-12-01-repeat`). Do not edit or
   delete existing lines. Record what you observed and leave the rest `null`; if you keep
   the full response, put it in `raw_response` and the answer lines are read from it.
3. Add a report to `reports:` in `benchmark.yaml` that lists the batches it covers (for
   the 18-run report: the preliminary batch and the new ones) and its own `path`.
4. Run `mblab benchmark report <dir> --report <new-report-id>` to write that report, then
   add `--check` to verify it. Omit `--report` to process all declared reports.

Earlier reports cover only their own batches, so they do not change when runs are added.
To inspect one or more batches directly, pass `--batch <name>` to `benchmark show` once
for each batch. For unequal numbers of repetitions, the comparison uses each condition's
fraction of fully correct runs and shows both denominators. Equal fractions describe only
the runs in those batches.
