# Experiment protocol: ChatGPT Model Selection Override

This is the procedure for running the bundled experiment
(`chatgpt-model-selection-override`). It is written so that two people following it on
different days produce comparable data. Background and definitions are in
[methodology.md](methodology.md).

**Scope.** You operate the official ChatGPT app by hand and record what it shows. Do not
automate the app, scrape it, intercept its traffic, or use unofficial APIs. Use only
neutral experiment prompts in fresh conversations.

## 0. What you need

- A ChatGPT account that offers every option used by the conditions (for the default
  template: *Instant* and *Pro*).
- The ChatGPT app on the platform named in `experiment.yaml` (default: `ios`).
- A stopwatch (a second phone or a watch; not the device running ChatGPT).
- A computer with Model Behavior Lab installed (see the README).
- Time: 72 runs. Slow options can take minutes per response, so plan several sessions.
  Each block of 24 runs (one repetition of everything) is a natural session.

## 1. Before collecting any data (pre-registration)

1. `mblab init my-study`
2. Edit `my-study/experiment.yaml`:
   - `platform`: the client you will use (`ios`, `android`, `web`, …). One study, one platform.
   - `selection_options[].label`: copy each option's name **exactly as your picker shows
     it**. If you are unsure what label the app shows *next to a response* for an option,
     add the text to `observed_label_aliases` now — not after seeing data.
   - `documented_facts`: add only statements you verified in official documentation, with
     the URL and the date you read it. Leave the list empty otherwise.
   - Optional extra conditions (for example *Extra High + Extra High*): add them now.
3. Run a pilot (§1a) to see which labels the app actually displays, and settle
   `label` and `observed_label_aliases` from it.
4. Do not change tasks, prompts, scoring rules, labels, aliases, or hypotheses from here
   on. If you must, bump `experiment_version` and treat the result as a new study.
5. `mblab plan -s my-study`. Note the definition hash it prints; it identifies exactly
   what you pre-registered.

## 1a. Pilot block (before the real study)

Which text the app shows next to a response — and where — is easiest to learn by trying.
But choosing aliases after seeing the real study's data is a post-hoc change. A pilot
separates the two:

1. After editing `platform` and the labels you can see in the picker:
   `mblab pilot -s my-study`. This creates `my-study-pilot/` — a copy of the definition
   with `experiment_id` `…-pilot`, `pilot: true`, one repetition (24 runs), and its own
   plan: run ids start with `pilot:` and the seed is the main seed + 1, so the pilot's
   order is not the main study's. Use `--repetitions` for more.
2. Run it exactly as in §2–§3 (`mblab next -s my-study-pilot --copy`, `mblab record …`).
   It is also a rehearsal: timing, copying responses, finding the app version.
3. `mblab labels -s my-study-pilot` lists every label the app displayed, verbatim, with
   the conditions and places it appeared and which option it matches. Texts that match
   no option are what the analysis would report as *unrecognized*.
4. Edit `my-study/experiment.yaml`: correct each option's `label` and add the texts the
   app shows for that option to `observed_label_aliases`. Write down anything else the
   pilot taught you (a notice that appears, a slow option) under protocol notes.
5. Continue with step 4 of §1 and plan the main study.

Pilot records are real observations but never part of the results: reports and the
dashboard label them PILOT, and the main study refuses to import them or a filled-in
pilot CSV (their run ids are not in its plan); `mblab validate` reports any that are
copied in by hand, and analysis leaves them out. Keep the pilot directory: it
documents how the labels and aliases were chosen. Do not reuse the same responses in
the main study — run its plan from the start.

## 2. Account and app settings

Keep these the same for the whole study and record them per run (`confounders`):

| Setting | Recommended | Record as |
|---|---|---|
| Memory (saved memories / chat-history reference) | off | `memory` |
| Custom instructions, personality presets | off | `custom_instructions` |
| Network | the same Wi-Fi throughout | `network` |
| App version | do not update during the study if you can avoid it | `client_version` on every run |

Find the app version in the app's settings (*About*). If it updates mid-study, record the
new version; `mblab validate` will flag the change and reports will list it.

## 3. Each run

1. On the computer: `mblab next -s my-study --copy`. It shows the run id, the condition,
   the settings to apply, and the exact prompt (copied to the clipboard).
2. In the app, **start a new conversation**.
3. Set the **default model selection** shown by `mblab next`.
4. Paste the prompt **exactly**. Do not edit it, add context, or attach anything.
5. Send:
   - Condition without a per-message selection: tap send normally.
   - Condition with a per-message selection: long-press send and choose the option shown.
   Start the stopwatch at the moment you send.
6. Stop the stopwatch when the response has finished (the stop control turns back into
   the send control). Do not interact with the conversation while it runs.
7. Record the run (`mblab record -s my-study`, the dashboard form, or the CSV template):

| Field | What to enter |
|---|---|
| Response status | `completed`, or `error` / `refused` / `incomplete` |
| Response text | The whole response, copied from the app. Do not tidy it up |
| Model label displayed | Exactly the text the app shows for this response, if any. Blank if none |
| Where it was displayed | e.g. "below the response", "regenerate menu" |
| Thinking indicator | `shown` / `not_shown` / `unknown`, and the text exactly as shown ("Thought for 1m 12s") |
| Stopwatch seconds | Your reading, or blank if you missed it |
| Time sent | When you tapped send, with a UTC offset (the CLI and form fill this in) |
| Confounders | Memory, custom instructions, tool use visible in the response, any usage-limit or model-switch notice, network |
| Protocol deviation | Anything that departed from this protocol (see §4). Blank otherwise |

Then move on with `mblab next`. Follow the planned order; `mblab validate` notes runs
done out of order.

## 4. Deviations, errors, and limits

- **Record, don't redo.** If the app returns an error or refuses, record it with that
  status. Do not resend until you get an answer: retrying selectively biases the data.
- **Deviations** — wrong default or per-message selection, not a new conversation, an
  edited prompt, an interruption — are recorded in `protocol_deviation`. Such runs are
  excluded from the primary analysis and listed in the report. Do not edit the
  condition fields to match what you actually did.
- **Usage-limit or model-switch notices** change what the client may be doing. Record
  them in `switch_notice`. If they keep appearing, pause the study and resume later
  rather than continuing under a limit.
- **Tool use.** Prompts ask for no tools. If the response visibly used one (search,
  code), record it in `tools_used`; do not discard the run.

## 5. After collection

```bash
mblab validate -s my-study        # fix errors; read every warning
mblab grade export -s my-study    # only if a task uses method: manual
mblab analyze -s my-study
mblab report -s my-study          # my-study/reports/report.md
mblab dashboard -s my-study
```

Read the *Limitations* and *Claims ledger* sections before writing anything about the
results. A pattern described as *consistent with the per-message selection* is not a
statement about which model ran on the server.

## 6. Sharing results

1. Run `mblab validate` and resolve every `possible_personal_data` warning. Read the
   response texts yourself as well; the scan is a safety net, not a guarantee.
2. Share the whole study directory: `experiment.yaml`, `plan.json`, `records.jsonl`,
   `grades.jsonl` (if any), and `reports/report.md`. With these, anyone can rerun
   `mblab validate` and `mblab report` and get the same numbers.
3. Open a *Result submission* issue in the repository with the definition hash, the
   platform, the client version(s), and the dates. Do not include account identifiers.

## 7. Variants

- **Stacked setting** (*Extra High + Extra High*): add an *Extra High* option and two
  conditions — Extra High with no per-message selection, and Extra High chosen twice.
  The analysis compares the two and reports *no difference detected* or *difference
  detected*; neither identifies a cause.
- **Another platform:** run a second study with a different `platform`, the same
  definition otherwise, and compare the two reports side by side.
- **More repetitions:** increase `repetitions` before planning. More runs narrow the
  intervals; they do not change what is observable.
