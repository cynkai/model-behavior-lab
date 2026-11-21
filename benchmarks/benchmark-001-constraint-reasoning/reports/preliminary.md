# Benchmark 01: ChatGPT Model Selection Override — Preliminary Results

> **Exploratory, preliminary results.** 6 runs: 6 conditions, 1 run each, one task, recorded by hand on 2026-11-22. The figures describe these runs only. They are not a measure of any model's or setting's accuracy, and no statistical test was run.

## Abstract

One constraint-reasoning task (ordering eight tasks under eight rules) was sent with the same prompt in the ChatGPT tablet app under 6 combinations of the default model selection and the per-message selection (long-press on send). 4 of 6 (66.7%) responses answered all three parts correctly. The other 2 gave a correct first order and a valid second order but miscounted the valid orders (A: 20, D: 25). The ground truth (23 valid orders, the first being ACBDEFGH) was computed independently by checking all 40,320 orders against the eight rules. C matched its per-message reference B. D matched its per-message reference A. F showed no difference from E. With so few runs, none of this is evidence about which model produced a response.

## Research Question

- In the ChatGPT mobile client, when the default model selection and the one-off model selection offered by long-pressing the send button are set differently, does the observable accuracy of the response differ?
- When the same model is selected in both places, is any further gain in reasoning performance observed?

## Hypotheses

- **H1** _(predicts the per-message pattern)_: When the per-message selection differs from the default selection, the observable properties of the response correspond to the per-message selection.
  - C (per-message Pro) has the same outcome as B (Pro), and D (per-message Instant) the same outcome as A (Instant).
- **H1-alt** _(predicts the default pattern)_: When the per-message selection differs from the default selection, the observable properties of the response correspond to the default selection.
  - C has the same outcome as A, and D the same outcome as B.
- **H2** _(predicts the no-difference pattern)_: Choosing the same setting as both the default and the per-message selection produces behaviour that is not distinguishable from choosing it once. (Not detecting a difference does not establish equivalence.)
  - F has the same outcome as E.

The hypotheses are stated, not tested: with this many runs the report only describes which pattern the outcomes match.

## Experimental Setup

| Item | Value |
|---|---|
| Service | chatgpt |
| Client | ChatGPT tablet app |
| Execution | by hand: the experimenter operated the app and recorded what it returned |
| Recorded on | 2026-11-22 |
| Batches | 2026-11-22-preliminary |
| Conversation | a new chat for every run |
| Prompt | the same text for every run (SHA-256 `7137220aefb77b4d…`) |
| Device | tablet |
| Device model | not recorded |
| Operating system | not recorded |
| App version | not recorded |
| Run order and times | not recorded |
| Response times | not recorded |
| Model label shown by the app | not recorded |
| Full response texts | not kept: the runs hold the experimenter's structured summaries |

Conditions:

| Condition | Default selection | Per-message selection | Role | Runs |
|---|---|---|---|---|
| A | Instant | none (normal send) | control | 1 |
| B | Pro | none (normal send) | control | 1 |
| C | Instant | Pro | test | 1 |
| D | Pro | Instant | test | 1 |
| E | Extra High | none (normal send) | control | 1 |
| F | Extra High | Extra High | test | 1 |

Selections are the labels the experimenter chose in the app. They name a setting, not the model that produced a response.

Not recorded:

- The tablet model and its operating system.
- The ChatGPT app version (build number).
- The time of each run and the order in which the runs were made.
- Response times.
- The model label, if any, that the app showed with each response.
- The full response texts: each run holds the experimenter's structured summary of the response.
- The account plan, memory, and custom-instruction settings.

## Benchmark Task

**Ordering eight tasks under eight rules**; category: logic and combinatorics; language: ko.

The prompt, verbatim:

```text
다음 논리 문제를 풀어라. 외부 도구, 웹 검색, 코드 실행은 사용하지 말고 주어진 조건만으로 해결하라.

서로 다른 작업 A, B, C, D, E, F, G, H를 각각 정확히 한 번 수행한다. 작업은 한 번에 하나씩 순서대로 진행한다.

다음 조건을 모두 만족해야 한다.

1. A는 D와 E보다 먼저 수행되어야 한다.
2. B는 D보다 먼저 수행되어야 한다.
3. C는 E와 F보다 먼저 수행되어야 한다.
4. D와 E는 모두 G보다 먼저 수행되어야 한다.
5. F와 G는 모두 H보다 먼저 수행되어야 한다.
6. A와 B는 서로 바로 이웃한 순서에 위치할 수 없다.
7. D와 F는 서로 바로 이웃한 순서에 위치할 수 없다.
8. C는 첫 번째부터 네 번째 위치 중 하나에 있어야 한다.

질문:

(1) 모든 조건을 만족하는 서로 다른 작업 순서는 총 몇 개인가?

(2) 그중 사전순으로 가장 앞서는 순서를 제시하라.

(3) (2)와 다른 유효한 순서를 하나 제시하라.

출력 형식:

총 경우의 수: [정수]
사전순 첫 번째 순서: [8글자]
다른 유효한 순서: [8글자]
검증 설명: [핵심 조건을 확인하는 간결한 설명]

확실하지 않은 경우에는 추측으로 단정하지 말고 불확실성을 명시하라.
```

_Text as provided by the experimenter on 2026-11-22. In that copy the first three output-format lines ended with a Markdown line-break backslash; they are stored without it._

| Answer | Answer line | Checked as |
|---|---|---|
| count | 총 경우의 수 | one number equal to the verified number of valid orders (numeric rule) |
| lexicographic_first | 사전순 첫 번째 순서 | the verified lexicographically first valid order |
| other_valid_order | 다른 유효한 순서 | an order that satisfies every rule and differs from the response's own answer for the first order |

Lines the prompt asks for beyond these (an explanation, for instance) are not scored: a response is judged by its answers against the verified ground truth, not by how convincing or long its explanation is.

## Ground Truth

Computed by `mblab.task_ordering`, which checks every one of the 8! = 40,320 orders against the eight rules, one rule at a time. It uses no response and no value from the run data; the reference answers in `benchmark.yaml` are checked against it.

| Quantity | Value |
|---|---|
| Valid orders | 23 |
| Lexicographically first valid order | ACBDEFGH |
| Valid orders by first task | A: 9, B: 6, C: 8 |
| Valid orders by first two tasks | AC: 9, BC: 6, CA: 5, CB: 2, CF: 1 |

All valid orders: `ACBDEFGH`, `ACBDEGFH`, `ACBEDGFH`, `ACBFEDGH`, `ACEBDGFH`, `ACEFBDGH`, `ACFBDEGH`, `ACFBEDGH`, `ACFEBDGH`, `BCADEFGH`, `BCADEGFH`, `BCAEDGFH`, `BCAFEDGH`, `BCFADEGH`, `BCFAEDGH`, `CAEBDGFH`, `CAEFBDGH`, `CAFBDEGH`, `CAFBEDGH`, `CAFEBDGH`, `CBFADEGH`, `CBFAEDGH`, `CFAEBDGH`.

Implied by the other rules: rule 8. Dropping it leaves the same 23 valid orders, so it does not change the answer.

## Observations

What each response gave, as recorded by the experimenter. The full response texts were not kept; the notes under the table are the experimenter's summaries, verbatim.

| Run | Condition | Default | Per-message | Count | First order | Second order |
|---|---|---|---|---|---|---|
| A:T1:r1 | A | Instant | none | 20 | ACBDEFGH | ACBDEGFH |
| B:T1:r1 | B | Pro | none | 23 | ACBDEFGH | ACBDEGFH |
| C:T1:r1 | C | Instant | Pro | 23 | ACBDEFGH | ACBDEGFH |
| D:T1:r1 | D | Pro | Instant | 25 | ACBDEFGH | ACBDEGFH |
| E:T1:r1 | E | Extra High | none | 23 | ACBDEFGH | ACBDEGFH |
| F:T1:r1 | F | Extra High | Extra High | 23 | ACBDEFGH | ACBDEGFH |

**A:T1:r1** (A):

- 총 경우의 수를 20개로 계산
- 첫 작업 기준 A=6, B=6, C=8로 분류
- 실제 A 시작 사례 9개 중 3개 누락

**B:T1:r1** (B):

- 총 경우의 수 23개
- AC=9, BC=6, C 시작=8로 분류
- 모든 최종 답변 정확

**C:T1:r1** (C):

- 총 경우의 수 23개
- AC=9, BC=6, CA=5, CB=2, CF=1로 분류
- 모든 최종 답변 정확

**D:T1:r1** (D):

- 총 경우의 수를 25개로 계산
- A=9, B=6, C=10으로 분류
- 실제 C 시작 사례 8개를 10개로 과대 계산

**E:T1:r1** (E):

- 총 경우의 수 23개
- F와 H를 제외한 6개 작업의 선행관계 순서 16개를 먼저 검토
- 이 중 F 삽입으로 조건을 만족시킬 수 있는 기본 순서 10개를 분류
- 각 기본 순서의 유효한 F 삽입 횟수를 합산하여 23개 산출
- 최종 답변 정확

**F:T1:r1** (F):

- 총 경우의 수 23개
- A와 B가 인접하지 않는 기본 순서 6개에서 19개
- F로 A와 B를 분리할 수 있는 기본 순서 4개에서 4개
- 19+4=23으로 계산
- 최종 답변 정확

## Results

| Run | Condition | Exact count | Lexicographic first | Second valid order | Full task success |
|---|---|---|---|---|---|
| A:T1:r1 | A | Fail (20) | Pass | Pass | Fail |
| B:T1:r1 | B | Pass (23) | Pass | Pass | Pass |
| C:T1:r1 | C | Pass (23) | Pass | Pass | Pass |
| D:T1:r1 | D | Fail (25) | Pass | Pass | Fail |
| E:T1:r1 | E | Pass (23) | Pass | Pass | Pass |
| F:T1:r1 | F | Pass (23) | Pass | Pass | Pass |

| Measure | Result |
|---|---|
| Runs | 6 |
| Full task success (all three answers correct) | 4 of 6 (66.7%) |
| Partially correct (some answers correct) | 2 of 6 (33.3%) |
| Exact count | 4 of 6 (66.7%) |
| Lexicographically first order | 6 of 6 (100%) |
| Second valid order | 6 of 6 (100%) |
| Valid orders given (both order answers) | 12 of 12 (100%) |

66.7% describes 6 runs of one task under 6 different settings. It is not the accuracy of any model or setting, and the runs are not a sample from one population.

### Comparisons

**Comparison 1: B vs C**

| Condition | Default | Per-message | Count | Full task success |
|---|---|---|---|---|
| B | Pro | none | 23 | 1 of 1 run fully correct |
| C | Instant | Pro | 23 | 1 of 1 run fully correct |
| A | Instant | none | 20 | 0 of 1 run fully correct |

C (default Instant, per-message Pro) has the same outcome as B: 1 of 1 run fully correct; the per-message reference B (default Pro, no per-message selection) had 1 of 1 run fully correct, the default reference A (default Instant, no per-message selection) had 0 of 1 run fully correct. This is consistent with the per-message selection being applied (H1), but it is not direct evidence for it. With one run per condition this describes single responses; it is not a test and not evidence of which model produced them.

**Comparison 2: A vs D**

| Condition | Default | Per-message | Count | Full task success |
|---|---|---|---|---|
| A | Instant | none | 20 | 0 of 1 run fully correct |
| D | Pro | Instant | 25 | 0 of 1 run fully correct |
| B | Pro | none | 23 | 1 of 1 run fully correct |

D (default Pro, per-message Instant) has the same outcome as A: 0 of 1 run fully correct; the per-message reference A (default Instant, no per-message selection) had 0 of 1 run fully correct, the default reference B (default Pro, no per-message selection) had 1 of 1 run fully correct. This is consistent with the per-message selection being applied (H1), but it is not direct evidence for it. Both answered the count wrongly, with different values: D gave 25 and A gave 20 (verified: 23). With one run per condition this describes single responses; it is not a test and not evidence of which model produced them.

**Comparison 3: E vs F**

| Condition | Default | Per-message | Count | Full task success |
|---|---|---|---|---|
| E | Extra High | none | 23 | 1 of 1 run fully correct |
| F | Extra High | Extra High | 23 | 1 of 1 run fully correct |

F (default Extra High, per-message Extra High) and E (default Extra High, no per-message selection) have the same outcome (1 of 1 run fully correct; 1 of 1 run fully correct), with counts 23 and 23. No gain from choosing Extra High twice was observed, which is consistent with H2. Not seeing a difference does not show that the two settings are the same: whether reasoning effort, reasoning tokens, or compute differed cannot be observed here. With one run per condition this describes single responses; it is not a test and not evidence of which model produced them.

## Error Analysis

Where each wrong answer came from, traced by comparing the intermediate counts in the experimenter's summary with the verifier's values.

| Run | Condition | Error type | Detail |
|---|---|---|---|
| A:T1:r1 | A | count too low | count 20 instead of 23 (3 fewer); valid orders that start with A: 6 reported, 9 verified (3 missing) |
| B:T1:r1 | B | none | every reported intermediate count matches the verifier |
| C:T1:r1 | C | none | every reported intermediate count matches the verifier |
| D:T1:r1 | D | count too high | count 25 instead of 23 (2 more); valid orders that start with C: 10 reported, 8 verified (2 too many) |
| E:T1:r1 | E | none | every reported intermediate count matches the verifier |
| F:T1:r1 | F | none | every reported intermediate count matches the verifier |

| Run | Intermediate count | Reported | Verified | Matches |
|---|---|---|---|---|
| A:T1:r1 | valid orders that start with A | 6 | 9 | no |
| A:T1:r1 | valid orders that start with B | 6 | 6 | yes |
| A:T1:r1 | valid orders that start with C | 8 | 8 | yes |
| B:T1:r1 | valid orders that start with A, C | 9 | 9 | yes |
| B:T1:r1 | valid orders that start with B, C | 6 | 6 | yes |
| B:T1:r1 | valid orders that start with C | 8 | 8 | yes |
| C:T1:r1 | valid orders that start with A, C | 9 | 9 | yes |
| C:T1:r1 | valid orders that start with B, C | 6 | 6 | yes |
| C:T1:r1 | valid orders that start with C, A | 5 | 5 | yes |
| C:T1:r1 | valid orders that start with C, B | 2 | 2 | yes |
| C:T1:r1 | valid orders that start with C, F | 1 | 1 | yes |
| D:T1:r1 | valid orders that start with A | 9 | 9 | yes |
| D:T1:r1 | valid orders that start with B | 6 | 6 | yes |
| D:T1:r1 | valid orders that start with C | 10 | 8 | no |
| E:T1:r1 | orders of A, B, C, D, E, G that keep the precedence rules among those six tasks | 16 | 16 | yes |
| E:T1:r1 | base orders into which F can be inserted (H last) to give a valid order | 10 | 10 | yes |
| F:T1:r1 | base orders with a place for F in which A and B are not next to each other | 6 | 6 | yes |
| F:T1:r1 | valid orders made from base orders in which A and B are not next to each other | 19 | 19 | yes |
| F:T1:r1 | base orders in which A and B are next to each other and F can be inserted between them | 4 | 4 | yes |
| F:T1:r1 | valid orders made from base orders in which A and B are next to each other | 4 | 4 | yes |

## Limitations

- 6 runs, 1 per condition, one task: no interval, test, or generalisation to other tasks, prompts, or days is possible.
- The answers come from the experimenter's structured summaries; the response texts were not kept and cannot be checked again.
- Selections are what the experimenter chose in the app. The model or reasoning effort behind a response is not observable.
- Run order and times were not recorded, so a change in the service during the session cannot be ruled out.
- No documented facts are recorded: nothing in this report relies on product documentation, and no product description is presented as observed here.

Not observable in this experiment:

- The backend model that produced each response.
- Which selection the service gives priority to (server-side routing).
- The reasoning effort actually applied to a response.
- How many reasoning tokens a response used.
- Whether choosing a setting twice stacks in any way (for example nested reasoning).
- Whether the account was part of a server-side experiment (A/B test).

## Next Experiments

Planned, not yet run (no results exist for them):

- Two more runs of every condition (repetitions 2 and 3), for 18 runs in all.
- More task types.
- Relating the settings to accuracy across those tasks.
- Recording the app version and the device for every run.

Further runs are added as new lines of `runs.jsonl` with the next repetition number and a new batch name; existing lines are not changed. A new report then lists the batches it covers in `benchmark.yaml`, and this report stays as it is.

## Reproduction

| Item | Value |
|---|---|
| Runs in this report | 6 (batches: 2026-11-22-preliminary) |
| SHA-256 of these runs | `cab09c4877bf7ffef2aa9fa6bccb0849196484b80c2d625eef9cf1cd6191cbf4` |
| SHA-256 of the prompt | `7137220aefb77b4d6b38280c59321638ff77fa8e4f114ef4b645d2e2dd0618d5` |

```bash
mblab benchmark show benchmarks/benchmark-001-constraint-reasoning
mblab benchmark report benchmarks/benchmark-001-constraint-reasoning --check
```

The first command recomputes the ground truth and scores every run; the second makes this report again and compares it with this file without writing anything.
