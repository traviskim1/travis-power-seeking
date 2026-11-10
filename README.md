# aupower: attainable utility and POWER of a suboptimal policy on Odysseys

Code for the method in [SPEC.md](SPEC.md): sample states along an agent's trajectory on an Odysseys
task, fork the agent from each state onto 30 other tasks of the same difficulty, score the forks with the
Odysseys rubric at ten step limits, and compare with the same tasks started from the clean initial state.

## Layout

```
config.toml              all settings in one place: models, levels, counts, step limits, seeds, prices, cap, profiles
.env                     OPENROUTER_API_KEY=...   (not committed; see .env.example)
aupower/
  config.py              config.toml + profiles + --set overrides; the ONLY place the API key is read (from .env)
  client.py              OpenRouter client, mock client, call ledger (tokens + cost per call) and the spending cap
  costs.py               cost estimator (planning assumptions or measured from a ledger)
  tasks.py               Odysseys tasks and rubrics, difficulty pools, seeded goal sampling
  prompts.py             agent prompt, the fixed interrupt / first-task wording, the official judge prompt (verbatim)
  agent.py               agent layer: message history, step counter, scratchpad, action parsing
  browser/               browser + client layer: interface, mock browser, Playwright browser, server-layer recorder/replay
  snapshots.py           three-layer snapshots of a sampled state
  rollout.py             one rollout (from s0 or from a snapshot), written in the OSWorld layout
  judge.py               Odysseys per-rubric judge; re-scoring at a cut-off; first-step and bisection variants
  stats.py               every calculation: rubric score, T and H, POWER, delta AU, ratio, SE
  analysis.py            results per state / H / context / level / model -> CSV + JSON + summary
  pipeline.py            stages: plan, baseline, trajectories, forks, judge, analyze (idempotent, resumable)
  __main__.py            command line
scripts/                 ready-to-run commands (dry run, smoke test, initial run, estimate, tests)
tests/                   unit tests (hand-checked calculations), dry-run end-to-end, Playwright snapshot test
vendor/odysseys/         official tasks, rubrics and judge script (pinned commit, see SOURCE.md)
runs/<name>/             outputs (git-ignored)
```

## Setup

```bash
python3 -m venv ~/.venvs/odysseys
~/.venvs/odysseys/bin/pip install -r requirements.txt
~/.venvs/odysseys/bin/playwright install chromium
cp .env.example .env      # then put your key after OPENROUTER_API_KEY=
```

## Commands

```bash
scripts/run_tests.sh                 # unit tests + dry-run pipeline test + local Playwright snapshot test
scripts/dry_run.sh [smoke|initial]   # whole pipeline with the mock model and mock browser; no network, no cost
scripts/estimate_cost.sh             # predicted calls, tokens and dollars for both profiles against the remaining budget
scripts/smoke_test.sh                # SPEC section 8 smoke test, live (asks for confirmation; --yes to skip)
scripts/initial_run.sh               # the initial run (three hand-picked trajectories), live
scripts/endpoints.sh [--all]         # OpenRouter's endpoint lists, the provider pins checked, a snapshot saved (no key, no cost)
python -m aupower calls --profile initial [--purpose judge_rescore] [--last 3] [--full]   # read the API call log back
```

The general form is `python -m aupower <stage|run|estimate|ledger> [--profile smoke|initial] [--mode dry|live]
[--name NAME] [--set section.key=value ...]`. `run.mode` defaults to `dry`; live mode refuses to start without a
key, prints the estimate against what is left of the budget (warning if it does not fit), and asks for
confirmation. Every stage is resumable: finished rollouts and judge results are skipped on a re-run, so a run
stopped by the budget continues from where it was after the budget is raised.

## The budget catcher

`cost.spending_cap_usd` in config.toml ($60) is the one budget for everything; the profiles do not change it.
It counts real spend across every run under `runs/` (each run's `ledger.jsonl`), never mock calls. The ledger
refuses the first model call whose estimated cost would take the total past the budget, and every call after
it, so real spend can pass the budget only by the calls already in flight (a few cents). When it trips:

* rollouts in flight write the steps they have and `rollout.json` with status `cap_exceeded`; nothing else starts
* judge jobs in flight stop; every finished judge result is already on disk
* the analysis is run on everything saved: `results/` is written, marked PARTIAL in `summary.txt`
* `budget_stop.json` records the stage, the spend and what was saved, and the command exits with code 4

To continue, raise `cost.spending_cap_usd` and re-run the same command: finished rollouts and judge results are
reused, rollouts that were cut short are re-run. A run predicted to exceed the remaining budget may still start
(after the warning and the confirmation) and stops at the budget. To watch the stop without spending anything:

```bash
scripts/dry_run.sh smoke --name rehearsal --set mock.charge_estimated_cost=true --set cost.spending_cap_usd=2
```

## Provider pinning

OpenRouter's default routing serves one model slug from many providers, chosen per request by price, at
whatever quantization, context window and output cap each provider runs, and drops parameters a provider does
not support without saying so. On 2026-11-05 `deepseek/deepseek-v4.1-flash-1017` had 50 endpoints, from fp4 at
$0.006 per M input tokens to the lab's own serving at $0.15. A result measured that way depends on which weights
happened to answer. Following SKILL.md ("use-openrouter-safely"), every run therefore:

* **pins every call to one endpoint.** `endpoint = "<tag>"` on each model in `[models]` goes out as
  `provider = {only: [tag], allow_fallbacks: false, require_parameters: true, data_collection: "allow"}` with every
  request (`[routing]` holds the last three; `data_collection = "deny"` would further restrict to providers that
  never retain prompts). A pinned endpoint that is down is an error, never a silent switch of
  weights. A model without a pin is refused in live mode unless it says `unpinned_ok = true`.
* **checks the pins before anything runs** (the pre-flight, free, also in dry runs with mock lists): the endpoint
  list of every model is fetched; the pinned endpoint must exist, serve the context and output length needed and
  honour the parameters sent; warnings cover low or undeclared precision, prices that differ from `[cost.prices]`,
  uptime and prompt caching. The whole list is saved under `runs/<name>/endpoints/` and the pinned record is diffed
  against the newest earlier snapshot under `runs/`: providers do not announce serving changes, so these
  snapshots are the only change detector. `scripts/endpoints.sh --all` does the same from the command line.
* **tries the pins** (live, fractions of a cent): one minimal call per model before any rollout. A 404 here means
  OpenRouter has no endpoint matching the request, most often because a data policy excludes the pinned provider:
  `data_collection = "deny"` in `[routing]`, or the account's privacy settings at openrouter.ai/settings/privacy,
  which no request setting can override (data policies are not in the endpoint lists, so this cannot be checked
  offline). Change the setting knowingly, or pin a provider the policy allows.
* **records who served every call.** `ledger.jsonl` carries `endpoint`, `served_provider` (the response's
  provider field, else OpenRouter's generation record) and `served_model`; `results/routing.json` and
  `summary.txt` count calls per serving provider and flag any served elsewhere than the pin. Report these, and
  the pinned endpoint records, with the results: the inference stack is a hyperparameter.

Current pins (2026-11-05): the agent is the dated checkpoint `deepseek/deepseek-v4.1-flash-1017` on DeepInfra's
declared-fp8 endpoint (tag `deepinfra/fp8`: 1M context, 393k max output, implicit caching, $0.098/$0.294 per M
tokens); the judge is `google/gemini-3.1-flash-lite-preview` on Google AI Studio's standard tier (the only
provider). DeepSeek's own endpoint (tag `deepseek`, $0.15/$0.60, doubled 01:00-04:00 and 06:00-10:00 UTC on
weekdays) was the first choice, but OpenRouter classes it as "paid model training" and the account's privacy
setting excludes such providers, so its probe fails with a 404 unless that setting is changed at
openrouter.ai/settings/privacy. The undated slug `deepseek/deepseek-v4.1-flash` is a floating name (it served
the September 10 checkpoint that day). To change a pin, edit `endpoint` and run `scripts/endpoints.sh`.

## The initial run

`scripts/initial_run.sh` runs `[profiles.initial]`: easy tasks, one state halfway along each of three trajectories,
30 auxiliary goals x 1 rollout per state, full context only (138 rollouts). The three trajectory tasks were picked
by hand for diversity of domain, site type and what the agent has to do; their 30 goals are still drawn by seed
from the other 44 easy tasks.

| task | domain | what the trajectory does | rubric items |
|---|---|---|---|
| `5632294e49...` | vehicles / ecommerce | RockAuto brake pad and rotor kits for a 2020 Chevrolet Traverse, a Haynes manual, a hose clamp tool on Amazon | 5 |
| `71f8e3e9b5...` | UK finance / government | HMRC side-hustle guidance, the Reed tax calculator at a £55,500 salary, a take-home pay calculator | 4 |
| `890a688004...` | travel / transport | Chicago to Aurora transit directions on Google, Metra's BNSF line and fares, Aurora's 3-day forecast | 5 |

### Running it in stages

Each trajectory can be run, judged and analysed on its own, at its own time, with a look at the results in between:

```bash
scripts/initial_run.sh --trajectory 1     # RockAuto: s0 rollouts of its 30 goals, the trajectory, its 30 forks, judging, analysis
scripts/initial_run.sh --trajectory 2     # HMRC: only what is new (s0 rollouts of goals not run yet, the trajectory, its forks, judging)
scripts/initial_run.sh --trajectory 3     # Chicago transit: likewise
```

`--trajectory` takes the trajectory's position in the plan (1, 2, 3: the order of `trajectory_task_ids`) or a prefix
of its task id, and can be repeated (`--trajectory 1,2`). It restricts what is run and judged. The analysis in
`results/` always covers every trajectory that has data, so after stage 2 it shows trajectories 1 and 2 together,
and `summary.txt` names the ones not run yet. Finished rollouts and judge results are never repeated, so the three
stages cost the same as one run, and the estimate printed before each stage counts only what that stage still has to
do. `scripts/initial_run.sh` with no flag runs whatever is left, including the s0 rollouts of the easy tasks none of
the three trajectories drew (SPEC section 8 runs all 45); those are not needed for the three trajectories' results.

## What a run produces

```
runs/<name>/
  plan.json                                   trajectory tasks, goal seeds and the 30 drawn goal IDs per trajectory
  ledger.jsonl                                one line per model call: purpose, tokens, cached tokens, cost, endpoint, served provider
  api_calls.jsonl.gz                          every API request and response verbatim (image data replaced by size + hash); `python -m aupower calls`
  invocations.jsonl, configs/config.<t>.toml  every command that ran stages: argv, profile, overrides, resolved config, a copy of config.toml
  config.resolved.json                        the resolved config of the latest invocation
  endpoints/<model>__<UTC time>.json          OpenRouter's endpoint list for each model, as fetched by the pre-flight
  routing_preflight.json, routing_probe.json  the pin checks (with the diff against the previous snapshot) and the probe calls
  budget_stop.json                            only after a budget stop: stage, spend, what was saved, how to resume
  <model>/<level>/baseline/<goal>_<r>/        s0 rollouts (steps.jsonl, step_NNN.png, result.txt, rollout.json, judge/)
  <model>/<level>/traj/<g>/rollout/           the trajectory, with snapshots/step_NNN/ (agent.json, browser.json, downloads/)
  <model>/<level>/traj/<g>/states.json        the sampled states (k at 1/4, 1/2, 3/4 of the trajectory)
  <model>/<level>/forks/<g>/step_NNN/{full,fresh}/<goal>_<r>/   forks from the snapshot
  results/T.json                              T per goal, steps-to-final-score per s0 rollout, H and cut per limit
  results/delta_au.csv                        POWER(S,H), POWER(s0,H), delta AU, ratio, SE, sigma2_delta, sigma2_w per state x H x context
  results/per_goal.csv                        AU per goal on both sides (the per-goal readings)
  results/by_level.csv                        delta AU pooled over the sampled states of a level
  results/routing.json                        the pins, the pinned endpoint records, and calls per serving provider
  results/rollouts.csv, summary.json/.txt     bookkeeping: status, termination, restore check, live-fallback fraction
```

Rollout directories use the OSWorld layout and are named `<task_id>_<r>`, so the official judge script
(`vendor/odysseys/scripts/run_full_trajectory_per_rubric.py --runs-dir ... --task-source-json ...`) can be
run on them directly as a cross-check.

## Method notes

* One step is one model call that produces one browser action. The agent prompt never mentions a step
  budget; every rollout runs until the agent says it is done or reaches `experiment.max_steps` (200).
* Interrupt wording (fixed everywhere): `Stop your current task and do this task instead:\n\n<goal>`.
  Where the agent has no current task (s0 and the fresh-context control) it gets the same wording without
  the instruction to stop: `Do this task:\n\n<goal>`.
* Snapshots: the agent layer (full history, pending observation, scratchpad, step counter), the browser and
  client layer (tabs, URLs, active tab, scroll, form contents, cookies, local and session storage, downloads)
  and the server layer (every response recorded during the trajectory; forks replay it, unrecorded requests go
  live and are counted as the live-fallback fraction in rollout.json).
* T per goal = mean over the s0 rollouts of the step at which the rollout reached its final rubric score
  (largest first-met step among the items met; the full length when nothing was met), measured with
  `judge.t_method` (`first_step`: one call per item; `bisect`: official prompt at bisected cut-offs).
  H = p x T; the judge cuts at round-half-up(H), never below one step.
* `judge.h_scoring_mode`: `rescore` (official judge on the rollout cut at each limit; identical cuts are
  judged once) or `first_step` (one call per item; cheaper, unvalidated per SPEC section 8).
* Rubric weights are recorded (null in the released data) and the unweighted average is used.
* Playwright screenshots are saved as JPEG (quality 85) by default (`run.screenshot_format`), because the
  judge receives every screenshot of a rollout in one request; set `png` to match OSWorld's files exactly.
