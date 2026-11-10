"""REPORT.md: summary first, one section per analysis with tables and figures, problems found, next steps."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)[:60]
    s = str(v).replace("|", "/").replace("\n", " ")
    return s if len(s) <= 70 else s[:67] + "..."


def md_table(rows: list[dict], cols: list[str] | None = None, max_rows: int = 40, source: str | None = None) -> str:
    if not rows:
        return "_(no rows)_\n" + (f"\nSource: {source}\n" if source else "")
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows[:max_rows]:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    if len(rows) > max_rows:
        out.append(f"\n_{len(rows) - max_rows} more rows in the CSV._")
    if source:
        out.append(f"\nSource: {source}")
    return "\n".join(out) + "\n"


def sign_changes(states: list[dict]) -> list[str]:
    pts = [s for s in states if s.get("G")]
    out = []
    for a, b in zip(pts, pts[1:]):
        if a["delta_au"] * b["delta_au"] < 0:
            out.append(f"{int(round(a['fraction'] * 100))}% -> {int(round(b['fraction'] * 100))}% ({a['delta_au']:+.3f} -> {b['delta_au']:+.3f})")
    return out


def recommend_H(states: list[dict], boot: list[dict], tsel: dict | None = None, R: int = 1) -> list[str]:
    pts = [s for s in states if s.get("G")]
    if not pts:
        return ["no scored state: no recommendation"]
    if R == 1 and tsel and tsel.get("n_distinct_cuts"):
        lo = next((s for s in pts if abs(s["fraction"] - tsel["fraction_lo"]) < 1e-9), None)
        hi = pts[-1]
        rec = (f"With r = 1 no single step limit is unbiased: T is the first-met step of the last item in the very s0 rollout that the fork is "
               f"compared with, so POWER(s0, T) is that rollout's own final score by construction and delta AU at 100% of T is biased against S, "
               f"while the cuts just below T leave out the item(s) that define T and are biased in S's favour. Report the whole curve and the bracket "
               f"[delta AU at {int(round(tsel['fraction_lo'] * 100))}% of T = {lo['delta_au']:+.3f}, at 100% of T = {hi['delta_au']:+.3f}]" if lo else
               f"With r = 1 no single step limit is unbiased (T comes from the compared s0 rollout itself); report the whole curve")
        rec += (f"; the s0 jump between those cuts ({tsel['mean_s0_gain']:+.3f} on average, {tsel['n_s0_gain_pos']} of {tsel['n_distinct_cuts']} goals) "
                f"against {tsel['mean_fork_gain']:+.3f} for the forks is the size of the artefact. For one number, take T from other s0 rollouts "
                f"(r >= 2, leave-one-out) or judge at fixed absolute step limits.")
        return [rec]
    last = pts[-1]
    reasons = []
    same_sign_from = None
    for s in pts:
        if all(x["delta_au"] * last["delta_au"] >= 0 for x in pts[pts.index(s):]):
            same_sign_from = s
            break
    bmap = {round(b["fraction"], 6): b for b in boot}
    sat = next((s for s in pts if s["power_s0"] >= 0.9 * last["power_s0"]), None)
    reasons.append(f"delta AU keeps the sign it has at 100% of T from {int(round(same_sign_from['fraction'] * 100))}% onward" if same_sign_from else
                   "delta AU changes sign all the way to 100% of T")
    if sat:
        reasons.append(f"the baseline curve reaches 90% of its 100%-of-T value by {int(round(sat['fraction'] * 100))}% of T "
                       f"(POWER(s0) {sat['power_s0']:.3f} vs {last['power_s0']:.3f})")
    sig = [s for s in pts if round(s["fraction"], 6) in bmap and (bmap[round(s["fraction"], 6)]["ci95_lo"] > 0 or bmap[round(s["fraction"], 6)]["ci95_hi"] < 0)]
    reasons.append("the 95% bootstrap interval excludes 0 at: " + (", ".join(f"{int(round(s['fraction'] * 100))}%" for s in sig) if sig else "no step limit"))
    rec = (f"Report delta AU at 100% of T (H = T, the full baseline horizon) as the headline, together with the whole curve; "
           f"treat values below {int(round(same_sign_from['fraction'] * 100))}% of T as horizon-sensitive." if same_sign_from and same_sign_from is not pts[0]
           else "Report delta AU at 100% of T as the headline together with the whole curve; the sign is stable across all step limits here.")
    return [rec] + reasons


def write_report(run, out_dir: Path, *, command: str, health: dict, claims: dict, judge: dict, traj: dict, stats: dict) -> Path:
    tdir = out_dir / "tables"
    L: list[str] = []
    cfg = run.config
    level = run.levels[0]
    planned = run.trajectories_planned(level)
    per_traj = traj["per_traj"]
    present = [e for e in per_traj.values() if e.get("states_table")]
    missing_traj = [e for e in per_traj.values() if e.get("status") == "not run"]
    unjudged_traj = [e for e in per_traj.values() if e.get("status") != "not run" and not e.get("states_table")]
    stop = run.budget_stop()
    n_roll = len(run.rollouts)
    R = cfg["experiment"]["rollouts_per_goal"]
    fresh = cfg["experiment"]["fresh_context_forks"]

    L.append(f"# Analysis of run `{run.run_dir.name}` ({cfg['run']['mode']} mode, profile {cfg.get('profile')!r})\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Reproduce with:\n\n```bash\n{command}\n```\n")
    L.append(f"Data: `{run.run_dir}` (plan.json, config.resolved.json, every rollout's rollout.json / steps.jsonl / judge/, "
             f"the sampled states' snapshots, results/, ledger.jsonl, api_calls.jsonl.gz, invocations.jsonl). "
             f"Tables are in `analysis/tables/`, figures in `analysis/figures/`. No model was called for this analysis.\n")
    L.append(f"**Design caveat that applies to every per-goal number below:** this run uses {R} rollout per goal and "
             f"{'no' if not fresh else 'a'} fresh-context control, so each goal's attainable utility on either side is a single sample "
             f"(SPEC.md section 4 plans r = 3 and a fresh-context fork). Per-goal deltas are therefore noisy readings, not estimates; "
             f"only the averages over goals (POWER, delta AU) carry the comparison."
             + (" With r = 1 each goal's step limit T is also taken from the very s0 rollout it is compared with, which biases delta AU at H = T "
                "against S and the cuts just below T in its favour (section 4)." if R == 1 else "") + "\n")

    # ---- scope / what is missing
    L.append("## Scope and what is missing\n")
    for e in per_traj.values():
        if e.get("states_table"):
            no_t = [g for g in run.goals_of(e["task_id"]) if run.goal_T(g)[0] is None]
            L.append(f"- trajectory {e['label']}: ran ({e['n_steps']} steps, ended by `{e['termination']}`), sampled state at k = {e['k']}, "
                     f"{len([r for r in e['per_goal'] if r['fraction'] == run.fractions[-1] and r['delta'] is not None])} of {len(run.goals_of(e['task_id']))} goals scored on both sides"
                     + (f" (excluded: {', '.join(g[:10] for g in no_t)}, no T because the first-step judging of its s0 rollout failed; see Problems found)." if no_t else "."))
        elif e.get("status") == "not run":
            why = "the budget stop happened before it" if stop else "it has not been run yet (the plan was to run the three trajectories one at a time, with a check in between; `--trajectory N` runs the next one)"
            L.append(f"- trajectory {e['label']}: **missing**: {why}.")
        else:
            if e.get("states"):
                L.append(f"- trajectory {e['label']}: ran (status {e.get('status')}, {e.get('n_steps')} steps) and its sampled state at step {e.get('k')} exists, "
                         f"but no fork of it has judge results yet ({'the budget stop came first' if stop else 'the forks or the judge stage did not finish'}); its state-level numbers are absent.")
            else:
                L.append(f"- trajectory {e['label']}: ran (status {e.get('status')}) but {e.get('note') or 'has no sampled state'}; see Problems found.")
    if stop:
        L.append(f"- The run was stopped by the spending cap during stage `{stop.get('stage')}`: {stop.get('reason')} (budget_stop.json). Everything below is computed from what was saved before the stop.")
    if run.missing:
        L.append(f"- {len(run.missing)} rollout directories have no rollout.json (interrupted or still in progress); they are excluded (health_rollouts.csv / Problems found).")
    L.append("")

    # ---- summary
    L.append("## Summary\n")
    bullets = []
    for e in present:
        st = [s for s in e["states_table"] if s.get("G")]
        last = st[-1] if st else None
        b = {round(x["fraction"], 6): x for x in stats["boot"].get(e["task_id"], [])}
        bl = b.get(round(last["fraction"], 6)) if last else None
        if last:
            bullets.append(f"Trajectory {e['label']}, state at step {e['k']} of {e['n_steps']}: delta AU at H = T is {last['delta_au']:+.3f} "
                           f"(POWER(S) {last['power_S']:.3f} vs POWER(s0) {last['power_s0']:.3f}, ratio {last['ratio']:.2f}" + (f", 95% bootstrap CI [{bl['ci95_lo']:+.3f}, {bl['ci95_hi']:+.3f}]" if bl else "") +
                           f") over {last['G']} goals; sign changes along H: {', '.join(sign_changes(st)) or 'none'} (tables/trajectory_{e['index']}_state.csv, bootstrap_trajectory_{e['index']}.csv)."
                           + (f" **Read with section 4:** with r = 1 the value at H = T is biased against S by construction; the bracket is "
                              f"[{next((x['delta_au'] for x in st if abs(x['fraction'] - e['t_selection']['fraction_lo']) < 1e-9), float('nan')):+.3f} at "
                              f"{int(round(e['t_selection']['fraction_lo'] * 100))}% of T, {last['delta_au']:+.3f} at 100%]." if R == 1 and e.get("t_selection", {}).get("n_distinct_cuts") else ""))
    if stats.get("pooled"):
        pl = [x for x in stats["pooled"] if abs(x["fraction"] - run.fractions[-1]) < 1e-9]
        if pl:
            bullets.append(f"Pooled over {pl[0]['n_states']} state(s) at H = T: delta AU {pl[0]['delta_au_pooled']:+.3f}, CI [{pl[0]['ci95_lo']:+.3f}, {pl[0]['ci95_hi']:+.3f}] (tables/bootstrap_pooled.csv).")
    hr = {r["stage"]: r for r in health["tables"]["health_rollouts.csv"]}
    done_n = sum(r["n"] for r in health["tables"]["health_rollouts.csv"] if r["status"] == "done")
    bullets.append(f"Run health: {done_n} of {n_roll} rollouts finished (status done); {health['notes'][0] if health['notes'] else ''} (tables/health_*.csv).")
    if health.get("attempts", {}).get("summary"):
        bullets.append(f"Launches: {health['attempts']['summary']} (tables/health_attempts.csv).")
    assum = {r["quantity"]: r for r in health["tables"]["health_assumptions.csv"]}
    sp = assum.get("steps per rollout (done rollouts)", {})
    it = assum.get("input tokens per agent step", {})
    ch = assum.get("cache hit rate of agent input", {})
    bullets.append(f"Planning assumptions vs actual: steps per rollout {sp.get('assumed')} assumed vs {sp.get('actual_mean')} mean (median {sp.get('actual_median')}); "
                   f"input tokens per step {it.get('assumed')} vs {it.get('actual_mean')}; cache hit rate {ch.get('assumed')} vs {ch.get('actual_mean')}; "
                   f"ledger spend ${health['total_cost']:.2f} (tables/health_assumptions.csv, health_cost_by_purpose.csv).")
    if health.get("provider_flag"):
        bullets.append("PROVIDER FLAG: " + health["provider_flag"])
    bullets.append(f"Claimed success vs rubric: {claims['n_gap']} of {claims['n_judged']} judged rollouts ended with `done` and no failure wording yet scored below 1 at full length, "
                   f"{claims['n_zero']} of them scored 0 (tables/claims.csv, claims_summary.csv).")
    jn = judge["notes"]
    bullets.append(f"Judge consistency: {jn[0] if jn else ''}; {jn[1] if len(jn) > 1 else ''} (tables/judge_monotonicity.csv, judge_first_step_vs_rescore.csv).")
    w = [r["fraction_after_final_score"] for r in traj["waste"] if r["fraction_after_final_score"] is not None]
    if w:
        bullets.append(f"Budget use: on average {sum(w) / len(w):.0%} of a rollout's steps come after its last rubric item was met (n = {len(w)}; tables/waste.csv, figures/waste.png).")
    for e in present:
        rec = recommend_H(e["states_table"], stats["boot"].get(e["task_id"], []), e.get("t_selection"), R)
        bullets.append(f"H to report (trajectory {e['index']}): {rec[0]}")
    excluded = [(e, [g for g in run.goals_of(e["task_id"]) if run.goal_T(g)[0] is None]) for e in present]
    for e, gs in excluded:
        if gs:
            bullets.append(f"Excluded from trajectory {e['index']}: {len(gs)} goal(s) with no T ({', '.join(g[:10] for g in gs)}): their s0 rollout has no "
                           f"first-step judgment (see Problems found for the API error), so G = {len(run.goals_of(e['task_id'])) - len(gs)}.")
    for n in health["notes"]:
        if n.startswith("served_from_recording"):
            bullets.append("Replay: " + n + ".")
    if claims.get("blocking_by_kind"):
        bullets.append("Bot blocking: " + "; ".join(f"{x['kind']}s with blocking wording {x['rollouts_with_block_text']} of {x['rollouts']} "
                                                    f"({x['block_step_rate']:.0%} of steps)" for x in claims["blocking_by_kind"]) +
                       "; empty `answer` in done actions: " + "; ".join(f"{x['kind']} {x['empty_answer']} of {x['done_rollouts']}" for x in claims.get("empty_by_kind", [])) +
                       " (tables/blocking_by_kind.csv, claims_empty_answer_by_kind.csv).")
    # cost of this stage and what the remaining trajectories would cost at the same rates
    costs = {r["purpose"]: r for r in health["tables"]["health_cost_by_purpose.csv"]}
    n_resc = sum(1 for r in run.rollouts if r.rescore)
    n_fs = sum(1 for r in run.by_kind("baseline") if r.first_step)
    c_resc = (costs.get("judge_rescore", {}).get("cost_usd") or 0) / n_resc if n_resc else None
    c_fs = (costs.get("judge_first_step", {}).get("cost_usd") or 0) / n_fs if n_fs else None
    c_agent = health["disk_cost"] / n_roll if n_roll else None
    cap = float(cfg["cost"]["spending_cap_usd"])
    spent = health["total_cost"]
    judge_total = (costs.get("judge_rescore", {}).get("cost_usd") or 0) + (costs.get("judge_first_step", {}).get("cost_usd") or 0)
    if c_resc is not None and c_agent is not None:
        proj = []
        for e in missing_traj:
            goals = run.goals_of(e["task_id"])
            new_b = [g for g in goals if not any(b.done for b in run.baselines(g))]
            est = c_agent + len(goals) * (c_agent + c_resc) + len(new_b) * (c_agent + c_resc + (c_fs or 0))
            proj.append(f"trajectory {e['index']} about ${est:.0f} ({len(new_b)} new baselines)")
        bullets.append(f"Cost: ledger ${spent:.2f} of the ${cap:.0f} cap (${cap - spent:.2f} left); judging is ${judge_total:.2f} of it "
                       f"(${c_resc:.2f} per judged rollout for the re-scoring at {len(run.fractions)} cuts" + (f", ${c_fs:.2f} per s0 rollout for first-step judging" if c_fs else "") +
                       f"), agent rollouts on disk ${health['disk_cost']:.2f} (${c_agent:.3f} each)" +
                       (f"; at these rates " + " and ".join(proj) + " would be needed" if proj else "") + " (tables/health_cost_by_purpose.csv).")
    if missing_traj:
        bullets.append(f"Missing: trajectories {', '.join(e['label'] for e in missing_traj)} ({'budget stop' if stop else 'not run yet'}); every cross-trajectory statement is from a single state.")
    if unjudged_traj:
        bullets.append(f"Incomplete: trajectories {', '.join(e['label'] for e in unjudged_traj)} ran but have no judged forks ({'the budget stop came first' if stop else 'forks or judging unfinished'}); no state-level numbers for them.")
    nprob = len(health["problems"]) + len(claims["problems"]) + len(judge["problems"]) + len(traj["problems"])
    bullets.append(f"Problems found: {nprob} (section at the end).")
    L += [f"- {b}" for b in bullets]
    L.append("")

    # ---- 1. health
    L.append("## 1. Run health\n")
    L.append("Rollouts by status and termination, per stage (source: every rollout.json):\n")
    L.append(md_table(health["tables"]["health_rollouts.csv"]))
    L.append("Fork restore checks and live-fallback fraction (source: rollout.json `restore` and `replay`, written by the browser layer):\n")
    fr = health["tables"]["health_forks_restore_replay.csv"]
    L.append(md_table(fr, cols=["goal_id", "k", "context", "status", "n_steps", "restore_url_matches", "tabs_restored", "restore_errors", "requests_total", "served_from_recording", "live_fallback_fraction"], max_rows=35, source="tables/health_forks_restore_replay.csv"))
    for n in health["notes"]:
        L.append(f"- {n}")
    L.append("\nCost and tokens by purpose (source: ledger.jsonl, live entries):\n")
    L.append(md_table(health["tables"]["health_cost_by_purpose.csv"], source="tables/health_cost_by_purpose.csv"))
    L.append("Planning assumptions (config.resolved.json `cost.planning`) vs measured (steps.jsonl per-call usage, rollout.json, ledger.jsonl):\n")
    L.append(md_table(health["tables"]["health_assumptions.csv"], source="tables/health_assumptions.csv"))
    if health["tables"]["health_estimates.csv"]:
        L.append("Estimates printed before each launch (source: the console logs next to the run directory):\n")
        L.append(md_table(health["tables"]["health_estimates.csv"]))
    if health["tables"].get("health_attempts.csv"):
        L.append("Launch attempts (source: the console log(s) next to the run directory, `runs/<name>*.console.log`; the pipeline's log lines are local time, "
                 "converted here with this machine's UTC offset; a launch without an exit code was stopped from outside):\n")
        L.append(md_table(health["tables"]["health_attempts.csv"], cols=["attempt", "label", "started_utc", "last_log_line_utc", "last_stage", "ended", "tracebacks",
                                                                         "error_lines", "minutes_idle_before_next_launch", "stalled"], source="tables/health_attempts.csv"))
    if health["tables"].get("health_archives.csv"):
        L.append("Rollout directories archived next to the run (`runs/<name>_*`, moved out by hand; nothing in them is used by this analysis):\n")
        L.append(md_table(health["tables"]["health_archives.csv"], cols=["archive", "rollout_dir", "kind", "n_steps", "last_action", "final", "has_rollout_json",
                                                                         "recording_entries_after_last_step", "last_step_utc"], max_rows=20, source="tables/health_archives.csv"))
        post_data_tb = [a for a in health["tables"]["health_attempts.csv"] if "post_data" in str(a.get("traceback_kinds", "")) and "UnicodeDecodeError" in str(a.get("traceback_kinds", ""))]
        if post_data_tb:
            L.append("The `UnicodeDecodeError ... in post_data` tracebacks come from Playwright's `request.post_data`, which decodes a request's POST body as UTF-8: "
                     "a binary body (0x1f 0x8b = gzip, used by telemetry beacons) raises inside the fork replay route handler (`_replay_route` in "
                     "`aupower/browser/playwright_browser.py`). Playwright then neither continues nor fulfils that request, so it stays pending until the page closes; "
                     "the agent's own navigations and page actions are not affected unless they send a binary body. Such requests are also missing from the "
                     "trajectory's recording (the recorder swallows the same error), so forks fetch them live. Fixed on 2026-11-06: the recorder and the "
                     "replay key on the raw bytes from `request.post_data_buffer`, and a failing lookup now falls back to the live web instead of leaving "
                     "the request pending. The trajectory and forks run before the fix were moved out of the run (archive table above) and are not used "
                     "here. The counts are in tables/health_attempts.csv (`traceback_kinds`).\n")
        stalled = [a for a in health["tables"]["health_attempts.csv"] if a.get("stalled")]
        unfinished = [a for a in health["tables"]["health_archives.csv"] if a["n_steps"] and not a["has_rollout_json"]]
        if stalled and unfinished:
            L.append("These two findings are one event: the browser teardown of a trajectory rollout waited without limit for response bodies that never "
                     "arrive (telemetry long-polls kept by the open pages), so the rollout never wrote its rollout.json and the run stalled after the "
                     "trajectory's final step. The wait is bounded since 2026-11-06 (`aupower/browser/playwright_browser.py`, `PlaywrightBrowser.close`); "
                     "the trajectory analysed below is the fresh sample from the relaunch, not the archived one.\n")
    L.append("Serving providers (source: ledger.jsonl `served_provider`, `endpoint`; the pin and its declared quantization from endpoints/*.json):\n")
    L.append(md_table(health["tables"]["health_providers.csv"]))
    if health.get("provider_flag"):
        L.append(f"**Flag:** {health['provider_flag']}\n")
    else:
        L.append(f"Every successful agent call was served through the pinned endpoint (declared quantization {health.get('pinned_quantization')!r}); nothing to flag.\n")
    L.append("Action failures (source: steps.jsonl `action == invalid` and `ok == false`):\n")
    L.append(md_table(health["tables"]["health_action_failures.csv"], max_rows=20, source="tables/health_action_failures.csv"))
    L.append("Judge result files with errors (source: judge/*.json `error`):\n")
    L.append(md_table(health["tables"]["health_judge_errors.csv"], max_rows=20, source="tables/health_judge_errors.csv"))
    L.append("API attempts that were not OK (source: api_calls.jsonl.gz):\n")
    L.append(md_table(health["tables"]["health_api_errors.csv"]))

    # ---- 2. trajectories
    L.append("## 2. What happened along each trajectory, and how it relates to power\n")
    L.append(f"Phases are labelled by rules on steps.jsonl (action, URL domain, repeated actions): `search` = on a search engine, "
             f"`form` = typing on another site, `reading` = clicks/scrolls/waits/navigation on a site, `stuck` = 3+ identical actions "
             f"in a row or 3+ consecutive failed actions, `invalid` = unparsable reply. The lower panel shows the mean averaged rubric "
             f"score over the {cfg['experiment']['n_aux_goals']} goals against the step limit H (x = mean cut over the goals, in steps after "
             f"the fork; labels = % of T), for forks from S and for the s0 baseline, with the fraction of goals scored perfect dotted.\n")
    if present:
        L.append("Combined figure: `figures/trajectories_combined.png`\n")
    for e in per_traj.values():
        L.append(f"### Trajectory {e['label']}\n")
        L.append(f"Task: {e['task_text'][:400]}{'...' if len(e['task_text']) > 400 else ''}\n")
        if not e.get("states_table"):
            L.append(f"_{'Not run.' if e.get('status') == 'not run' else e.get('note') or e.get('status')}_\n")
            continue
        L.append(f"![trajectory {e['index']}]({e['figure']})\n")
        L.append(f"Phases (source: tables/trajectory_{e['index']}_phases.csv):\n")
        L.append(md_table(e["segments"], cols=["start", "end", "n", "label"], max_rows=40))
        s = e["state_at_S"]
        L.append(f"**State at S (k = {s['k']}; source: steps.jsonl rows <= k and {Path(s['snapshot_dir']).name}/browser.json, downloads/):**\n")
        L.append(f"- sites visited in order: {' > '.join(s['domains_visited_in_order']) or 'none'}")
        L.append(f"- searches typed: {'; '.join(x for x in s['searches_typed'] if x) or 'none'}")
        L.append(f"- tabs open at S: {s['n_tabs']}: " + "; ".join(f"[{t['index']}{' active' if t['active'] else ''}] {t['url']}" for t in s["tabs_open_at_S"]))
        L.append(f"- cookies: {s['cookies_total']} on {len(s['cookie_domains'])} domains ({', '.join(list(s['cookie_domains'])[:8])}); local storage origins: {len(s['local_storage_origins'])}")
        L.append(f"- downloads: {', '.join(s['downloads']) if s['downloads'] else 'none'}; half-filled forms: {len(s['half_filled_forms'])} tab(s) with typed values"
                 + (": " + "; ".join(f"{t['url']}: {[fv.get('name') or fv.get('id') for fv in t['form_values']][:5]}" for t in s["half_filled_forms"]) if s["half_filled_forms"] else ""))
        if s["information_found_candidates"]:
            L.append("- information the agent noted before S (thoughts containing found/noted/price): " + " / ".join(f"step {x['step']}: {x['text'][:140]}" for x in s["information_found_candidates"][-4:]))
        L.append(f"- last thought before S: {s['last_thought_before_S'][:300]}\n")
        L.append(f"Per goal: delta AU (fork minus baseline, single rollouts each) at each H, and whether the goal shares a site visited before S "
                 f"(domains of the goal's own s0 rollout or names in its text) or a category with the trajectory task (source: tables/trajectory_{e['index']}_goal_delta.csv):\n")
        cols = ["goal_id", "shares_site", "shared_sites_with_pre_S", "shares_category_with_g"] + [f"delta_{int(round(p * 100))}" for p in run.fractions]
        L.append(md_table(e["overlap"], cols=cols, max_rows=35))
        L.append("Related vs unrelated goals, mean delta AU by H (source: tables/trajectory_{0}_overlap_summary.csv):\n".format(e["index"]))
        L.append(md_table(e["overlap_summary"], cols=["grouping", "fraction", "n_related", "mean_delta_related", "n_unrelated", "mean_delta_unrelated", "difference"], max_rows=30))
        os_ = [r for r in e["overlap_summary"] if r["grouping"] == "either" and abs(r["fraction"] - run.fractions[-1]) < 1e-9]
        if os_ and os_[0]["mean_delta_related"] is not None and os_[0]["mean_delta_unrelated"] is not None:
            L.append(f"At H = T, the {os_[0]['n_related']} goals that share a site or category with what the agent did before S changed by {os_[0]['mean_delta_related']:+.3f} on average "
                     f"and the {os_[0]['n_unrelated']} unrelated goals by {os_[0]['mean_delta_unrelated']:+.3f} (difference {os_[0]['difference']:+.3f}). "
                     f"With one rollout per goal this is a description of this run, not evidence of a mechanism.\n")

    # ---- 3. claims
    L.append("## 3. Claimed success vs rubric\n")
    L.append("A rollout 'claims completion' when it ends with the `done` action and neither its answer nor its last thought contains failure "
             "wording (rule in analysis/au_analysis/rules.py, `FAIL_RE`). The score is the averaged rubric score of the re-scoring at the largest "
             "cut the judge scored (judge/rescore_cut<cut>.json): the last step when the rollout is no longer than its goal's T, otherwise the "
             "cut at H = T (the pipeline judges only at the H cuts).\n")
    for n in claims["notes"]:
        L.append(f"- {n}")
    L.append("\nCounts and rates (source: tables/claims_summary.csv):\n")
    L.append(md_table(claims["tables"]["claims_summary.csv"], max_rows=60))
    L.append("Examples (largest gaps; source: tables/claims.csv, the rollout's steps.jsonl and judge/rescore_cut<n>.json):\n")
    for ex in claims["examples"]:
        L.append(f"- **{ex['kind']} {ex['rollout_id']}** ({ex['n_steps']} steps, score {ex['score_full']:.2f}). Task: _{ex['task_text'][:200]}..._ "
                 f"Claim: \"{ex['claim_text'][:350]}\"")
        for u in ex["unmet"]:
            L.append(f"    - unmet {u['rubric_id']}: {u['requirement'][:160]} — judge: {u['judge'][:260]}")
    L.append("\nThe reverse: rubric fully met but the agent never said `done`, or kept going long after the last item was met (source: tables/claims_reverse.csv):\n")
    L.append(md_table(claims["tables"]["claims_reverse.csv"], cols=["rollout_id", "kind", "n_steps", "termination", "last_item_met_at", "steps_after", "never_said_done", "kept_going_long_after", "first_met_source"], max_rows=25))
    L.append("Goal persistence in forks: the final text mentions 2+ terms specific to the trajectory task (the stronger signal), or the fork "
             "visits a site the trajectory used that this goal's own s0 rollout and text do not (the looser signal; shared utility sites can trigger it). "
             "Source: tables/goal_persistence.csv.\n")
    L.append(md_table([r for r in claims["tables"]["goal_persistence.csv"] if r["flagged_by_terms"]], cols=["rollout_id", "goal_id", "n_steps", "g_terms_in_final_text", "g_specific_sites_revisited", "final_text"], max_rows=20))
    n_sites_only = sum(1 for r in claims["tables"]["goal_persistence.csv"] if r["flagged_by_sites"] and not r["flagged_by_terms"])
    L.append(f"\n{n_sites_only} further forks are flagged by the site signal only (see the CSV).\n")

    # ---- 4. horizon
    L.append("## 4. Horizon effects\n")
    if present:
        L.append("![horizon](figures/horizon_delta_ratio.png)\n")
    for e in present:
        st = e["states_table"]
        L.append(f"Trajectory {e['label']}: delta AU and ratio by step limit (source: tables/trajectory_{e['index']}_state.csv; bands from tables/bootstrap_trajectory_{e['index']}.csv):\n")
        L.append(md_table(st, cols=["fraction", "mean_H", "mean_cut", "power_S", "power_s0", "delta_au", "ratio", "se_observed", "G"]))
        L.append(f"Sign changes between consecutive step limits: {', '.join(sign_changes(st)) or 'none'}.\n")
        for r in recommend_H(st, stats["boot"].get(e["task_id"], []), e.get("t_selection"), R):
            L.append(f"- {r}")
        L.append("")
        ts = e.get("t_selection") or {}
        if ts.get("n_distinct_cuts"):
            L.append(f"**How T is tied to the baseline (source: tables/trajectory_{e['index']}_t_selection.csv; first-met steps from judge/first_step.json):** "
                     f"T is the step at which the s0 rollout first met its last item, so at the cut H = T that rollout's score includes every item it ever "
                     f"met, and at the cut just below T it lacks the item(s) first met at step T. Per goal, {ts['mean_items_at_T_share']:.0%} of the rubric "
                     f"items on average are first met exactly at T. Between the {int(round(ts['fraction_lo'] * 100))}% and the 100% cut the s0 score rises by "
                     f"{ts['mean_s0_gain']:+.3f} on average ({ts['n_s0_gain_pos']} of {ts['n_distinct_cuts']} goals; the exceptions are goals where the "
                     f"re-scoring judge did not confirm the first-step judge's item) and the fork score by {ts['mean_fork_gain']:+.3f} "
                     f"({ts['n_fork_gain_pos']} of {ts['n_distinct_cuts']}). {ts['n_forks_longer_than_T']} of {ts['n_goals']} forks are longer than T, so "
                     f"their score at H = T is not their final score either (the pipeline judges only at the H cuts). Among the {ts['n_fully_judged']} forks "
                     f"judged at their own last step, the fork's final score averages {ts['mean_fork_final_fully_judged']:.3f} against "
                     f"{ts['mean_s0_final_fully_judged']:.3f} for the same goals' s0 rollouts (difference {ts['mean_delta_fully_judged']:+.3f}); that subset "
                     f"is selected on fork length and is not an estimate either.\n")
            L.append(md_table(e.get("t_selection_rows") or [], cols=["goal_id", "T", "m", "items_first_met_at_T", "cut_lo", "cut_hi", "s0_score_lo", "s0_score_hi",
                                                                      "fork_score_lo", "fork_score_hi", "fork_steps", "fork_judged_at_own_end"],
                              max_rows=40, source=f"tables/trajectory_{e['index']}_t_selection.csv"))
    L.append("Steps spent after the final rubric score was reached (source: tables/waste.csv; first-met steps from judge/first_step.json for "
             "s0 rollouts and from the re-scored cuts for forks, whose resolution is the set of judged cuts):\n")
    if traj["waste"]:
        L.append("![waste](figures/waste.png)\n")
        for kind in ("baseline", "fork"):
            vals = [r["fraction_after_final_score"] for r in traj["waste"] if r["kind"] == kind and r["fraction_after_final_score"] is not None]
            zero = [r for r in traj["waste"] if r["kind"] == kind and r["items_met"] == 0]
            if vals:
                L.append(f"- {kind}: mean {sum(vals) / len(vals):.0%}, median {sorted(vals)[len(vals) // 2]:.0%} of the steps come after the last item met (n = {len(vals)}); "
                         f"{len(zero)} rollouts met no item at all (their whole length counts as T per SPEC section 1).")
    L.append("")

    # ---- 5. judge
    L.append("## 5. Judge consistency\n")
    for n in judge["notes"]:
        L.append(f"- {n}")
    L.append("\nViolations of monotonicity (met at a shorter cut, unmet at a longer one; source: tables/judge_monotonicity.csv):\n")
    L.append(md_table(judge["tables"]["judge_monotonicity.csv"], cols=["rollout_id", "rubric_id", "met_at_cut", "unmet_at_cut", "n_steps", "reasoning_at_longer_cut"], max_rows=30))
    L.append("First-step judging vs re-scored cuts, per rubric item of the s0 rollouts (source: tables/judge_first_step_vs_rescore.csv):\n")
    L.append(md_table([r for r in judge["tables"]["judge_first_step_vs_rescore.csv"] if r["verdict"] not in ("agree", "agree_never_met")],
                      cols=["rollout_id", "rubric_id", "n_steps", "first_step_judging", "rescore_first_met_cut", "judged_cuts", "verdict"], max_rows=30))
    L.append("T per goal: from first-step judging (used by the pipeline), from the re-scored cuts, and as written in results/T.json (source: tables/judge_T_comparison.csv):\n")
    L.append(md_table(judge["tables"]["judge_T_comparison.csv"], max_rows=45))

    # ---- 6. statistics
    L.append("## 6. Statistics\n")
    L.append(f"Paired bootstrap over goals ({stats['n_boot']:,} resamples, seed {stats['seed']}): goals are resampled with replacement and delta AU "
             f"recomputed; the interval is the 2.5th to 97.5th percentile. With one rollout per goal the between-goal spread of the differences "
             f"contains the rollout-to-rollout noise, so these intervals cover both sources together.\n")
    for e in present:
        L.append(f"Trajectory {e['label']} (source: tables/bootstrap_trajectory_{e['index']}.csv):\n")
        L.append(md_table(stats["boot"].get(e["task_id"], []), cols=["fraction", "G", "delta_au", "ci95_lo", "ci95_hi", "p_boot_gt_0", "sd_goals"]))
    if stats.get("pooled"):
        L.append("Pooled across the states present (mean of the per-state delta AU, goals resampled within each state; source: tables/bootstrap_pooled.csv):\n")
        L.append(md_table(stats["pooled"]))
    L.append("Goals needed to detect a delta AU of 0.05 and 0.10 (paired test, alpha 0.05 two-sided, power 0.8), from the observed variance of the "
             "per-goal differences (source: tables/power_analysis_trajectory_*.csv). With r = 1 the within-goal variance sigma2_w cannot be separated "
             "from sigma2_delta (SPEC.md eq. 10), so these counts assume more goals at r = 1; more rollouts per goal would reduce the noise part only, "
             "by an amount this run cannot measure (the spec's pilot with r = 3 is what measures it).\n")
    for e in present:
        L.append(md_table(stats["power"].get(e["task_id"], []), cols=["fraction", "G", "var_observed_diff", "sd_observed_diff", "goals_needed_delta_0.05", "goals_needed_delta_0.10"]))

    # ---- 7. paths
    L.append("## 7. Baseline vs fork on the same goal\n")
    for e in present:
        L.append(f"Trajectory {e['label']} (source: tables/trajectory_{e['index']}_paths.csv; domains from steps.jsonl URLs, tab reuse from `switch_tab` "
                 f"to a tab index that existed at S, in {Path(e['state_at_S']['snapshot_dir']).name}/browser.json):\n")
        rows = [r for r in e["paths"] if "steps_s0" in r]
        L.append("Scores are at the largest judged cut of each rollout (cut_judged_* in the CSV).\n")
        L.append(md_table(rows, cols=["goal_id", "steps_s0", "steps_fork", "diff_fork_minus_s0", "score_s0_full", "score_fork_full", "fork_starts_on", "fork_switched_to_inherited_tab", "fork_used_pre_S_site", "first_common_domain", "fork_step_reaching_it", "s0_step_reaching_it"], max_rows=35))
        if rows:
            n_reuse = sum(1 for r in rows if r["fork_switched_to_inherited_tab"] or r["fork_used_pre_S_site"])
            L.append(f"{n_reuse} of {len(rows)} forks reused an inherited tab or revisited a site from before S; mean steps {sum(r['steps_s0'] for r in rows) / len(rows):.1f} (s0) vs "
                     f"{sum(r['steps_fork'] for r in rows) / len(rows):.1f} (fork).\n")
            ex = sorted(rows, key=lambda r: -abs(r["diff_fork_minus_s0"]))[:3]
            L.append("Largest differences, with the first thought on each side (the reason as the agent stated it):\n")
            for r in ex:
                L.append(f"- goal {r['goal_id'][:10]}: s0 {r['steps_s0']} steps (score {_fmt(r['score_s0_full'])}), fork {r['steps_fork']} steps (score {_fmt(r['score_fork_full'])}). "
                         f"s0 path: {r['domains_s0'][:160]}. fork path: {r['domains_fork'][:160]}. s0 first thought: \"{r['first_thought_s0'][:160]}\" fork first thought: \"{r['first_thought_fork'][:160]}\"")
            L.append("")

    # ---- problems
    L.append("## Problems found\n")
    allp = [("health", p) for p in health["problems"]] + [("trajectories/recomputation", p) for p in traj["problems"]] + \
           [("claims", p) for p in claims["problems"]] + [("judge", p) for p in judge["problems"]]
    if not allp:
        L.append("None detected by the checks above.\n")
    for area, p in allp:
        L.append(f"- [{area}] {p}")
    L.append("")

    # ---- next steps
    L.append("## Suggested next steps\n")
    nxt = []
    if unjudged_traj:
        nxt.append(f"Complete the forks and judging of trajectories {', '.join(e['label'] for e in unjudged_traj)} by re-running the same command"
                   f"{' after raising the budget' if stop else ''}; finished rollouts are reused.")
    if missing_traj and not stop:
        nxt.append(f"Run the remaining trajectories ({', '.join(e['label'] for e in missing_traj)}) with `scripts/initial_run.sh --trajectory N`; this report regenerates with the same command.")
    if sp.get("actual_mean") and it.get("actual_mean"):
        nxt.append(f"Replace the planning assumptions in config.toml with the measured values (steps per rollout {sp['actual_mean']}, input tokens per step {it['actual_mean']}, "
                   f"cache hit rate {ch.get('actual_mean')}) before estimating the full design; `python -m aupower estimate --from-ledger runs/{run.run_dir.name}/ledger.jsonl` does this.")
    nxt.append("Run r = 3 rollouts per goal on at least one state (the spec's pilot) to separate rollout noise from between-goal variance; the goal counts in section 6 assume r = 1.")
    nxt.append("Add the fresh-context forks before interpreting any delta AU as a property of the world state rather than of the longer context.")
    if judge["problems"]:
        nxt.append("Investigate the judge inconsistencies in section 5 before relying on T or on the smaller step limits; consider the bisection mode (judge.t_method = bisect) for T.")
    if claims["problems"]:
        nxt.append("Fix the empty `answer` in done actions (the agent writes its answer in the thought): the judge should see the stated final answer, and claims analysis should not depend on the thought.")
    if any(e.get("t_selection", {}).get("n_distinct_cuts") for e in present) and R == 1:
        nxt.append("Decouple T from the compared rollout before reading delta AU at H = T as a result: with r >= 2 define each goal's T from the other s0 rollouts "
                   "(leave-one-out), or judge every rollout at fixed absolute step limits shared by all goals; with the present data report the bracket in section 4.")
    if any("no judge/first_step.json" in p for p in health["problems"]):
        nxt.append("Keep judge payloads under the provider's 20 MB request limit (a 200-step rollout with every screenshot exceeded it): send fewer or smaller "
                   "screenshots to the judge (judge.max_images, a lower screenshot_quality or a smaller viewport), which also cuts the judge cost, the largest item.")
    if c_resc is not None and missing_traj and proj:
        nxt.append(f"Budget: ${cap - spent:.2f} of the cap is left and the remaining trajectories would cost " + " and ".join(proj) +
                   " at this run's rates (re-scoring at 10 cuts with every screenshot dominates); raise cost.spending_cap_usd or reduce the judge payload before stage 2.")
    for n in nxt:
        L.append(f"- {n}")
    L.append("")
    out = out_dir / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
