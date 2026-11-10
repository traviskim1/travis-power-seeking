"""Analysis 1: run health. Status and termination counts, restore and replay checks, cost and tokens against
the planning assumptions, provider check, parse failures and judge errors."""
from __future__ import annotations

import collections
import re
import statistics
from pathlib import Path
from typing import Any

from .load import RunData


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def run_health(run: RunData) -> dict[str, Any]:
    tables: dict[str, list[dict]] = {}
    problems: list[str] = []
    notes: list[str] = []

    # ---- 1a. rollouts by status and termination, per stage
    rows = []
    for kind in ("baseline", "trajectory", "fork"):
        rs = run.by_kind(kind)
        c = collections.Counter((r.status, r.termination) for r in rs)
        for (status, term), n in sorted(c.items()):
            rows.append({"stage": kind, "status": status, "termination": term, "n": n})
        if not rs:
            rows.append({"stage": kind, "status": "-", "termination": "-", "n": 0})
    tables["health_rollouts.csv"] = rows
    for m in run.missing:
        problems.append(f"{m['kind']}: {m['dir']} has no rollout.json ({m['why']})")
    for r in run.rollouts:
        if r.status != "done":
            problems.append(f"{r.kind} {r.rollout_id} ended with status {r.status}: {str(r.error or '')[:160]}")

    # ---- 1b. restore checks and live fallback per fork
    frows = []
    for r in run.by_kind("fork"):
        rest = r.restore or {}
        rep = r.replay or {}
        frows.append({"rollout_id": r.rollout_id, "goal_id": r.goal_id, "trajectory": r.traj_id, "k": r.k, "context": r.context,
                      "status": r.status, "n_steps": r.n_steps,
                      "restore_url_matches": rest.get("active_tab_url_matches"), "tabs_restored": rest.get("tabs_restored"),
                      "restore_errors": "; ".join(rest.get("errors") or []), "expected_url": rest.get("expected_url"),
                      "actual_url": rest.get("actual_url"), "requests_total": rep.get("requests_total"),
                      "served_from_recording": rep.get("served_from_recording"), "live_fallback": rep.get("live_fallback"),
                      "live_fallback_fraction": rep.get("live_fallback_fraction")})
        if rest and rest.get("active_tab_url_matches") is False:
            problems.append(f"fork {r.rollout_id}: restored active tab {rest.get('actual_url')!r} differs from the snapshot {rest.get('expected_url')!r}")
        if rest and rest.get("errors"):
            problems.append(f"fork {r.rollout_id}: restore errors {rest['errors']}")
    tables["health_forks_restore_replay.csv"] = frows
    lff = [x["live_fallback_fraction"] for x in frows if x["live_fallback_fraction"] is not None]
    if frows:
        notes.append(f"{len(frows)} forks: restore URL matched in {sum(1 for x in frows if x['restore_url_matches'] is True)}, "
                     f"mismatched in {sum(1 for x in frows if x['restore_url_matches'] is False)}, unknown in "
                     f"{sum(1 for x in frows if x['restore_url_matches'] is None)}; live-fallback fraction mean "
                     f"{_mean(lff):.3f}, median {_median(lff):.3f}, min {min(lff):.3f}, max {max(lff):.3f}" if lff else
                     f"{len(frows)} forks; no replay statistics recorded")
    served = sorted(x["served_from_recording"] for x in frows if x["served_from_recording"] is not None)
    if served:
        med = served[len(served) // 2]
        if served[-1] - served[0] <= max(50, 0.3 * med):
            notes.append(f"served_from_recording is nearly the same for every fork (min {served[0]}, median {med}, max {served[-1]}): it is the restore of the "
                         f"tab(s) at S; what a fork fetches afterwards belongs to its own goal's sites, which the trajectory's recording cannot contain, "
                         f"so the live-fallback fraction measures how much of a fork is new browsing, not a failure of the replay")

    # ---- 1c. cost and tokens: actual vs planning assumptions
    led = run.ledger
    live = [e for e in led if not e.get("mock")]
    by_purpose: dict[str, dict[str, float]] = {}
    for e in live:
        p = by_purpose.setdefault(e.get("purpose", "?"), {"calls": 0, "ok": 0, "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0})
        p["calls"] += 1
        p["ok"] += 1 if e.get("status") == "ok" else 0
        p["prompt"] += int(e.get("prompt_tokens") or 0)
        p["cached"] += int(e.get("cached_tokens") or 0)
        p["completion"] += int(e.get("completion_tokens") or 0)
        p["cost"] += float(e.get("cost_usd") or 0.0)
    total_cost = sum(p["cost"] for p in by_purpose.values())
    crow = [{"purpose": k, "calls": int(v["calls"]), "ok": int(v["ok"]), "prompt_tokens": int(v["prompt"]), "cached_tokens": int(v["cached"]),
             "completion_tokens": int(v["completion"]), "cost_usd": round(v["cost"], 4),
             "cache_hit_rate": round(v["cached"] / v["prompt"], 4) if v["prompt"] else None,
             "prompt_per_call": round(v["prompt"] / v["calls"]) if v["calls"] else None,
             "completion_per_call": round(v["completion"] / v["calls"]) if v["calls"] else None,
             "cost_per_call": round(v["cost"] / v["calls"], 6) if v["calls"] else None}
            for k, v in sorted(by_purpose.items())]
    crow.append({"purpose": "TOTAL", "calls": sum(int(v["calls"]) for v in by_purpose.values()), "ok": sum(int(v["ok"]) for v in by_purpose.values()),
                 "prompt_tokens": sum(int(v["prompt"]) for v in by_purpose.values()), "cached_tokens": sum(int(v["cached"]) for v in by_purpose.values()),
                 "completion_tokens": sum(int(v["completion"]) for v in by_purpose.values()), "cost_usd": round(total_cost, 4),
                 "cache_hit_rate": None, "prompt_per_call": None, "completion_per_call": None, "cost_per_call": None})
    tables["health_cost_by_purpose.csv"] = crow

    # rollouts on disk: steps and tokens per step (from steps.jsonl, which carries the per-call usage)
    done = [r for r in run.rollouts if r.done]
    steps_per = [r.n_steps for r in done]
    per_step_prompt, per_step_cached, per_step_completion, per_step_cost = [], [], [], []
    for r in done:
        for s in r.steps:
            per_step_prompt.append(int(s.get("prompt_tokens") or 0))
            per_step_cached.append(int(s.get("cached_tokens") or 0))
            per_step_completion.append(int(s.get("completion_tokens") or 0))
            per_step_cost.append(float(s.get("cost_usd") or 0.0))
    plan = run.config["cost"]["planning"]
    disk_cost = sum(float((r.usage or {}).get("cost_usd") or 0.0) for r in run.rollouts)
    arows = [
        {"quantity": "steps per rollout (done rollouts)", "assumed": plan["steps_per_rollout"], "actual_mean": round(_mean(steps_per), 1) if steps_per else None,
         "actual_median": _median(steps_per), "actual_max": max(steps_per) if steps_per else None, "n": len(steps_per), "source": "rollout.json"},
        {"quantity": "input tokens per agent step", "assumed": plan["input_tokens_per_step"], "actual_mean": round(_mean(per_step_prompt)) if per_step_prompt else None,
         "actual_median": _median(per_step_prompt), "actual_max": max(per_step_prompt) if per_step_prompt else None, "n": len(per_step_prompt), "source": "steps.jsonl"},
        {"quantity": "output tokens per agent step", "assumed": plan["output_tokens_per_step"], "actual_mean": round(_mean(per_step_completion)) if per_step_completion else None,
         "actual_median": _median(per_step_completion), "actual_max": max(per_step_completion) if per_step_completion else None, "n": len(per_step_completion), "source": "steps.jsonl"},
        {"quantity": "cache hit rate of agent input", "assumed": plan["cache_fraction"],
         "actual_mean": round(sum(per_step_cached) / sum(per_step_prompt), 4) if sum(per_step_prompt) else None, "actual_median": None, "actual_max": None,
         "n": len(per_step_prompt), "source": "steps.jsonl"},
        {"quantity": "cost per agent step (USD)", "assumed": None, "actual_mean": round(_mean(per_step_cost), 6) if per_step_cost else None,
         "actual_median": round(_median(per_step_cost), 6) if per_step_cost else None, "actual_max": round(max(per_step_cost), 6) if per_step_cost else None,
         "n": len(per_step_cost), "source": "steps.jsonl"},
    ]
    jp = by_purpose.get("judge_rescore") or {}
    jf = by_purpose.get("judge_first_step") or {}
    for label, p in (("judge re-score", jp), ("judge first-step", jf)):
        if p.get("calls"):
            arows.append({"quantity": f"input tokens per {label} call", "assumed": plan["judge_input_tokens_full_trajectory"],
                          "actual_mean": round(p["prompt"] / p["calls"]), "actual_median": None, "actual_max": None, "n": int(p["calls"]), "source": "ledger.jsonl"})
            arows.append({"quantity": f"output tokens per {label} call", "assumed": plan["judge_output_tokens_per_call"],
                          "actual_mean": round(p["completion"] / p["calls"]), "actual_median": None, "actual_max": None, "n": int(p["calls"]), "source": "ledger.jsonl"})
    tables["health_assumptions.csv"] = arows

    # the estimates printed before the run (console logs) and the ledger totals
    estimates = []
    for p in run.console_logs():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"predicted total: \$([0-9.,]+)", text):
            estimates.append({"console_log": str(p), "predicted_total_usd": float(m.group(1).replace(",", ""))})
    tables["health_estimates.csv"] = estimates
    stop = run.budget_stop()
    cost_note = (f"ledger (all live calls of this run directory, including calls of rollouts later archived): ${total_cost:.2f} over "
                 f"{sum(int(v['calls']) for v in by_purpose.values())} calls; rollouts on disk account for ${disk_cost:.2f} of agent spend")
    if estimates:
        cost_note += f"; the estimate printed before the last launch was ${estimates[-1]['predicted_total_usd']:.2f} ({Path(estimates[-1]['console_log']).name})"
    if stop:
        cost_note += f"; BUDGET STOP recorded during stage {stop.get('stage')!r}: {stop.get('reason')}"
        problems.append(f"budget stop during stage {stop.get('stage')!r}: {stop.get('reason')}")
    notes.append(cost_note)

    # ---- 1d. provider check
    prov_rows = []
    agent_model = run.plan["models"][0]
    bad: dict[str, list[str]] = collections.defaultdict(list)
    for e in live:
        if e.get("status") != "ok":
            continue
        served = e.get("served_provider")
        endpoint = e.get("endpoint")
        if e.get("purpose") == "agent_step" or e.get("model") == agent_model:
            if (served or "").lower() != "deepinfra" or endpoint != "deepinfra/fp8":
                bad[f"{served}|{endpoint}"].append(str((e.get("meta") or {}).get("rollout_id") or e.get("purpose")))
    counter = collections.Counter((e.get("model"), e.get("endpoint"), e.get("served_provider")) for e in live if e.get("status") == "ok")
    for (model, endpoint, served), n in sorted(counter.items(), key=lambda kv: -kv[1]):
        prov_rows.append({"model": model, "pinned_endpoint": endpoint, "served_provider": served, "calls": n})
    tables["health_providers.csv"] = prov_rows
    provider_flag = None
    if bad:
        affected = sorted({rid for ids in bad.values() for rid in ids})
        provider_flag = (f"{sum(len(v) for v in bad.values())} agent calls were NOT served by DeepInfra through deepinfra/fp8 "
                         f"({', '.join(bad)}); rollouts affected: {', '.join(affected[:40])}{' ...' if len(affected) > 40 else ''}")
        problems.append(provider_flag)
    quant = None
    for p in sorted((run.run_dir / "endpoints").glob("*deepseek*.json")):
        try:
            import json
            d = json.loads(p.read_text(encoding="utf-8"))
            for ep in (d.get("data") or {}).get("endpoints") or []:
                if ep.get("tag") == "deepinfra/fp8":
                    quant = ep.get("quantization")
        except Exception:
            pass

    # ---- 1e. parse failures, malformed actions, judge errors, API errors
    inv_rows = []
    n_invalid = n_steps_all = n_failed_actions = n_watchdog = 0
    watchdog_rollouts = []
    for r in run.rollouts:
        inv = [s for s in r.steps if str(s.get("action")) == "invalid"]
        failed = [s for s in r.steps if s.get("ok") is False and str(s.get("action")) != "invalid"]
        wd = [s for s in r.steps if str(s.get("result") or "").startswith("Browser step timed out")]
        n_invalid += len(inv)
        n_failed_actions += len(failed)
        n_watchdog += len(wd)
        if wd:
            watchdog_rollouts.append(r.rollout_id)
        n_steps_all += len(r.steps)
        if inv or failed:
            inv_rows.append({"rollout_id": r.rollout_id, "kind": r.kind, "n_steps": r.n_steps, "invalid_actions": len(inv),
                             "failed_actions": len(failed), "watchdog_timeouts": len(wd),
                             "first_invalid_error": str(((inv[0].get("arguments") or {}).get("error") if inv else "") or "")[:160],
                             "first_failed_result": str(failed[0].get("result") if failed else "")[:160]})
    tables["health_action_failures.csv"] = inv_rows
    judge_err = []
    expected_missing = 0
    for r in run.rollouts:
        for cut, jr in r.rescore.items():
            if jr.get("error"):
                judge_err.append({"rollout_id": r.rollout_id, "file": f"judge/rescore_cut{cut:03d}.json", "error": str(jr["error"])[:200]})
        fs = r.first_step
        if fs and fs.get("error"):
            judge_err.append({"rollout_id": r.rollout_id, "file": "judge/first_step.json", "error": str(fs["error"])[:200]})
    tables["health_judge_errors.csv"] = judge_err
    # s0 rollouts with no first-step judgment: the goal has no T and the pipeline excludes it
    no_t = [r for r in run.by_kind("baseline") if r.done and not r.first_step]
    if no_t:
        ids = {r.rollout_id for r in no_t}
        errs: dict[str, list[str]] = collections.defaultdict(list)
        for e in run.api_calls():
            rid = (e.get("meta") or {}).get("rollout_id")
            if e.get("purpose") == "judge_first_step" and rid in ids and (e.get("status") != 200 or e.get("error")):
                errs[rid].append(str(e.get("error") or e.get("status"))[:220])
        for r in no_t:
            msg = (f"goal {r.goal_id[:10]}: its s0 rollout ({r.n_steps} steps) has no judge/first_step.json, so the goal has no T and is excluded "
                   f"from every state-level number (G is one smaller)")
            if errs.get(r.rollout_id):
                msg += f"; the API log holds {len(errs[r.rollout_id])} failed judge_first_step attempt(s), e.g. {errs[r.rollout_id][0]}"
            problems.append(msg + " (api_calls.jsonl.gz)")
    api_err = collections.Counter()
    retries = 0
    n_api = 0
    for e in run.api_calls():
        n_api += 1
        if e.get("status") != 200 or e.get("error"):
            api_err[f"{e.get('kind')} status {e.get('status')}"] += 1
        if (e.get("attempt") or 1) > 1:
            retries += 1
    tables["health_api_errors.csv"] = [{"what": k, "n": v} for k, v in sorted(api_err.items())]
    notes.append(f"actions: {n_steps_all} steps in {len(run.rollouts)} rollouts; {n_invalid} unparsable model replies (action 'invalid'), "
                 f"{n_failed_actions} actions the browser could not perform; judge result files with errors: {len(judge_err)}; "
                 f"API attempts logged: {n_api}, of which {sum(api_err.values())} not OK and {retries} retries")
    if n_invalid:
        problems.append(f"{n_invalid} of {n_steps_all} agent replies could not be parsed into an action ({n_invalid / max(1, n_steps_all):.1%}); see health_action_failures.csv")
    unresponsive = [r for r in run.rollouts if "BrowserUnresponsive" in str(r.error or "")]
    if n_watchdog or unresponsive:
        notes.append(f"browser watchdog: {n_watchdog} action(s) timed out in {len(watchdog_rollouts)} rollout(s); "
                     f"{len(unresponsive)} rollout(s) ended as 'error' because the browser stopped answering (steps.jsonl result text; rollout.json error)")
        problems.append(f"browser watchdog fired {n_watchdog} time(s) in {len(watchdog_rollouts)} rollout(s) and {len(unresponsive)} rollout(s) ended unresponsive: "
                        f"{', '.join((watchdog_rollouts + [r.rollout_id for r in unresponsive])[:10])} (tables/health_action_failures.csv)")
    # ---- 1f. launch attempts (console logs next to the run) and rollouts archived next to it (runs/<name>_*)
    from .attempts import archived_rollouts, launch_attempts
    att = launch_attempts(run)
    arch = archived_rollouts(run)
    tables["health_attempts.csv"] = att["rows"]
    tables["health_archives.csv"] = arch["rows"]
    notes += att["notes"] + arch["notes"]
    problems += att["problems"] + arch["problems"]
    return {"tables": tables, "notes": notes, "problems": problems, "provider_flag": provider_flag, "pinned_quantization": quant,
            "total_cost": total_cost, "disk_cost": disk_cost, "n_live_calls": len(live), "attempts": att, "archives": arch}
