"""Analysis 3: what the agent claimed at the end versus what the rubric says; the reverse; goal persistence."""
from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import Any

from .load import RunData, Rollout
from .rules import BLOCK_RE, claim_class, entity_tokens, domain_of


def _unmet_items(r: Rollout) -> list[dict]:
    c = r.largest_judged_cut
    jr = r.judge_at(c) if c is not None else None
    if not jr:
        return []
    return [it for it in jr.get("items", []) if int(it.get("score", 0)) == 0]


def _final_text(r: Rollout) -> str:
    fs = r.final_step or {}
    return (r.answer or "").strip() or str(fs.get("response") or "").strip()


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def blocking_tables(run: RunData) -> tuple[list[dict], list[dict]]:
    """Bot-blocking wording (rules.BLOCK_RE) in the agent's thoughts or action results, per kind and per start hour."""
    per = []
    for r in run.rollouts:
        if r.kind == "trajectory":
            continue
        n = sum(1 for st in r.steps if BLOCK_RE.search(str(st.get("response") or "") + " " + str(st.get("result") or "")))
        per.append((r, n))
    by_kind = []
    for kind in ("baseline", "fork"):
        xs = [(r, n) for r, n in per if r.kind == kind]
        if not xs:
            continue
        steps = sum(len(r.steps) for r, _ in xs)
        by_kind.append({"kind": kind, "rollouts": len(xs), "rollouts_with_block_text": sum(1 for _, n in xs if n), "block_steps": sum(n for _, n in xs),
                        "steps": steps, "block_step_rate": round(sum(n for _, n in xs) / steps, 3) if steps else None,
                        "mean_final_score_with_blocks": _mean([r.final_score for r, n in xs if n]),
                        "mean_final_score_without": _mean([r.final_score for r, n in xs if not n])})
    hours = collections.defaultdict(list)
    for r, n in per:
        hours[datetime.fromtimestamp(float(r.started_at or 0), timezone.utc).strftime("%Y-%m-%d %Hh UTC")].append((r, n))
    by_hour = []
    for h, xs in sorted(hours.items()):
        steps = sum(len(r.steps) for r, _ in xs)
        by_hour.append({"start_hour_utc": h, "rollouts": len(xs), "baseline": sum(1 for r, _ in xs if r.kind == "baseline"),
                        "fork": sum(1 for r, _ in xs if r.kind == "fork"), "rollouts_with_block_text": sum(1 for _, n in xs if n),
                        "block_step_rate": round(sum(n for _, n in xs) / steps, 3) if steps else None})
    return by_kind, by_hour


def run_claims(run: RunData) -> dict[str, Any]:
    tables: dict[str, list[dict]] = {}
    problems: list[str] = []
    notes: list[str] = []
    rows = []
    n_empty_answer = 0
    for r in run.rollouts:
        if r.kind == "trajectory" or not r.done:
            continue
        fs = r.final_step
        cls = claim_class(fs, r.answer)
        score = r.final_score
        if fs and str(fs.get("action")) == "done" and not (r.answer or "").strip():
            n_empty_answer += 1
        unmet = _unmet_items(r)
        rows.append({"rollout_id": r.rollout_id, "kind": r.kind, "trajectory": r.traj_id or "", "k": r.k if r.k is not None else "",
                     "goal_id": r.goal_id, "n_steps": r.n_steps, "termination": r.termination, "claim": cls,
                     "answer_empty": not (r.answer or "").strip(), "score_full": score, "cut_judged": r.largest_judged_cut,
                     "judged_to_the_end": r.judged_to_the_end,
                     "n_items": len((r.judge_at(r.largest_judged_cut) or {}).get("items", [])) if r.largest_judged_cut is not None else 0, "n_unmet": len(unmet),
                     "unmet_items": " ".join(it["rubric_id"] for it in unmet),
                     "final_text": _final_text(r)[:400].replace("\n", " ")})
    tables["claims.csv"] = rows
    judged = [x for x in rows if x["score_full"] is not None]
    gap = [x for x in judged if x["claim"] == "claims_complete" and x["score_full"] < 1]
    zero = [x for x in gap if x["score_full"] == 0]

    # counts and rates per kind, per trajectory, per goal
    def summarise(group_key, label):
        out = []
        groups = collections.defaultdict(list)
        for x in judged:
            groups[group_key(x)].append(x)
        for g, xs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            cc = [x for x in xs if x["claim"] == "claims_complete"]
            gp = [x for x in cc if x["score_full"] < 1]
            out.append({"group": label, "value": g, "judged_rollouts": len(xs), "claims_complete": len(cc),
                        "claims_complete_but_score_below_1": len(gp), "of_which_score_0": sum(1 for x in gp if x["score_full"] == 0),
                        "rate_gap_among_claims": round(len(gp) / len(cc), 3) if cc else None,
                        "claims_partial_or_failure": sum(1 for x in xs if x["claim"] == "claims_partial_or_failure"),
                        "no_done": sum(1 for x in xs if x["claim"] == "no_done"),
                        "mean_score_full": round(sum(x["score_full"] for x in xs) / len(xs), 3)})
        return out
    summary = summarise(lambda x: x["kind"], "kind") + summarise(lambda x: f"{x['kind']}:{x['trajectory'][:10] or 's0'}", "kind:trajectory") \
        + summarise(lambda x: x["goal_id"][:10], "goal")
    tables["claims_summary.csv"] = summary
    n_end = sum(1 for x in judged if x["judged_to_the_end"])
    notes.append(f"scores are taken at the largest cut the judge scored ({n_end} of {len(judged)} rollouts were judged at their last step; the rest "
                 f"are longer than their goal's T, so their final score is the score at H = T)")
    notes.append(f"{len(judged)} judged rollouts (baseline + forks): {sum(1 for x in judged if x['claim'] == 'claims_complete')} ended with done "
                 f"and no failure wording, {len(gap)} of those scored below 1 at full length and {len(zero)} scored 0; "
                 f"{sum(1 for x in judged if x['claim'] == 'claims_partial_or_failure')} said done with failure wording; "
                 f"{sum(1 for x in judged if x['claim'] == 'no_done')} never said done")
    ea = []
    for kind in sorted({x["kind"] for x in rows}):
        xs = [x for x in rows if x["kind"] == kind and x["termination"] == "done"]
        emp = [x for x in xs if x["answer_empty"]]
        non = [x for x in xs if not x["answer_empty"]]
        ea.append({"kind": kind, "done_rollouts": len(xs), "empty_answer": len(emp), "rate": round(len(emp) / len(xs), 3) if xs else None,
                   "mean_score_empty_answer": _mean([x["score_full"] for x in emp]), "mean_score_with_answer": _mean([x["score_full"] for x in non])})
    tables["claims_empty_answer_by_kind.csv"] = ea
    if n_empty_answer:
        by = "; ".join(f"{x['kind']}: {x['empty_answer']} of {x['done_rollouts']} done rollouts (mean score {x['mean_score_empty_answer']} vs "
                       f"{x['mean_score_with_answer']} with an answer)" for x in ea)
        problems.append(f"{n_empty_answer} rollouts ended with a done action whose answer argument was empty (the agent wrote its answer "
                        f"in the thought and left args.answer blank); the judge then sees only the thought text (steps.jsonl 'response'); "
                        f"by kind: {by} (tables/claims_empty_answer_by_kind.csv)")
    bk, bh = blocking_tables(run)
    tables["blocking_by_kind.csv"] = bk
    tables["blocking_by_hour.csv"] = bh
    if bk:
        notes.append("bot-blocking wording (Cloudflare, CAPTCHA, access denied, ...; rules.BLOCK_RE) in thoughts or action results: " +
                     "; ".join(f"{x['kind']}: {x['rollouts_with_block_text']} of {x['rollouts']} rollouts, {x['block_step_rate']:.0%} of steps "
                               f"(mean final score {x['mean_final_score_with_blocks']} with blocks vs {x['mean_final_score_without']} without)" for x in bk) +
                     f"; by start hour: " + ", ".join(f"{x['start_hour_utc'][11:]} {x['rollouts_with_block_text']}/{x['rollouts']}" for x in bh) +
                     " (tables/blocking_by_kind.csv, blocking_by_hour.csv)")

    # examples: largest gaps first (claims complete, lowest score), with the judge's reasoning on the unmet items
    examples = []
    for x in sorted(gap, key=lambda x: (x["score_full"], -x["n_steps"]))[:5]:
        r = next(rr for rr in run.rollouts if rr.rollout_id == x["rollout_id"])
        unmet = _unmet_items(r)
        examples.append({"rollout_id": r.rollout_id, "kind": r.kind, "goal_id": r.goal_id, "n_steps": r.n_steps, "score_full": x["score_full"],
                         "claim_text": _final_text(r)[:600], "task_text": run.task(r.goal_id).text[:300],
                         "unmet": [{"rubric_id": it["rubric_id"], "requirement": it["requirement"][:220], "judge": it["final_reasoning"][:400]} for it in unmet[:4]]})

    # the reverse: rubric fully met but the agent never said done, or kept going long after
    rev = []
    for r in run.rollouts:
        if r.kind == "trajectory" or not r.done or r.final_score is None or r.final_score < 1:
            continue
        fm = r.first_met_from_first_step() or r.first_met_from_rescore()
        met = [s for s in (fm or {}).values() if s is not None]
        last_met = max(met) if met else None
        after = (r.n_steps - last_met) if last_met is not None else None
        never_done = r.termination != "done"
        long_after = after is not None and after > max(5, 0.25 * r.n_steps)
        if never_done or long_after:
            rev.append({"rollout_id": r.rollout_id, "kind": r.kind, "goal_id": r.goal_id, "n_steps": r.n_steps, "termination": r.termination,
                        "last_item_met_at": last_met, "steps_after": after, "never_said_done": never_done, "kept_going_long_after": long_after,
                        "first_met_source": "first_step.json" if r.first_met_from_first_step() else "rescore cuts (resolution = the judged cuts)"})
    tables["claims_reverse.csv"] = rev
    notes.append(f"reverse cases (rubric fully met): {sum(1 for x in rev if x['never_said_done'])} never said done, "
                 f"{sum(1 for x in rev if x['kept_going_long_after'])} kept going more than max(5 steps, 25% of the rollout) after the last item was met")

    # goal persistence in forks: does the final text (or the path) return to the trajectory task g?
    pers = []
    for r in run.by_kind("fork"):
        if not r.done or not r.traj_id:
            continue
        g_text = run.task(r.traj_id).text
        y_text = run.task(r.goal_id).text
        g_only = entity_tokens(g_text) - entity_tokens(y_text)
        final = _final_text(r).lower()
        hits = sorted(t for t in g_only if t in final)
        # sites of g that y's own text and y's s0 rollout never mention or visit
        g_sites = {domain_of(s.get("url") or "") for s in (run.trajectory(r.traj_id).steps if run.trajectory(r.traj_id) else [])}
        y_sites = {domain_of(s.get("url") or "") for b in run.baselines(r.goal_id) for s in b.steps}
        y_named = entity_tokens(y_text)
        g_specific_sites = {d for d in g_sites if d and d not in y_sites and not any(n.split(".")[0] in d for n in y_named)}
        fork_sites = [domain_of(s.get("url") or "") for s in r.steps]
        revisits = sorted({d for d in fork_sites[1:] if d in g_specific_sites})     # the first step is on the inherited page
        flagged_terms = len(hits) >= 2
        flagged_sites = bool(revisits)
        pers.append({"rollout_id": r.rollout_id, "trajectory": r.traj_id, "goal_id": r.goal_id, "n_steps": r.n_steps,
                     "g_terms_in_final_text": " ".join(hits), "n_g_terms": len(hits), "g_specific_sites_revisited": " ".join(revisits),
                     "flagged_by_terms": flagged_terms, "flagged_by_sites": flagged_sites, "flagged": flagged_terms or flagged_sites,
                     "final_text": _final_text(r)[:300].replace("\n", " ")})
    tables["goal_persistence.csv"] = pers
    if pers:
        notes.append(f"goal persistence (rule-based, goal_persistence.csv): {sum(1 for x in pers if x['flagged_by_terms'])} of {len(pers)} forks "
                     f"mention 2+ terms specific to the trajectory task in their final text; {sum(1 for x in pers if x['flagged_by_sites'])} "
                     f"visited, after the first step, a site that the trajectory used but this goal's own s0 rollout and text do not")
    return {"tables": tables, "notes": notes, "problems": problems, "examples": examples, "n_gap": len(gap), "n_zero": len(zero), "n_judged": len(judged),
            "empty_by_kind": ea, "blocking_by_kind": bk}
