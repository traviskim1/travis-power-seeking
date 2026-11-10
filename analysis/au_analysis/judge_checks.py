"""Analysis 5: judge consistency. Monotonicity of verdicts across cuts; first-step judging versus re-scoring."""
from __future__ import annotations

from typing import Any

from .load import RunData


def run_judge_checks(run: RunData) -> dict[str, Any]:
    tables: dict[str, list[dict]] = {}
    problems: list[str] = []
    notes: list[str] = []

    # ---- monotonicity: met at a shorter cut but not at a longer one
    viol = []
    n_pairs = 0
    n_rollouts_with_cuts = 0
    for r in run.rollouts:
        cuts = sorted(c for c, jr in r.rescore.items() if not jr.get("error"))
        if len(cuts) < 2:
            continue
        n_rollouts_with_cuts += 1
        verdicts = {c: {it["rubric_id"]: int(it["score"]) for it in r.rescore[c].get("items", [])} for c in cuts}
        items = sorted({rid for c in cuts for rid in verdicts[c]})
        for rid in items:
            seq = [(c, verdicts[c].get(rid)) for c in cuts if rid in verdicts[c]]
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    n_pairs += 1
                    if seq[i][1] == 1 and seq[j][1] == 0:
                        viol.append({"rollout_id": r.rollout_id, "kind": r.kind, "goal_id": r.goal_id, "rubric_id": rid,
                                     "met_at_cut": seq[i][0], "unmet_at_cut": seq[j][0], "n_steps": r.n_steps,
                                     "reasoning_at_longer_cut": str(next((it["final_reasoning"] for it in r.rescore[seq[j][0]]["items"] if it["rubric_id"] == rid), ""))[:300]})
    tables["judge_monotonicity.csv"] = viol
    distinct = {(v["rollout_id"], v["rubric_id"]) for v in viol}
    notes.append(f"monotonicity: {len(viol)} violating cut pairs out of {n_pairs} item/cut pairs in {n_rollouts_with_cuts} rollouts "
                 f"({len(distinct)} distinct rollout/item combinations)")
    if viol:
        problems.append(f"{len(distinct)} rollout/item combinations where the judge met an item at a shorter cut but not at a longer one "
                        f"(judge_monotonicity.csv); the official verdict should never decrease with more steps")

    # ---- T from first-step judging vs the step at which re-scored items first appear (s0 rollouts)
    rows = []
    for r in run.by_kind("baseline"):
        fs = r.first_met_from_first_step()
        if fs is None or not r.rescore:
            continue
        rs = r.first_met_from_rescore()
        cuts = sorted(c for c, jr in r.rescore.items() if not jr.get("error"))
        full_cut = max(cuts) if cuts else None
        for rid, f in fs.items():
            rs_first = rs.get(rid)
            full_met = None
            if full_cut is not None:
                full_met = next((int(it["score"]) for it in r.rescore[full_cut].get("items", []) if it["rubric_id"] == rid), None)
            # the two judgings agree when: never met in both; or met in both and the smallest met cut is the first cut >= f
            if f is None and rs_first is None:
                verdict = "agree_never_met"
            elif f is None and rs_first is not None:
                verdict = "first_step_missed_it"
            elif f is not None and rs_first is None:
                verdict = "rescore_never_met_it" if full_met == 0 else "rescore_has_no_cut_at_or_after_first_step"
            else:
                expected = next((c for c in cuts if c >= f), None)
                if expected is not None and rs_first == expected:
                    verdict = "agree"
                elif rs_first < f:
                    verdict = "rescore_met_earlier"
                else:
                    verdict = "rescore_met_later"
            rows.append({"rollout_id": r.rollout_id, "goal_id": r.goal_id, "rubric_id": rid, "n_steps": r.n_steps,
                         "first_step_judging": f, "rescore_first_met_cut": rs_first, "judged_cuts": " ".join(map(str, cuts)), "verdict": verdict})
    tables["judge_first_step_vs_rescore.csv"] = rows
    if rows:
        import collections
        c = collections.Counter(x["verdict"] for x in rows)
        notes.append("first-step vs re-score, per rubric item of the s0 rollouts: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        bad = sum(v for k, v in c.items() if k not in ("agree", "agree_never_met"))
        if bad:
            problems.append(f"{bad} of {len(rows)} rubric items disagree between the first-step judging (used for T) and the re-scored cuts "
                            f"(judge_first_step_vs_rescore.csv)")
    # T comparison per goal
    trows = []
    for goal in sorted({r.goal_id for r in run.by_kind("baseline")}):
        T_fs, stf = run.goal_T(goal)
        T_rs_vals = []
        for b in run.baselines(goal):
            if not b.done or not b.rescore:
                continue
            rs = b.first_met_from_rescore()
            met = [s for s in rs.values() if s is not None]
            T_rs_vals.append(max(met) if met else b.n_steps)
        T_rs = sum(T_rs_vals) / len(T_rs_vals) if T_rs_vals else None
        pipe = (run.results_json("T.json") or {}).get(run.levels[0], {}).get(goal, {})
        trows.append({"goal_id": goal, "T_first_step": T_fs, "T_from_rescore_cuts": T_rs, "T_pipeline_results": pipe.get("T"),
                      "difference_rescore_minus_first_step": (T_rs - T_fs) if (T_rs is not None and T_fs is not None) else None})
    tables["judge_T_comparison.csv"] = trows
    mism = [x for x in trows if x["T_first_step"] is not None and x["T_pipeline_results"] is not None and abs(x["T_first_step"] - x["T_pipeline_results"]) > 1e-6]
    if mism:
        problems.append(f"T recomputed from first_step.json differs from results/T.json for {len(mism)} goals (judge_T_comparison.csv): possible bug")
    return {"tables": tables, "notes": notes, "problems": problems}
