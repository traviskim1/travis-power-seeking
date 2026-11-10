"""The method's quantities recomputed from the judge files (SPEC.md equations 7-10), independently of the
pipeline's results/, and the comparison with what the pipeline wrote."""
from __future__ import annotations

import math
from typing import Any

from .load import RunData, Rollout


def per_goal_table(run: RunData, traj_id: str, k: int, context: str = "full") -> list[dict]:
    """One row per goal x fraction: cut, fork score, s0 score, difference (single rollouts: r = 1 here)."""
    rows = []
    R = int(run.config["experiment"]["rollouts_per_goal"])
    for goal in run.goals_of(traj_id):
        cuts = run.cuts_for(goal)
        T, _ = run.goal_T(goal)
        if cuts is None:
            continue
        s0 = [b for b in run.baselines(goal) if b.done]
        fk = [f for f in run.forks(traj_id, k, context) if f.goal_id == goal and f.done]
        for p in run.fractions:
            cut = cuts[p]
            s0_scores = [x for x in (b.score_at(cut) for b in s0) if x is not None]
            fk_scores = [x for x in (f.score_at(cut) for f in fk) if x is not None]
            au_s0 = sum(s0_scores) / len(s0_scores) if s0_scores else None
            au_S = sum(fk_scores) / len(fk_scores) if fk_scores else None
            rows.append({"trajectory": traj_id, "k": k, "context": context, "goal_id": goal, "fraction": p, "T": T, "H": p * T, "cut": cut,
                         "au_S": au_S, "au_s0": au_s0, "delta": (au_S - au_s0) if (au_S is not None and au_s0 is not None) else None,
                         "n_S": len(fk_scores), "n_s0": len(s0_scores), "R_configured": R,
                         "steps_s0": s0[0].n_steps if s0 else None, "steps_fork": fk[0].n_steps if fk else None})
    return rows


def state_table(per_goal: list[dict]) -> list[dict]:
    """POWER(S,H), POWER(s0,H), delta AU and the ratio per fraction, over the goals with both sides (eq. 5, 9)."""
    out = []
    fractions = sorted({r["fraction"] for r in per_goal})
    for p in fractions:
        rows = [r for r in per_goal if r["fraction"] == p and r["delta"] is not None]
        if not rows:
            out.append({"fraction": p, "G": 0})
            continue
        pS = sum(r["au_S"] for r in rows) / len(rows)
        p0 = sum(r["au_s0"] for r in rows) / len(rows)
        diffs = [r["delta"] for r in rows]
        var = (sum((d - (pS - p0)) ** 2 for d in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else None
        out.append({"fraction": p, "mean_H": sum(r["H"] for r in rows) / len(rows), "mean_cut": sum(r["cut"] for r in rows) / len(rows),
                    "power_S": pS, "power_s0": p0, "delta_au": pS - p0, "ratio": (pS / p0) if p0 else None,
                    "se_observed": math.sqrt(var / len(rows)) if var is not None else None, "var_observed": var, "G": len(rows),
                    "perfect_S": sum(1 for r in rows if r["au_S"] >= 1) / len(rows), "perfect_s0": sum(1 for r in rows if r["au_s0"] >= 1) / len(rows)})
    return out


def compare_with_pipeline(run: RunData, per_goal: list[dict], states: list[dict], traj_id: str, k: int, context: str) -> list[str]:
    """Differences between the recomputation and results/per_goal.csv + results/delta_au.csv (bug flags)."""
    problems = []
    pg = [r for r in run.results_csv("per_goal.csv") if r["trajectory_task_id"] == traj_id and int(r["step_k"]) == k and r["context"] == context]
    mine = {(r["goal_id"], round(r["fraction"], 6)): r for r in per_goal if r["delta"] is not None}
    n_cmp = 0
    for r in pg:
        key = (r["goal_id"], round(float(r["fraction"]), 6))
        m = mine.get(key)
        if m is None:
            problems.append(f"results/per_goal.csv has goal {r['goal_id'][:10]} at p={r['fraction']} that the recomputation could not score")
            continue
        n_cmp += 1
        for col, mk in (("cut", "cut"), ("au_S", "au_S"), ("au_s0", "au_s0")):
            if abs(float(r[col]) - float(m[mk])) > 1e-6:
                problems.append(f"per_goal.csv vs recomputation differ for goal {r['goal_id'][:10]} p={r['fraction']} {col}: {r[col]} vs {m[mk]}")
    if pg and len(mine) != len(pg):
        problems.append(f"per_goal.csv has {len(pg)} rows for this state, the recomputation {len(mine)} (goals with both sides)")
    da = [r for r in run.results_csv("delta_au.csv") if r["trajectory_task_id"] == traj_id and int(r["step_k"]) == k and r["context"] == context]
    for r in da:
        s = next((x for x in states if abs(x["fraction"] - float(r["fraction"])) < 1e-9 and x.get("G")), None)
        if s is None:
            continue
        if abs(float(r["delta_au"]) - s["delta_au"]) > 1e-6:
            problems.append(f"delta_au.csv vs recomputation at p={r['fraction']}: {r['delta_au']} vs {s['delta_au']:.6f}")
    return problems


def bootstrap_delta(per_goal: list[dict], n_boot: int = 10000, seed: int = 0) -> list[dict]:
    """Paired bootstrap over goals (resampled with replacement) of delta AU at each fraction."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for p in sorted({r["fraction"] for r in per_goal}):
        d = np.array([r["delta"] for r in per_goal if r["fraction"] == p and r["delta"] is not None], dtype=float)
        if d.size == 0:
            continue
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"fraction": p, "G": int(d.size), "delta_au": float(d.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
                    "p_boot_gt_0": float((boots > 0).mean()), "sd_goals": float(d.std(ddof=1)) if d.size > 1 else None,
                    "n_boot": n_boot, "seed": seed})
    return out


def pooled_bootstrap(per_goal_by_state: dict[str, list[dict]], n_boot: int = 10000, seed: int = 0) -> list[dict]:
    """Pooled delta AU across states (mean of the per-state estimates), bootstrapping goals within each state."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    fractions = sorted({r["fraction"] for rows in per_goal_by_state.values() for r in rows})
    for p in fractions:
        arrays = []
        for st, rows in per_goal_by_state.items():
            d = np.array([r["delta"] for r in rows if r["fraction"] == p and r["delta"] is not None], dtype=float)
            if d.size:
                arrays.append(d)
        if not arrays:
            continue
        boots = np.zeros(n_boot)
        for d in arrays:
            idx = rng.integers(0, d.size, size=(n_boot, d.size))
            boots += d[idx].mean(axis=1)
        boots /= len(arrays)
        est = float(np.mean([d.mean() for d in arrays]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"fraction": p, "n_states": len(arrays), "delta_au_pooled": est, "ci95_lo": float(lo), "ci95_hi": float(hi),
                    "p_boot_gt_0": float((boots > 0).mean()), "n_boot": n_boot, "seed": seed})
    return out


def sample_size_table(per_goal: list[dict], effects=(0.05, 0.10), alpha: float = 0.05, power: float = 0.8) -> list[dict]:
    """Goals needed to detect a delta AU of a given size with a paired test, from the observed between-goal
    variance of the differences (with one rollout per goal this variance contains the rollout noise too)."""
    from math import ceil
    z_a, z_b = 1.959964, 0.841621     # two-sided alpha = 0.05, power = 0.8
    out = []
    for p in sorted({r["fraction"] for r in per_goal}):
        d = [r["delta"] for r in per_goal if r["fraction"] == p and r["delta"] is not None]
        if len(d) < 2:
            continue
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        row = {"fraction": p, "G": len(d), "var_observed_diff": var, "sd_observed_diff": math.sqrt(var)}
        for eff in effects:
            row[f"goals_needed_delta_{eff:.2f}"] = ceil((z_a + z_b) ** 2 * var / eff ** 2) if var > 0 else 1
        # if rollout noise were the spec's worst case (sd <= 0.5 per rollout, both sides sampled): what r would buy
        row["note"] = "variance of per-goal differences; with r = 1 it includes rollout-to-rollout noise (sigma2_w cannot be separated)"
        out.append(row)
    return out


def t_selection_table(run: RunData, traj_id: str, k: int, context: str = "full") -> tuple[list[dict], dict]:
    """With r = 1 a goal's T is the first-met step of the last item in the very s0 rollout that the fork is compared with:
    at the cut H = T that rollout holds every item it ever met (by construction), and at the cut just below T it lacks
    the item(s) first met at step T. Per goal: both sides' scores at the two largest cuts, the gains between them, the
    items first met exactly at T (judge/first_step.json), and whether the fork was judged at its own last step."""
    fr = run.fractions
    rows: list[dict] = []
    if len(fr) < 2:
        return rows, {}
    p_lo, p_hi = fr[-2], fr[-1]
    for goal in run.goals_of(traj_id):
        T, _ = run.goal_T(goal)
        cuts = run.cuts_for(goal)
        s0 = [b for b in run.baselines(goal) if b.done]
        fk = [f for f in run.forks(traj_id, k, context) if f.goal_id == goal and f.done]
        if T is None or cuts is None or not s0:
            continue
        b, f = s0[0], (fk[0] if fk else None)
        c_lo, c_hi = cuts[p_lo], cuts[p_hi]
        fm = b.first_met_from_first_step() or {}
        m = len(run.task(goal).rubrics)
        at_T = sum(1 for v in fm.values() if v is not None and v == T)
        s0_lo, s0_hi = b.score_at(c_lo), b.score_at(c_hi)
        f_lo, f_hi = (f.score_at(c_lo), f.score_at(c_hi)) if f else (None, None)
        rows.append({"goal_id": goal, "T": T, "m": m, "items_first_met_at_T": at_T, "cut_lo": c_lo, "cut_hi": c_hi, "fraction_lo": p_lo,
                     "s0_score_lo": s0_lo, "s0_score_hi": s0_hi, "s0_gain": (s0_hi - s0_lo) if None not in (s0_lo, s0_hi) else None,
                     "fork_score_lo": f_lo, "fork_score_hi": f_hi, "fork_gain": (f_hi - f_lo) if None not in (f_lo, f_hi) else None,
                     "fork_steps": f.n_steps if f else None, "fork_judged_at_own_end": (f.n_steps <= c_hi) if f else None,
                     "delta_hi": (f_hi - s0_hi) if None not in (f_hi, s0_hi) else None})
    distinct = [r for r in rows if r["cut_lo"] != r["cut_hi"] and r["s0_gain"] is not None and r["fork_gain"] is not None]
    full = [r for r in rows if r["fork_judged_at_own_end"] and r["delta_hi"] is not None]
    summary = {"fraction_lo": p_lo, "n_goals": len(rows), "n_distinct_cuts": len(distinct),
               "mean_s0_gain": (sum(r["s0_gain"] for r in distinct) / len(distinct)) if distinct else None,
               "n_s0_gain_pos": sum(1 for r in distinct if r["s0_gain"] > 0),
               "mean_fork_gain": (sum(r["fork_gain"] for r in distinct) / len(distinct)) if distinct else None,
               "n_fork_gain_pos": sum(1 for r in distinct if r["fork_gain"] > 0),
               "mean_items_at_T_share": (sum(r["items_first_met_at_T"] / r["m"] for r in rows) / len(rows)) if rows else None,
               "n_forks_longer_than_T": sum(1 for r in rows if r["fork_steps"] is not None and not r["fork_judged_at_own_end"]),
               "n_fully_judged": len(full),
               "mean_fork_final_fully_judged": (sum(r["fork_score_hi"] for r in full) / len(full)) if full else None,
               "mean_s0_final_fully_judged": (sum(r["s0_score_hi"] for r in full) / len(full)) if full else None,
               "mean_delta_fully_judged": (sum(r["delta_hi"] for r in full) / len(full)) if full else None}
    return rows, summary
