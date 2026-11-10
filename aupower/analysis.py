"""Assemble the measurements: T and the step limits per goal, then the POWER estimate, delta AU, the
ratio and the standard error per sampled state, per step limit H, per context, per difficulty level and
per model (SPEC.md equations 5, 8, 9 and 10)."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config, ModelSpec
from .judge import judge_cache_path, load_judge_result, score_from_first_steps
from .layout import Layout
from .rollout import load_result
from .stats import (StepLimit, au_goal, delta_au, goal_T, power_estimate, ratio, se_delta_au, step_limits,
                    steps_to_final_score, mean, sample_variance)
from .tasks import TaskIndex

log = logging.getLogger("aupower.analysis")


@dataclass
class GoalLimits:
    goal_id: str
    T: float
    limits: list[StepLimit]
    steps_to_final: dict[str, int]
    t_method: str

    def cut_for(self, fraction: float) -> int:
        for lim in self.limits:
            if abs(lim.fraction - fraction) < 1e-9:
                return lim.cut
        raise KeyError(fraction)

    def H_for(self, fraction: float) -> float:
        for lim in self.limits:
            if abs(lim.fraction - fraction) < 1e-9:
                return lim.H
        raise KeyError(fraction)

    def to_dict(self) -> dict:
        return {"goal_id": self.goal_id, "T": self.T, "t_method": self.t_method, "n_rollouts": len(self.steps_to_final),
                "steps_to_final_score": self.steps_to_final,
                "limits": [{"fraction": l.fraction, "H": round(l.H, 6), "cut": l.cut} for l in self.limits]}


def goals_used(plan: dict, level: str) -> list[str]:
    return sorted({g for t in plan["levels"][level]["trajectories"] for g in t["goals"]})


def compute_goal_limits(cfg: Config, layout: Layout, plan: dict, level: str) -> dict[str, GoalLimits]:
    """T for each goal from the reference model's s0 rollouts, and the ten step limits H = p x T."""
    ref = cfg.models.t_reference
    mode = "bisect" if cfg.judge.t_method == "bisect" else "first_step"
    out: dict[str, GoalLimits] = {}
    for goal in goals_used(plan, level):
        stf: dict[str, int] = {}
        for r in range(cfg.experiment.rollouts_per_goal):
            d = layout.baseline_dir(ref, level, goal, r)
            res = load_result(d)
            if res is None or res.status != "done":
                continue
            jr = load_judge_result(judge_cache_path(d, mode))
            if jr is None:
                continue
            stf[layout.baseline_id(ref, level, goal, r)] = steps_to_final_score(jr.first_steps, jr.n_steps_full)
        if not stf:
            continue
        T = goal_T(list(stf.values()))
        out[goal] = GoalLimits(goal, T, step_limits(T, cfg.experiment.step_limit_fractions), stf, mode)
    return out


def rollout_score(cfg: Config, rollout_dir: Path, cut: int) -> float | None:
    """The averaged rubric score of a finished rollout within ``cut`` steps, from the judge files."""
    res = load_result(rollout_dir)
    if res is None or res.status != "done":
        return None
    if cfg.judge.h_scoring_mode == "rescore":
        eff = max(0, min(int(cut), res.n_steps))
        jr = load_judge_result(judge_cache_path(rollout_dir, "rescore", eff))
        return jr.average_rubric_score if jr is not None else None
    jr = load_judge_result(judge_cache_path(rollout_dir, "first_step"))
    return score_from_first_steps(jr, cut) if jr is not None else None


def load_states(layout: Layout, model: ModelSpec, level: str, g: str) -> list[dict]:
    p = layout.states_path(model, level, g)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("states", [])


def rollout_inventory(run_dir: Path) -> list[dict]:
    """Every rollout.json under the run: status, termination, steps, restore check, live-fallback fraction."""
    rows = []
    for p in sorted(Path(run_dir).rglob("rollout.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        spec = d.get("spec", {})
        replay = d.get("replay") or {}
        restore = d.get("restore") or {}
        rows.append({
            "rollout_id": spec.get("rollout_id"), "kind": spec.get("kind"), "level": spec.get("level"),
            "model": spec.get("model"), "context": spec.get("context"), "status": d.get("status"),
            "termination": d.get("termination"), "n_steps": d.get("n_steps"),
            "restore_url_matches": restore.get("active_tab_url_matches"),
            "live_fallback_fraction": replay.get("live_fallback_fraction"),
            "requests_total": replay.get("requests_total"), "cost_usd": (d.get("usage") or {}).get("cost_usd"),
            "duration_s": d.get("duration_s"),
        })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def analyze(cfg: Config, tasks: TaskIndex, run_dir: Path, plan: dict,
            extra_notes: list[str] | None = None) -> dict[str, Any]:
    """Compute everything that can be computed from the rollouts and judge results on disk. Goals or
    rollouts without results are left out and counted (goals_missing, notes), so a run stopped by the
    budget still gets its partial results."""
    layout = Layout(run_dir)
    out_dir = layout.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    R = cfg.experiment.rollouts_per_goal
    contexts = ["full"] + (["fresh"] if cfg.experiment.fresh_context_forks else [])
    fractions = list(cfg.experiment.step_limit_fractions)

    state_rows: list[dict] = []
    goal_rows: list[dict] = []
    T_out: dict[str, Any] = {}
    notes: list[str] = list(extra_notes or [])

    for level in plan["levels"]:
        trajectories = plan["levels"][level]["trajectories"]
        not_run = [(i, t["task_id"]) for i, t in enumerate(trajectories, 1)
                   if all(load_result(layout.traj_rollout_dir(m, level, t["task_id"])) is None for m in cfg.models.agents)]
        if not_run:
            notes.append(f"{level}: trajectories not run yet: " + ", ".join(f"{i} ({tid[:10]})" for i, tid in not_run)
                         + " (run them with --trajectory N)")
        with_data = [t for t in trajectories if any(load_states(layout, m, level, t["task_id"]) for m in cfg.models.agents)]
        limits = compute_goal_limits(cfg, layout, plan, level)
        T_out[level] = {g: gl.to_dict() for g, gl in limits.items()}
        used = sorted({y for t in with_data for y in t["goals"]})
        missing_T = [g for g in used if g not in limits]
        if missing_T:
            notes.append(f"{level}: {len(missing_T)} goal(s) have no T (no judged s0 rollouts) and are excluded")
        for model in cfg.models.agents:
            s0_cache: dict[tuple[str, int], list[float]] = {}

            def s0_scores(goal: str, cut: int) -> list[float]:
                key = (goal, cut)
                if key not in s0_cache:
                    vals = [rollout_score(cfg, layout.baseline_dir(model, level, goal, r), cut) for r in range(R)]
                    s0_cache[key] = [v for v in vals if v is not None]
                return s0_cache[key]

            for traj in plan["levels"][level]["trajectories"]:
                g = traj["task_id"]
                states = load_states(layout, model, level, g)
                if not states:
                    if load_result(layout.traj_rollout_dir(model, level, g)) is not None:   # ran, but saved no state
                        notes.append(f"{model.id}/{level}/{g[:8]}: no sampled states")
                    continue
                goals = [y for y in traj["goals"] if y in limits]
                for st in states:
                    k = int(st["step"])
                    for context in contexts:
                        for p in fractions:
                            scores_S: dict[str, list[float]] = {}
                            scores_0: dict[str, list[float]] = {}
                            n_missing = 0
                            for y in goals:
                                cut = limits[y].cut_for(p)
                                sS = [rollout_score(cfg, layout.fork_dir(model, level, g, k, context, y, r), cut) for r in range(R)]
                                sS = [v for v in sS if v is not None]
                                s0 = s0_scores(y, cut)
                                if not sS or not s0:
                                    n_missing += 1
                                    continue
                                scores_S[y] = sS
                                scores_0[y] = s0
                                goal_rows.append({
                                    "model": model.id, "level": level, "trajectory_task_id": g, "step_k": k,
                                    "context": context, "fraction": p, "goal_id": y, "T": round(limits[y].T, 4),
                                    "H": round(limits[y].H_for(p), 4), "cut": cut,
                                    "au_S": round(au_goal(sS), 6), "au_s0": round(au_goal(s0), 6),
                                    "delta": round(au_goal(sS) - au_goal(s0), 6), "n_S": len(sS), "n_s0": len(s0),
                                    "scores_S": json.dumps(sS), "scores_s0": json.dumps(s0),
                                })
                            if not scores_S:
                                continue
                            au_S = {y: au_goal(v) for y, v in scores_S.items()}
                            au_0 = {y: au_goal(v) for y, v in scores_0.items()}
                            pS = power_estimate(au_S)
                            p0 = power_estimate(au_0)
                            se = se_delta_au(scores_S, scores_0)
                            state_rows.append({
                                "model": model.id, "level": level, "trajectory_task_id": g, "step_k": k,
                                "trajectory_steps": st.get("n_steps"), "state_fraction": st.get("fraction"),
                                "context": context, "fraction": p,
                                "mean_H": round(mean([limits[y].H_for(p) for y in scores_S]), 4),
                                "mean_cut": round(mean([limits[y].cut_for(p) for y in scores_S]), 4),
                                "power_S": round(pS, 6), "power_s0": round(p0, 6),
                                "delta_au": round(delta_au(pS, p0), 6),
                                "ratio": (round(ratio(pS, p0), 6) if ratio(pS, p0) is not None else None),
                                "se": (round(se.se, 6) if se.se is not None else None),
                                "sigma2_delta": (round(se.sigma2_delta, 6) if se.sigma2_delta is not None else None),
                                "sigma2_w": (round(se.sigma2_w, 6) if se.sigma2_w is not None else None),
                                "var_observed": (round(se.var_observed, 6) if se.var_observed is not None else None),
                                "G": se.G, "r": round(se.r, 3), "goals_missing": n_missing,
                            })

    # ---- pooled over the sampled states of a level (goals are a shared random effect across trajectories) ----
    pooled: list[dict] = []
    keyed: dict[tuple, list[dict]] = {}
    for row in state_rows:
        keyed.setdefault((row["model"], row["level"], row["context"], row["fraction"]), []).append(row)
    for (model_id, level, context, p), rows in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        d = [r["delta_au"] for r in rows]
        rt = [r["ratio"] for r in rows if r["ratio"] is not None]
        pooled.append({
            "model": model_id, "level": level, "context": context, "fraction": p, "n_states": len(rows),
            "mean_power_S": round(mean([r["power_S"] for r in rows]), 6),
            "mean_power_s0": round(mean([r["power_s0"] for r in rows]), 6),
            "mean_delta_au": round(mean(d), 6),
            "se_across_states": (round((sample_variance(d) / len(d)) ** 0.5, 6) if len(d) >= 2 else None),
            "mean_ratio": (round(mean(rt), 6) if rt else None),
            "mean_H": round(mean([r["mean_H"] for r in rows]), 4),
        })

    inventory = rollout_inventory(run_dir)
    status_counts: dict[str, int] = {}
    term_counts: dict[str, int] = {}
    live_fracs = [r["live_fallback_fraction"] for r in inventory if r["live_fallback_fraction"] is not None]
    restore_mismatch = sum(1 for r in inventory if r["restore_url_matches"] is False)
    for r in inventory:
        status_counts[str(r["status"])] = status_counts.get(str(r["status"]), 0) + 1
        if r["status"] == "done":
            term_counts[str(r["termination"])] = term_counts.get(str(r["termination"]), 0) + 1

    summary = {
        "run": str(run_dir), "profile": cfg.profile, "mode": cfg.run.mode, "judge_h_scoring_mode": cfg.judge.h_scoring_mode,
        "t_method": cfg.judge.t_method, "rollouts": {"by_status": status_counts, "by_termination": term_counts,
                                                     "restore_url_mismatches": restore_mismatch,
                                                     "mean_live_fallback_fraction": (round(mean(live_fracs), 4) if live_fracs else None)},
        "n_state_rows": len(state_rows), "n_goal_rows": len(goal_rows), "notes": notes,
    }

    _write_csv(out_dir / "delta_au.csv", state_rows)
    _write_csv(out_dir / "per_goal.csv", goal_rows)
    _write_csv(out_dir / "by_level.csv", pooled)
    _write_csv(out_dir / "rollouts.csv", inventory)
    (out_dir / "T.json").write_text(json.dumps(T_out, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({**summary, "pooled": pooled}, indent=1), encoding="utf-8")
    return {"summary": summary, "state_rows": state_rows, "pooled": pooled, "T": T_out}


def render_summary(cfg: Config, result: dict[str, Any]) -> str:
    """A readable summary: per model/level/context, delta AU per sampled state at each step limit."""
    s = result["summary"]
    lines = [f"=== Results: {s['run']} (mode={s['mode']}, judge mode={s['judge_h_scoring_mode']}, T via {s['t_method']}) ===",
             f"rollouts by status: {s['rollouts']['by_status']}; by termination: {s['rollouts']['by_termination']}; "
             f"restore URL mismatches: {s['rollouts']['restore_url_mismatches']}; "
             f"mean live-fallback fraction: {s['rollouts']['mean_live_fallback_fraction']}"]
    for n in s["notes"]:
        lines.append(f"note: {n}")
    fractions = list(cfg.experiment.step_limit_fractions)
    for level, goals in result["T"].items():
        if goals:
            Ts = [g["T"] for g in goals.values()]
            lines.append(f"[{level}] T over {len(Ts)} goals: mean {mean(Ts):.2f} steps, min {min(Ts):.1f}, max {max(Ts):.1f}")
    groups: dict[tuple, list[dict]] = {}
    for row in result["state_rows"]:
        groups.setdefault((row["model"], row["level"], row["context"]), []).append(row)
    header = f"{'state (g, k)':24}" + "".join(f"{int(round(p * 100)):>7}%" for p in fractions)
    for (model_id, level, context), rows in sorted(groups.items()):
        lines.append("")
        lines.append(f"--- {model_id} | {level} | {context} context: delta AU = POWER(S,H) - POWER(s0,H) ---")
        lines.append(header)
        states = sorted({(r["trajectory_task_id"], r["step_k"]) for r in rows})
        for g, k in states:
            cells = []
            for p in fractions:
                m = next((r for r in rows if r["trajectory_task_id"] == g and r["step_k"] == k and abs(r["fraction"] - p) < 1e-9), None)
                cells.append(f"{m['delta_au']:>+8.3f}" if m else f"{'':>8}")
            lines.append(f"{g[:8] + ' k=' + str(k):24}" + "".join(cells))
        pooled = [r for r in result["pooled"] if r["model"] == model_id and r["level"] == level and r["context"] == context]
        if pooled:
            cells = "".join(f"{next((r['mean_delta_au'] for r in pooled if abs(r['fraction'] - p) < 1e-9), float('nan')):>+8.3f}" for p in fractions)
            lines.append(f"{'mean over states':24}" + cells)
            cells = "".join(f"{next((r['mean_power_s0'] for r in pooled if abs(r['fraction'] - p) < 1e-9), float('nan')):>8.3f}" for p in fractions)
            lines.append(f"{'POWER(s0,H)':24}" + cells)
            ses = [next((r['se'] for r in rows if abs(r['fraction'] - p) < 1e-9 and r['se'] is not None), None) for p in fractions]
            lines.append(f"{'SE (first state)':24}" + "".join(f"{se:>8.3f}" if se is not None else f"{'n/a':>8}" for se in ses))
    return "\n".join(lines)
