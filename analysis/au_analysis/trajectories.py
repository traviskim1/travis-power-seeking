"""Analysis 2 (timelines, horizon charts, state at S, per-goal delta with site/topic overlap), analysis 4
(horizon effects, waste) and analysis 7 (baseline vs fork on the same goal). Figures via matplotlib (Agg)."""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .load import RunData, Rollout  # noqa: E402
from .measures import per_goal_table, state_table  # noqa: E402
from .measures import t_selection_table
from .rules import domain_of, entity_tokens, phase_segments, site_names  # noqa: E402

COLORS = {"search": "#4C72B0", "reading": "#55A868", "form": "#C44E52", "stuck": "#8172B2", "invalid": "#937860", "done": "#333333", "other": "#B5B5B5"}


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------

def _draw_timeline(ax, segs: list[dict], n_steps: int, k: int | None, title: str) -> None:
    for sg in segs:
        ax.barh(0, sg["end"] - sg["start"] + 1, left=sg["start"] - 0.5, color=COLORS.get(sg["category"], "#B5B5B5"), edgecolor="white", linewidth=0.5)
        if sg["n"] >= max(3, n_steps // 25):
            ax.text((sg["start"] + sg["end"]) / 2, 0, sg["domain"] if sg["category"] != "done" else "done", ha="center", va="center", fontsize=7,
                    color="white" if sg["category"] in ("search", "form", "stuck", "done") else "black", rotation=0)
    if k is not None:
        ax.axvline(k + 0.5, color="black", linestyle="--", linewidth=1.2)
        ax.text(k + 0.5, 0.45, f" S (k={k})", fontsize=8, va="bottom")
    ax.set_xlim(0.5, max(n_steps, 1) + 0.5)
    ax.set_ylim(-0.6, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("trajectory step")
    ax.set_title(title, fontsize=9, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, list(COLORS), ncol=7, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.55), frameon=False)


def _draw_horizon(ax, states: list[dict], title: str) -> None:
    pts = [s for s in states if s.get("G")]
    if not pts:
        ax.text(0.5, 0.5, "no scored goals", ha="center", transform=ax.transAxes)
        return
    x = [s["mean_cut"] for s in pts]
    ax.plot(x, [s["power_S"] for s in pts], "o-", color="#C44E52", label="forks from S (mean rubric score)")
    ax.plot(x, [s["power_s0"] for s in pts], "s-", color="#4C72B0", label="baseline s0 (mean rubric score)")
    ax.plot(x, [s["perfect_S"] for s in pts], "o:", color="#C44E52", alpha=0.5, markersize=3, label="forks: fraction of goals perfect")
    ax.plot(x, [s["perfect_s0"] for s in pts], "s:", color="#4C72B0", alpha=0.5, markersize=3, label="baseline: fraction perfect")
    for s in pts:
        ax.annotate(f"{int(round(s['fraction'] * 100))}%", (s["mean_cut"], s["power_S"]), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
    ax.set_xlabel("step limit H (mean cut over the goals, steps after the fork); labels: % of T")
    ax.set_ylabel("proportion of auxiliary goals completed")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def trajectory_figure(path: Path, traj_label: str, segs: list[dict], n_steps: int, k: int | None, states: list[dict], G: int) -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [1, 2.6]})
    _draw_timeline(a1, segs, n_steps, k, f"trajectory {traj_label}: phases from the agent's actions and URLs (rules), S = sampled state")
    _draw_horizon(a2, states, f"attainable utility vs step limit H: forks from S vs s0 baseline on the same {G} goals (1 rollout per goal)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def combined_figure(path: Path, panels: list[dict]) -> None:
    n = len(panels)
    fig, axes = plt.subplots(2 * n, 1, figsize=(11, 5.6 * n), gridspec_kw={"height_ratios": [1, 2.4] * n})
    if n == 1:
        axes = list(axes)
    for i, pnl in enumerate(panels):
        _draw_timeline(axes[2 * i], pnl["segs"], pnl["n_steps"], pnl["k"], f"trajectory {pnl['label']}")
        _draw_horizon(axes[2 * i + 1], pnl["states"], f"AU vs H, trajectory {pnl['label']} ({pnl['G']} goals)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def horizon_figure(path: Path, states_by_label: dict[str, list[dict]], boot_by_label: dict[str, list[dict]]) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, states in states_by_label.items():
        pts = [s for s in states if s.get("G")]
        xs = [s["fraction"] * 100 for s in pts]
        a1.plot(xs, [s["delta_au"] for s in pts], "o-", label=label)
        boots = {round(b["fraction"], 6): b for b in boot_by_label.get(label, [])}
        lo = [boots[round(s["fraction"], 6)]["ci95_lo"] if round(s["fraction"], 6) in boots else s["delta_au"] for s in pts]
        hi = [boots[round(s["fraction"], 6)]["ci95_hi"] if round(s["fraction"], 6) in boots else s["delta_au"] for s in pts]
        a1.fill_between(xs, lo, hi, alpha=0.15)
        a2.plot(xs, [s["ratio"] if s["ratio"] is not None else float("nan") for s in pts], "o-", label=label)
    a1.axhline(0, color="black", linewidth=0.8)
    a1.set_xlabel("step limit H as % of T")
    a1.set_ylabel("delta AU = POWER(S,H) - POWER(s0,H)")
    a1.set_title("delta AU vs H (band: 95% paired bootstrap over goals)", fontsize=9)
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    a2.axhline(1, color="black", linewidth=0.8)
    a2.set_xlabel("step limit H as % of T")
    a2.set_ylabel("ratio POWER(S,H) / POWER(s0,H)")
    a2.set_title("ratio vs H", fontsize=9)
    a2.legend(fontsize=7)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def waste_figure(path: Path, waste_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for kind, color in (("baseline", "#4C72B0"), ("fork", "#C44E52")):
        vals = [r["fraction_after_final_score"] for r in waste_rows if r["kind"] == kind and r["fraction_after_final_score"] is not None]
        if vals:
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.6, color=color, label=f"{kind} (n={len(vals)})")
    ax.set_xlabel("fraction of the rollout after the last rubric item was met")
    ax.set_ylabel("rollouts")
    ax.legend(fontsize=8)
    ax.set_title("steps spent after the final rubric score was reached", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Analysis 2
# --------------------------------------------------------------------------------------------

def state_at_S(run: RunData, traj: Rollout, k: int) -> dict[str, Any]:
    """What the agent had done by S, from steps.jsonl (<= k) and the snapshot at k."""
    steps = [s for s in traj.steps if int(s.get("step_num", 0)) <= k]
    domains = []
    for s in steps:
        d = domain_of(s.get("url") or "")
        if d and (not domains or domains[-1] != d):
            domains.append(d)
    searches = [str((s.get("arguments") or {}).get("text") or "") for s in steps if s.get("action") == "type" and domain_of(s.get("url") or "") in ("google.com", "bing.com", "duckduckgo.com")]
    typed = [{"step": s["step_num"], "domain": domain_of(s.get("url") or ""), "text": str((s.get("arguments") or {}).get("text") or "")[:80]}
             for s in steps if s.get("action") == "type"]
    snap = run.snapshot(traj.goal_id, k)
    b = snap.get("browser") or {}
    tabs = [{"index": i, "url": t.get("url"), "title": (t.get("title") or "")[:80], "active": i == b.get("active_tab"),
             "form_values": [fv for fv in (t.get("form_values") or []) if str(fv.get("value") or "").strip()]} for i, t in enumerate(b.get("tabs") or [])]
    cookies = (b.get("storage_state") or {}).get("cookies") or []
    cookie_domains = collections.Counter(str(c.get("domain") or "").lstrip(".") for c in cookies)
    origins = [o.get("origin") for o in (b.get("storage_state") or {}).get("origins") or []]
    info = []
    for s in steps:
        txt = str(s.get("response") or "")
        if any(w in txt.lower() for w in ("found", "noted", "price", "$", "£", "confirmed", "shows")) and len(txt) > 30:
            info.append({"step": s["step_num"], "text": txt[:220].replace("\n", " ")})
    return {"k": k, "n_steps_trajectory": traj.n_steps, "domains_visited_in_order": domains, "searches_typed": searches[:20], "typed": typed[:30],
            "tabs_open_at_S": tabs, "n_tabs": len(tabs), "cookies_total": len(cookies), "cookie_domains": dict(cookie_domains.most_common(25)),
            "local_storage_origins": origins, "downloads": snap.get("downloads"), "half_filled_forms": [t for t in tabs if t["form_values"]],
            "information_found_candidates": info[-12:], "last_thought_before_S": str(steps[-1].get("response") or "")[:400] if steps else "",
            "snapshot_dir": str(snap.get("dir")), "snapshot_exists": snap.get("exists")}


def goal_overlap_table(run: RunData, traj: Rollout, k: int, per_goal: list[dict]) -> list[dict]:
    """Per goal: delta AU at each fraction, and whether the goal shares sites or topics with what the agent did before S."""
    pre = [s for s in traj.steps if int(s.get("step_num", 0)) <= k]
    traj_domains = {domain_of(s.get("url") or "") for s in pre} - {""}
    g_task = run.task(traj.goal_id)
    g_cats = set(g_task.categories or [])
    rows = []
    for goal in run.goals_of(traj.goal_id):
        y = run.task(goal)
        y_domains = {domain_of(s.get("url") or "") for b in run.baselines(goal) for s in b.steps} - {""}
        named = site_names(y.text)
        shared_sites = sorted((traj_domains & y_domains) | {d for d in traj_domains if any(n.split(".")[0] in d for n in named if len(n) > 3)})
        shared_sites = [d for d in shared_sites if d not in ("google.com",)]
        shared_cats = sorted(g_cats & set(y.categories or []))
        row = {"goal_id": goal, "categories": " | ".join(y.categories or []), "shares_category_with_g": bool(shared_cats),
               "shared_sites_with_pre_S": " ".join(shared_sites), "shares_site": bool(shared_sites),
               "related": bool(shared_sites) or bool(shared_cats)}
        for r in per_goal:
            if r["goal_id"] == goal:
                row[f"delta_{int(round(r['fraction'] * 100))}"] = r["delta"]
                row[f"cut_{int(round(r['fraction'] * 100))}"] = r["cut"]
        rows.append(row)
    return rows


def overlap_summary(rows: list[dict], fractions: list[float]) -> list[dict]:
    out = []
    for key, label in (("shares_site", "shares a site visited before S"), ("shares_category_with_g", "shares a category with g"), ("related", "either")):
        for p in fractions:
            col = f"delta_{int(round(p * 100))}"
            a = [r[col] for r in rows if r.get(key) and r.get(col) is not None]
            b = [r[col] for r in rows if not r.get(key) and r.get(col) is not None]
            out.append({"grouping": label, "fraction": p, "n_related": len(a), "mean_delta_related": statistics.mean(a) if a else None,
                        "n_unrelated": len(b), "mean_delta_unrelated": statistics.mean(b) if b else None,
                        "difference": (statistics.mean(a) - statistics.mean(b)) if a and b else None})
    return out


# --------------------------------------------------------------------------------------------
# Analysis 4: waste; Analysis 7: paths
# --------------------------------------------------------------------------------------------

def waste_table(run: RunData) -> list[dict]:
    rows = []
    for r in run.rollouts:
        if r.kind == "trajectory" or not r.done:
            continue
        fm = r.first_met_from_first_step()
        src = "first_step.json"
        if fm is None:
            fm = r.first_met_from_rescore() if r.rescore else None
            src = "rescore cuts"
        if not fm:
            continue
        met = [s for s in fm.values() if s is not None]
        last = max(met) if met else None
        rows.append({"rollout_id": r.rollout_id, "kind": r.kind, "goal_id": r.goal_id, "n_steps": r.n_steps, "termination": r.termination,
                     "items_met": len(met), "items_total": len(fm), "last_item_met_at": last,
                     "steps_after_final_score": (r.n_steps - last) if last is not None else None,
                     "fraction_after_final_score": ((r.n_steps - last) / r.n_steps) if (last is not None and r.n_steps) else None,
                     "first_met_source": src})
    return rows


def path_table(run: RunData, traj: Rollout, k: int, context: str = "full") -> list[dict]:
    """Baseline vs fork on the same goal: steps, scores, tab reuse, shared domains, divergence."""
    pre_domains = [domain_of(s.get("url") or "") for s in traj.steps if int(s.get("step_num", 0)) <= k]
    snap = run.snapshot(traj.goal_id, k)
    n_tabs_at_S = len(((snap.get("browser") or {}).get("tabs") or []))
    rows = []
    for goal in run.goals_of(traj.goal_id):
        b = next((x for x in run.baselines(goal) if x.done), None)
        f = next((x for x in run.forks(traj.goal_id, k, context) if x.goal_id == goal and x.done), None)
        if b is None or f is None:
            rows.append({"goal_id": goal, "note": "baseline or fork missing"})
            continue
        bd = [domain_of(s.get("url") or "") for s in b.steps]
        fd = [domain_of(s.get("url") or "") for s in f.steps]
        switch_old = [s for s in f.steps if s.get("action") == "switch_tab" and int(((s.get("arguments") or {}).get("index") or 0)) < n_tabs_at_S]
        reused_inherited = bool(switch_old) or (len(fd) > 1 and fd[1] == fd[0] and fd[0] in set(pre_domains) and fd[0] not in set(bd[:1]))
        common = [d for d in dict.fromkeys(fd) if d in set(bd)]
        first_common = None
        for i, d in enumerate(fd):
            if d in set(bd):
                first_common = (i + 1, bd.index(d) + 1, d)
                break
        rows.append({"goal_id": goal, "steps_s0": b.n_steps, "steps_fork": f.n_steps, "diff_fork_minus_s0": f.n_steps - b.n_steps,
                     "score_s0_full": b.final_score, "score_fork_full": f.final_score, "cut_judged_s0": b.largest_judged_cut, "cut_judged_fork": f.largest_judged_cut,
                     "termination_s0": b.termination, "termination_fork": f.termination,
                     "fork_starts_on": fd[0] if fd else "", "fork_switched_to_inherited_tab": bool(switch_old),
                     "fork_used_pre_S_site": bool(set(fd[1:]) & set(pre_domains) - {"google.com"}),
                     "pre_S_sites_used_by_fork": " ".join(sorted(set(fd[1:]) & set(pre_domains) - {"google.com"})),
                     "first_common_domain": first_common[2] if first_common else "", "fork_step_reaching_it": first_common[0] if first_common else None,
                     "s0_step_reaching_it": first_common[1] if first_common else None, "domains_s0": " > ".join(dict.fromkeys(bd)),
                     "domains_fork": " > ".join(dict.fromkeys(fd)), "first_thought_s0": str(b.steps[0].get("response") or "")[:200].replace("\n", " ") if b.steps else "",
                     "first_thought_fork": str(f.steps[0].get("response") or "")[:200].replace("\n", " ") if f.steps else ""})
    return rows


def run_trajectories(run: RunData, out_dir: Path, boot_by_state: dict[str, list[dict]] | None = None) -> dict[str, Any]:
    """Everything per trajectory, plus the combined and horizon figures. Returns tables, figures, notes, problems."""
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict]] = {}
    notes: list[str] = []
    problems: list[str] = []
    per_traj: dict[str, Any] = {}
    panels = []
    level = run.levels[0]
    context = "full"
    for i, t in enumerate(run.trajectories_planned(level), 1):
        g = t["task_id"]
        label = f"{i} ({g[:10]})"
        traj = run.trajectory(g)
        entry: dict[str, Any] = {"index": i, "task_id": g, "label": label, "task_text": run.task(g).text}
        if traj is None:
            entry["status"] = "not run"
            per_traj[g] = entry
            continue
        states = run.states_of(g)
        entry["status"] = traj.status
        entry["n_steps"] = traj.n_steps
        entry["termination"] = traj.termination
        segs = phase_segments(traj.steps)
        tables[f"trajectory_{i}_phases.csv"] = [{"start": s["start"], "end": s["end"], "n": s["n"], "category": s["category"], "domain": s["domain"], "label": s["label"]} for s in segs]
        entry["segments"] = segs
        entry["states"] = states
        if not states:
            entry["note"] = "trajectory ran but no sampled state was saved"
            trajectory_figure(figs / f"trajectory_{i}.png", label, segs, traj.n_steps, None, [], 0)
            per_traj[g] = entry
            continue
        k = int(states[0]["step"])
        entry["k"] = k
        pg = per_goal_table(run, g, k, context)
        st = state_table(pg)
        entry["per_goal"] = pg
        entry["states_table"] = st
        tables[f"trajectory_{i}_per_goal.csv"] = pg
        tables[f"trajectory_{i}_state.csv"] = st
        ts_rows, ts_sum = t_selection_table(run, g, k, context)
        entry["t_selection"] = ts_sum
        entry["t_selection_rows"] = ts_rows
        tables[f"trajectory_{i}_t_selection.csv"] = ts_rows
        R = int(run.config["experiment"]["rollouts_per_goal"])
        if R == 1 and ts_sum.get("n_distinct_cuts"):
            problems.append(f"trajectory {label}: with r = 1 the goal's T comes from the same s0 rollout the fork is compared with, so POWER(s0, H = T) "
                            f"is that rollout's own final score by construction: between the {int(round(ts_sum['fraction_lo'] * 100))}% and the 100% cut the "
                            f"s0 score rises by {ts_sum['mean_s0_gain']:+.3f} on average ({ts_sum['n_s0_gain_pos']} of {ts_sum['n_distinct_cuts']} goals) against "
                            f"{ts_sum['mean_fork_gain']:+.3f} for the forks ({ts_sum['n_fork_gain_pos']} of {ts_sum['n_distinct_cuts']}); delta AU at H = T is biased "
                            f"against S and the cuts just below T in its favour (tables/trajectory_{i}_t_selection.csv)")
        problems += [f"trajectory {i}: {p}" for p in __import__("au_analysis.measures", fromlist=["compare_with_pipeline"]).compare_with_pipeline(run, pg, st, g, k, context)]
        s_at = state_at_S(run, traj, k)
        entry["state_at_S"] = s_at
        (out_dir / "tables" / f"trajectory_{i}_state_at_S.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "tables" / f"trajectory_{i}_state_at_S.json").write_text(json.dumps(s_at, indent=1, default=str), encoding="utf-8")
        ov = goal_overlap_table(run, traj, k, pg)
        tables[f"trajectory_{i}_goal_delta.csv"] = ov
        tables[f"trajectory_{i}_overlap_summary.csv"] = overlap_summary(ov, run.fractions)
        entry["overlap"] = ov
        entry["overlap_summary"] = tables[f"trajectory_{i}_overlap_summary.csv"]
        paths = path_table(run, traj, k, context)
        tables[f"trajectory_{i}_paths.csv"] = paths
        entry["paths"] = paths
        G = max((s.get("G", 0) for s in st), default=0)
        trajectory_figure(figs / f"trajectory_{i}.png", label, segs, traj.n_steps, k, st, G)
        panels.append({"label": label, "segs": segs, "n_steps": traj.n_steps, "k": k, "states": st, "G": G})
        entry["figure"] = f"figures/trajectory_{i}.png"
        per_traj[g] = entry
    if panels:
        combined_figure(figs / "trajectories_combined.png", panels)
    # horizon figure over the states that exist
    states_by_label = {e["label"]: e["states_table"] for e in per_traj.values() if e.get("states_table")}
    if states_by_label:
        horizon_figure(figs / "horizon_delta_ratio.png", states_by_label, boot_by_state or {})
    waste = waste_table(run)
    tables["waste.csv"] = waste
    if waste:
        waste_figure(figs / "waste.png", waste)
    return {"tables": tables, "notes": notes, "problems": problems, "per_traj": per_traj, "waste": waste}
