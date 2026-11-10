"""Cost estimator (SPEC.md section 8): predicts calls, tokens and dollars before a run, from the planning
assumptions in config.toml or from measurements in a previous run's ledger, and checks the spending cap."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .tasks import TaskIndex


@dataclass
class CostLine:
    item: str
    assumption: str
    calls: float
    input_tokens: float
    cached_tokens: float
    output_tokens: float
    cost_usd: float


@dataclass
class CostEstimate:
    profile: str | None
    counts: dict[str, Any]
    lines: list[CostLine]
    total_usd: float
    cap_usd: float
    assumptions: dict[str, Any] = field(default_factory=dict)
    spent_usd: float = 0.0                                   # already counted against the budget (this run + other runs)
    spent_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    @property
    def within_cap(self) -> bool:
        """True if the predicted cost fits in what is left of the budget."""
        return self.total_usd <= self.remaining_usd

    def to_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "counts": self.counts, "assumptions": self.assumptions,
                "lines": [line.__dict__ for line in self.lines], "total_usd": self.total_usd,
                "cap_usd": self.cap_usd, "spent_usd": self.spent_usd, "remaining_usd": self.remaining_usd,
                "within_cap": self.within_cap}

    def render(self) -> str:
        c = self.counts
        rows = [f"Cost estimate{' for profile ' + repr(self.profile) if self.profile else ''}",
                f"  rollouts: {c['rollouts_total']:,} total = {c['baseline_rollouts']:,} baseline (s0) + "
                f"{c['trajectory_runs']:,} trajectory + {c['fork_rollouts']:,} forks "
                f"({c['sampled_states']} sampled states x {c['n_aux_goals']} goals x {c['rollouts_per_goal']} rollouts"
                f"{' x 2 contexts' if c['fresh_context_forks'] else ''}); {c['n_models']} model(s), levels {c['levels']}",
                f"  judge calls: {c['judge_calls_total']:,}",
                *([f"  scope: {c['stage']['scope']}; already on disk: "
                   + ", ".join(f"{v} {k}" for k, v in c['stage']['done'].items())
                   + "; still to judge: " + ", ".join(f"{v} {k}" for k, v in c['stage']['to_judge'].items())]
                  if c.get("stage") else []),
                "", f"  {'item':38} {'assumption':58} {'calls':>9} {'input tok':>13} {'cached tok':>12} {'output tok':>11} {'USD':>10}"]
        for line in self.lines:
            rows.append(f"  {line.item:38} {line.assumption[:58]:58} {line.calls:>9,.0f} {line.input_tokens:>13,.0f} "
                        f"{line.cached_tokens:>12,.0f} {line.output_tokens:>11,.0f} {line.cost_usd:>10,.2f}")
        rows.append("")
        rows.append(f"  predicted total: ${self.total_usd:,.2f}")
        d = self.spent_detail
        detail = (f" (this run ${d.get('spent_this_run_usd', 0.0):,.2f}, other runs ${d.get('spent_other_runs_usd', 0.0):,.2f})"
                  if d else "")
        rows.append(f"  budget (cost.spending_cap_usd): ${self.cap_usd:,.2f}; already spent ${self.spent_usd:,.2f}{detail}; "
                    f"remaining ${self.remaining_usd:,.2f}")
        if self.within_cap:
            rows.append("  -> fits within the remaining budget")
        else:
            rows.append("  -> EXCEEDS the remaining budget: the run would stop when spend reaches the budget, "
                        "keeping everything collected until then")
        return "\n".join(rows)


def rollout_counts(cfg: Config, tasks: TaskIndex) -> dict[str, Any]:
    e = cfg.experiment
    n_models = len(cfg.models.agents)
    n_states_per_traj = len(e.state_fractions)
    per_level = {}
    baseline = traj = forks = states = 0
    for level in e.levels:
        pool = tasks.pool(level, exclude=e.excluded_task_ids)
        n_pool = len(pool)
        n_traj = min(e.n_trajectories, n_pool) if not e.trajectory_task_ids else len(e.trajectory_task_ids)
        n_goal_tasks = n_pool if e.baseline_all_tasks else min(n_pool, e.n_aux_goals * n_traj)
        b = n_goal_tasks * e.rollouts_per_goal
        s = n_traj * n_states_per_traj
        f = s * e.n_aux_goals * e.rollouts_per_goal * (2 if e.fresh_context_forks else 1)
        per_level[level] = {"pool": n_pool, "trajectories": n_traj, "baseline_rollouts": b, "sampled_states": s,
                            "fork_rollouts": f}
        baseline += b
        traj += n_traj
        forks += f
        states += s
    return {
        "levels": list(e.levels), "n_models": n_models, "per_level": per_level,
        "baseline_rollouts": baseline * n_models, "trajectory_runs": traj * n_models,
        "fork_rollouts": forks * n_models, "sampled_states": states * n_models,
        "rollouts_total": (baseline + traj + forks) * n_models,
        "n_aux_goals": e.n_aux_goals, "rollouts_per_goal": e.rollouts_per_goal,
        "fresh_context_forks": e.fresh_context_forks, "n_step_limits": len(e.step_limit_fractions),
        "mean_step_limit_fraction": sum(e.step_limit_fractions) / len(e.step_limit_fractions),
        "baseline_rollouts_reference_model": baseline,
    }


def mean_rubric_items(cfg: Config, tasks: TaskIndex) -> float:
    items = [t.m for level in cfg.experiment.levels for t in tasks.pool(level)]
    return sum(items) / len(items) if items else 6.0


def measure_from_ledger(path: Path) -> dict[str, float]:
    """Replace planning assumptions with measurements from a run's ledger."""
    agent_calls = judge_calls = 0
    agent_in = agent_out = agent_cached = 0
    judge_in = judge_out = 0
    rollouts: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("status") != "ok":
                continue
            if e.get("purpose") == "agent_step":
                agent_calls += 1
                agent_in += e.get("prompt_tokens", 0)
                agent_out += e.get("completion_tokens", 0)
                agent_cached += e.get("cached_tokens", 0)
                rid = (e.get("meta") or {}).get("rollout_id")
                if rid:
                    rollouts.add(rid)
            elif str(e.get("purpose", "")).startswith("judge"):
                judge_calls += 1
                judge_in += e.get("prompt_tokens", 0)
                judge_out += e.get("completion_tokens", 0)
    out: dict[str, float] = {}
    if agent_calls:
        out["input_tokens_per_step"] = agent_in / agent_calls
        out["output_tokens_per_step"] = agent_out / agent_calls
        out["cache_fraction"] = agent_cached / agent_in if agent_in else 0.0
        if rollouts:
            out["steps_per_rollout"] = agent_calls / len(rollouts)
    if judge_calls:
        out["judge_input_tokens_full_trajectory"] = judge_in / judge_calls
        out["judge_output_tokens_per_call"] = judge_out / judge_calls
    return out


def estimate(cfg: Config, tasks: TaskIndex, measured: dict[str, float] | None = None, *,
             spent_usd: float = 0.0, spent_detail: dict[str, Any] | None = None,
             counts_override: dict[str, Any] | None = None) -> CostEstimate:
    """Predict the cost of a run from the config, or, with ``counts_override`` (``planned_work`` in the pipeline),
    of what a staged or resumed run still has to do."""
    pl = cfg.cost.planning
    a: dict[str, float] = {
        "steps_per_rollout": pl.steps_per_rollout, "input_tokens_per_step": pl.input_tokens_per_step,
        "output_tokens_per_step": pl.output_tokens_per_step, "cache_fraction": pl.cache_fraction,
        "judge_input_tokens_full_trajectory": pl.judge_input_tokens_full_trajectory,
        "judge_output_tokens_per_call": pl.judge_output_tokens_per_call,
    }
    if measured:
        a.update({k: v for k, v in measured.items() if k in a})
    counts = rollout_counts(cfg, tasks)
    if counts_override:
        counts.update(counts_override)
    m_items = mean_rubric_items(cfg, tasks)
    a["mean_rubric_items"] = m_items
    lines: list[CostLine] = []

    # ---- agent ----
    n_roll = counts["rollouts_total"]
    agent_cost = 0.0
    for spec in cfg.models.agents:
        p = cfg.cost.price_for(spec.id)
        share = n_roll / len(cfg.models.agents)
        in_tok = share * a["steps_per_rollout"] * a["input_tokens_per_step"]
        out_tok = share * a["steps_per_rollout"] * a["output_tokens_per_step"]
        cached = in_tok * a["cache_fraction"]
        cost = ((in_tok - cached) * p.input_per_m + cached * p.cached_input_per_m + out_tok * p.output_per_m) / 1e6
        cost_nocache = (in_tok * p.input_per_m + out_tok * p.output_per_m) / 1e6
        lines.append(CostLine(f"agent {spec.id}"[:38],
                              f"{a['steps_per_rollout']:.0f} steps x {a['input_tokens_per_step']:,.0f} in / "
                              f"{a['output_tokens_per_step']:,.0f} out, {a['cache_fraction']:.0%} cached",
                              share * a["steps_per_rollout"], in_tok, cached, out_tok, cost))
        lines.append(CostLine("  (same, no caching, for reference)", "0% cached", 0, 0, 0, 0, cost_nocache))
        agent_cost += cost

    # ---- judge ----
    jp = cfg.cost.price_for(cfg.models.judge.id)
    full = a["judge_input_tokens_full_trajectory"]
    jout = a["judge_output_tokens_per_call"]
    judge_calls = 0.0
    judge_cost = 0.0

    # T measurement on the reference model's s0 rollouts (section 1): one call per rubric item, or bisection
    n_ref_baseline = counts["baseline_rollouts_reference_model"]
    if cfg.judge.t_method == "first_step":
        calls_T = n_ref_baseline * m_items
        tok_T = calls_T * full
        label_T = "T: first step per item (s0 rollouts)"
        assum_T = f"{n_ref_baseline:,} s0 rollouts x {m_items:.1f} items x 1 call, full trajectory"
    else:
        per_item = math.ceil(math.log2(max(2, a["steps_per_rollout"]))) + 1
        calls_T = n_ref_baseline * m_items * per_item
        tok_T = calls_T * full * 0.5
        label_T = "T: bisection (s0 rollouts)"
        assum_T = f"{n_ref_baseline:,} s0 rollouts x {m_items:.1f} items x {per_item} cuts, ~50% length"
    cost_T = (tok_T * jp.input_per_m + calls_T * jout * jp.output_per_m) / 1e6
    lines.append(CostLine(label_T, assum_T, calls_T, tok_T, 0, calls_T * jout, cost_T))
    judge_calls += calls_T
    judge_cost += cost_T

    # scoring at the step limits H (section 1 / 6): every baseline and fork rollout
    n_scored = counts.get("baseline_to_judge", counts["baseline_rollouts"]) + counts.get("forks_to_judge", counts["fork_rollouts"])
    nH = counts["n_step_limits"]
    pbar = counts["mean_step_limit_fraction"]
    if cfg.judge.h_scoring_mode == "rescore":
        calls_H = n_scored * m_items * nH
        tok_H = calls_H * full * pbar
        label_H = "H scoring: re-score at each step limit"
        assum_H = f"{n_scored:,} rollouts x {m_items:.1f} items x {nH} limits, ~{pbar:.0%} of full length"
    else:
        n_new = n_scored - n_ref_baseline  # the reference model's s0 rollouts already have first-step results
        calls_H = max(0, n_new) * m_items
        tok_H = calls_H * full
        label_H = "H scoring: first step per item"
        assum_H = f"{max(0, n_new):,} rollouts x {m_items:.1f} items x 1 call, full trajectory"
    cost_H = (tok_H * jp.input_per_m + calls_H * jout * jp.output_per_m) / 1e6
    lines.append(CostLine(label_H, assum_H, calls_H, tok_H, 0, calls_H * jout, cost_H))
    judge_calls += calls_H
    judge_cost += cost_H

    counts["judge_calls_total"] = int(round(judge_calls))
    total = agent_cost + judge_cost
    return CostEstimate(profile=cfg.profile, counts=counts, lines=lines, total_usd=total,
                        cap_usd=cfg.cost.spending_cap_usd, assumptions=a, spent_usd=float(spent_usd),
                        spent_detail=dict(spent_detail or {}))
