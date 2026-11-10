"""The stages of the method (SPEC.md sections 4 and 8), each idempotent and resumable:

plan          choose trajectory tasks and draw the auxiliary goals (fixed seeds, IDs recorded)
baseline      s0 rollouts: every goal task as the agent's first task, r rollouts each
trajectories  run each trajectory task g with a snapshot after every step; choose the sampled states
forks         from every sampled state: 30 goals x r rollouts, full context and fresh context
judge         T on the s0 rollouts, then the rubric at every step limit for every rollout
analyze       POWER, delta AU, ratio and SE per state, H, context, level and model

The budget catcher: the ledger refuses the first model call that would take real spend past
cost.spending_cap_usd, and every call after it. Rollouts and judge jobs in flight save what they have and
exit, nothing else starts, the analysis is run on everything saved so far (marked PARTIAL), and
budget_stop.json records the stop. Re-running the same command after raising the budget continues from there.

Provider pinning (SKILL.md): before any model call the pre-flight fetches every model's endpoint list, checks
its pin, keeps the list with the run and diffs it against earlier snapshots; live runs then make one minimal
call per model through its pin (the probe). Every call carries the pin; who served it goes to the ledger and
to results/routing.json.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from .analysis import analyze, compute_goal_limits, render_summary, rollout_inventory
from .client import SpendingCapExceeded, make_client, make_ledger
from .config import Config
from .judge import judge_at_cut, judge_bisect_first_steps, judge_first_steps
from .layout import Layout
from .prompts import first_task_message, interrupt_message
from .rollout import (RolloutSpec, RunContext, StartFromScratch, StartFromSnapshot, load_result, run_rollout)
from .routing import (PROBE_FILE, RoutingError, preflight_problems, preflight_records, probe_pins, routing_notes,
                      run_preflight, write_routing_summary)
from .snapshots import load_snapshot
from .stats import sampled_state_steps
from .tasks import TaskIndex, derive_seed, load_tasks, sample_aux_goals, sample_trajectory_tasks

log = logging.getLogger("aupower.pipeline")
STAGES = ("plan", "baseline", "trajectories", "forks", "judge", "analyze")
SPENDING_STAGES = ("baseline", "trajectories", "forks", "judge")
GATHER_CHUNK = 256      # jobs handed to asyncio.gather at a time; later chunks are never started after a budget stop


def build_plan(cfg: Config, tasks: TaskIndex) -> dict[str, Any]:
    """The plan (trajectory tasks and the goals drawn for each, from fixed seeds) without touching the disk."""
    e = cfg.experiment
    plan: dict[str, Any] = {"created_at": time.time(), "seed_base": cfg.seeds.base,
                            "models": [m.id for m in cfg.models.agents], "judge": cfg.models.judge.id,
                            "t_reference_model": cfg.models.t_reference.id, "levels": {}}
    for level in e.levels:
        pool = tasks.pool(level, exclude=e.excluded_task_ids)
        pool_ids = sorted(t.task_id for t in pool)
        if e.trajectory_task_ids:
            traj_ids = [t for t in e.trajectory_task_ids if t in pool_ids]
        else:
            traj_ids = sample_trajectory_tasks(pool, min(e.n_trajectories, len(pool)),
                                               derive_seed(cfg.seeds.base, "trajectories", level))
        trajectories = []
        for g in traj_ids:
            seed = derive_seed(cfg.seeds.base, "goals", level, g)
            goals = sample_aux_goals(pool, g, e.n_aux_goals, seed)
            trajectories.append({"task_id": g, "goal_seed": seed, "goals": goals})
        used = sorted({y for t in trajectories for y in t["goals"]})
        plan["levels"][level] = {"pool": pool_ids, "trajectories": trajectories,
                                 "baseline_tasks": pool_ids if e.baseline_all_tasks else used}
    return plan


def resolve_trajectories(plan: dict, specs: list[str] | None) -> dict[str, list[str]] | None:
    """Which trajectories a run is restricted to (``--trajectory``): each spec is a 1-based position in the plan's
    trajectory list ("1", "2", ...; the order of experiment.trajectory_task_ids) or a prefix of a task id; specs may
    be repeated or comma-separated. None or empty = every trajectory. Returns {level: [task ids]}."""
    if not specs:
        return None
    items = [x.strip() for spec in specs for x in str(spec).split(",") if x.strip()]
    if not items:
        return None
    out: dict[str, list[str]] = {}
    matched: set[str] = set()
    for level, lp in plan["levels"].items():
        ids = [t["task_id"] for t in lp["trajectories"]]
        chosen: list[str] = []
        for item in items:
            if item.isdigit():
                i = int(item)
                if not 1 <= i <= len(ids):
                    raise ValueError(f"--trajectory {item}: level {level!r} has {len(ids)} trajectories, numbered 1..{len(ids)}")
                tid = ids[i - 1]
            else:
                hits = [t for t in ids if t.startswith(item)]
                if len(hits) > 1:
                    raise ValueError(f"--trajectory {item!r} matches {len(hits)} trajectory tasks at level {level!r}")
                if not hits:
                    continue
                tid = hits[0]
            matched.add(item)
            if tid not in chosen:
                chosen.append(tid)
        out[level] = chosen
    unknown = [x for x in items if x not in matched]
    if unknown:
        raise ValueError(f"--trajectory {unknown}: no trajectory task of the plan starts with that; the plan has "
                         + "; ".join(f"{lvl}: " + ", ".join(f"{i + 1}={t['task_id'][:10]}" for i, t in enumerate(lp["trajectories"]))
                                     for lvl, lp in plan["levels"].items()))
    return out


def planned_work(cfg: Config, tasks: TaskIndex, plan: dict, selected: dict[str, list[str]] | None, run_dir: Path) -> dict[str, Any]:
    """What a run restricted to ``selected`` trajectories (None = all) still has to do, given what is on disk
    under run_dir: rollouts to run and rollouts to judge, by kind, plus the counts the cost estimate needs. The
    stages themselves skip finished rollouts and judge results; this only predicts."""
    layout = Layout(run_dir)
    e = cfg.experiment
    R = e.rollouts_per_goal
    contexts = ["full"] + (["fresh"] if e.fresh_context_forks else [])
    todo = {"baseline": 0, "trajectory": 0, "fork": 0}
    done = {"baseline": 0, "trajectory": 0, "fork": 0}
    to_judge = {"baseline": 0, "fork": 0}
    ref_id = cfg.models.t_reference.id
    ref_to_judge = 0
    states = 0
    scope: dict[str, list[str]] = {}

    def is_done(d: Path) -> bool:
        r = load_result(d)
        return r is not None and r.status == "done"

    def is_judged(d: Path) -> bool:
        j = d / "judge"
        return j.is_dir() and any(j.glob("*.json"))

    for level, lp in plan["levels"].items():
        keep = None if selected is None else set(selected.get(level, []))
        trajs = [t for t in lp["trajectories"] if keep is None or t["task_id"] in keep]
        scope[level] = [t["task_id"] for t in trajs]
        base_tasks = list(lp["baseline_tasks"]) if keep is None else sorted({y for t in trajs for y in t["goals"]})
        for model in cfg.models.agents:
            for y in base_tasks:
                for r in range(R):
                    d = layout.baseline_dir(model, level, y, r)
                    if is_done(d):
                        done["baseline"] += 1
                        judged = is_judged(d)
                    else:
                        todo["baseline"] += 1
                        judged = False
                    if not judged:
                        to_judge["baseline"] += 1
                        if model.id == ref_id:
                            ref_to_judge += 1
            for traj in trajs:
                g = traj["task_id"]
                if is_done(layout.traj_rollout_dir(model, level, g)):
                    done["trajectory"] += 1
                else:
                    todo["trajectory"] += 1
                sp = layout.states_path(model, level, g)
                ks = [int(st["step"]) for st in json.loads(sp.read_text(encoding="utf-8"))["states"]] if sp.exists() else None
                n_states = len(ks) if ks is not None else len(e.state_fractions)
                states += n_states
                for goal_id in traj["goals"]:
                    for context in contexts:
                        for r in range(R):
                            if ks is None:                   # the trajectory has not run: its forks are all ahead
                                todo["fork"] += n_states
                                to_judge["fork"] += n_states
                                continue
                            for k in ks:
                                d = layout.fork_dir(model, level, g, k, context, goal_id, r)
                                if is_done(d):
                                    done["fork"] += 1
                                    if not is_judged(d):
                                        to_judge["fork"] += 1
                                else:
                                    todo["fork"] += 1
                                    to_judge["fork"] += 1
    parts = []
    for lvl, ids in scope.items():
        all_ids = [t["task_id"] for t in plan["levels"][lvl]["trajectories"]]
        if selected is None:
            parts.append(f"{lvl}: all {len(all_ids)} trajectories")
        else:
            parts.append(f"{lvl}: trajectory " + ", ".join(f"{all_ids.index(t) + 1} of {len(all_ids)} ({t[:10]})" for t in ids))
    scope_text = "; ".join(parts)
    return {"scope": scope, "scope_text": scope_text, "todo": todo, "done": done, "to_judge": to_judge,
            "counts_override": {
                "baseline_rollouts": todo["baseline"], "trajectory_runs": todo["trajectory"], "fork_rollouts": todo["fork"],
                "sampled_states": states, "rollouts_total": sum(todo.values()),
                "baseline_rollouts_reference_model": ref_to_judge,
                "baseline_to_judge": to_judge["baseline"], "forks_to_judge": to_judge["fork"],
                "stage": {"scope": scope_text, "done": dict(done), "to_judge": dict(to_judge)}}}


class Pipeline:
    def __init__(self, cfg: Config, trajectories: list[str] | None = None):
        self.cfg = cfg
        self.trajectory_specs = list(trajectories or [])        # --trajectory values; resolved against the plan
        self.selected: dict[str, list[str]] | None = None       # {level: [task ids]}; None = every trajectory
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.layout = Layout(self.run_dir)
        self.tasks = TaskIndex(load_tasks(cfg.tasks_path))
        self.ledger = make_ledger(cfg, self.layout.ledger_path)
        self.client = make_client(cfg, self.ledger)
        self.ctx = RunContext(cfg, self.tasks, self.client, self.ledger)
        self.preflight_report: dict[str, Any] | None = None
        (self.run_dir / "config.resolved.json").write_text(json.dumps(cfg.to_dict(), indent=1), encoding="utf-8")
        log.info("run directory %s (mode=%s, browser=%s, client=%s)", self.run_dir, cfg.run.mode,
                 cfg.run.effective_browser, "mock" if self.client.is_mock else "openrouter")
        log.info("budget $%.2f (cost.spending_cap_usd): %s; remaining $%.4f", self.ledger.cap_usd,
                 self.ledger.describe(), self.ledger.remaining_usd)

    async def close(self) -> None:
        await self.client.close()

    # ---------------------------------------------------------------- provider pinning
    async def preflight(self) -> dict[str, Any]:
        """Endpoint lists fetched, pins checked, snapshot saved and diffed (free; the mock list in dry runs).
        The report says whether every pin can be used; run_stages refuses to start when one cannot."""
        self.preflight_report = await run_preflight(self.client, self.cfg, self.run_dir)
        return self.preflight_report

    async def probe(self) -> dict[str, Any]:
        """One minimal call per model through its pin before any rollout (live: fractions of a cent)."""
        if not self.cfg.routing.probe:
            log.warning("routing.probe is off: the pins are not tried before the rollouts")
            return {"skipped": True}
        path = self.run_dir / PROBE_FILE
        pins = {m.id: m.endpoint for m in self.cfg.models.agents + [self.cfg.models.judge]}
        if path.exists():                     # resuming: the same pins were proven in this run already
            try:
                earlier = json.loads(path.read_text(encoding="utf-8"))
                done = earlier.get("results") or {}
                if all(done.get(mid, {}).get("ok") and done[mid].get("endpoint") == pin for mid, pin in pins.items()):
                    log.info("routing probe skipped: the same pins were proven in this run at %s",
                             time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(earlier.get("ts") or 0))))
                    return done
            except Exception:
                pass
        results = await probe_pins(self.client, self.cfg, preflight_records(self.preflight_report or {}))
        path.write_text(json.dumps({"ts": time.time(), "results": results}, indent=1), encoding="utf-8")
        return results

    # ---------------------------------------------------------------- plan
    def plan(self) -> dict[str, Any]:
        cfg = self.cfg
        path = self.layout.plan_path
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
            if plan.get("seed_base") != cfg.seeds.base:
                raise RuntimeError(f"{path} was made with seed {plan.get('seed_base')} but the config has "
                                   f"{cfg.seeds.base}; use a different run name or delete the plan")
        else:
            plan = build_plan(cfg, self.tasks)
            path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
            log.info("plan written: %s", {lvl: {"trajectories": len(v["trajectories"]), "baseline_tasks": len(v["baseline_tasks"])}
                                           for lvl, v in plan["levels"].items()})
        if self.selected is None and self.trajectory_specs:
            self.selected = resolve_trajectories(plan, self.trajectory_specs)
            log.info("restricted to trajectories %s (the analysis still covers every trajectory with data)",
                     {lvl: [t[:10] for t in ids] for lvl, ids in self.selected.items()})
        return plan

    def _trajectories(self, level: str, lp: dict) -> list[dict]:
        """The plan's trajectories at a level, or only the selected ones."""
        if self.selected is None:
            return list(lp["trajectories"])
        keep = set(self.selected.get(level, []))
        return [t for t in lp["trajectories"] if t["task_id"] in keep]

    def _baseline_tasks(self, level: str, lp: dict) -> list[str]:
        """The s0 rollouts to run: every planned baseline task, or, when restricted to some trajectories, the
        goals those trajectories use (the rest can be topped up later by a run without --trajectory)."""
        if self.selected is None:
            return list(lp["baseline_tasks"])
        return sorted({y for t in self._trajectories(level, lp) for y in t["goals"]})

    # ---------------------------------------------------------------- helpers
    async def _gather_stoppable(self, coros: list) -> list:
        """asyncio.gather in chunks. Once the budget has tripped, the remaining chunks are not started at
        all (their coroutines are closed unrun), so the run winds down as soon as the jobs in flight end."""
        results: list = []
        for i in range(0, len(coros), GATHER_CHUNK):
            chunk = coros[i:i + GATHER_CHUNK]
            if self.ctx.stop.is_set():
                for c in chunk:
                    c.close()
                continue
            results.extend(await asyncio.gather(*chunk, return_exceptions=True))
        return results

    async def _run_many(self, jobs: list[tuple[RolloutSpec, Any]], label: str) -> list:
        if not jobs:
            log.info("%s: nothing to run", label)
            return []
        todo = [(s, st) for s, st in jobs if not (load_result(s.dir) and load_result(s.dir).status == "done")]
        log.info("%s: %d rollouts (%d already done)", label, len(jobs), len(jobs) - len(todo))
        t0 = time.time()
        results = await self._gather_stoppable([run_rollout(self.ctx, s, st) for s, st in todo])
        caps = [r for r in results if isinstance(r, SpendingCapExceeded)]
        errors = [r for r in results if isinstance(r, Exception) and not isinstance(r, SpendingCapExceeded)]
        done = sum(1 for r in results if not isinstance(r, Exception) and r.status == "done")
        failed = sum(1 for r in results if not isinstance(r, Exception) and r.status != "done")
        log.info("%s: %d done, %d failed, %d exceptions, %d stopped or skipped by the budget in %.1fs; spend so far $%.4f",
                 label, done, failed, len(errors), len(caps), time.time() - t0, self.ledger.spent_usd)
        for exc in errors[:5]:
            log.error("%s: %r", label, exc)
        if caps:
            raise caps[0]
        return results

    def _start_url(self, task) -> str:
        return self.cfg.experiment.start_url or task.website

    # ---------------------------------------------------------------- stages
    async def stage_baseline(self) -> None:
        plan = self.plan()
        jobs = []
        for model in self.cfg.models.agents:
            for level, lp in plan["levels"].items():
                for task_id in self._baseline_tasks(level, lp):
                    task = self.tasks[task_id]
                    for r in range(self.cfg.experiment.rollouts_per_goal):
                        spec = RolloutSpec(rollout_id=self.layout.baseline_id(model, level, task_id, r),
                                           dir=self.layout.baseline_dir(model, level, task_id, r), model=model, level=level,
                                           goal_id=task_id, kind="baseline", rollout_index=r)
                        jobs.append((spec, StartFromScratch(self._start_url(task), first_task_message(task.text))))
        await self._run_many(jobs, "baseline (s0)")

    async def stage_trajectories(self) -> None:
        plan = self.plan()
        jobs = []
        for model in self.cfg.models.agents:
            for level, lp in plan["levels"].items():
                for traj in self._trajectories(level, lp):
                    g = traj["task_id"]
                    task = self.tasks[g]
                    spec = RolloutSpec(rollout_id=self.layout.traj_id(model, level, g),
                                       dir=self.layout.traj_rollout_dir(model, level, g), model=model, level=level,
                                       goal_id=g, kind="trajectory", snapshot_every_step=True)
                    jobs.append((spec, StartFromScratch(self._start_url(task), first_task_message(task.text))))
        await self._run_many(jobs, "trajectories")
        # choose the sampled states
        for model in self.cfg.models.agents:
            for level, lp in plan["levels"].items():
                for traj in self._trajectories(level, lp):
                    g = traj["task_id"]
                    res = load_result(self.layout.traj_rollout_dir(model, level, g))
                    if res is None or res.status != "done":
                        continue
                    ks = sampled_state_steps(res.n_steps, self.cfg.experiment.state_fractions)
                    states = []
                    for f, k in zip(self.cfg.experiment.state_fractions, ks):
                        snap_dir = self.layout.snapshot_dir(model, level, g, k)
                        if snap_dir.exists():
                            states.append({"step": k, "fraction": f, "n_steps": res.n_steps, "snapshot_dir": str(snap_dir)})
                    self.layout.states_path(model, level, g).write_text(
                        json.dumps({"task_id": g, "model": model.id, "n_steps": res.n_steps, "termination": res.termination,
                                    "states": states}, indent=1), encoding="utf-8")
                    log.info("%s/%s/%s: %d steps, sampled states at k=%s", model.slug, level, g[:8], res.n_steps, ks)

    def _fork_jobs(self, plan: dict) -> list[tuple[RolloutSpec, StartFromSnapshot]]:
        jobs = []
        contexts = ["full"] + (["fresh"] if self.cfg.experiment.fresh_context_forks else [])
        for model in self.cfg.models.agents:
            for level, lp in plan["levels"].items():
                for traj in self._trajectories(level, lp):
                    g = traj["task_id"]
                    sp = self.layout.states_path(model, level, g)
                    if not sp.exists():
                        continue
                    for st in json.loads(sp.read_text(encoding="utf-8"))["states"]:
                        k = int(st["step"])
                        snapshot = load_snapshot(self.layout.snapshot_dir(model, level, g, k))
                        for context in contexts:
                            for goal_id in traj["goals"]:
                                goal = self.tasks[goal_id]
                                msg = interrupt_message(goal.text) if context == "full" else first_task_message(goal.text)
                                for r in range(self.cfg.experiment.rollouts_per_goal):
                                    spec = RolloutSpec(rollout_id=self.layout.fork_id(model, level, g, k, context, goal_id, r),
                                                       dir=self.layout.fork_dir(model, level, g, k, context, goal_id, r),
                                                       model=model, level=level, goal_id=goal_id, kind="fork", rollout_index=r,
                                                       context=context, origin={"trajectory_task_id": g, "step": k})
                                    jobs.append((spec, StartFromSnapshot(snapshot, context, msg)))
        return jobs

    async def stage_forks(self) -> None:
        plan = self.plan()
        await self._run_many(self._fork_jobs(plan), "forks")

    async def stage_judge(self) -> None:
        plan = self.plan()
        cfg = self.cfg
        R = cfg.experiment.rollouts_per_goal
        ref = cfg.models.t_reference
        contexts = ["full"] + (["fresh"] if cfg.experiment.fresh_context_forks else [])

        # 1. T: first-met steps on the reference model's s0 rollouts of every goal in use
        t_jobs = []
        for level, lp in plan["levels"].items():
            for goal_id in sorted({y for t in self._trajectories(level, lp) for y in t["goals"]}):
                task = self.tasks[goal_id]
                for r in range(R):
                    d = self.layout.baseline_dir(ref, level, goal_id, r)
                    res = load_result(d)
                    if res is None or res.status != "done":
                        continue
                    rid = self.layout.baseline_id(ref, level, goal_id, r)
                    if cfg.judge.t_method == "bisect":
                        t_jobs.append(judge_bisect_first_steps(self.ctx, d, task, rid))
                    else:
                        t_jobs.append(judge_first_steps(self.ctx, d, task, rid))
        log.info("judge: measuring T on %d s0 rollouts", len(t_jobs))
        await self._gather_judge(t_jobs, "judge T")

        # 2. scores at every step limit for every s0 and fork rollout. Jobs go goal by goal (the goal's s0
        #    rollouts, then its forks from every state and context), so that a budget stop leaves whole
        #    goals scored on both sides for every state rather than every goal half scored.
        jobs = []
        n_rollouts = 0
        for level, lp in plan["levels"].items():
            limits = compute_goal_limits(cfg, self.layout, plan, level)
            for model in cfg.models.agents:
                targets: list[tuple[Path, str, str]] = []   # (dir, rollout_id, goal_id)
                for traj in self._trajectories(level, lp):
                    g = traj["task_id"]
                    sp = self.layout.states_path(model, level, g)
                    states = [int(st["step"]) for st in json.loads(sp.read_text(encoding="utf-8"))["states"]] if sp.exists() else []
                    for y in traj["goals"]:
                        for r in range(R):
                            targets.append((self.layout.baseline_dir(model, level, y, r), self.layout.baseline_id(model, level, y, r), y))
                        for k in states:
                            for context in contexts:
                                for r in range(R):
                                    targets.append((self.layout.fork_dir(model, level, g, k, context, y, r),
                                                    self.layout.fork_id(model, level, g, k, context, y, r), y))
                seen: set[Path] = set()
                for d, rid, y in targets:
                    if d in seen or y not in limits:
                        continue
                    seen.add(d)
                    res = load_result(d)
                    if res is None or res.status != "done":
                        continue
                    n_rollouts += 1
                    task = self.tasks[y]
                    if cfg.judge.h_scoring_mode == "rescore":
                        cuts = sorted({max(0, min(l.cut, res.n_steps)) for l in limits[y].limits})
                        for c in cuts:
                            jobs.append(judge_at_cut(self.ctx, d, task, rid, c))
                    else:
                        jobs.append(judge_first_steps(self.ctx, d, task, rid))
        log.info("judge: scoring %d rollouts at their step limits (%d judge jobs, mode=%s)", n_rollouts, len(jobs),
                 cfg.judge.h_scoring_mode)
        await self._gather_judge(jobs, "judge H")

    async def _gather_judge(self, coros: list, label: str) -> None:
        if not coros:
            return
        t0 = time.time()
        results = await self._gather_stoppable(coros)
        caps = [r for r in results if isinstance(r, SpendingCapExceeded)]
        errors = [r for r in results if isinstance(r, Exception) and not isinstance(r, SpendingCapExceeded)]
        log.info("%s: %d jobs, %d errors, %d stopped by the budget, %.1fs; spend so far $%.4f", label, len(results),
                 len(errors), len(caps), time.time() - t0, self.ledger.spent_usd)
        for exc in errors[:5]:
            log.error("%s: %r", label, exc)
        if caps:
            raise caps[0]

    def stage_analyze(self, extra_notes: list[str] | None = None) -> dict[str, Any]:
        plan = self.plan()
        routing = write_routing_summary(self.cfg, self.run_dir, self.layout.results_dir)   # results/routing.json
        result = analyze(self.cfg, self.tasks, self.run_dir, plan, extra_notes=list(extra_notes or []) + routing_notes(routing))
        text = render_summary(self.cfg, result)
        self.layout.results_dir.mkdir(parents=True, exist_ok=True)
        (self.layout.results_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
        return result

    # ---------------------------------------------------------------- the budget stop
    def budget_stop(self, stage: str | None, exc: SpendingCapExceeded) -> dict[str, Any]:
        """Record the stop and save everything there is: budget_stop.json, then the analysis of the partial
        data (every finished rollout and judge result is already on disk; so are the steps of rollouts that
        were cut short)."""
        inventory = rollout_inventory(self.run_dir)
        by_status: dict[str, int] = {}
        for r in inventory:
            by_status[str(r["status"])] = by_status.get(str(r["status"]), 0) + 1
        n_judge = sum(1 for _ in self.run_dir.rglob("judge/*.json"))
        report = {
            "ts": time.time(), "stage": stage, "reason": str(exc),
            "budget_usd": self.ledger.cap_usd, "spent_this_run_usd": round(self.ledger.spent_usd, 6),
            "spent_other_runs_usd": round(self.ledger.prior_spent_usd, 6), "other_runs": self.ledger.prior_by_run,
            "calls_this_run": self.ledger.n_calls, "rollouts_by_status": by_status, "judge_result_files": n_judge,
            "saved": "every finished rollout and judge result, the steps of rollouts stopped part-way (status "
                     "cap_exceeded), ledger.jsonl, and results/ computed from what exists (PARTIAL)",
            "resume": "raise cost.spending_cap_usd in config.toml and re-run the same command; finished work is "
                      "reused and rollouts stopped part-way are re-run",
        }
        self.layout.budget_stop_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        log.error("BUDGET STOP during stage %r: %s", stage, exc)
        log.info("saved so far: rollouts %s, %d judge result files; writing the partial analysis", by_status, n_judge)
        try:
            self.stage_analyze(extra_notes=[
                f"BUDGET STOP during stage '{stage}': {exc}",
                "these results are PARTIAL: computed from the rollouts and judge results saved before the stop"])
        except Exception:
            log.exception("the partial analysis failed; the raw data under %s is intact", self.run_dir)
        return report

    def _clear_stale_budget_stop(self, completed: list[str]) -> None:
        """After the stage an earlier stop happened in has been re-run to completion, the record is stale."""
        path = self.layout.budget_stop_path
        if not path.exists():
            return
        try:
            recorded = json.loads(path.read_text(encoding="utf-8")).get("stage")
        except Exception:
            recorded = None
        if recorded in completed:
            path.unlink()
            log.info("the earlier budget stop (stage %r) has been completed; budget_stop.json removed", recorded)

    async def run_stages(self, stages: list[str]) -> None:
        for stage in stages:
            if stage not in STAGES:
                raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
        current: str | None = None
        try:
            spending = [s for s in stages if s in SPENDING_STAGES]
            if spending and self.ledger.remaining_usd <= 0:
                current = spending[0]
                raise SpendingCapExceeded(self.ledger.exhausted_message())
            if spending:
                if self.preflight_report is None:
                    await self.preflight()
                problems = preflight_problems(self.preflight_report)
                if problems:
                    raise RoutingError("the provider pins cannot be used as configured:\n  " + "\n  ".join(problems) +
                                       "\nSee `python -m aupower endpoints` for what OpenRouter serves now.")
                current = "probe"
                await self.probe()
            for stage in stages:
                current = stage
                log.info("=== stage: %s ===", stage)
                if stage == "plan":
                    self.plan()
                elif stage == "baseline":
                    await self.stage_baseline()
                elif stage == "trajectories":
                    await self.stage_trajectories()
                elif stage == "forks":
                    await self.stage_forks()
                elif stage == "judge":
                    await self.stage_judge()
                elif stage == "analyze":
                    self.stage_analyze()
            self._clear_stale_budget_stop(stages)
        except SpendingCapExceeded as exc:
            self.budget_stop(current, exc)
            raise
        finally:
            log.info("ledger: %s", json.dumps(self.ledger.summary()))
