"""Read-only loaders for a run directory: plan, resolved config, tasks, every rollout with its steps and judge
results, the sampled states and their snapshots, the ledger, the API call log and the pipeline's results/."""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from aupower.tasks import TaskIndex, load_tasks  # noqa: E402  (the pipeline's task loader; read-only)


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Rollout:
    rollout_id: str
    kind: str                       # baseline | trajectory | fork
    dir: Path
    model_slug: str
    level: str
    goal_id: str
    traj_id: str | None             # the trajectory a fork comes from (= goal_id for a trajectory rollout)
    k: int | None
    context: str | None
    r: int
    status: str
    termination: str
    n_steps: int
    answer: str
    error: str | None
    restore: dict | None
    replay: dict
    usage: dict
    duration_s: float
    started_at: float

    @property
    def done(self) -> bool:
        return self.status == "done"

    @cached_property
    def steps(self) -> list[dict]:
        rows = read_jsonl(self.dir / "steps.jsonl")
        return sorted(rows, key=lambda r: int(r.get("step_num", 0)))

    @property
    def final_step(self) -> dict | None:
        return self.steps[-1] if self.steps else None

    @cached_property
    def rescore(self) -> dict[int, dict]:
        """{cut: judge result} from judge/rescore_cutNNN.json (results with an error are kept, flagged)."""
        out: dict[int, dict] = {}
        for p in sorted((self.dir / "judge").glob("rescore_cut*.json")):
            m = re.search(r"rescore_cut(\d+)\.json$", p.name)
            if not m:
                continue
            try:
                out[int(m.group(1))] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                out[int(m.group(1))] = {"error": f"unreadable: {exc}", "items": []}
        return out

    @cached_property
    def first_step(self) -> dict | None:
        p = self.dir / "judge" / "first_step.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"error": f"unreadable: {exc}", "items": []}

    def effective_cut(self, cut: int) -> int:
        return max(0, min(int(cut), self.n_steps))

    def judge_at(self, cut: int) -> dict | None:
        r = self.rescore.get(self.effective_cut(cut))
        return None if r is None or r.get("error") else r

    def score_at(self, cut: int) -> float | None:
        r = self.judge_at(cut)
        return None if r is None else float(r["average_rubric_score"])

    def verdicts_at(self, cut: int) -> dict[str, int] | None:
        r = self.judge_at(cut)
        return None if r is None else {it["rubric_id"]: int(it["score"]) for it in r.get("items", [])}

    @property
    def full_score(self) -> float | None:
        """The score of the re-scoring at the rollout's last step, when the pipeline judged that cut (it judges at
        the H cuts only, so a rollout longer than its goal's T has no judgment at full length)."""
        return self.score_at(self.n_steps)

    @property
    def largest_judged_cut(self) -> int | None:
        cuts = [c for c, jr in self.rescore.items() if not jr.get("error")]
        return max(cuts) if cuts else None

    @property
    def final_score(self) -> float | None:
        """The score at the largest judged cut (equal to the full-length score when that cut is the last step)."""
        c = self.largest_judged_cut
        return None if c is None else self.score_at(c)

    @property
    def judged_to_the_end(self) -> bool | None:
        c = self.largest_judged_cut
        return None if c is None else c >= self.n_steps

    def first_met_from_rescore(self) -> dict[str, int | None]:
        """Per item, the smallest rescore cut at which it is met (None = never met at any judged cut)."""
        out: dict[str, int | None] = {}
        for cut in sorted(self.rescore):
            r = self.rescore[cut]
            if r.get("error"):
                continue
            for it in r.get("items", []):
                rid = it["rubric_id"]
                out.setdefault(rid, None)
                if int(it["score"]) == 1 and out[rid] is None:
                    out[rid] = cut
        return out

    def first_met_from_first_step(self) -> dict[str, int | None] | None:
        fs = self.first_step
        if not fs or fs.get("error"):
            return None
        return {it["rubric_id"]: it.get("first_step") for it in fs.get("items", [])}

    def domains(self) -> list[str]:
        from .rules import domain_of
        return [domain_of(s.get("url") or "") for s in self.steps]


@dataclass
class RunData:
    run_dir: Path
    plan: dict
    config: dict
    tasks: TaskIndex
    rollouts: list[Rollout]
    states: dict[tuple[str, str, str], list[dict]]     # (model_slug, level, traj_id) -> states
    missing: list[dict] = field(default_factory=list)   # expected rollouts without a rollout.json

    # ---- lookups ----
    @property
    def model_slug(self) -> str:
        return self.plan["models"][0].replace("/", "__").replace(":", "_")

    @property
    def levels(self) -> list[str]:
        return list(self.plan["levels"])

    def by_kind(self, kind: str) -> list[Rollout]:
        return [r for r in self.rollouts if r.kind == kind]

    def baselines(self, goal_id: str) -> list[Rollout]:
        return sorted([r for r in self.rollouts if r.kind == "baseline" and r.goal_id == goal_id], key=lambda r: r.r)

    def trajectory(self, traj_id: str) -> Rollout | None:
        return next((r for r in self.rollouts if r.kind == "trajectory" and r.goal_id == traj_id), None)

    def forks(self, traj_id: str, k: int | None = None, context: str | None = None) -> list[Rollout]:
        return [r for r in self.rollouts if r.kind == "fork" and r.traj_id == traj_id
                and (k is None or r.k == k) and (context is None or r.context == context)]

    def fork(self, traj_id: str, k: int, context: str, goal_id: str, r: int = 0) -> Rollout | None:
        return next((x for x in self.forks(traj_id, k, context) if x.goal_id == goal_id and x.r == r), None)

    def trajectories_planned(self, level: str) -> list[dict]:
        return list(self.plan["levels"][level]["trajectories"])

    def goals_of(self, traj_id: str) -> list[str]:
        for level in self.levels:
            for t in self.plan["levels"][level]["trajectories"]:
                if t["task_id"] == traj_id:
                    return list(t["goals"])
        return []

    def level_of(self, traj_id: str) -> str:
        for level in self.levels:
            if any(t["task_id"] == traj_id for t in self.plan["levels"][level]["trajectories"]):
                return level
        return self.levels[0]

    def task(self, task_id: str):
        return self.tasks[task_id]

    def states_of(self, traj_id: str) -> list[dict]:
        return self.states.get((self.model_slug, self.level_of(traj_id), traj_id), [])

    def snapshot_dir(self, traj_id: str, k: int) -> Path:
        return self.run_dir / self.model_slug / self.level_of(traj_id) / "traj" / traj_id / "rollout" / "snapshots" / f"step_{k:03d}"

    def snapshot(self, traj_id: str, k: int) -> dict[str, Any]:
        d = self.snapshot_dir(traj_id, k)
        out: dict[str, Any] = {"dir": d, "exists": d.exists(), "browser": None, "meta": None, "downloads": []}
        if not d.exists():
            return out
        for name in ("browser", "meta"):
            p = d / f"{name}.json"
            if p.exists():
                try:
                    out[name] = json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    out[name] = {"error": str(exc)}
        dl = d / "downloads"
        if dl.exists():
            out["downloads"] = sorted(str(p.relative_to(dl)) for p in dl.rglob("*") if p.is_file())
        return out

    # ---- pipeline outputs, logs ----
    def results_csv(self, name: str) -> list[dict]:
        return read_csv(self.run_dir / "results" / name)

    def results_json(self, name: str) -> Any:
        p = self.run_dir / "results" / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @cached_property
    def ledger(self) -> list[dict]:
        return read_jsonl(self.run_dir / "ledger.jsonl")

    @cached_property
    def invocations(self) -> list[dict]:
        return read_jsonl(self.run_dir / "invocations.jsonl")

    def api_calls(self) -> Iterator[dict]:
        p = self.run_dir / "api_calls.jsonl.gz"
        if not p.exists():
            return iter(())

        def gen():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        return gen()

    def console_logs(self) -> list[Path]:
        return sorted(self.run_dir.parent.glob(f"{self.run_dir.name}*.console.log"))

    def budget_stop(self) -> dict | None:
        p = self.run_dir / "budget_stop.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    # ---- step limits per goal, recomputed from the judge files (SPEC.md section 1) ----
    @cached_property
    def fractions(self) -> list[float]:
        return list(self.config["experiment"]["step_limit_fractions"])

    def goal_T(self, goal_id: str) -> tuple[float | None, dict[str, int]]:
        """T = mean over the goal's s0 rollouts of the steps to their final rubric score (max first-met step;
        full length when nothing was met), from judge/first_step.json. Returns (T, {rollout_id: steps})."""
        stf: dict[str, int] = {}
        for b in self.baselines(goal_id):
            if not b.done:
                continue
            fm = b.first_met_from_first_step()
            if fm is None:
                continue
            met = [s for s in fm.values() if s is not None]
            stf[b.rollout_id] = max(met) if met else b.n_steps
        if not stf:
            return None, stf
        return sum(stf.values()) / len(stf), stf

    def cuts_for(self, goal_id: str) -> dict[float, int] | None:
        T, _ = self.goal_T(goal_id)
        if T is None:
            return None
        return {p: max(1, round_half_up(p * T)) for p in self.fractions}


# --------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------

def _load_rollout(path: Path, kind: str, model_slug: str, level: str, goal_id: str, traj_id: str | None,
                  k: int | None, context: str | None, r: int) -> Rollout | None:
    rj = path / "rollout.json"
    if not rj.exists():
        return None
    try:
        d = json.loads(rj.read_text(encoding="utf-8"))
    except Exception:
        return None
    spec = d.get("spec") or {}
    return Rollout(rollout_id=spec.get("rollout_id") or path.name, kind=kind, dir=path, model_slug=model_slug, level=level,
                   goal_id=goal_id, traj_id=traj_id, k=k, context=context, r=r, status=str(d.get("status")),
                   termination=str(d.get("termination")), n_steps=int(d.get("n_steps") or 0), answer=str(d.get("answer") or ""),
                   error=d.get("error"), restore=d.get("restore"), replay=d.get("replay") or {}, usage=d.get("usage") or {},
                   duration_s=float(d.get("duration_s") or 0.0), started_at=float(d.get("started_at") or 0.0))


def load_run(run_dir: Path, tasks_path: Path | None = None) -> RunData:
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    tp = Path(tasks_path) if tasks_path else PROJECT / config["run"]["tasks_path"]
    tasks = TaskIndex(load_tasks(tp))
    rollouts: list[Rollout] = []
    states: dict[tuple[str, str, str], list[dict]] = {}
    missing: list[dict] = []
    for model_id in plan["models"]:
        slug = model_id.replace("/", "__").replace(":", "_")
        for level, lp in plan["levels"].items():
            base = run_dir / slug / level
            # baselines (every task dir found, planned or not)
            for d in sorted((base / "baseline").glob("*_*")) if (base / "baseline").exists() else []:
                goal, _, r = d.name.rpartition("_")
                ro = _load_rollout(d, "baseline", slug, level, goal, None, None, None, int(r))
                if ro:
                    rollouts.append(ro)
                else:
                    missing.append({"kind": "baseline", "dir": str(d), "why": "directory without rollout.json (interrupted or in progress)"})
            for t in lp["trajectories"]:
                g = t["task_id"]
                td = base / "traj" / g / "rollout"
                ro = _load_rollout(td, "trajectory", slug, level, g, g, None, None, 0)
                if ro:
                    rollouts.append(ro)
                elif td.exists():
                    missing.append({"kind": "trajectory", "dir": str(td), "why": "directory without rollout.json (interrupted or in progress)"})
                sp = base / "traj" / g / "states.json"
                if sp.exists():
                    states[(slug, level, g)] = json.loads(sp.read_text(encoding="utf-8")).get("states", [])
                fd = base / "forks" / g
                if fd.exists():
                    for sd in sorted(fd.glob("step_*")):
                        k = int(sd.name.split("_")[1])
                        for cd in sorted(sd.iterdir()):
                            if not cd.is_dir():
                                continue
                            for d in sorted(cd.glob("*_*")):
                                goal, _, r = d.name.rpartition("_")
                                ro = _load_rollout(d, "fork", slug, level, goal, g, k, cd.name, int(r))
                                if ro:
                                    rollouts.append(ro)
                                else:
                                    missing.append({"kind": "fork", "dir": str(d), "why": "directory without rollout.json (interrupted or in progress)"})
    return RunData(run_dir=run_dir, plan=plan, config=config, tasks=tasks, rollouts=rollouts, states=states, missing=missing)
