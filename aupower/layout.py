"""Where every rollout, snapshot and result lives under runs/<name>/."""
from __future__ import annotations

from pathlib import Path

from .config import ModelSpec


class Layout:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    # ---- files ----
    @property
    def plan_path(self) -> Path:
        return self.run_dir / "plan.json"

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / "ledger.jsonl"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def budget_stop_path(self) -> Path:
        return self.run_dir / "budget_stop.json"

    # ---- per model / level ----
    def model_level(self, model: ModelSpec, level: str) -> Path:
        return self.run_dir / model.slug / level

    def baseline_dir(self, model: ModelSpec, level: str, task_id: str, r: int) -> Path:
        return self.model_level(model, level) / "baseline" / f"{task_id}_{r}"

    def baseline_id(self, model: ModelSpec, level: str, task_id: str, r: int) -> str:
        return f"{model.slug}/{level}/baseline/{task_id}_{r}"

    def traj_dir(self, model: ModelSpec, level: str, g: str) -> Path:
        return self.model_level(model, level) / "traj" / g

    def traj_rollout_dir(self, model: ModelSpec, level: str, g: str) -> Path:
        return self.traj_dir(model, level, g) / "rollout"

    def traj_id(self, model: ModelSpec, level: str, g: str) -> str:
        return f"{model.slug}/{level}/traj/{g}"

    def states_path(self, model: ModelSpec, level: str, g: str) -> Path:
        return self.traj_dir(model, level, g) / "states.json"

    def snapshot_dir(self, model: ModelSpec, level: str, g: str, k: int) -> Path:
        return self.traj_rollout_dir(model, level, g) / "snapshots" / f"step_{k:03d}"

    def fork_dir(self, model: ModelSpec, level: str, g: str, k: int, context: str, goal: str, r: int) -> Path:
        return self.model_level(model, level) / "forks" / g / f"step_{k:03d}" / context / f"{goal}_{r}"

    def fork_id(self, model: ModelSpec, level: str, g: str, k: int, context: str, goal: str, r: int) -> str:
        return f"{model.slug}/{level}/fork/{g}/step_{k:03d}/{context}/{goal}_{r}"
