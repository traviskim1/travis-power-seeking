#!/usr/bin/env python
"""Analyse an aupower run directory (read-only) and write REPORT.md, figures/ and tables/ under <run>/analysis/.

    python analysis/run_analysis.py --run runs/initial

Nothing is sent to any model; only files on disk are read. The pipeline, config.toml and the run's existing
files are not modified."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from au_analysis.claims import run_claims  # noqa: E402
from au_analysis.health import run_health  # noqa: E402
from au_analysis.judge_checks import run_judge_checks  # noqa: E402
from au_analysis.load import load_run  # noqa: E402
from au_analysis.measures import bootstrap_delta, pooled_bootstrap, sample_size_table  # noqa: E402
from au_analysis.report import write_csv, write_report  # noqa: E402
from au_analysis.trajectories import horizon_figure, run_trajectories  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/initial", help="run directory (default runs/initial)")
    ap.add_argument("--out", default=None, help="output directory (default <run>/analysis)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    run_dir = Path(args.run).resolve()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "analysis"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir)
    print(f"loaded {len(run.rollouts)} rollouts from {run_dir} ({len(run.missing)} directories without rollout.json)")

    health = run_health(run)
    claims = run_claims(run)
    judge = run_judge_checks(run)
    traj = run_trajectories(run, out_dir)
    boot: dict[str, list[dict]] = {}
    power: dict[str, list[dict]] = {}
    per_goal_by_state: dict[str, list[dict]] = {}
    for g, e in traj["per_traj"].items():
        if not e.get("per_goal"):
            continue
        boot[g] = bootstrap_delta(e["per_goal"], n_boot=args.n_boot, seed=args.seed)
        power[g] = sample_size_table(e["per_goal"])
        per_goal_by_state[g] = e["per_goal"]
        traj["tables"][f"bootstrap_trajectory_{e['index']}.csv"] = boot[g]
        traj["tables"][f"power_analysis_trajectory_{e['index']}.csv"] = power[g]
    pooled = pooled_bootstrap(per_goal_by_state, n_boot=args.n_boot, seed=args.seed) if per_goal_by_state else []
    if pooled:
        traj["tables"]["bootstrap_pooled.csv"] = pooled
    states_by_label = {e["label"]: e["states_table"] for e in traj["per_traj"].values() if e.get("states_table")}
    if states_by_label:
        horizon_figure(out_dir / "figures" / "horizon_delta_ratio.png", states_by_label,
                       {e["label"]: boot[g] for g, e in traj["per_traj"].items() if g in boot})
    stats = {"boot": boot, "power": power, "pooled": pooled, "n_boot": args.n_boot, "seed": args.seed}

    for part in (health, claims, judge, traj):
        for name, rows in part["tables"].items():
            write_csv(out_dir / "tables" / name, rows)
    command = f"python analysis/run_analysis.py --run {Path(args.run)}" + (f" --out {args.out}" if args.out else "")
    report = write_report(run, out_dir, command=command, health=health, claims=claims, judge=judge, traj=traj, stats=stats)
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
