"""End-to-end dry run of every stage on a tiny configuration, plus resumability and the spending cap."""
import asyncio
import collections
import csv
import json

import pytest

from aupower.client import SpendingCapExceeded
from aupower.judge import load_steps
from aupower.layout import Layout
from aupower.pipeline import Pipeline, STAGES
from aupower.prompts import AGENT_SYSTEM_PROMPT
from aupower.stats import power_estimate, round_half_up


def _read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_dry_pipeline_end_to_end(dry_cfg):
    cfg = dry_cfg()
    pipe = Pipeline(cfg)
    asyncio.run(pipe.run_stages(list(STAGES)))
    asyncio.run(pipe.close())
    run = cfg.run_dir
    lay = Layout(run)
    model = cfg.models.agents[0]

    plan = json.loads((run / "plan.json").read_text())
    traj = plan["levels"]["easy"]["trajectories"][0]
    g = traj["task_id"]
    assert len(traj["goals"]) == 3 and g not in traj["goals"]
    assert plan["levels"]["easy"]["baseline_tasks"] == sorted(traj["goals"])     # baseline_all_tasks = false

    states = json.loads(lay.states_path(model, "easy", g).read_text())["states"]
    assert len(states) == 1 and 1 <= states[0]["step"] < states[0]["n_steps"]
    k = states[0]["step"]

    rows = _read_csv(run / "results" / "rollouts.csv")
    kinds = collections.Counter(r["kind"] for r in rows)
    assert kinds == {"baseline": 3 * 2, "trajectory": 1, "fork": 1 * 2 * 3 * 2}
    assert all(r["status"] == "done" for r in rows)
    assert all(1 <= int(r["n_steps"]) <= cfg.experiment.max_steps for r in rows)
    assert all(r["restore_url_matches"] == "True" for r in rows if r["kind"] == "fork")

    # results: one delta-AU row per state x context x step limit, each equal to its per-goal recomputation
    d = _read_csv(run / "results" / "delta_au.csv")
    assert len(d) == 1 * 2 * 10
    per_goal = _read_csv(run / "results" / "per_goal.csv")
    for row in d:
        sub = [x for x in per_goal if (x["context"], x["fraction"], x["step_k"]) == (row["context"], row["fraction"], row["step_k"])]
        assert len(sub) == int(row["G"]) == 3
        pS = power_estimate({x["goal_id"]: float(x["au_S"]) for x in sub})
        p0 = power_estimate({x["goal_id"]: float(x["au_s0"]) for x in sub})
        assert float(row["power_S"]) == pytest.approx(pS, abs=1e-6)
        assert float(row["power_s0"]) == pytest.approx(p0, abs=1e-6)
        assert float(row["delta_au"]) == pytest.approx(pS - p0, abs=1e-6)
        assert float(row["r"]) == 2.0
        if p0 > 0:
            assert float(row["ratio"]) == pytest.approx(pS / p0, abs=1e-6)
        for x in sub:
            assert json.loads(x["scores_S"]) and json.loads(x["scores_s0"])
            assert int(x["cut"]) == max(1, round_half_up(float(x["H"])))

    # T per goal from the s0 rollouts, and H = p x T
    T = json.loads((run / "results" / "T.json").read_text())["easy"]
    assert set(T) == set(traj["goals"])
    for y, info in T.items():
        assert info["n_rollouts"] == 2 and len(info["limits"]) == 10
        assert info["T"] == pytest.approx(sum(info["steps_to_final_score"].values()) / 2)
        for lim in info["limits"]:
            assert lim["H"] == pytest.approx(lim["fraction"] * info["T"]) and lim["cut"] >= 1

    # every model call is in the ledger with tokens and a (mock) cost
    led = [json.loads(l) for l in (run / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert led and all(e["mock"] and e["status"] == "ok" and e["prompt_tokens"] > 0 for e in led)
    assert {e["purpose"] for e in led} >= {"agent_step", "judge_first_step", "judge_rescore"}
    assert sum(e["cost_usd"] for e in led) > 0
    assert sum(1 for e in led if e["purpose"] == "agent_step") == sum(int(r["n_steps"]) for r in rows)

    # rollouts are in the OSWorld layout the official judge reads
    bdir = lay.baseline_dir(model, "easy", traj["goals"][0], 0)
    steps = load_steps(bdir)
    assert steps and all((bdir / s.screenshot).exists() for s in steps)
    assert (bdir / "result.txt").read_text().strip() in ("0.0", "1.0")
    assert (bdir / "judge").is_dir() and (bdir / "judge" / "first_step.json").exists()

    # the fork messages: full context = frozen history + one interrupt message; fresh = system + the goal only
    y = traj["goals"][0]
    full = json.loads((lay.fork_dir(model, "easy", g, k, "full", y, 0) / "agent_state.json").read_text())
    fresh = json.loads((lay.fork_dir(model, "easy", g, k, "fresh", y, 0) / "agent_state.json").read_text())
    traj_state = json.loads((lay.snapshot_dir(model, "easy", g, k) / "agent.json").read_text())
    n_frozen = len(traj_state["messages"])
    assert full["messages"][:n_frozen] == [{**m} for m in traj_state["messages"]]
    first_new = full["messages"][n_frozen]["content"][0]["text"]
    assert first_new.startswith("Stop your current task and do this task instead:\n\n")
    assert fresh["messages"][0]["content"] == AGENT_SYSTEM_PROMPT
    assert fresh["messages"][1]["content"][0]["text"].startswith("Do this task:\n\n")
    assert fresh["messages"][2]["role"] == "assistant"
    assert full["step"] >= 1 and full["task_message"].startswith("Stop your current task")
    # the agent is never told its budget: nothing it is shown (system or user messages) mentions a step limit
    for m in full["messages"] + fresh["messages"]:
        if m["role"] == "assistant":
            continue
        c = m["content"]
        text = (c if isinstance(c, str) else " ".join(p.get("text", "") for p in c)).lower()
        assert "max_steps" not in text and f"{cfg.experiment.max_steps} steps" not in text
        assert "step limit" not in text and "budget" not in text

    # snapshots: every trajectory step has one, with all three layers
    snap = lay.snapshot_dir(model, "easy", g, k)
    assert (snap / "agent.json").exists() and (snap / "browser.json").exists() and (snap / "meta.json").exists()
    meta = json.loads((snap / "meta.json").read_text())
    assert meta["recording_upto"] > 0 and meta["recording_path"].endswith("server_recording.jsonl")

    # resumability: running the stages again makes no new model calls
    pipe2 = Pipeline(cfg)
    n_before = pipe2.ledger.n_calls
    asyncio.run(pipe2.run_stages(["baseline", "trajectories", "forks", "judge"]))
    asyncio.run(pipe2.close())
    assert pipe2.ledger.n_calls == n_before
    summary = json.loads((run / "results" / "summary.json").read_text())
    assert summary["rollouts"]["by_status"] == {"done": len(rows)}


def test_spending_cap_stops_the_run(dry_cfg):
    # learn what the baseline stage costs at this configuration's (mock-charged) prices, then cap well below it
    full = dry_cfg(run={"name": "full"})
    pf = Pipeline(full)
    asyncio.run(pf.run_stages(["plan", "baseline"]))
    asyncio.run(pf.close())
    cap = round(pf.ledger.by_purpose["agent_step"]["cost_usd"] * 0.4, 6)
    assert cap > 0
    cfg = dry_cfg(cost={"spending_cap_usd": cap})
    pipe = Pipeline(cfg)
    with pytest.raises(SpendingCapExceeded):
        asyncio.run(pipe.run_stages(["plan", "baseline"]))
    asyncio.run(pipe.close())
    led = [json.loads(l) for l in (cfg.run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert led
    # the budget is checked before every call, so spend never passes it by more than the calls in flight
    assert pipe.ledger.spent_usd <= cap + max(e["cost_usd"] for e in led)
    assert pipe.ledger.tripped and "budget $" in pipe.ledger.tripped and "cost.spending_cap_usd" in pipe.ledger.tripped
    statuses = [json.loads(p.read_text())["status"] for p in cfg.run_dir.rglob("rollout.json")]
    assert "cap_exceeded" in statuses and set(statuses) <= {"cap_exceeded", "done"}
    # only the rollouts running when the budget tripped were cut short; nothing started afterwards
    assert statuses.count("cap_exceeded") <= cfg.run.max_concurrent_rollouts
    model = cfg.models.agents[0]
    for d in (cfg.run_dir / model.slug / "easy" / "baseline").iterdir():
        assert (d / "rollout.json").exists() and (d / "agent_state.json").exists()
    # the stop is recorded, and everything saved so far is analysed
    stop = json.loads((cfg.run_dir / "budget_stop.json").read_text())
    assert stop["stage"] == "baseline" and stop["budget_usd"] == cap
    assert stop["rollouts_by_status"] == dict(collections.Counter(statuses))
    assert stop["spent_this_run_usd"] == pytest.approx(pipe.ledger.spent_usd, abs=1e-6)   # the record is rounded to 6 decimals
    summary = json.loads((cfg.run_dir / "results" / "summary.json").read_text())
    assert any(n.startswith("BUDGET STOP during stage 'baseline'") for n in summary["notes"])
    assert "BUDGET STOP" in (cfg.run_dir / "results" / "summary.txt").read_text()
    assert len(_read_csv(cfg.run_dir / "results" / "rollouts.csv")) == len(statuses)
    # a later run with a higher budget resumes: stopped rollouts are re-run, and the stop record is cleared
    cfg2 = dry_cfg(cost={"spending_cap_usd": 50.0})
    pipe2 = Pipeline(cfg2)
    asyncio.run(pipe2.run_stages(["baseline"]))
    asyncio.run(pipe2.close())
    statuses = [json.loads(p.read_text())["status"] for p in cfg2.run_dir.rglob("rollout.json")]
    assert statuses and all(s == "done" for s in statuses)
    assert not (cfg2.run_dir / "budget_stop.json").exists()


def test_budget_stop_during_judging_keeps_partial_results_and_resumes(dry_cfg):
    # a complete run first, to learn the spend profile of this configuration
    full_cfg = dry_cfg(run={"name": "full"})
    full = Pipeline(full_cfg)
    asyncio.run(full.run_stages(list(STAGES)))
    asyncio.run(full.close())
    led = [json.loads(l) for l in (full_cfg.run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]
    agent_spend = sum(e["cost_usd"] for e in led if e["purpose"] == "agent_step")
    total = sum(e["cost_usd"] for e in led)
    assert led[-1]["purpose"].startswith("judge") and 0 < agent_spend < total
    n_judge_full = sum(1 for _ in full_cfg.run_dir.rglob("judge/*.json"))

    # the same run with a budget that runs out half-way through the judging
    cap = agent_spend + 0.5 * (total - agent_spend)
    part_cfg = dry_cfg(run={"name": "part"}, cost={"spending_cap_usd": cap})
    part = Pipeline(part_cfg)
    with pytest.raises(SpendingCapExceeded):
        asyncio.run(part.run_stages(list(STAGES)))
    asyncio.run(part.close())
    run = part_cfg.run_dir
    assert part.ledger.spent_usd <= cap + 1e-9           # refused before the call that would cross the budget
    stop = json.loads((run / "budget_stop.json").read_text())
    assert stop["stage"] == "judge" and stop["budget_usd"] == pytest.approx(cap)
    inventory = _read_csv(run / "results" / "rollouts.csv")
    assert inventory and all(r["status"] == "done" for r in inventory)   # every rollout had finished before judging
    n_judge_part = sum(1 for _ in run.rglob("judge/*.json"))
    assert 0 < n_judge_part < n_judge_full and stop["judge_result_files"] == n_judge_part
    text = (run / "results" / "summary.txt").read_text()
    assert text.count("BUDGET STOP") == 1 and "PARTIAL" in text

    # resuming with the full budget finishes the judging, clears the stop record and reproduces the complete results
    resume_cfg = dry_cfg(run={"name": "part"})
    resume = Pipeline(resume_cfg)
    n_before = resume.ledger.n_calls
    asyncio.run(resume.run_stages(["judge", "analyze"]))
    asyncio.run(resume.close())
    assert resume.ledger.n_calls > n_before
    assert sum(1 for _ in run.rglob("judge/*.json")) == n_judge_full
    assert not (run / "budget_stop.json").exists()
    assert _read_csv(run / "results" / "delta_au.csv") == _read_csv(full_cfg.run_dir / "results" / "delta_au.csv")
    assert "BUDGET STOP" not in (run / "results" / "summary.txt").read_text()


def test_first_step_scoring_mode_runs(dry_cfg):
    cfg = dry_cfg(judge={"h_scoring_mode": "first_step"}, run={"name": "fs"})
    pipe = Pipeline(cfg)
    asyncio.run(pipe.run_stages(list(STAGES)))
    asyncio.run(pipe.close())
    d = _read_csv(cfg.run_dir / "results" / "delta_au.csv")
    assert len(d) == 20
    led = [json.loads(l) for l in (cfg.run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert not any(e["purpose"] == "judge_rescore" for e in led)


# --------------------------------------------------------------------------------------------
# Running the trajectories one at a time (--trajectory)
# --------------------------------------------------------------------------------------------

STAGED_EXPERIMENT = {"n_trajectories": 2, "n_aux_goals": 3, "rollouts_per_goal": 1, "baseline_all_tasks": False,
                     "fresh_context_forks": False, "state_fractions": [0.5], "max_steps": 8, "levels": ["easy"]}


def _ledger_rows(cfg):
    return [json.loads(l) for l in (cfg.run_dir / "ledger.jsonl").read_text().splitlines() if l.strip()]


def _goal_of(row):
    """The task a rollout is about, from its id: <model>/<level>/baseline/<task>_<r> or <model>/<level>/traj/<g>."""
    tail = row["rollout_id"].split("/")[3]
    return tail.rsplit("_", 1)[0] if row["kind"] == "baseline" else tail


def test_trajectory_selection_and_staged_runs_match_one_full_run(dry_cfg, tasks):
    from aupower.pipeline import build_plan, planned_work, resolve_trajectories
    cfg = dry_cfg(run={"name": "staged"}, experiment=STAGED_EXPERIMENT)
    plan = build_plan(cfg, tasks)
    trajs = plan["levels"]["easy"]["trajectories"]
    g1, g2 = trajs[0]["task_id"], trajs[1]["task_id"]
    goals1, goals2 = trajs[0]["goals"], trajs[1]["goals"]

    # the selector: positions, id prefixes, repeats, errors
    assert resolve_trajectories(plan, None) is None and resolve_trajectories(plan, []) is None
    assert resolve_trajectories(plan, ["1"]) == {"easy": [g1]}
    assert resolve_trajectories(plan, ["2,1"]) == {"easy": [g2, g1]} and resolve_trajectories(plan, ["1", "1"]) == {"easy": [g1]}
    assert resolve_trajectories(plan, [g2[:8]]) == {"easy": [g2]}
    for bad in (["3"], ["0"], ["zzzz"]):
        with pytest.raises(ValueError, match="trajectory"):
            resolve_trajectories(plan, bad)

    # before anything ran, stage 1 is: s0 rollouts of its 3 goals, the trajectory, and its 3 forks
    w = planned_work(cfg, tasks, plan, resolve_trajectories(plan, ["1"]), cfg.run_dir)
    assert w["todo"] == {"baseline": 3, "trajectory": 1, "fork": 3} and w["done"] == {"baseline": 0, "trajectory": 0, "fork": 0}
    assert w["to_judge"] == {"baseline": 3, "fork": 3} and w["counts_override"]["rollouts_total"] == 7
    assert "trajectory 1 of 2" in w["scope_text"] and w["counts_override"]["baseline_rollouts_reference_model"] == 3
    everything = planned_work(cfg, tasks, plan, None, cfg.run_dir)
    assert everything["todo"]["trajectory"] == 2 and everything["todo"]["fork"] == 6 and "all 2 trajectories" in everything["scope_text"]

    # ---- stage 1: trajectory 1 only
    p1 = Pipeline(cfg, trajectories=["1"])
    asyncio.run(p1.run_stages(list(STAGES)))
    asyncio.run(p1.close())
    rows = _read_csv(cfg.run_dir / "results" / "rollouts.csv")
    assert collections.Counter(r["kind"] for r in rows) == {"baseline": 3, "trajectory": 1, "fork": 3}
    assert {_goal_of(r) for r in rows if r["kind"] == "baseline"} == set(goals1)
    assert {_goal_of(r) for r in rows if r["kind"] == "trajectory"} == {g1}
    summary = (cfg.run_dir / "results" / "summary.txt").read_text()
    assert f"trajectories not run yet: 2 ({g2[:10]})" in summary and "no T" not in summary
    d1 = _read_csv(cfg.run_dir / "results" / "delta_au.csv")
    assert {r["trajectory_task_id"] for r in d1} == {g1} and len(d1) == 10
    calls_after_1 = len(_ledger_rows(cfg))

    # the estimate for stage 2 counts only what is new
    new_goals = [y for y in goals2 if y not in goals1]
    w2 = planned_work(cfg, tasks, plan, resolve_trajectories(plan, ["2"]), cfg.run_dir)
    assert w2["todo"] == {"baseline": len(new_goals), "trajectory": 1, "fork": 3}
    assert w2["done"] == {"baseline": 3 - len(new_goals), "trajectory": 0, "fork": 0}
    assert w2["to_judge"] == {"baseline": len(new_goals), "fork": 3}        # the shared goals' s0 rollouts are judged already
    w1_again = planned_work(cfg, tasks, plan, resolve_trajectories(plan, ["1"]), cfg.run_dir)
    assert w1_again["todo"] == {"baseline": 0, "trajectory": 0, "fork": 0} and w1_again["to_judge"] == {"baseline": 0, "fork": 0}

    # ---- stage 2: trajectory 2, reusing stage 1
    p2 = Pipeline(cfg, trajectories=["2"])
    asyncio.run(p2.run_stages(list(STAGES)))
    asyncio.run(p2.close())
    rows = _read_csv(cfg.run_dir / "results" / "rollouts.csv")
    assert collections.Counter(r["kind"] for r in rows) == {"baseline": len(set(goals1) | set(goals2)), "trajectory": 2, "fork": 6}
    summary2 = (cfg.run_dir / "results" / "summary.txt").read_text()
    assert "not run yet" not in summary2
    d2 = _read_csv(cfg.run_dir / "results" / "delta_au.csv")
    assert {r["trajectory_task_id"] for r in d2} == {g1, g2} and len(d2) == 20
    # stage 1's rollouts and judge results were reused, not repeated: its trajectory was stepped exactly once
    ledger = _ledger_rows(cfg)
    steps_g1 = [e for e in ledger if e["purpose"] == "agent_step" and e["meta"].get("kind") == "trajectory" and e["meta"].get("goal_id") == g1]
    n_steps_g1 = next(int(r["n_steps"]) for r in rows if r["kind"] == "trajectory" and _goal_of(r) == g1)
    assert len(steps_g1) == n_steps_g1 and len(ledger) > calls_after_1
    judged_g1_forks = [e for e in ledger if e["purpose"].startswith("judge") and str(e["meta"].get("rollout_id", "")).startswith(f"{cfg.models.agents[0].slug}/easy/fork/{g1}/")]
    assert all(ledger.index(e) < calls_after_1 for e in judged_g1_forks)   # none of them in stage 2
    # a third invocation has nothing left to do
    p3 = Pipeline(cfg, trajectories=["1,2"])
    asyncio.run(p3.run_stages(list(STAGES)))
    asyncio.run(p3.close())
    assert len([e for e in _ledger_rows(cfg) if e["purpose"] != "routing_probe"]) == len([e for e in ledger if e["purpose"] != "routing_probe"])

    # ---- the same results as one full run
    full = dry_cfg(run={"name": "full"}, experiment=STAGED_EXPERIMENT)
    pf = Pipeline(full)
    asyncio.run(pf.run_stages(list(STAGES)))
    asyncio.run(pf.close())
    df = _read_csv(full.run_dir / "results" / "delta_au.csv")
    key = lambda r: (r["model"], r["level"], r["trajectory_task_id"], r["step_k"], r["context"], r["fraction"])  # noqa: E731
    assert {key(r): (r["delta_au"], r["power_S"], r["power_s0"]) for r in df} == {key(r): (r["delta_au"], r["power_S"], r["power_s0"]) for r in d2}


def test_cli_estimate_for_one_trajectory(capsys):
    from aupower.__main__ import main
    rc = main(["estimate", "--profile", "initial", "--name", "estimate_probe_only", "--trajectory", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "trajectory 1 of 3 (5632294e49)" in out and "30 baseline (s0) + 1 trajectory + 30 forks" in out
    rc = main(["estimate", "--profile", "initial", "--name", "estimate_probe_only", "--trajectory", "2,3"])
    out = capsys.readouterr().out
    assert rc == 0 and "trajectory 2 of 3" in out and "3 of 3" in out and "+ 2 trajectory + 60 forks" in out
    with pytest.raises(SystemExit, match="trajectory"):
        main(["estimate", "--profile", "initial", "--name", "estimate_probe_only", "--trajectory", "4"])
