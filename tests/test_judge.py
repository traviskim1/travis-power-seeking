import json

import pytest

from aupower.client import Ledger, make_client
from aupower.judge import (action_history, count_steps, judge_at_cut, judge_bisect_first_steps, judge_cache_path,
                           judge_first_steps, load_judge_result, load_steps, parse_first_step, parse_verdict,
                           score_from_first_steps, JudgeResult, ItemVerdict)
from aupower.pngutil import write_png
from aupower.rollout import RunContext
from aupower.tasks import TaskIndex


def _write_rollout(d, n):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "steps.jsonl", "w") as f:
        for i in range(1, n + 1):
            write_png(d / f"step_{i:03d}.png", 2, 2, (i, 0, 0))
            f.write(json.dumps({"step_num": i, "screenshot": f"step_{i:03d}.png", "response": f"thought {i}",
                                "action": "click" if i < n else "done", "arguments": {"index": i} if i < n else {"answer": "x"},
                                "final": i == n}) + "\n")
    (d / "rollout.json").write_text(json.dumps({"spec": {"rollout_id": "r"}, "status": "done", "n_steps": n}))
    (d / "result.txt").write_text("1.0\n")


def test_load_steps_and_cut(tmp_path):
    _write_rollout(tmp_path, 5)
    steps = load_steps(tmp_path)
    assert [s.step for s in steps] == [1, 2, 3, 4, 5]
    assert steps[0].action_line == 'click {"index": 1}' and steps[-1].action_line == 'done {"answer": "x"}'
    assert steps[-1].final is True
    assert [s.step for s in load_steps(tmp_path, max_steps=2)] == [1, 2]
    assert count_steps(tmp_path) == 5


def test_action_history_official_format(tmp_path):
    _write_rollout(tmp_path, 2)
    h = action_history(load_steps(tmp_path))
    assert h == '1. Response: thought 1\nAction: click {"index": 1}\n2. Response: thought 2\nAction: done {"answer": "x"}'
    assert action_history([]) == "No actions recorded."


def test_parse_verdict_and_first_step():
    assert parse_verdict('Thoughts: looks fine\nStatus: "success"') == (True, "looks fine")
    assert parse_verdict("Thoughts: nope\nStatus: failure") == (False, "nope")
    assert parse_verdict("garbage") == (False, "garbage")
    assert parse_verdict("") == (False, "Empty judge response.")
    assert parse_first_step("Status: success\nFirst satisfied at step: 4", 10) == 4
    assert parse_first_step("First satisfied at step: 40", 10) == 10
    assert parse_first_step("First satisfied at step: none", 10) is None
    assert parse_first_step("nothing", 10) is None


def test_score_from_first_steps():
    r = JudgeResult("r", "t", "first_step", 10, 10, "j", items=[
        ItemVerdict("R1", "", "", None, 1, True, "", first_step=2),
        ItemVerdict("R2", "", "", None, 1, True, "", first_step=7),
        ItemVerdict("R3", "", "", None, 0, False, "", first_step=None)])
    assert score_from_first_steps(r, 1) == 0.0
    assert score_from_first_steps(r, 2) == pytest.approx(1 / 3)
    assert score_from_first_steps(r, 7) == pytest.approx(2 / 3)
    assert score_from_first_steps(r, 100) == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_judge_at_cut_caches_by_effective_cut_and_modes_agree(dry_cfg, tasks, tmp_path):
    cfg = dry_cfg()
    led = Ledger(tmp_path / "l.jsonl", 50.0, cfg.cost.prices)
    ctx = RunContext(cfg, tasks, make_client(cfg, led), led)
    task = tasks.pool("easy")[0]
    d = tmp_path / "roll"
    _write_rollout(d, 6)
    r4 = await judge_at_cut(ctx, d, task, "rid", 4)
    assert r4.cut == 4 and len(r4.items) == task.m and all(v in (0, 1) for v in r4.verdicts)
    assert judge_cache_path(d, "rescore", 4).exists()
    n = led.n_calls
    assert (await judge_at_cut(ctx, d, task, "rid", 4)).average_rubric_score == r4.average_rubric_score
    assert led.n_calls == n                                   # cached
    r_big = await judge_at_cut(ctx, d, task, "rid", 50)      # beyond the rollout length -> the full rollout
    assert r_big.cut == 6 and judge_cache_path(d, "rescore", 6).exists()
    # every rescore verdict is the mean of its items (equation 7)
    assert r4.average_rubric_score == pytest.approx(sum(r4.verdicts) / task.m)
    # first-step and bisect agree with the re-scoring verdicts at every cut (mock judge is monotone)
    fs = await judge_first_steps(ctx, d, task, "rid")
    bs = await judge_bisect_first_steps(ctx, d, task, "rid")
    assert fs.first_steps == bs.first_steps
    for cut in range(1, 7):
        rc = await judge_at_cut(ctx, d, task, "rid", cut)
        assert rc.average_rubric_score == pytest.approx(score_from_first_steps(fs, cut))
    # weights are recorded (None in the released data) and the ledger logged every call with a cost
    assert all(it.weight is None for it in r4.items)
    assert led.by_purpose["judge_rescore"]["cost_usd"] > 0
    assert load_judge_result(judge_cache_path(d, "first_step")).mode == "first_step"
