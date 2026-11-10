"""The per-step browser watchdog (rollout.py): a browser call that never returns costs one step, not the rollout."""
import asyncio
import json

import pytest

from aupower.browser.mock import MockBrowser
from aupower.client import Ledger, make_client
from aupower.prompts import first_task_message
from aupower.rollout import RolloutSpec, RunContext, StartFromScratch, run_rollout


class HangingBrowser(MockBrowser):
    """Hangs on the first action(s) and on observation(s) after the initial one; counts page recoveries."""

    def __init__(self, run_cfg, *, hang_actions=1, hang_observes=1):
        super().__init__(run_cfg, seed=1)
        self.hang_actions, self.hang_observes = hang_actions, hang_observes
        self.n_act = self.n_obs = self.recoveries = 0

    async def act(self, action):
        self.n_act += 1
        if self.n_act <= self.hang_actions:
            await asyncio.sleep(3600)
        return await super().act(action)

    async def observe(self, screenshot_path=None):
        self.n_obs += 1
        if 2 <= self.n_obs <= 1 + self.hang_observes:
            await asyncio.sleep(3600)
        return await super().observe(screenshot_path)

    async def recover_active_page(self):
        self.recoveries += 1


def _setup(dry_cfg, tasks, tmp_path, **hang):
    cfg = dry_cfg(run={"action_timeout_s": 0.1})          # watchdog limit = 1 s (its floor)
    led = Ledger(tmp_path / "l.jsonl", 50.0, cfg.cost.prices)
    ctx = RunContext(cfg, tasks, make_client(cfg, led), led)
    holder = {}

    def make_browser(rollout_id):
        holder["browser"] = HangingBrowser(cfg.run, **hang)
        return holder["browser"]

    ctx.make_browser = make_browser
    task = tasks.pool("easy")[0]
    spec = RolloutSpec(rollout_id="m/easy/baseline/x_0", dir=tmp_path / "roll", model=cfg.models.agents[0], level="easy",
                       goal_id=task.task_id, kind="baseline")
    start = StartFromScratch(start_url="https://www.google.com", task_message=first_task_message("find the cheapest kettle"))
    return ctx, spec, start, holder


@pytest.mark.asyncio
async def test_hung_action_and_observation_cost_one_step(dry_cfg, tasks, tmp_path):
    ctx, spec, start, holder = _setup(dry_cfg, tasks, tmp_path, hang_actions=1, hang_observes=1)
    res = await run_rollout(ctx, spec, start)
    assert res.status == "done" and res.termination == "done"
    rows = [json.loads(l) for l in (spec.dir / "steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["ok"] is False and rows[0]["result"].startswith("Browser step timed out after 1 s")
    assert all(not r["result"].startswith("Browser step timed out") for r in rows[1:])
    assert holder["browser"].recoveries == 1 and res.n_steps == len(rows) > 1


@pytest.mark.asyncio
async def test_browser_that_never_answers_ends_the_rollout_as_error(dry_cfg, tasks, tmp_path):
    ctx, spec, start, holder = _setup(dry_cfg, tasks, tmp_path, hang_actions=0, hang_observes=99)
    res = await run_rollout(ctx, spec, start)
    assert res.status == "error" and "BrowserUnresponsive" in (res.error or "")
    assert holder["browser"].recoveries == 3 and (spec.dir / "rollout.json").exists()
