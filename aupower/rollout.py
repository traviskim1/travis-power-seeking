"""One rollout: the agent acts on a goal from s0 or from a restored snapshot, for up to max_steps steps,
and the trajectory is written in the OSWorld layout the official Odysseys judge reads (steps.jsonl,
step screenshots, result.txt)."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .agent import AgentState, ActionParseError, build_api_messages, parse_action, user_turn_text
from .browser import Browser, Recorder, ReplayIndex, make_browser
from .browser.base import ActionResult
from .client import ModelClient, SpendingCapExceeded, ModelCallError, Ledger
from .config import Config, ModelSpec
from .snapshots import StateSnapshot, save_snapshot
from .tasks import TaskIndex, derive_seed

log = logging.getLogger("aupower.rollout")


@dataclass
class RolloutSpec:
    rollout_id: str
    dir: Path
    model: ModelSpec
    level: str
    goal_id: str
    kind: str                          # "baseline" | "trajectory" | "fork"
    rollout_index: int = 0
    context: str | None = None         # "full" | "fresh" (forks)
    origin: dict | None = None         # {"trajectory_task_id": ..., "step": k} (forks)
    snapshot_every_step: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dir"] = str(self.dir)
        d["model"] = self.model.id
        return d


@dataclass
class StartFromScratch:
    start_url: str
    task_message: str


@dataclass
class StartFromSnapshot:
    snapshot: StateSnapshot
    context: str                       # "full" | "fresh"
    task_message: str                  # the interrupt message (full) or first-task message (fresh)


@dataclass
class RolloutResult:
    spec: dict
    status: str                        # "done" | "error" | "cap_exceeded"
    n_steps: int = 0
    termination: str = ""              # "done" | "max_steps" | "error"
    answer: str | None = None
    error: str | None = None
    restore: dict | None = None
    replay: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    duration_s: float = 0.0
    started_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RunContext:
    """Shared services for a run: config, tasks, model client, ledger, concurrency limits."""

    def __init__(self, cfg: Config, tasks: TaskIndex, client: ModelClient, ledger: Ledger):
        self.cfg = cfg
        self.tasks = tasks
        self.client = client
        self.ledger = ledger
        self.rollout_sem = asyncio.Semaphore(max(1, cfg.run.max_concurrent_rollouts))
        self.judge_sem = asyncio.Semaphore(max(1, cfg.run.max_concurrent_judge_calls))
        self.stop = asyncio.Event()
        self.stop_reason: str | None = None

    def request_stop(self, reason: str) -> None:
        """End the run: nothing starts another model call; rollouts in flight save their state and exit."""
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()

    def make_browser(self, rollout_id: str) -> Browser:
        return make_browser(self.cfg, seed=derive_seed(self.cfg.seeds.base, "browser", rollout_id))


def load_result(rollout_dir: Path) -> RolloutResult | None:
    p = rollout_dir / "rollout.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return RolloutResult(**d)
    except Exception:
        return None


def rollout_is_done(rollout_dir: Path) -> bool:
    r = load_result(rollout_dir)
    return r is not None and r.status == "done"


class BrowserUnresponsive(Exception):
    """The browser produced no observation within the watchdog limit, three times in a row."""


def _step_timeout(cfg) -> float:
    """Four action timeouts (120 s at the default 30 s): long enough for any page that answers at all."""
    return max(1.0, 4.0 * float(cfg.run.action_timeout_s))


async def _act_with_watchdog(ctx: RunContext, spec: RolloutSpec, browser, action, step: int) -> ActionResult:
    """Every browser action is bounded. Playwright's own timeouts cover clicks and navigations but not every call (a hung
    renderer never answers `evaluate`), and the first live run left four forks waiting for hours (2026-11-06)."""
    t = _step_timeout(ctx.cfg)
    try:
        return await asyncio.wait_for(browser.act(action), timeout=t)
    except asyncio.TimeoutError:
        log.warning("%s step %d: browser action %s timed out after %.0f s", spec.rollout_id, step, action.name, t)
        return ActionResult(False, f"Browser step timed out after {t:.0f} s: the page did not respond to '{action.name}'. "
                                   f"Try a different action, another tab or another site.")


async def _observe_with_watchdog(ctx: RunContext, spec: RolloutSpec, browser, path: Path, step: int):
    """An observation is bounded too; a page that stops answering is replaced (same URL, new tab) and observed again."""
    t = _step_timeout(ctx.cfg)
    for attempt in range(1, 4):
        try:
            return await asyncio.wait_for(browser.observe(path), timeout=t)
        except asyncio.TimeoutError:
            log.warning("%s step %d: observation timed out after %.0f s (attempt %d of 3); replacing the active page",
                        spec.rollout_id, step, t, attempt)
            try:
                await asyncio.wait_for(browser.recover_active_page(), timeout=t)
            except Exception as exc:
                log.warning("%s step %d: page recovery failed: %s: %s", spec.rollout_id, step, type(exc).__name__, exc)
    raise BrowserUnresponsive(f"no observation within {t:.0f} s, three times in a row, at step {step}")


def _clear_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


async def run_rollout(ctx: RunContext, spec: RolloutSpec, start: StartFromScratch | StartFromSnapshot) -> RolloutResult:
    """Run (or skip, if already complete) one rollout. Never raises except for SpendingCapExceeded, which
    also means the rollout was not started or was stopped part-way because the budget ran out."""
    existing = load_result(spec.dir)
    if existing is not None:
        if existing.status == "done":
            return existing
        if not ctx.cfg.run.retry_failed_rollouts:
            return existing
    if ctx.stop.is_set():
        raise SpendingCapExceeded(ctx.stop_reason or "run stopped")
    async with ctx.rollout_sem:
        if ctx.stop.is_set():   # the budget ran out while this rollout waited for a slot: start nothing, write nothing
            raise SpendingCapExceeded(ctx.stop_reason or "run stopped")
        return await _run_rollout_inner(ctx, spec, start)


async def _run_rollout_inner(ctx: RunContext, spec: RolloutSpec, start: StartFromScratch | StartFromSnapshot) -> RolloutResult:
    cfg = ctx.cfg
    _clear_dir(spec.dir)
    t0 = time.time()
    result = RolloutResult(spec=spec.to_dict(), status="error", started_at=t0)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "calls": 0}
    browser = ctx.make_browser(spec.rollout_id)
    recorder: Recorder | None = None
    replay: ReplayIndex | None = None
    downloads_dir = spec.dir / "downloads"
    steps_path = spec.dir / "steps.jsonl"
    agent: AgentState | None = None
    n_steps = 0
    try:
        # ---- browser and server layers ----
        if spec.snapshot_every_step:
            recorder = Recorder(spec.dir / "server_recording.jsonl")
        storage_state = None
        if isinstance(start, StartFromSnapshot):
            snap = start.snapshot
            bsnap = snap.load_browser()
            storage_state = bsnap.storage_state or None
            if snap.recording_path is not None and Path(snap.recording_path).exists():
                replay = ReplayIndex.load(Path(snap.recording_path), upto=snap.recording_upto)
            if snap.downloads_dir.exists():
                shutil.copytree(snap.downloads_dir, downloads_dir, dirs_exist_ok=True)
            await browser.start(storage_state=storage_state, downloads_dir=downloads_dir, replay=replay, recorder=recorder)
            report = await browser.restore(bsnap)
            result.restore = report.to_dict()
            if not report.active_tab_url_matches:
                log.warning("%s: restored active tab URL %r does not match the record %r", spec.rollout_id,
                            report.actual_url, report.expected_url)
        else:
            await browser.start(start_url=start.start_url, downloads_dir=downloads_dir, recorder=recorder)

        # ---- agent layer ----
        if isinstance(start, StartFromSnapshot) and start.context == "full":
            agent = start.snapshot.load_agent()
            agent.image_root = str(spec.dir)
            agent.pending = None           # superseded by the observation taken after the restore
        else:
            agent = AgentState.fresh(spec.model.id, spec.dir)
        agent.step = 0                     # the goal gets the full step budget (section 4)
        agent.task_message = start.task_message
        ext = browser.screenshot_ext
        obs0 = await browser.observe(spec.dir / f"initial{ext}")
        agent.append_user(user_turn_text(start.task_message, obs0.render(), None), f"initial{ext}")

        # ---- loop ----
        termination = "max_steps"
        answer = None
        with open(steps_path, "w", encoding="utf-8") as steps_f:
            for step in range(1, cfg.experiment.max_steps + 1):
                if agent.pending is not None:
                    p = agent.pending
                    agent.append_user(user_turn_text(None, p["text"], p.get("result")), p.get("screenshot"))
                    agent.pending = None
                agent.step = step
                if ctx.stop.is_set():   # the budget ran out elsewhere: keep what this rollout has and stop here
                    raise SpendingCapExceeded(ctx.stop_reason or "run stopped")
                messages = build_api_messages(agent, history_screenshots=cfg.run.history_screenshots,
                                              supports_images=spec.model.supports_images, image_root=None)
                resp = await ctx.client.chat(model=spec.model.id, messages=messages, max_tokens=spec.model.max_tokens,
                                             temperature=spec.model.temperature, purpose="agent_step",
                                             meta={"rollout_id": spec.rollout_id, "step": step, "kind": spec.kind,
                                                   "level": spec.level, "goal_id": spec.goal_id, "context": spec.context})
                usage["prompt_tokens"] += resp.usage.prompt_tokens
                usage["completion_tokens"] += resp.usage.completion_tokens
                usage["cached_tokens"] += resp.usage.cached_tokens
                usage["cost_usd"] += resp.usage.cost_usd
                usage["calls"] += 1
                agent.append_assistant(resp.text)
                try:
                    action = parse_action(resp.text)
                except ActionParseError as exc:
                    action = None
                    act_result = ActionResult(False, f"Invalid action: {exc}. Reply with exactly one action in the JSON form described at the start.")
                if action is not None:
                    act_result = await _act_with_watchdog(ctx, spec, browser, action, step)
                shot = f"step_{step:03d}{ext}"
                obs = await _observe_with_watchdog(ctx, spec, browser, spec.dir / shot, step)
                response_text = (action.thought if action is not None and action.thought else resp.text.strip())[:4000]
                if action is not None and action.name == "done":
                    response_text = (response_text + "\nFinal answer: " + action.args.get("answer", "")).strip()
                row = {
                    "step_num": step, "screenshot": shot, "response": response_text,
                    "action": action.name if action is not None else "invalid",
                    "arguments": action.args if action is not None else {"error": act_result.message},
                    "final": bool(act_result.final), "ok": act_result.ok, "result": act_result.message, "url": obs.url,
                    "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                    "cached_tokens": resp.usage.cached_tokens, "cost_usd": resp.usage.cost_usd, "ts": time.time(),
                }
                steps_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                steps_f.flush()
                n_steps = step
                agent.set_pending(obs, act_result.message)
                if spec.snapshot_every_step:
                    bsnap = await browser.snapshot()
                    save_snapshot(spec.dir / "snapshots" / f"step_{step:03d}", step=step, agent=agent, browser_snap=bsnap,
                                  downloads_src=downloads_dir, recording_path=recorder.path if recorder else None,
                                  recording_upto=recorder.count if recorder else 0, image_root=spec.dir,
                                  meta={"url": obs.url, "screenshot": shot, "rollout_id": spec.rollout_id})
                if act_result.final:
                    termination = "done"
                    answer = act_result.answer
                    break
        result.status = "done"
        result.termination = termination
        result.answer = answer
    except SpendingCapExceeded as exc:
        result.status = "cap_exceeded"   # steps so far are on disk; the rollout is re-run after the budget is raised
        result.error = str(exc)
        result.termination = "error"
        ctx.request_stop(str(exc))
        raise
    except (ModelCallError, Exception) as exc:
        result.status = "error"
        result.termination = "error"
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        log.error("%s failed: %s: %s", spec.rollout_id, type(exc).__name__, exc)
    finally:
        try:
            if agent is not None:
                agent.save(spec.dir / "agent_state.json")
            result.replay = browser.replay_stats()
        except Exception:
            pass
        try:
            await browser.close()
        except Exception:
            pass
        result.n_steps = n_steps
        result.usage = usage
        result.duration_s = time.time() - t0
        spec.dir.mkdir(parents=True, exist_ok=True)
        (spec.dir / "rollout.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        if result.status == "done":
            # OSWorld writes the evaluator's score here; the official judge only requires a finite number.
            (spec.dir / "result.txt").write_text("1.0\n" if result.termination == "done" else "0.0\n", encoding="utf-8")
    return result
