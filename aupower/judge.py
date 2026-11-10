"""Scoring rollouts with the Odysseys rubric (SPEC.md section 6).

The judge is the official per-rubric judge: for each rubric item, one call with the task, the item, the
action history and every screenshot, answered with a binary verdict. The trajectory is read from
steps.jsonl exactly as the official script reads it. A rollout is scored at a step limit H by cutting it
at that step (rows with step_num <= H) before building the judge input, which is what the official
script's --max-steps option does.

Two extra modes support the method:
* ``first_step``: one call per item asking also for the first step at which the item was met (used to
  measure T on the s0 rollouts; SPEC.md section 8 asks for it to be validated against re-scoring).
* ``bisect``: finds the first step with the official prompt by bisecting the cut-off (the official
  verdict is "satisfied if any step satisfies it", so it is monotone in the cut).
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .client import SpendingCapExceeded, image_data_url
from .prompts import (FINAL_JUDGMENT_MAX_COMPLETION_TOKENS, FIRST_STEP_JUDGMENT_SYSTEM, FULL_TRAJ_JUDGMENT_SYSTEM,
                      first_step_user_text, judge_user_text)
from .stats import averaged_rubric_score, perfect, weighted_rubric_score
from .tasks import Task

log = logging.getLogger("aupower.judge")

_STATUS_RE = re.compile(r'Status:\s*["\']?(success|failure)["\']?', re.IGNORECASE)
_THOUGHTS_RE = re.compile(r"Thoughts:\s*(.+?)(?:Status:|$)", re.DOTALL)
_FIRST_STEP_RE = re.compile(r"First satisfied at step:\s*\**\s*[\"']?(\d+|none)", re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# Reading a rollout the way the official script does
# --------------------------------------------------------------------------------------------

@dataclass
class StepRecord:
    step: int
    screenshot: str
    response: str
    action_line: str
    final: bool


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def load_steps(rollout_dir: Path, max_steps: int | None = None) -> list[StepRecord]:
    """Group steps.jsonl rows by step_num (rows with step_num > max_steps are dropped), merging the
    response and action text as the official judge script does."""
    path = Path(rollout_dir) / "steps.jsonl"
    if not path.exists():
        return []
    grouped: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, row in enumerate(rows):
        step = int(row.get("step_num", i) or i)
        if max_steps is not None and max_steps > 0 and step > max_steps:
            continue
        g = grouped.setdefault(step, {"screenshot": "", "response": "", "_actions": [], "final": False})
        if row.get("screenshot") or row.get("screenshot_file"):
            g["screenshot"] = str(row.get("screenshot") or row.get("screenshot_file"))
        response = str(row.get("response") or "").strip()
        if response:
            g["response"] = response
        action = str(row.get("action") or "").strip()
        if action and action.lower() != "screenshot":
            arguments = row.get("arguments")
            if isinstance(arguments, dict):
                arguments = {k: v for k, v in arguments.items() if k != "action"}
                text = f"{action} {json.dumps(arguments, ensure_ascii=True, sort_keys=True)}" if arguments else action
            else:
                text = action
        else:
            text = str(row.get("action_line") or "").strip()
        _append_unique(g["_actions"], text)
        g["final"] = g["final"] or row.get("final") is True
    return [StepRecord(step=s, screenshot=g["screenshot"], response=g["response"], action_line=" -> ".join(g["_actions"]),
                       final=g["final"]) for s, g in sorted(grouped.items())]


def count_steps(rollout_dir: Path) -> int:
    steps = load_steps(rollout_dir)
    return max((s.step for s in steps), default=0)


def action_history(steps: list[StepRecord]) -> str:
    lines = []
    for i, step in enumerate(steps, start=1):
        parts = []
        if step.response.strip():
            parts.append(f"Response: {step.response.strip()}")
        if step.action_line.strip():
            parts.append(f"Action: {step.action_line.strip()}")
        if parts:
            lines.append(f"{i}. " + "\n".join(parts))
    return "\n".join(lines) if lines else "No actions recorded."


def screenshot_paths(rollout_dir: Path, steps: list[StepRecord], max_images: int = 0) -> list[Path]:
    resolved = []
    for step in steps:
        if not step.screenshot:
            continue
        raw = Path(step.screenshot)
        candidates = [raw] if raw.is_absolute() else [rollout_dir / raw.name, rollout_dir / raw, raw]
        match = next((p for p in candidates if p.exists()), None)
        if match is not None:
            resolved.append(match)
    return resolved[-max_images:] if max_images > 0 else resolved


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------

@dataclass
class ItemVerdict:
    rubric_id: str
    requirement: str
    verification: str
    weight: float | None
    score: int
    success: bool
    final_reasoning: str
    first_step: int | None = None     # only in first-step / bisect results

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeResult:
    rollout_id: str
    task_id: str
    mode: str                          # "rescore" | "first_step" | "bisect"
    cut: int                           # number of steps the judge saw
    n_steps_full: int                  # full rollout length
    judge_model: str
    items: list[ItemVerdict] = field(default_factory=list)
    average_rubric_score: float = 0.0
    weighted_rubric_score: float | None = None
    perfect: bool = False
    n_screenshots_sent: int = 0
    error: str | None = None

    @property
    def verdicts(self) -> list[int]:
        return [it.score for it in self.items]

    @property
    def first_steps(self) -> dict[str, int | None]:
        return {it.rubric_id: it.first_step for it in self.items}

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeResult":
        items = [ItemVerdict(**it) for it in d.get("items", [])]
        return cls(rollout_id=d["rollout_id"], task_id=d["task_id"], mode=d["mode"], cut=int(d["cut"]),
                   n_steps_full=int(d["n_steps_full"]), judge_model=d.get("judge_model", ""), items=items,
                   average_rubric_score=float(d.get("average_rubric_score", 0.0)),
                   weighted_rubric_score=d.get("weighted_rubric_score"), perfect=bool(d.get("perfect", False)),
                   n_screenshots_sent=int(d.get("n_screenshots_sent", 0)), error=d.get("error"))


def parse_verdict(text: str) -> tuple[bool, str]:
    """The official parse: Status success/failure (missing status = failure), and the Thoughts text."""
    status = _STATUS_RE.search(text)
    thoughts = _THOUGHTS_RE.search(text)
    success = bool(status and status.group(1).lower() == "success")
    reasoning = thoughts.group(1).strip() if thoughts else (text.strip() or "Empty judge response.")
    return success, reasoning


def parse_first_step(text: str, n_steps: int) -> int | None:
    m = _FIRST_STEP_RE.search(text)
    if not m or m.group(1).lower() == "none":
        return None
    v = int(m.group(1))
    if v < 1:
        return None
    return min(v, n_steps)


def _finish(result: JudgeResult, task: Task) -> JudgeResult:
    if result.items:
        v = result.verdicts
        result.average_rubric_score = round(averaged_rubric_score(v), 6)
        result.weighted_rubric_score = weighted_rubric_score(v, task.weights)
        result.perfect = perfect(v)
    return result


def judge_cache_path(rollout_dir: Path, mode: str, cut: int | None = None) -> Path:
    d = Path(rollout_dir) / "judge"
    if mode == "rescore":
        return d / f"rescore_cut{cut:03d}.json"
    return d / f"{mode}.json"


def load_judge_result(path: Path) -> JudgeResult | None:
    if not path.exists():
        return None
    try:
        r = JudgeResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        return None if r.error else r
    except Exception:
        return None


def _save(path: Path, result: JudgeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


def build_messages(task: Task, item, steps: list[StepRecord], shots: list[Path], mode: str) -> list[dict]:
    history = action_history(steps)
    if mode == "first_step":
        system = FIRST_STEP_JUDGMENT_SYSTEM
        text = first_step_user_text(task.text.strip(), item.id, item.requirement, item.verification, history,
                                    len(shots), len(steps))
    else:
        system = FULL_TRAJ_JUDGMENT_SYSTEM
        text = judge_user_text(task.text.strip(), item.id, item.requirement, item.verification, history,
                               len(shots), len(steps))
    content: list[dict] = [{"type": "text", "text": text}]
    for p in shots:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(p), "detail": "high"}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


# --------------------------------------------------------------------------------------------
# Judge calls
# --------------------------------------------------------------------------------------------

async def _judge_items(ctx, rollout_dir: Path, task: Task, rollout_id: str, steps: list[StepRecord], mode: str,
                       cut: int, n_full: int) -> JudgeResult:
    cfg = ctx.cfg
    shots = screenshot_paths(rollout_dir, steps, cfg.judge.max_images)
    judge = cfg.models.judge
    result = JudgeResult(rollout_id=rollout_id, task_id=task.task_id, mode=mode, cut=cut, n_steps_full=n_full,
                         judge_model=judge.id, n_screenshots_sent=len(shots))
    purpose = "judge_first_step" if mode == "first_step" else "judge_rescore"

    async def one(item) -> ItemVerdict:
        async with ctx.judge_sem:
            if ctx.stop.is_set():   # the budget ran out: no further judge call
                raise SpendingCapExceeded(ctx.stop_reason or "run stopped")
            # built only while a slot is held, so at most max_concurrent_judge_calls screenshot payloads exist at once
            messages = build_messages(task, item, steps, shots, mode)
            try:
                resp = await ctx.client.chat(model=judge.id, messages=messages, max_tokens=FINAL_JUDGMENT_MAX_COMPLETION_TOKENS,
                                             temperature=judge.temperature, purpose=purpose,
                                             meta={"rollout_id": rollout_id, "rubric_id": item.id, "cut": cut,
                                                   "n_steps_full": n_full, "task_id": task.task_id})
            except SpendingCapExceeded as exc:
                ctx.request_stop(str(exc))
                raise
        success, reasoning = parse_verdict(resp.text)
        first = None
        if mode == "first_step":
            first = parse_first_step(resp.text, len(steps)) if success else None
            if success and first is None:
                first = len(steps)   # met, but no usable step reported: the latest step the judge saw
        return ItemVerdict(rubric_id=item.id, requirement=item.requirement, verification=item.verification,
                           weight=item.weight, score=1 if success else 0, success=success, final_reasoning=reasoning,
                           first_step=first)

    result.items = list(await asyncio.gather(*(one(item) for item in task.rubrics)))
    return _finish(result, task)


async def judge_at_cut(ctx, rollout_dir: Path, task: Task, rollout_id: str, cut: int) -> JudgeResult:
    """Official per-rubric verdicts on the rollout cut at ``cut`` steps (cached per effective cut)."""
    rollout_dir = Path(rollout_dir)
    n_full = count_steps(rollout_dir)
    eff = max(0, min(int(cut), n_full))
    path = judge_cache_path(rollout_dir, "rescore", eff)
    cached = load_judge_result(path)
    if cached is not None:
        return cached
    steps = load_steps(rollout_dir, max_steps=eff) if eff > 0 else []
    if not steps:
        result = JudgeResult(rollout_id=rollout_id, task_id=task.task_id, mode="rescore", cut=eff, n_steps_full=n_full,
                             judge_model=ctx.cfg.models.judge.id)
        result.items = [ItemVerdict(it.id, it.requirement, it.verification, it.weight, 0, False,
                                    "Empty trajectory - no steps recorded.") for it in task.rubrics]
        result = _finish(result, task)
        _save(path, result)
        return result
    result = await _judge_items(ctx, rollout_dir, task, rollout_id, steps, "rescore", eff, n_full)
    _save(path, result)
    return result


async def judge_first_steps(ctx, rollout_dir: Path, task: Task, rollout_id: str) -> JudgeResult:
    """One call per rubric item on the full rollout, reporting the first step each item was met."""
    rollout_dir = Path(rollout_dir)
    path = judge_cache_path(rollout_dir, "first_step")
    cached = load_judge_result(path)
    if cached is not None:
        return cached
    steps = load_steps(rollout_dir)
    n_full = max((s.step for s in steps), default=0)
    if not steps:
        result = JudgeResult(rollout_id=rollout_id, task_id=task.task_id, mode="first_step", cut=0, n_steps_full=0,
                             judge_model=ctx.cfg.models.judge.id)
        result.items = [ItemVerdict(it.id, it.requirement, it.verification, it.weight, 0, False,
                                    "Empty trajectory - no steps recorded.") for it in task.rubrics]
        result = _finish(result, task)
        _save(path, result)
        return result
    result = await _judge_items(ctx, rollout_dir, task, rollout_id, steps, "first_step", n_full, n_full)
    _save(path, result)
    return result


async def judge_bisect_first_steps(ctx, rollout_dir: Path, task: Task, rollout_id: str) -> JudgeResult:
    """First-met step per item using only the official prompt: judge the full rollout, then bisect the
    cut-off for each item that is met (the official verdict is monotone in the cut)."""
    rollout_dir = Path(rollout_dir)
    path = judge_cache_path(rollout_dir, "bisect")
    cached = load_judge_result(path)
    if cached is not None:
        return cached
    n_full = count_steps(rollout_dir)
    full = await judge_at_cut(ctx, rollout_dir, task, rollout_id, n_full)
    items: list[ItemVerdict] = []
    for it in full.items:
        if not it.success:
            items.append(ItemVerdict(**{**it.to_dict(), "first_step": None}))
            continue
        lo, hi = 1, n_full            # invariant: met at hi; search the smallest met cut
        while lo < hi:
            mid = (lo + hi) // 2
            r = await judge_at_cut(ctx, rollout_dir, task, rollout_id, mid)
            met = next((x.success for x in r.items if x.rubric_id == it.rubric_id), False)
            if met:
                hi = mid
            else:
                lo = mid + 1
        items.append(ItemVerdict(**{**it.to_dict(), "first_step": hi}))
    result = JudgeResult(rollout_id=rollout_id, task_id=task.task_id, mode="bisect", cut=n_full, n_steps_full=n_full,
                         judge_model=ctx.cfg.models.judge.id, items=items, n_screenshots_sent=full.n_screenshots_sent)
    result = _finish(result, task)
    _save(path, result)
    return result


def score_from_first_steps(result: JudgeResult, cut: int) -> float:
    """Averaged rubric score at a cut, derived from first-met steps (item met iff first_step <= cut)."""
    verdicts = [1 if (it.first_step is not None and it.first_step <= cut) else 0 for it in result.items]
    return averaged_rubric_score(verdicts)
