"""Agent layer (SPEC.md section 4): the message history, step counter and scratchpad; serialization for
snapshots; conversion to API messages; and parsing of the model's action."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .browser.base import Action, Observation, normalize_action
from .client import image_data_url
from .prompts import AGENT_SYSTEM_PROMPT, OBSERVATION_HEADER, ACTION_RESULT_HEADER


@dataclass
class AgentState:
    """Everything the agent layer freezes at a sampled state.

    ``messages`` holds the full history: system prompt, the task message, and every observation and
    action so far. Image parts refer to screenshot files by path (relative to the rollout directory)
    rather than embedding bytes; :func:`build_api_messages` turns them into data URLs when calling the
    model. ``pending`` is the observation taken after the last action that the model has not seen yet.
    """
    model_id: str
    messages: list[dict] = field(default_factory=list)
    step: int = 0
    task_message: str = ""
    pending: dict | None = None
    scratchpad: dict = field(default_factory=dict)
    image_root: str = ""

    # ---- construction ----
    @classmethod
    def fresh(cls, model_id: str, image_root: Path | str) -> "AgentState":
        return cls(model_id=model_id, messages=[{"role": "system", "content": AGENT_SYSTEM_PROMPT}],
                   image_root=str(image_root))

    # ---- history ----
    def append_user(self, text: str, screenshot: str | None = None) -> None:
        parts: list[dict] = [{"type": "text", "text": text}]
        if screenshot:
            parts.append({"type": "image_file", "path": screenshot})
        self.messages.append({"role": "user", "content": parts})

    def append_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def set_pending(self, obs: Observation, result_message: str | None) -> None:
        self.pending = {"text": obs.render(), "screenshot": obs.screenshot, "result": result_message,
                        "url": obs.url}

    # ---- serialization ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        return cls(model_id=d["model_id"], messages=list(d["messages"]), step=int(d.get("step", 0)),
                   task_message=d.get("task_message", ""), pending=d.get("pending"),
                   scratchpad=dict(d.get("scratchpad", {})), image_root=d.get("image_root", ""))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AgentState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def n_screenshots(self) -> int:
        return sum(1 for m in self.messages if isinstance(m.get("content"), list)
                   for p in m["content"] if p.get("type") == "image_file")


def user_turn_text(task_message: str | None, obs_text: str, result_message: str | None) -> str:
    """The text of one user turn: optional task message, optional result of the last action, observation."""
    blocks = []
    if task_message:
        blocks.append(task_message)
    if result_message:
        blocks.append(f"{ACTION_RESULT_HEADER}\n{result_message}")
    blocks.append(f"{OBSERVATION_HEADER}\n{obs_text}")
    return "\n\n".join(blocks)


def build_api_messages(state: AgentState, *, history_screenshots: int, supports_images: bool,
                       image_root: Path | None = None) -> list[dict]:
    """Convert the stored history into OpenAI-style messages. Only the last ``history_screenshots``
    screenshots are attached as images; older ones are replaced by a short note, so the text history is
    kept in full while the image context stays bounded."""
    root = Path(image_root) if image_root is not None else Path(state.image_root or ".")
    # indices of image parts in order
    image_positions: list[tuple[int, int]] = []
    for mi, m in enumerate(state.messages):
        c = m.get("content")
        if isinstance(c, list):
            for pi, p in enumerate(c):
                if p.get("type") == "image_file":
                    image_positions.append((mi, pi))
    keep = set(image_positions[-history_screenshots:]) if (supports_images and history_screenshots > 0) else set()
    out: list[dict] = []
    for mi, m in enumerate(state.messages):
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": m["role"], "content": c})
            continue
        parts: list[dict] = []
        for pi, p in enumerate(c):
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image_file":
                if (mi, pi) in keep:
                    path = Path(p.get("root") or root) / p["path"]
                    if path.exists():
                        parts.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
                    else:
                        parts.append({"type": "text", "text": "[screenshot unavailable]"})
                else:
                    parts.append({"type": "text", "text": "[earlier screenshot omitted]"})
        out.append({"role": m["role"], "content": parts})
    return out


# --------------------------------------------------------------------------------------------
# Action parsing
# --------------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _json_objects(text: str) -> list[str]:
    """All top-level {...} substrings in order (brace matching, string-aware)."""
    out = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    return out


class ActionParseError(ValueError):
    pass


def parse_action(text: str) -> Action:
    """Extract the single action from the model's reply. Raises ActionParseError with a message that is
    fed back to the model as the result of an invalid step."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates += [c for c in _json_objects(text) if c not in candidates]
    last_err = "no JSON object with an 'action' field found"
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = f"invalid JSON: {exc}"
            continue
        if not isinstance(obj, dict) or "action" not in obj:
            continue
        try:
            action = normalize_action(obj["action"], obj.get("args") or {})
        except ValueError as exc:
            raise ActionParseError(str(exc)) from None
        action.thought = str(obj.get("thought", "") or "")
        action.raw = text
        return action
    raise ActionParseError(last_err)
