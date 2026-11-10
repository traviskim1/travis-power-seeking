"""The browser interface shared by the mock and Playwright implementations, and the action vocabulary."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .recorder import Recorder, ReplayIndex


# --------------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------------

ACTION_NAMES = ("navigate", "click", "type", "scroll", "back", "forward", "new_tab", "switch_tab", "close_tab",
                "press", "wait", "done")


@dataclass
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    raw: str = ""

    def line(self) -> str:
        """The action as one line, in the form the official judge builds from steps.jsonl."""
        import json
        return f"{self.name} {json.dumps(self.args, ensure_ascii=True, sort_keys=True)}" if self.args else self.name


def normalize_action(name: str, args: dict | None) -> Action:
    """Validate an action name and its arguments; raises ValueError with a message for the agent."""
    args = dict(args or {})
    name = str(name).strip().lower()
    if name not in ACTION_NAMES:
        raise ValueError(f"unknown action {name!r}; use one of {', '.join(ACTION_NAMES)}")

    def need_int(key: str) -> int:
        if key not in args:
            raise ValueError(f"action {name!r} needs an integer argument {key!r}")
        try:
            return int(args[key])
        except (TypeError, ValueError):
            raise ValueError(f"argument {key!r} of action {name!r} must be an integer") from None

    out: dict[str, Any] = {}
    if name == "navigate":
        url = str(args.get("url", "")).strip()
        if not url:
            raise ValueError("navigate needs a 'url'")
        if not url.lower().startswith(("http://", "https://")):
            url = "https://" + url
        out["url"] = url
    elif name == "click":
        out["index"] = need_int("index")
    elif name == "type":
        out["index"] = need_int("index")
        out["text"] = str(args.get("text", ""))
        out["press_enter"] = bool(args.get("press_enter", False))
    elif name == "scroll":
        d = str(args.get("direction", "down")).lower()
        if d not in ("up", "down"):
            raise ValueError("scroll direction must be 'up' or 'down'")
        out["direction"] = d
        if "amount" in args and args["amount"] is not None:
            out["amount"] = int(args["amount"])
    elif name == "new_tab":
        url = str(args.get("url", "") or "").strip()
        if url and not url.lower().startswith(("http://", "https://")):
            url = "https://" + url
        if url:
            out["url"] = url
    elif name == "switch_tab":
        out["index"] = need_int("index")
    elif name == "close_tab":
        if "index" in args and args["index"] is not None:
            out["index"] = int(args["index"])
    elif name == "press":
        key = str(args.get("key", "")).strip()
        if not key:
            raise ValueError("press needs a 'key'")
        out["key"] = key
    elif name == "wait":
        try:
            out["seconds"] = min(10.0, max(0.0, float(args.get("seconds", 1))))
        except (TypeError, ValueError):
            raise ValueError("wait 'seconds' must be a number") from None
    elif name == "done":
        out["answer"] = str(args.get("answer", ""))
    return Action(name=name, args=out)


@dataclass
class ActionResult:
    ok: bool
    message: str
    final: bool = False
    answer: str | None = None


# --------------------------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------------------------

@dataclass
class Element:
    index: int
    tag: str
    text: str = ""
    role: str = ""
    type: str = ""
    href: str = ""
    name: str = ""
    in_viewport: bool = True

    def render(self) -> str:
        bits = [f"[{self.index}] <{self.tag}>"]
        if self.role:
            bits.append(f"role={self.role}")
        if self.type:
            bits.append(f"type={self.type}")
        if self.name:
            bits.append(f"name={self.name}")
        if self.text:
            bits.append(f'"{self.text}"')
        if self.href:
            bits.append(f"href={self.href}")
        if not self.in_viewport:
            bits.append("(below the fold)")
        return " ".join(bits)


@dataclass
class TabInfo:
    index: int
    url: str
    title: str
    active: bool


@dataclass
class Observation:
    url: str
    title: str
    tabs: list[TabInfo]
    elements: list[Element]
    text: str
    screenshot: str | None = None     # path of the screenshot file, relative to the rollout directory
    scroll_y: int = 0
    doc_height: int = 0
    viewport_height: int = 0
    note: str = ""

    def render(self) -> str:
        lines = ["Open tabs:"]
        for t in self.tabs:
            lines.append(f"  tab {t.index}. {'(active) ' if t.active else ''}{t.title[:80]} - {t.url[:160]}")
        lines.append(f"Active tab URL: {self.url}")
        lines.append(f"Page title: {self.title}")
        if self.doc_height and self.viewport_height:
            lines.append(f"Scroll position: {self.scroll_y}px of {self.doc_height}px (viewport {self.viewport_height}px)")
        if self.note:
            lines.append(f"Note: {self.note}")
        lines.append("Interactive elements:")
        if self.elements:
            lines.extend(e.render() for e in self.elements)
        else:
            lines.append("(none found)")
        lines.append("Visible page text (excerpt):")
        lines.append(self.text.strip() if self.text.strip() else "(empty)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------------------------
# Snapshots (browser and client state)
# --------------------------------------------------------------------------------------------

@dataclass
class TabState:
    url: str
    title: str = ""
    scroll_x: int = 0
    scroll_y: int = 0
    form_values: list[dict] = field(default_factory=list)      # [{i, name, id, type, value, checked}]
    session_storage: dict[str, str] = field(default_factory=dict)


@dataclass
class BrowserSnapshot:
    tabs: list[TabState]
    active_tab: int
    storage_state: dict               # cookies + local storage (Playwright storage_state() format)
    downloads: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tabs": [asdict(t) for t in self.tabs], "active_tab": self.active_tab,
                "storage_state": self.storage_state, "downloads": list(self.downloads), "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "BrowserSnapshot":
        return cls(tabs=[TabState(**t) for t in d["tabs"]], active_tab=int(d["active_tab"]),
                   storage_state=d.get("storage_state", {}), downloads=list(d.get("downloads", [])),
                   extra=dict(d.get("extra", {})))


@dataclass
class RestoreReport:
    active_tab_url_matches: bool
    expected_url: str
    actual_url: str
    tabs_restored: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def same_url(a: str, b: str) -> bool:
    def norm(u: str) -> str:
        u = u.split("#", 1)[0]
        return u[:-1] if u.endswith("/") else u
    return norm(a) == norm(b)


# --------------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------------

class Browser(ABC):
    """One browser session (one rollout). Tabs are addressed by index in creation order."""

    screenshot_ext: str = ".png"

    @abstractmethod
    async def start(self, *, start_url: str | None = None, storage_state: dict | None = None,
                    downloads_dir: Path | None = None, replay: ReplayIndex | None = None,
                    recorder: Recorder | None = None) -> None: ...

    @abstractmethod
    async def observe(self, screenshot_path: Path | None = None) -> Observation: ...

    @abstractmethod
    async def act(self, action: Action) -> ActionResult: ...

    @abstractmethod
    async def snapshot(self) -> BrowserSnapshot: ...

    @abstractmethod
    async def restore(self, snap: BrowserSnapshot) -> RestoreReport: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def recover_active_page(self) -> None:
        """Replace the active page when it stops answering (a no-op where that cannot happen)."""
        return None

    @abstractmethod
    def replay_stats(self) -> dict: ...
