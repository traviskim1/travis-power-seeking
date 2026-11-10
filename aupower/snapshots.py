"""Three-layer snapshots of a sampled state S (SPEC.md section 4): the agent layer (message history,
scratchpad, step counter), the browser and client layer, and a pointer into the trajectory's server
recording. A snapshot directory is read-only once written; every fork restores from it, so no rollout
ever starts from where another one ended."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentState
from .browser.base import BrowserSnapshot


@dataclass
class StateSnapshot:
    dir: Path
    step: int
    recording_path: Path | None
    recording_upto: int
    meta: dict

    @property
    def agent_path(self) -> Path:
        return self.dir / "agent.json"

    @property
    def browser_path(self) -> Path:
        return self.dir / "browser.json"

    @property
    def downloads_dir(self) -> Path:
        return self.dir / "downloads"

    def load_agent(self) -> AgentState:
        return AgentState.load(self.agent_path)

    def load_browser(self) -> BrowserSnapshot:
        return BrowserSnapshot.from_dict(json.loads(self.browser_path.read_text(encoding="utf-8")))


def save_snapshot(snap_dir: Path, *, step: int, agent: AgentState, browser_snap: BrowserSnapshot,
                  downloads_src: Path | None, recording_path: Path | None, recording_upto: int,
                  image_root: Path, meta: dict | None = None) -> StateSnapshot:
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Screenshot references in the frozen history must stay resolvable from a fork's own directory, so
    # each image part records the absolute directory that holds it.
    agent_dict = agent.to_dict()
    for m in agent_dict["messages"]:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "image_file" and not p.get("root"):
                    p["root"] = str(Path(image_root).resolve())
    if agent_dict.get("pending") and agent_dict["pending"].get("screenshot"):
        agent_dict["pending"]["screenshot_root"] = str(Path(image_root).resolve())
    (snap_dir / "agent.json").write_text(json.dumps(agent_dict, ensure_ascii=False, indent=1), encoding="utf-8")
    (snap_dir / "browser.json").write_text(json.dumps(browser_snap.to_dict(), ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    dl = snap_dir / "downloads"
    if dl.exists():
        shutil.rmtree(dl)
    if downloads_src is not None and downloads_src.exists():
        shutil.copytree(downloads_src, dl)
    else:
        dl.mkdir(exist_ok=True)
    full_meta = {"step": step, "created_at": time.time(), "recording_path": str(recording_path) if recording_path else None,
                 "recording_upto": recording_upto, **(meta or {})}
    (snap_dir / "meta.json").write_text(json.dumps(full_meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return StateSnapshot(dir=snap_dir, step=step, recording_path=recording_path, recording_upto=recording_upto,
                         meta=full_meta)


def load_snapshot(snap_dir: Path) -> StateSnapshot:
    meta = json.loads((Path(snap_dir) / "meta.json").read_text(encoding="utf-8"))
    rp = meta.get("recording_path")
    return StateSnapshot(dir=Path(snap_dir), step=int(meta["step"]), recording_path=Path(rp) if rp else None,
                         recording_upto=int(meta.get("recording_upto", 0)), meta=meta)
