"""Browser and client layer (SPEC.md section 4): a common interface, a mock implementation for dry runs,
a Playwright implementation, and the server-layer recorder/replayer."""
from __future__ import annotations

from ..config import Config
from .base import (Action, ActionResult, Browser, BrowserSnapshot, Element, Observation, RestoreReport,
                   TabInfo, TabState, normalize_action, ACTION_NAMES)
from .recorder import Recorder, ReplayIndex, ReplayStats


def make_browser(cfg: Config, seed: int = 0) -> Browser:
    kind = cfg.run.effective_browser
    if kind == "mock":
        from .mock import MockBrowser
        return MockBrowser(cfg.run, seed=seed)
    if kind == "playwright":
        from .playwright_browser import PlaywrightBrowser
        return PlaywrightBrowser(cfg.run)
    raise ValueError(f"unknown browser kind {kind!r}")


__all__ = ["Action", "ActionResult", "Browser", "BrowserSnapshot", "Element", "Observation", "RestoreReport",
           "TabInfo", "TabState", "normalize_action", "ACTION_NAMES", "Recorder", "ReplayIndex", "ReplayStats",
           "make_browser"]
