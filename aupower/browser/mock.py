"""A deterministic fake web for dry runs. No network. Pages, links, forms, cookies, local and session
storage, tabs and downloads behave consistently enough to exercise every snapshot and restore path."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, quote

from ..config import RunConfig
from ..pngutil import write_png
from ..tasks import derive_seed
from .base import (Action, ActionResult, Browser, BrowserSnapshot, Element, Observation, RestoreReport, TabInfo,
                   TabState, same_url)
from .recorder import Recorder, ReplayIndex, ReplayStats

_WORDS = ["search", "results", "item", "detail", "cart", "login", "article", "compare", "review", "booking",
          "flights", "hotels", "episode", "cast", "recipe", "price"]
_LOREM = ("the page lists several options with prices and ratings", "click an item to see its details",
          "sign in to save your selection", "results are sorted by relevance", "a summary appears at the top",
          "the cast and crew are listed below", "the air date is shown in the sidebar", "add items to the cart")


@dataclass
class MockTab:
    url: str
    scroll_y: int = 0
    form_values: list[dict] = field(default_factory=list)
    session: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    forward: list[str] = field(default_factory=list)


def _colour(s: str) -> tuple[int, int, int]:
    h = hashlib.sha1(s.encode("utf-8")).digest()
    return (h[0], h[1], h[2])


class MockBrowser(Browser):
    def __init__(self, run_cfg: RunConfig, seed: int = 0):
        self.cfg = run_cfg
        self.seed = seed
        self.tabs: list[MockTab] = []
        self.active = 0
        self.cookies: dict[str, str] = {}
        self.local: dict[str, dict[str, str]] = {}
        self.downloads_dir: Path | None = None
        self.recorder: Recorder | None = None
        self.replay: ReplayIndex | None = None
        self._stats = ReplayStats()
        self.closed = False

    # ---- fake web ----
    def _page(self, url: str) -> dict:
        rng = random.Random(derive_seed(self.seed, "mockpage", url))
        parsed = urlparse(url)
        host = parsed.netloc or "www.google.com"
        path = parsed.path.strip("/")
        name = host.replace("www.", "").split(".")[0].title()
        title = f"{name} - {path.replace('-', ' ').title() or 'Home'}"
        if parsed.query:
            title += " - Search results"
        elements: list[Element] = []
        idx = 0
        elements.append(Element(idx, "input", "Search", type="search", name="q")); idx += 1
        elements.append(Element(idx, "button", "Search")); idx += 1
        if "google" in host and not path:
            for h in ["https://www.shopmock.com", "https://www.wikimock.org", "https://travelmock.net",
                      "https://www.newsmock.com", "https://recipesmock.io"]:
                elements.append(Element(idx, "a", h.split("//")[1], href=f"{h}/{rng.choice(_WORDS)}-{rng.randint(1, 50)}"))
                idx += 1
        else:
            for j in range(rng.randint(3, 7)):
                word = rng.choice(_WORDS)
                elements.append(Element(idx, "a", f"{word.title()} {j + 1}", href=f"https://{host}/{word}-{rng.randint(1, 50)}",
                                        in_viewport=j < 4))
                idx += 1
            elements.append(Element(idx, "button", rng.choice(["Add to cart", "Sign in", "Download PDF", "Show more"]))); idx += 1
        text = f"Welcome to {title}. " + ". ".join(rng.choice(_LOREM) for _ in range(12)) + "."
        return {"title": title, "text": text, "elements": elements, "height": 1600 + 400 * rng.randint(0, 5)}

    def _request(self, method: str, url: str) -> None:
        page = self._page(url)
        if self.recorder is not None:
            self.recorder.add(method=method, url=url, status=200, headers={"content-type": "text/html"},
                              body=f"<html><title>{page['title']}</title></html>".encode("utf-8"), mime="text/html")
        if self.replay is not None:
            entry = self.replay.take(method, url, None)
            if entry is not None:
                self._stats.log_served(url)
            else:
                self._stats.log_live(url)

    def _load(self, tab: MockTab, url: str, push_history: bool = True) -> None:
        if push_history and tab.url:
            tab.history.append(tab.url)
            tab.forward.clear()
        tab.url = url
        tab.scroll_y = 0
        tab.form_values = []
        host = urlparse(url).netloc
        origin = f"https://{host}"
        self.cookies[f"session_{host}"] = str(len(tab.history) + 1)
        self.local.setdefault(origin, {})["last_path"] = urlparse(url).path
        tab.session[f"visits_{host}"] = str(int(tab.session.get(f"visits_{host}", "0")) + 1)
        self._request("GET", url)

    @property
    def tab(self) -> MockTab:
        self.active = min(self.active, len(self.tabs) - 1)
        return self.tabs[self.active]

    # ---- interface ----
    async def start(self, *, start_url: str | None = None, storage_state: dict | None = None,
                    downloads_dir: Path | None = None, replay: ReplayIndex | None = None,
                    recorder: Recorder | None = None) -> None:
        self.downloads_dir = downloads_dir
        self.replay = replay
        self.recorder = recorder
        if replay is not None:
            self._stats = replay.stats
        if storage_state:
            for c in storage_state.get("cookies", []):
                self.cookies[c["name"]] = c["value"]
            for o in storage_state.get("origins", []):
                self.local[o["origin"]] = {kv["name"]: kv["value"] for kv in o.get("localStorage", [])}
        if start_url:
            tab = MockTab(url="")
            self.tabs.append(tab)
            self._load(tab, start_url, push_history=False)
            self.active = 0

    async def observe(self, screenshot_path: Path | None = None) -> Observation:
        tab = self.tab
        page = self._page(tab.url)
        shot = None
        if screenshot_path is not None:
            write_png(screenshot_path, 64, 40, _colour(tab.url), stripe=_colour(page["title"]))
            shot = Path(screenshot_path).name
        tabs = [TabInfo(i, t.url, self._page(t.url)["title"], i == self.active) for i, t in enumerate(self.tabs)]
        return Observation(url=tab.url, title=page["title"], tabs=tabs, elements=page["elements"], text=page["text"],
                           screenshot=shot, scroll_y=tab.scroll_y, doc_height=page["height"],
                           viewport_height=self.cfg.viewport[1])

    async def act(self, action: Action) -> ActionResult:
        tab = self.tab
        page = self._page(tab.url)
        a = action.args
        n = action.name
        if n == "navigate":
            self._load(tab, a["url"])
            return ActionResult(True, f"Navigated to {tab.url}")
        if n == "click":
            idx = a["index"]
            if not 0 <= idx < len(page["elements"]):
                return ActionResult(False, f"No element with index {idx} on this page (0..{len(page['elements']) - 1}).")
            el = page["elements"][idx]
            if el.tag == "a":
                self._load(tab, el.href)
                return ActionResult(True, f"Clicked link {idx}; now at {tab.url}")
            if el.tag == "button":
                if el.text == "Download PDF" and self.downloads_dir is not None:
                    self.downloads_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"download_{len(list(self.downloads_dir.iterdir())) + 1}.pdf"
                    (self.downloads_dir / fname).write_bytes(b"%PDF-1.4 mock\n")
                    return ActionResult(True, f"Clicked {el.text!r}; file {fname} downloaded")
                if el.text == "Search":
                    q = next((f["value"] for f in tab.form_values if f.get("name") == "q"), "")
                    self._load(tab, f"https://{urlparse(tab.url).netloc}/search?q={quote(q)}")
                    return ActionResult(True, f"Searched; now at {tab.url}")
                self.cookies[f"clicked_{el.text.lower().replace(' ', '_')}"] = "1"
                return ActionResult(True, f"Clicked button {el.text!r}")
            return ActionResult(True, f"Focused element {idx}")
        if n == "type":
            idx = a["index"]
            if not 0 <= idx < len(page["elements"]) or page["elements"][idx].tag not in ("input", "textarea"):
                return ActionResult(False, f"Element {idx} is not a text field.")
            el = page["elements"][idx]
            tab.form_values = [f for f in tab.form_values if f["i"] != idx]
            tab.form_values.append({"i": idx, "name": el.name, "id": "", "type": el.type or "text", "value": a["text"],
                                    "checked": False})
            if a.get("press_enter"):
                self._load(tab, f"https://{urlparse(tab.url).netloc}/search?q={quote(a['text'])}")
                return ActionResult(True, f"Typed and submitted; now at {tab.url}")
            return ActionResult(True, f"Typed into element {idx}")
        if n == "scroll":
            amount = a.get("amount") or int(self.cfg.viewport[1] * 0.8)
            tab.scroll_y = max(0, min(page["height"] - self.cfg.viewport[1], tab.scroll_y + (amount if a["direction"] == "down" else -amount)))
            return ActionResult(True, f"Scrolled {a['direction']} to {tab.scroll_y}px")
        if n == "back":
            if not tab.history:
                return ActionResult(False, "No previous page in this tab.")
            tab.forward.append(tab.url)
            prev = tab.history.pop()
            self._load(tab, prev, push_history=False)
            return ActionResult(True, f"Went back to {tab.url}")
        if n == "forward":
            if not tab.forward:
                return ActionResult(False, "No forward page in this tab.")
            nxt = tab.forward.pop()
            tab.history.append(tab.url)
            self._load(tab, nxt, push_history=False)
            return ActionResult(True, f"Went forward to {tab.url}")
        if n == "new_tab":
            t = MockTab(url="")
            self.tabs.append(t)
            self.active = len(self.tabs) - 1
            self._load(t, a.get("url") or "about:blank", push_history=False)
            return ActionResult(True, f"Opened tab {self.active} at {t.url}")
        if n == "switch_tab":
            idx = a["index"]
            if not 0 <= idx < len(self.tabs):
                return ActionResult(False, f"No tab with index {idx}.")
            self.active = idx
            return ActionResult(True, f"Switched to tab {idx}")
        if n == "close_tab":
            idx = a.get("index", self.active)
            if not 0 <= idx < len(self.tabs):
                return ActionResult(False, f"No tab with index {idx}.")
            self.tabs.pop(idx)
            if not self.tabs:
                t = MockTab(url="")
                self.tabs.append(t)
                self._load(t, "about:blank", push_history=False)
            self.active = min(self.active, len(self.tabs) - 1)
            return ActionResult(True, f"Closed tab {idx}")
        if n == "press":
            return ActionResult(True, f"Pressed {a['key']}")
        if n == "wait":
            return ActionResult(True, "Waited.")
        if n == "done":
            return ActionResult(True, "Task finished.", final=True, answer=a.get("answer", ""))
        return ActionResult(False, f"Unsupported action {n}")

    async def snapshot(self) -> BrowserSnapshot:
        tabs = [TabState(url=t.url, title=self._page(t.url)["title"], scroll_x=0, scroll_y=t.scroll_y,
                         form_values=[dict(f) for f in t.form_values], session_storage=dict(t.session))
                for t in self.tabs]
        storage = {
            "cookies": [{"name": k, "value": v, "domain": k.split("_", 1)[-1], "path": "/"} for k, v in sorted(self.cookies.items())],
            "origins": [{"origin": o, "localStorage": [{"name": k, "value": v} for k, v in sorted(kv.items())]}
                        for o, kv in sorted(self.local.items())],
        }
        downloads = sorted(p.name for p in self.downloads_dir.iterdir()) if self.downloads_dir and self.downloads_dir.exists() else []
        return BrowserSnapshot(tabs=tabs, active_tab=self.active, storage_state=storage, downloads=downloads,
                               extra={"history": [list(t.history) for t in self.tabs]})

    async def restore(self, snap: BrowserSnapshot) -> RestoreReport:
        self.tabs = []
        histories = snap.extra.get("history", [])
        for i, ts in enumerate(snap.tabs):
            t = MockTab(url=ts.url)
            t.scroll_y = ts.scroll_y
            t.form_values = [dict(f) for f in ts.form_values]
            t.session = dict(ts.session_storage)
            t.history = list(histories[i]) if i < len(histories) else []
            self.tabs.append(t)
            self._request("GET", ts.url)
        if not self.tabs:
            self.tabs.append(MockTab(url="about:blank"))
        self.active = min(snap.active_tab, len(self.tabs) - 1)
        expected = snap.tabs[snap.active_tab].url if snap.tabs else "about:blank"
        return RestoreReport(active_tab_url_matches=same_url(self.tab.url, expected), expected_url=expected,
                             actual_url=self.tab.url, tabs_restored=len(snap.tabs))

    async def close(self) -> None:
        self.closed = True
        if self.recorder is not None:
            self.recorder.close()

    def replay_stats(self) -> dict:
        return self._stats.to_dict() if self.replay is not None else {}
