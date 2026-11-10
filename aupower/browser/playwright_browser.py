"""Playwright (Chromium) implementation of the browser and client layer.

What is frozen at a snapshot (SPEC.md section 4, "lighter fallback"): open tabs and their URLs, the active
tab, scroll positions, form contents, cookies and logins and local storage (Playwright ``storage_state``),
session storage, and downloaded files. The server layer is recorded from every response the context
receives and replayed to forks through a route handler; unrecorded requests go to the live web and are
counted (see ``recorder.py``).
"""
from __future__ import annotations

import asyncio
import logging
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

from ..config import RunConfig
from ..pngutil import write_png
from .base import (Action, ActionResult, Browser, BrowserSnapshot, Element, Observation, RestoreReport, TabInfo,
                   TabState, same_url)
from .recorder import Recorder, ReplayIndex, ReplayStats

log = logging.getLogger(__name__)

OBSERVE_JS = r"""
(opts) => {
  const maxElements = opts.maxElements, textChars = opts.textChars;
  const sel = 'a[href], button, input, select, textarea, summary, [role="button"], [role="link"], [role="tab"], [role="menuitem"], [role="option"], [role="checkbox"], [role="radio"], [role="combobox"], [role="textbox"], [onclick], [contenteditable="true"]';
  document.querySelectorAll('[data-aupower-idx]').forEach(e => e.removeAttribute('data-aupower-idx'));
  const out = []; let idx = 0;
  const vh = window.innerHeight || 800;
  for (const el of document.querySelectorAll(sel)) {
    if (idx >= maxElements) break;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const cs = window.getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || cs.opacity === '0') continue;
    if (r.bottom < -vh || r.top > 3 * vh) continue;
    el.setAttribute('data-aupower-idx', String(idx));
    const label = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('alt') || '').trim().replace(/\s+/g, ' ').slice(0, 80);
    out.push({index: idx, tag: el.tagName.toLowerCase(), text: label, role: el.getAttribute('role') || '',
              type: el.getAttribute('type') || '', href: el.tagName === 'A' ? (el.getAttribute('href') || '').slice(0, 120) : '',
              name: el.getAttribute('name') || '', in_viewport: r.top < vh && r.bottom > 0});
    idx++;
  }
  const text = ((document.body ? document.body.innerText : '') || '').replace(/\n{3,}/g, '\n\n');
  return {elements: out, text: text.slice(0, textChars), scrollY: Math.round(window.scrollY),
          docHeight: Math.round(document.documentElement.scrollHeight), viewportHeight: vh};
}
"""

SNAPSHOT_JS = r"""
() => {
  const forms = [];
  document.querySelectorAll('input, textarea, select').forEach((el, i) => {
    const type = (el.type || '').toLowerCase();
    if (type === 'hidden' || type === 'submit' || type === 'button' || type === 'file') return;
    const checked = (type === 'checkbox' || type === 'radio') ? !!el.checked : false;
    if (el.value || checked) forms.push({i: i, name: el.name || '', id: el.id || '', type: type, value: el.value || '', checked: checked});
  });
  const session = {};
  try { for (let i = 0; i < sessionStorage.length; i++) { const k = sessionStorage.key(i); session[k] = sessionStorage.getItem(k); } } catch (e) {}
  return {scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY), forms: forms, session: session};
}
"""

RESTORE_JS = r"""
(state) => {
  const els = Array.from(document.querySelectorAll('input, textarea, select'));
  let applied = 0;
  for (const f of (state.forms || [])) {
    let el = null;
    if (f.id) el = document.getElementById(f.id);
    if (!el && f.name) el = els.find(e => e.name === f.name && (e.type || '').toLowerCase() === f.type);
    if (!el && f.i < els.length) el = els[f.i];
    if (!el) continue;
    const type = (el.type || '').toLowerCase();
    if (type === 'checkbox' || type === 'radio') { el.checked = !!f.checked; }
    else if (type !== 'file') { el.value = f.value; }
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    applied++;
  }
  window.scrollTo(state.scrollX || 0, state.scrollY || 0);
  return applied;
}
"""

_DROP_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"}


class _Runtime:
    """One Chromium process shared by every PlaywrightBrowser in the process."""
    pw = None
    browser = None
    refs = 0
    lock: asyncio.Lock | None = None

    @classmethod
    async def acquire(cls, headless: bool):
        if cls.lock is None:
            cls.lock = asyncio.Lock()
        async with cls.lock:
            if cls.browser is None:
                from playwright.async_api import async_playwright
                cls.pw = await async_playwright().start()
                cls.browser = await cls.pw.chromium.launch(headless=headless)
            cls.refs += 1
            return cls.browser

    @classmethod
    async def release(cls):
        if cls.lock is None:
            return
        async with cls.lock:
            cls.refs -= 1
            if cls.refs <= 0 and cls.browser is not None:
                try:
                    await asyncio.wait_for(cls.browser.close(), timeout=30)
                except Exception:
                    pass
                finally:
                    cls.browser = None
                    if cls.pw is not None:
                        await cls.pw.stop()
                        cls.pw = None
                cls.refs = 0
        if cls.refs <= 0 and cls.browser is None:
            cls.lock = None   # a later asyncio.run() gets a fresh lock bound to its own loop


def _post_bytes(request) -> bytes | None:
    """The raw POST body of a request, or None. Playwright's `request.post_data` decodes the body as UTF-8 and raises
    on a binary body (gzip-compressed telemetry posts, for instance; seen 2026-11-06), so the recording and the replay
    key on the raw bytes instead."""
    try:
        return request.post_data_buffer
    except Exception:
        return None


class PlaywrightBrowser(Browser):
    def __init__(self, run_cfg: RunConfig):
        self.cfg = run_cfg
        self.context = None
        self.active = 0
        self.recorder: Recorder | None = None
        self.replay: ReplayIndex | None = None
        self._stats = ReplayStats()
        self.downloads_dir: Path | None = None
        self._pending: set[asyncio.Task] = set()
        self.screenshot_ext = ".jpg" if run_cfg.screenshot_format == "jpeg" else ".png"

    # ---- lifecycle ----
    async def start(self, *, start_url: str | None = None, storage_state: dict | None = None,
                    downloads_dir: Path | None = None, replay: ReplayIndex | None = None,
                    recorder: Recorder | None = None) -> None:
        browser = await _Runtime.acquire(self.cfg.headless)
        kwargs: dict = {"viewport": {"width": int(self.cfg.viewport[0]), "height": int(self.cfg.viewport[1])},
                        "accept_downloads": True, "ignore_https_errors": True}
        if storage_state:
            kwargs["storage_state"] = storage_state
        self.context = await browser.new_context(**kwargs)
        timeout_ms = int(self.cfg.action_timeout_s * 1000)
        self.context.set_default_timeout(timeout_ms)
        self.context.set_default_navigation_timeout(timeout_ms)
        self.downloads_dir = downloads_dir
        if downloads_dir is not None:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder
        self.replay = replay
        if replay is not None:
            self._stats = replay.stats
            await self.context.route("**/*", self._replay_route)
        if recorder is not None:
            self.context.on("response", self._on_response)
        self.context.on("page", self._attach_page)
        if start_url:
            page = await self.context.new_page()
            await self._goto(page, start_url)
            self.active = 0

    async def close(self) -> None:
        # Teardown is bounded. A page can hold a response open indefinitely (telemetry long-polls, event streams),
        # so a pending ``_record`` task's ``response.body()`` may never resolve; waiting on it without a limit hung
        # the first live trajectory rollout for hours after its final step (2026-11-05). In-flight recordings and
        # downloads get one action timeout to finish, the rest are cancelled, and closing the context is bounded too.
        grace = max(1.0, float(self.cfg.action_timeout_s))
        await self._settle_pending(grace)
        if self.context is not None:
            try:
                await asyncio.wait_for(self.context.close(), timeout=grace)
            except Exception:
                pass
            self.context = None
        await self._settle_pending(1.0)   # tasks started while the context was closing fail fast now that it is closed
        if self.recorder is not None:
            self.recorder.close()
        await _Runtime.release()

    async def recover_active_page(self) -> None:
        """Replace the active page when it stops answering: a new tab at the same URL, then the old one closed (each
        step bounded, errors ignored), so a hung renderer costs the rollout one step rather than everything after it."""
        if self.context is None:
            return
        old = self.page
        url = old.url
        new = await self.context.new_page()
        try:
            await self._goto(new, url)
        except Exception:
            pass
        try:
            await asyncio.wait_for(old.close(), timeout=10)
        except Exception:
            pass
        pages = self.pages
        self.active = pages.index(new) if new in pages else max(0, len(pages) - 1)

    async def _settle_pending(self, timeout: float) -> None:
        """Wait up to ``timeout`` seconds for the pending recording/download tasks, then cancel what is left."""
        if not self._pending:
            return
        _, pending = await asyncio.wait(set(self._pending), timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ---- helpers ----
    @property
    def pages(self) -> list:
        return list(self.context.pages) if self.context is not None else []

    @property
    def page(self):
        pages = self.pages
        self.active = max(0, min(self.active, len(pages) - 1))
        return pages[self.active]

    def _attach_page(self, page) -> None:
        if self.downloads_dir is not None:
            def on_download(d):
                task = asyncio.ensure_future(d.save_as(self.downloads_dir / d.suggested_filename))
                self._pending.add(task)
                task.add_done_callback(self._pending.discard)
            page.on("download", on_download)

    def _on_response(self, response) -> None:
        task = asyncio.ensure_future(self._record(response))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _record(self, response) -> None:
        if self.recorder is None:
            return
        req = response.request
        try:
            body = await response.body()
        except Exception:
            body = None
        try:
            headers = await response.all_headers()
        except Exception:
            headers = dict(response.headers)
        try:
            self.recorder.add(method=req.method, url=response.url, status=response.status, headers=headers, body=body,
                              post_data=_post_bytes(req), mime=headers.get("content-type", ""))
        except Exception:
            pass

    async def _replay_route(self, route) -> None:
        # A route handler that raises leaves its request pending for ever (Playwright neither continues nor fulfils it),
        # so every path out of here ends in fulfill or continue.
        req = route.request
        try:
            entry = self.replay.take(req.method, req.url, _post_bytes(req)) if self.replay is not None else None
            if entry is not None and entry.get("body_b64") is not None:
                headers = {k: v for k, v in entry.get("headers", {}).items() if k.lower() not in _DROP_HEADERS}
                try:
                    await route.fulfill(status=int(entry["status"]), headers=headers, body=ReplayIndex.body_bytes(entry))
                    self._stats.log_served(req.url)
                    return
                except Exception:
                    pass
        except Exception as exc:
            log.warning("replay lookup failed for %s %s (%s: %s); fetching it live", req.method, req.url[:160], type(exc).__name__, exc)
        self._stats.log_live(req.url)
        try:
            await route.continue_()
        except Exception:
            pass

    async def _goto(self, page, url: str) -> None:
        try:
            await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # timeouts: keep whatever loaded
            if "Timeout" not in type(exc).__name__:
                raise

    async def _settle(self, page) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        await asyncio.sleep(0.3)

    async def _title(self, page) -> str:
        try:
            return await page.title()
        except Exception:
            return ""

    # ---- interface ----
    async def observe(self, screenshot_path: Path | None = None) -> Observation:
        page = self.page
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        try:
            data = await page.evaluate(OBSERVE_JS, {"maxElements": int(self.cfg.max_elements),
                                                    "textChars": int(self.cfg.page_text_chars)})
        except Exception as exc:
            data = {"elements": [], "text": f"(page could not be read: {type(exc).__name__})", "scrollY": 0,
                    "docHeight": 0, "viewportHeight": int(self.cfg.viewport[1])}
        shot = None
        if screenshot_path is not None:
            try:
                if str(screenshot_path).lower().endswith((".jpg", ".jpeg")):
                    await page.screenshot(path=str(screenshot_path), timeout=15000, type="jpeg",
                                          quality=int(self.cfg.screenshot_quality))
                else:
                    await page.screenshot(path=str(screenshot_path), timeout=15000)
            except Exception:
                write_png(screenshot_path, 64, 40, (200, 200, 200))
            shot = Path(screenshot_path).name
        tabs = []
        for i, p in enumerate(self.pages):
            tabs.append(TabInfo(i, p.url, await self._title(p), i == self.active))
        elements = [Element(index=int(e["index"]), tag=e["tag"], text=e.get("text", ""), role=e.get("role", ""),
                            type=e.get("type", ""), href=e.get("href", ""), name=e.get("name", ""),
                            in_viewport=bool(e.get("in_viewport", True))) for e in data.get("elements", [])]
        return Observation(url=page.url, title=await self._title(page), tabs=tabs, elements=elements,
                           text=data.get("text", ""), screenshot=shot, scroll_y=int(data.get("scrollY", 0)),
                           doc_height=int(data.get("docHeight", 0)), viewport_height=int(data.get("viewportHeight", 0)))

    async def act(self, action: Action) -> ActionResult:
        a = action.args
        n = action.name
        page = self.page
        try:
            if n == "navigate":
                await self._goto(page, a["url"])
                return ActionResult(True, f"Navigated to {page.url}")
            if n in ("click", "type"):
                loc = page.locator(f'[data-aupower-idx="{a["index"]}"]')
                if await loc.count() == 0:
                    return ActionResult(False, f"No element with index {a['index']} on the current page.")
                loc = loc.first
                if n == "click":
                    await loc.click()
                    await self._settle(page)
                    return ActionResult(True, f"Clicked element {a['index']}; now at {page.url}")
                await loc.fill(a["text"])
                if a.get("press_enter"):
                    await loc.press("Enter")
                    await self._settle(page)
                return ActionResult(True, f"Typed into element {a['index']}" + (f"; now at {page.url}" if a.get("press_enter") else ""))
            if n == "scroll":
                amount = int(a.get("amount") or int(self.cfg.viewport[1] * 0.8))
                dy = amount if a["direction"] == "down" else -amount
                y = await page.evaluate("(dy) => { window.scrollBy(0, dy); return Math.round(window.scrollY); }", dy)
                return ActionResult(True, f"Scrolled {a['direction']} to {y}px")
            if n == "back":
                r = await page.go_back(wait_until="domcontentloaded")
                return ActionResult(r is not None, f"Went back to {page.url}" if r is not None else "No previous page in this tab.")
            if n == "forward":
                r = await page.go_forward(wait_until="domcontentloaded")
                return ActionResult(r is not None, f"Went forward to {page.url}" if r is not None else "No forward page in this tab.")
            if n == "new_tab":
                p = await self.context.new_page()
                self.active = len(self.pages) - 1
                if a.get("url"):
                    await self._goto(p, a["url"])
                return ActionResult(True, f"Opened tab {self.active} at {p.url}")
            if n == "switch_tab":
                idx = a["index"]
                if not 0 <= idx < len(self.pages):
                    return ActionResult(False, f"No tab with index {idx}.")
                self.active = idx
                await self.page.bring_to_front()
                return ActionResult(True, f"Switched to tab {idx} ({self.page.url})")
            if n == "close_tab":
                idx = a.get("index", self.active)
                pages = self.pages
                if not 0 <= idx < len(pages):
                    return ActionResult(False, f"No tab with index {idx}.")
                await pages[idx].close()
                if not self.pages:
                    await self.context.new_page()
                self.active = max(0, min(self.active, len(self.pages) - 1))
                return ActionResult(True, f"Closed tab {idx}")
            if n == "press":
                await page.keyboard.press(a["key"])
                await self._settle(page)
                return ActionResult(True, f"Pressed {a['key']}")
            if n == "wait":
                await asyncio.sleep(float(a.get("seconds", 1)))
                return ActionResult(True, "Waited.")
            if n == "done":
                return ActionResult(True, "Task finished.", final=True, answer=a.get("answer", ""))
            return ActionResult(False, f"Unsupported action {n}")
        except Exception as exc:
            kind = type(exc).__name__
            return ActionResult(False, (f"Timed out: {exc}" if "Timeout" in kind else f"Action failed ({kind}): {exc}")[:400])

    async def snapshot(self) -> BrowserSnapshot:
        storage = await self.context.storage_state()
        tabs = []
        for p in self.pages:
            try:
                st = await p.evaluate(SNAPSHOT_JS)
            except Exception:
                st = {}
            tabs.append(TabState(url=p.url, title=await self._title(p), scroll_x=int(st.get("scrollX", 0)),
                                 scroll_y=int(st.get("scrollY", 0)), form_values=list(st.get("forms", [])),
                                 session_storage=dict(st.get("session", {}))))
        downloads = sorted(f.name for f in self.downloads_dir.iterdir()) if self.downloads_dir and self.downloads_dir.exists() else []
        return BrowserSnapshot(tabs=tabs, active_tab=self.active, storage_state=storage, downloads=downloads)

    async def restore(self, snap: BrowserSnapshot) -> RestoreReport:
        errors: list[str] = []
        for i, tab in enumerate(snap.tabs):
            page = await self.context.new_page()
            if tab.session_storage:
                parsed = urlparse(tab.url)
                origin = f"{parsed.scheme}://{parsed.netloc}"
                await page.add_init_script(
                    "(() => { if (location.origin === " + json.dumps(origin) + ") { const d = "
                    + json.dumps(tab.session_storage) + "; for (const k in d) { try { sessionStorage.setItem(k, d[k]); } catch (e) {} } } })();")
            try:
                await self._goto(page, tab.url)
            except Exception as exc:
                errors.append(f"tab {i} ({tab.url}): {type(exc).__name__}: {exc}"[:300])
                continue
            try:
                await page.evaluate(RESTORE_JS, {"scrollX": tab.scroll_x, "scrollY": tab.scroll_y, "forms": tab.form_values})
            except Exception as exc:
                errors.append(f"tab {i} in-page state: {type(exc).__name__}: {exc}"[:300])
        if not self.pages:
            await self.context.new_page()
        self.active = max(0, min(snap.active_tab, len(self.pages) - 1))
        try:
            await self.page.bring_to_front()
        except Exception:
            pass
        expected = snap.tabs[snap.active_tab].url if snap.tabs else "about:blank"
        actual = self.page.url
        return RestoreReport(active_tab_url_matches=same_url(actual, expected), expected_url=expected,
                             actual_url=actual, tabs_restored=len(snap.tabs), errors=errors)

    def replay_stats(self) -> dict:
        return self._stats.to_dict() if self.replay is not None else {}
