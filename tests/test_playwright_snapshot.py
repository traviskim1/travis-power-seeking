"""Integration test of the real browser layer: record a session against a local site, snapshot it, stop the
server, then restore the snapshot into a new browser that can only load pages from the recording."""
import asyncio
import http.server
import json
import threading

import pytest

pytest.importorskip("playwright")

from aupower.browser.base import normalize_action  # noqa: E402
from aupower.browser.recorder import Recorder, ReplayIndex  # noqa: E402

SITE = {
    "/": ("<html><head><title>Home</title></head><body><h1>Home</h1>"
          "<a href='/page2'>Page two</a>"
          "<form><input id='q' name='q' type='text' placeholder='query'>"
          "<label><input type='checkbox' name='agree'> agree</label><button type='button'>Go</button></form>"
          "<div style='height:3000px'>tall</div>"
          "<script>localStorage.setItem('seen','1');"
          "if(!sessionStorage.getItem('ss')) sessionStorage.setItem('ss','set-by-page-'+Date.now()+'-'+Math.random());</script>"
          "</body></html>"),
    "/page2": ("<html><head><title>Page Two</title></head><body><h1>Two</h1><a href='/'>home</a>"
               "<textarea name='notes' placeholder='notes'></textarea></body></html>"),
}


# a binary (gzip magic bytes) and a text POST from the page; the server echoes length:first-four-bytes-hex
POST_SCRIPT = ("async () => { const r = await fetch('/echo', {method: 'POST', body: new Uint8Array([0x1f, 0x8b, 0x08, 0x00, 0xff, 0xfe])});"
               " const t = await fetch('/echo', {method: 'POST', body: 'plain text'}); return [await r.text(), await t.text()]; }")
EXPECTED_POSTS = ["6:1f8b0800", "10:706c6169"]


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        body = SITE.get(path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"not found")
            return
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if path == "/":
            self.send_header("Set-Cookie", "sid=42; Path=/")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(n)
        out = f"{len(data)}:{data[:4].hex()}".encode("ascii")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class _Server:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def site():
    s = _Server()
    yield s
    s.stop()


def test_playwright_snapshot_restore_and_replay(site, tmp_path, dry_cfg):
    cfg = dry_cfg(run={"browser": "playwright", "headless": True, "action_timeout_s": 15})
    from aupower.browser.playwright_browser import PlaywrightBrowser

    async def record():
        rec = Recorder(tmp_path / "rec.jsonl")
        b = PlaywrightBrowser(cfg.run)
        try:
            await b.start(start_url=site.url + "/", downloads_dir=tmp_path / "dl", recorder=rec)
        except Exception as exc:
            pytest.skip(f"Chromium not available: {exc}")
        obs = await b.observe(tmp_path / "o.png")
        assert obs.title == "Home" and (tmp_path / "o.png").exists()
        assert await b.page.evaluate(POST_SCRIPT) == EXPECTED_POSTS      # both POST bodies get recorded
        assert b.screenshot_ext == ".jpg"
        await b.observe(tmp_path / "o2.jpg")
        assert (tmp_path / "o2.jpg").read_bytes()[:2] == b"\xff\xd8" and (tmp_path / "o.png").read_bytes()[:4] == b"\x89PNG"
        q = next(e.index for e in obs.elements if e.name == "q")
        assert (await b.act(normalize_action("type", {"index": q, "text": "hello"}))).ok
        assert (await b.act(normalize_action("scroll", {"direction": "down", "amount": 500}))).ok
        assert (await b.act(normalize_action("new_tab", {"url": site.url + "/page2"}))).ok
        obs2 = await b.observe(None)
        assert obs2.title == "Page Two"
        ta = next(e.index for e in obs2.elements if e.tag == "textarea")
        assert (await b.act(normalize_action("type", {"index": ta, "text": "note text"}))).ok
        assert (await b.act(normalize_action("switch_tab", {"index": 0}))).ok
        snap = await b.snapshot()
        await b.close()
        return snap, rec

    snap, rec = asyncio.run(record())
    assert [t.url for t in snap.tabs] == [site.url + "/", site.url + "/page2"] and snap.active_tab == 0
    assert snap.tabs[0].scroll_y == 500
    assert any(f["name"] == "q" and f["value"] == "hello" for f in snap.tabs[0].form_values)
    assert any(f["name"] == "notes" and f["value"] == "note text" for f in snap.tabs[1].form_values)
    ss_value = snap.tabs[0].session_storage["ss"]
    assert ss_value.startswith("set-by-page-")
    assert any(c["name"] == "sid" and c["value"] == "42" for c in snap.storage_state["cookies"])
    assert any(kv["name"] == "seen" for o in snap.storage_state["origins"] for kv in o["localStorage"])
    assert rec.count >= 2
    rows = [json.loads(l) for l in rec.path.read_text(encoding="utf-8").splitlines() if l.strip()]
    posts = [r for r in rows if r["method"] == "POST" and r["url"].endswith("/echo")]
    assert len(posts) == 2 and sum(1 for r in posts if r.get("post_b64")) == 1 and any(r.get("post") == "plain text" for r in posts)

    site.stop()   # from here on, only the recording can serve the pages

    async def restore():
        replay = ReplayIndex.load(rec.path, upto=rec.count)
        b = PlaywrightBrowser(cfg.run)
        await b.start(storage_state=snap.storage_state, downloads_dir=tmp_path / "dl2", replay=replay)
        rep = await b.restore(snap)
        obs = await b.observe(tmp_path / "r.png")
        posted = await b.page.evaluate(POST_SCRIPT)          # served from the recording: the site is down
        snap2 = await b.snapshot()
        stats = b.replay_stats()
        await b.close()
        return rep, obs, snap2, stats, posted

    rep, obs, snap2, stats, posted = asyncio.run(restore())
    assert posted == EXPECTED_POSTS and not any(u.endswith("/echo") for u in stats["live_urls"])
    assert rep.active_tab_url_matches and rep.tabs_restored == 2 and not rep.errors
    assert obs.title == "Home" and obs.url == site.url + "/"
    assert [t.url for t in snap2.tabs] == [t.url for t in snap.tabs] and snap2.active_tab == 0
    assert snap2.tabs[0].scroll_y == 500
    assert any(f["name"] == "q" and f["value"] == "hello" for f in snap2.tabs[0].form_values)
    assert any(f["name"] == "notes" and f["value"] == "note text" for f in snap2.tabs[1].form_values)
    assert snap2.tabs[0].session_storage["ss"] == ss_value          # restored before the page script ran
    assert any(c["name"] == "sid" for c in snap2.storage_state["cookies"])
    assert stats["served_from_recording"] >= 2
    assert stats["live_fallback_fraction"] is None or stats["live_fallback_fraction"] < 0.5
