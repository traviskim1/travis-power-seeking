"""Snapshot and restore of all three layers with the mock browser (SPEC.md section 4)."""
import asyncio
import base64
import json

from aupower.agent import AgentState, user_turn_text
from aupower.browser.base import normalize_action
from aupower.browser.mock import MockBrowser
from aupower.browser.recorder import Recorder, ReplayIndex
from aupower.prompts import first_task_message
from aupower.snapshots import load_snapshot, save_snapshot


async def _drive(tmp_path, cfg):
    rec = Recorder(tmp_path / "rec.jsonl")
    b = MockBrowser(cfg.run, seed=1)
    await b.start(start_url="https://www.google.com", downloads_dir=tmp_path / "dl", recorder=rec)
    obs0 = await b.observe(tmp_path / "initial.png")
    assert obs0.screenshot == "initial.png" and (tmp_path / "initial.png").read_bytes()[:4] == b"\x89PNG"
    assert (await b.act(normalize_action("type", {"index": 0, "text": "ponies cast"}))).ok
    assert (await b.act(normalize_action("click", {"index": 2}))).ok          # a link: navigates
    assert (await b.act(normalize_action("scroll", {"index": 0, "direction": "down"}))).ok
    assert (await b.act(normalize_action("new_tab", {"url": "https://www.wikimock.org/article-3"}))).ok
    assert (await b.act(normalize_action("type", {"index": 0, "text": "draft text"}))).ok
    assert (await b.act(normalize_action("switch_tab", {"index": 0}))).ok
    snap = await b.snapshot()
    agent = AgentState.fresh("m", tmp_path)
    agent.append_user(user_turn_text(first_task_message("g"), obs0.render(), None), "initial.png")
    agent.append_assistant("a1")
    obs = await b.observe(tmp_path / "step_001.png")
    agent.set_pending(obs, "ok")
    ss = save_snapshot(tmp_path / "snap", step=1, agent=agent, browser_snap=snap, downloads_src=tmp_path / "dl",
                       recording_path=rec.path, recording_upto=rec.count, image_root=tmp_path)
    await b.close()
    return snap, ss, rec


def test_mock_snapshot_restore_round_trip(dry_cfg, tmp_path):
    cfg = dry_cfg()
    snap, ss, rec = asyncio.run(_drive(tmp_path, cfg))
    assert len(snap.tabs) == 2 and snap.active_tab == 0
    assert snap.tabs[0].url.startswith("https://www.shopmock.com/") and snap.tabs[0].scroll_y > 0
    assert snap.tabs[1].form_values[0]["value"] == "draft text"
    assert snap.storage_state["cookies"] and snap.storage_state["origins"]
    assert rec.count == 3                                   # google, shopmock, wikimock

    loaded = load_snapshot(ss.dir)
    assert loaded.step == 1 and loaded.recording_upto == 3
    bsnap = loaded.load_browser()
    assert bsnap.to_dict() == snap.to_dict()

    async def _restore():
        replay = ReplayIndex.load(loaded.recording_path, upto=loaded.recording_upto)
        assert len(replay) == 3
        b2 = MockBrowser(cfg.run, seed=99)
        await b2.start(storage_state=bsnap.storage_state, downloads_dir=tmp_path / "dl2", replay=replay)
        rep = await b2.restore(bsnap)
        snap2 = await b2.snapshot()
        stats = b2.replay_stats()
        # the fork keeps acting: a page outside the recording is a logged live fallback
        await b2.act(normalize_action("navigate", {"url": "https://www.newsmock.com/fresh-1"}))
        stats_after = b2.replay_stats()
        await b2.close()
        return rep, snap2, stats, stats_after

    rep, snap2, stats, stats_after = asyncio.run(_restore())
    assert rep.active_tab_url_matches and rep.tabs_restored == 2 and rep.actual_url == snap.tabs[0].url
    assert [t.url for t in snap2.tabs] == [t.url for t in snap.tabs]
    assert snap2.active_tab == 0
    assert snap2.tabs[0].scroll_y == snap.tabs[0].scroll_y
    assert snap2.tabs[1].form_values == snap.tabs[1].form_values
    assert snap2.tabs[0].session_storage == snap.tabs[0].session_storage
    assert snap2.storage_state == snap.storage_state
    assert stats["served_from_recording"] == 2 and stats["live_fallback"] == 0
    assert stats_after["live_fallback"] == 1 and stats_after["live_urls"] == ["https://www.newsmock.com/fresh-1"]
    assert stats_after["live_fallback_fraction"] == 1 / 3

    # agent layer: the frozen history ends with the last action; the observation after it is pending;
    # screenshot references carry the trajectory directory so a fork can resolve them
    agent = loaded.load_agent()
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.pending["screenshot"] == "step_001.png" and agent.pending["result"] == "ok"
    assert agent.messages[1]["content"][1]["root"] == str(tmp_path.resolve())
    assert (ss.dir / "downloads").is_dir()


def test_mock_browser_tab_and_history_actions(dry_cfg, tmp_path):
    cfg = dry_cfg()

    async def go():
        b = MockBrowser(cfg.run, seed=3)
        await b.start(start_url="https://www.google.com")
        assert (await b.act(normalize_action("navigate", {"url": "https://travelmock.net/flights-1"}))).ok
        assert (await b.act(normalize_action("back", {}))).ok and b.tab.url == "https://www.google.com"
        assert (await b.act(normalize_action("forward", {}))).ok and b.tab.url == "https://travelmock.net/flights-1"
        assert not (await b.act(normalize_action("switch_tab", {"index": 5}))).ok
        assert not (await b.act(normalize_action("click", {"index": 99}))).ok
        r = await b.act(normalize_action("done", {"answer": "42"}))
        assert r.final and r.answer == "42"
        obs = await b.observe(None)
        assert "[0] <input>" in obs.render() and "Active tab URL: https://travelmock.net/flights-1" in obs.render()
        await b.close()

    asyncio.run(go())


def test_recorder_keys_binary_post_bodies(tmp_path):
    """Binary POST bodies (gzip-compressed telemetry, for instance) are recorded and replayed keyed on their raw bytes;
    text bodies keep the historical `post` field, so earlier recordings still load."""
    rec = Recorder(tmp_path / "rec.jsonl")
    gz = b"\x1f\x8b\x08\x00" + bytes(range(256))
    rec.add(method="POST", url="https://t.example/beacon", status=204, headers={}, body=b"", post_data=gz)
    rec.add(method="POST", url="https://t.example/json", status=200, headers={}, body=b"{}", post_data=b'{"a": 1}')
    rec.add(method="POST", url="https://t.example/json", status=200, headers={}, body=b"[]", post_data='{"a": 1}')
    rec.add(method="GET", url="https://t.example/", status=200, headers={}, body=b"<html>", post_data=None)
    rec.close()
    rows = [json.loads(l) for l in (tmp_path / "rec.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["post"] is None and base64.b64decode(rows[0]["post_b64"]) == gz
    assert rows[1]["post"] == '{"a": 1}' and "post_b64" not in rows[1] and rows[3]["post"] is None
    idx = ReplayIndex.load(rec.path)
    assert idx.take("POST", "https://t.example/beacon", gz)["status"] == 204
    assert idx.take("POST", "https://t.example/beacon", b"other") is None
    assert ReplayIndex.body_bytes(idx.take("POST", "https://t.example/json", b'{"a": 1}')) == b"{}"
    assert ReplayIndex.body_bytes(idx.take("POST", "https://t.example/json", '{"a": 1}')) == b"[]"   # str and bytes key alike
    assert idx.take("GET", "https://t.example/", None)["status"] == 200
