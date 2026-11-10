import json
from pathlib import Path

import pytest

from aupower.agent import ActionParseError, AgentState, build_api_messages, parse_action, user_turn_text
from aupower.browser.base import normalize_action
from aupower.pngutil import write_png


def test_parse_action_from_fenced_json():
    text = 'I will click.\n\n```json\n{"thought": "see button", "action": "click", "args": {"index": 3}}\n```'
    a = parse_action(text)
    assert a.name == "click" and a.args == {"index": 3} and a.thought == "see button"
    assert a.line() == 'click {"index": 3}'


def test_parse_action_from_bare_json_and_last_object_wins():
    text = 'Options: {"action": "scroll", "args": {"direction": "up"}} ... final: {"action": "navigate", "args": {"url": "example.com"}}'
    a = parse_action(text)
    assert a.name == "navigate" and a.args["url"] == "https://example.com"


def test_parse_action_errors():
    with pytest.raises(ActionParseError):
        parse_action("no json here")
    with pytest.raises(ActionParseError):
        parse_action('{"action": "fly", "args": {}}')
    with pytest.raises(ActionParseError):
        parse_action('{"action": "click", "args": {"index": "three"}}')


def test_normalize_action():
    assert normalize_action("type", {"index": "2", "text": "hi", "press_enter": 1}).args == {"index": 2, "text": "hi", "press_enter": True}
    assert normalize_action("wait", {"seconds": 99}).args == {"seconds": 10.0}
    assert normalize_action("done", {"answer": "x"}).args == {"answer": "x"}
    assert normalize_action("back", None).args == {}
    with pytest.raises(ValueError):
        normalize_action("scroll", {"direction": "left"})


def test_user_turn_text_layout():
    t = user_turn_text("Do this task:\n\nX", "OBS", "Clicked")
    assert t.startswith("Do this task:\n\nX\n\nResult of your last action:\nClicked\n\nCurrent browser state:\nOBS")
    assert user_turn_text(None, "OBS", None) == "Current browser state:\nOBS"


def test_build_api_messages_keeps_only_recent_images(tmp_path):
    for i in range(4):
        write_png(tmp_path / f"s{i}.png", 4, 4, (i, i, i))
    st = AgentState.fresh("m", tmp_path)
    for i in range(4):
        st.append_user(f"obs {i}", f"s{i}.png")
        st.append_assistant(f"act {i}")
    msgs = build_api_messages(st, history_screenshots=2, supports_images=True)
    images = [p for m in msgs if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]
    omitted = [p for m in msgs if isinstance(m["content"], list) for p in m["content"] if p["type"] == "text" and "omitted" in p["text"]]
    assert len(images) == 2 and len(omitted) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[0] == {"role": "system", "content": st.messages[0]["content"]}
    # text-only models get no images at all
    msgs = build_api_messages(st, history_screenshots=2, supports_images=False)
    assert not [p for m in msgs if isinstance(m["content"], list) for p in m["content"] if p["type"] == "image_url"]


def test_build_api_messages_honours_per_image_root(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    write_png(other / "a.png", 2, 2, (1, 2, 3))
    st = AgentState.fresh("m", tmp_path)
    st.messages.append({"role": "user", "content": [{"type": "text", "text": "x"},
                                                    {"type": "image_file", "path": "a.png", "root": str(other)}]})
    msgs = build_api_messages(st, history_screenshots=1, supports_images=True)
    assert msgs[-1]["content"][1]["type"] == "image_url"


def test_agent_state_round_trip(tmp_path):
    st = AgentState.fresh("m", tmp_path)
    st.append_user("task", "initial.png")
    st.append_assistant("act")
    st.step = 3
    st.scratchpad["notes"] = "n"
    st.save(tmp_path / "agent.json")
    back = AgentState.load(tmp_path / "agent.json")
    assert back.to_dict() == st.to_dict()
    assert back.n_screenshots() == 1
