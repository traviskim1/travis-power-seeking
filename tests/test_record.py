"""The records: the API call log (every request and response) and the invocation log."""
import gzip
import json

import pytest

from aupower.record import API_LOG_NAME, INVOCATIONS_NAME, CallLog, log_invocation, read_calls, redact_images


def test_redact_images_keeps_text_and_replaces_data_urls():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": [{"type": "text", "text": "look"},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                                         {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
    out = redact_images(msgs)
    assert out[0] == msgs[0] and out[1]["content"][0] == {"type": "text", "text": "look"}
    assert out[1]["content"][1]["image_url"]["url"].startswith("<data:image/jpeg;base64 omitted: 4 chars, sha256 ")
    assert out[1]["content"][2]["image_url"]["url"] == "https://x/y.png"
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:")       # the original is untouched


def test_call_log_round_trip(tmp_path):
    log = CallLog(tmp_path / API_LOG_NAME)
    log.write(kind="chat", purpose="agent_step", meta={"rollout_id": "r", "step": 1}, model="m", endpoint="p/fp8", attempt=1,
              request={"url": "u", "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}},
              status=200, latency_s=0.5, response={"id": "gen-1", "choices": [{"message": {"content": "OK"}}]}, request_id="gen-1")
    log.write(kind="generation", attempt=1, request={"url": "g"}, status=200, latency_s=0.1, response={"data": {"provider_name": "P"}})
    log.write(kind="chat", purpose="judge_rescore", meta={}, model="j", endpoint="q", attempt=2, request={"url": "u", "body": {}},
              status=429, latency_s=1.0, response={"error": "slow down"}, error="HTTP 429")
    assert log.n == 3
    with gzip.open(tmp_path / API_LOG_NAME, "rt") as f:
        lines = [json.loads(l) for l in f]
    assert [e["kind"] for e in lines] == ["chat", "generation", "chat"] and lines[0]["request"]["body"]["messages"][0]["content"] == "hi"
    assert lines[0]["response"]["choices"][0]["message"]["content"] == "OK" and lines[2]["status"] == 429 and lines[2]["error"] == "HTTP 429"
    assert [e["purpose"] for e in read_calls(tmp_path / API_LOG_NAME, kind="chat")] == ["agent_step", "judge_rescore"]
    assert read_calls(tmp_path / API_LOG_NAME, purpose="judge_rescore")[0]["attempt"] == 2
    assert len(read_calls(tmp_path / API_LOG_NAME, limit=1)) == 1 and read_calls(tmp_path / "missing.gz") == []


async def test_openrouter_client_logs_every_attempt(tmp_path, dry_cfg):
    import httpx
    from aupower.client import Ledger, ModelCallError, OpenRouterClient
    cfg = dry_cfg(run={"mode": "live"})
    agent = cfg.models.agents[0]
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json={"id": "gen-7", "model": agent.id, "provider": "DeepInfra",
                                         "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 1, "cost": 0.000001}})

    log_path = tmp_path / API_LOG_NAME
    client = OpenRouterClient("k", Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices), routing=cfg.routing,
                              models={agent.id: agent}, transport=httpx.MockTransport(handler), api_log=log_path)
    try:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]}]
        resp = await client.chat(model=agent.id, messages=msgs, max_tokens=4, temperature=1.0, purpose="agent_step",
                                 meta={"rollout_id": "r", "step": 1})
        assert resp.text == "OK" and resp.served_provider == "DeepInfra"
        entries = read_calls(log_path)
        assert [(e["attempt"], e["status"]) for e in entries] == [(1, 429), (2, 200)]
        first, second = entries
        assert first["error"].startswith("HTTP 429") and first["response"] == {"error": {"message": "rate limited"}}
        body = second["request"]["body"]
        assert body["model"] == agent.id and body["max_tokens"] == 4 and body["temperature"] == 1.0
        assert body["provider"]["only"] == [agent.endpoint] and body["messages"][0]["content"][0]["text"] == "hi"
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("<data:image/jpeg;base64 omitted: 4 chars")
        assert second["response"]["provider"] == "DeepInfra" and second["request_id"] == "gen-7" and second["purpose"] == "agent_step"
        assert second["meta"] == {"rollout_id": "r", "step": 1} and second["endpoint"] == agent.endpoint
    finally:
        await client.close()


def test_invocation_log_records_command_and_config(dry_cfg):
    cfg = dry_cfg()
    entry = log_invocation(cfg, ["run", "--profile", "initial", "--trajectory", "1"], stages=["plan", "baseline"], trajectories=["1"])
    lines = [json.loads(l) for l in (cfg.run_dir / INVOCATIONS_NAME).read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["argv"] == ["run", "--profile", "initial", "--trajectory", "1"]
    assert lines[0]["stages"] == ["plan", "baseline"] and lines[0]["trajectories"] == ["1"] and lines[0]["mode"] == "dry"
    assert lines[0]["config_resolved"]["models"]["agents"][0]["endpoint"] == cfg.models.agents[0].endpoint
    assert lines[0]["config_resolved"]["cost"]["spending_cap_usd"] == cfg.cost.spending_cap_usd
    assert lines[0]["api_key"] in ("present", "absent") and "sk-" not in json.dumps(lines[0])
    copy = entry["config_toml_copy"]
    assert copy and copy.startswith(str(cfg.run_dir / "configs")) and "spending_cap_usd" in open(copy).read()
    log_invocation(cfg, ["forks"], stages=["forks"])
    assert len((cfg.run_dir / INVOCATIONS_NAME).read_text().splitlines()) == 2
