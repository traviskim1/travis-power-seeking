"""Provider pinning (SKILL.md): the pin sent with every call, the pre-flight checks, snapshots and diffs, the
probe, and the record of who served each call."""
import json

import pytest

from aupower.client import Ledger, ModelCallError, OpenRouterClient, make_client
from aupower.config import ModelSpec, RoutingConfig, config_from_dict
from aupower.pipeline import Pipeline
from aupower.routing import (PREFLIGHT_FILE, PROBE_FILE, RoutingError, check_pin, describe_diff, diff_pinned, find_endpoint,
                             latest_snapshot, load_snapshot, previous_snapshots, probe_pins, render_preflight, routing_notes,
                             routing_summary, run_preflight, save_snapshot, served_matches_pin)


def _ep(tag, provider, quant="fp8", ctx=1_048_576, maxout=393_216, prompt=0.15, completion=0.60, cache=0.003,
        params=("max_tokens", "temperature", "seed"), caching=True, status=0):
    return {"name": f"{provider} | m-20261017", "model_id": "m", "model_name": "M", "context_length": ctx,
            "pricing": {"prompt": f"{prompt / 1e6:.12f}", "completion": f"{completion / 1e6:.12f}", "input_cache_read": f"{cache / 1e6:.12f}"},
            "provider_name": provider, "tag": tag, "service_tier": None, "quantization": quant, "max_completion_tokens": maxout,
            "max_prompt_tokens": None, "supported_parameters": list(params), "status": status, "uptime_last_30m": 100,
            "supports_implicit_caching": caching}


def _list(*eps, modalities=("text", "image")):
    return {"id": "m", "name": "M", "architecture": {"input_modalities": list(modalities)}, "endpoints": list(eps)}


PROVIDER = "DeepInfra"      # the fake provider name used for the configured pin in these tests


def _pinned(cfg, **kw):
    """The configured agent pin as an endpoint record: its tag and the config's prices, unless overridden."""
    m = cfg.models.agents[0]
    pr = cfg.cost.prices[m.id]
    args = dict(prompt=pr.input_per_m, completion=pr.output_per_m, cache=pr.cached_input_per_m)
    args.update(kw)
    return _ep(m.endpoint, PROVIDER, **args)


# --------------------------------------------------------------------------------------------
# the pin in the request, and the config rules around it
# --------------------------------------------------------------------------------------------

def test_provider_preferences_carry_the_pin_and_the_rules():
    r = RoutingConfig()
    assert r.provider_preferences(ModelSpec(id="x", endpoint="deepinfra/fp8")) == {
        "require_parameters": True, "data_collection": "deny", "only": ["deepinfra/fp8"], "allow_fallbacks": False}
    assert r.provider_preferences(ModelSpec(id="x")) == {"require_parameters": True, "data_collection": "deny"}
    assert ModelSpec(id="x", endpoint="google-ai-studio/flex").provider_slug == "google-ai-studio"


def test_live_mode_needs_a_pin_or_an_explicit_opt_out(dry_cfg):
    agent = dry_cfg().models.agents[0].id
    with pytest.raises(ValueError, match="no `endpoint` pin"):
        dry_cfg(run={"mode": "live"}, models={"agents": [{"id": agent}]})
    assert dry_cfg(run={"mode": "live"}, models={"agents": [{"id": agent, "unpinned_ok": True}]}).models.agents[0].unpinned_ok
    assert dry_cfg(models={"agents": [{"id": agent}]}).models.agents[0].endpoint == ""     # dry mode: no pin needed
    with pytest.raises(ValueError, match="data_collection"):
        dry_cfg(routing={"data_collection": "maybe"})


# --------------------------------------------------------------------------------------------
# the pre-flight checks
# --------------------------------------------------------------------------------------------

def test_check_pin_passes_a_matching_endpoint(dry_cfg):
    cfg = dry_cfg()
    m = cfg.models.agents[0]
    assert m.endpoint and "/" in m.endpoint                      # a full endpoint tag, provider/variant
    check = check_pin(m, _list(_pinned(cfg), _ep("x/fp4", "X", quant="fp4")), cfg)
    assert check.ok and check.warnings == [] and check.record["provider_name"] == PROVIDER
    pr = cfg.cost.prices[m.id]
    assert check.record["pricing_per_m"] == {"prompt": pr.input_per_m, "completion": pr.output_per_m, "input_cache_read": pr.cached_input_per_m}
    assert [a["tag"] for a in check.available] == [m.endpoint, "x/fp4"]


def test_check_pin_finds_the_problems(dry_cfg):
    cfg = dry_cfg()
    m = cfg.models.agents[0]
    missing = check_pin(m, _list(_ep("other/fp8", "Other")), cfg)
    assert not missing.ok and "not in OpenRouter's current list" in missing.problems[0] and "other/fp8" in missing.problems[0]
    unpinned = check_pin(ModelSpec(id=m.id), _list(_ep("a", "A")), cfg)
    assert not unpinned.ok and "not pinned" in unpinned.problems[0]
    opted = check_pin(ModelSpec(id=m.id, unpinned_ok=True), _list(_ep("a", "A")), cfg)
    assert opted.ok and "unpinned_ok" in opted.warnings[0]
    small = check_pin(m, _list(_pinned(cfg, maxout=1000)), cfg)
    assert any("caps output" in p for p in small.problems)
    short = check_pin(m, _list(_pinned(cfg, ctx=8000)), cfg)
    assert any("context" in p for p in short.problems)
    no_temp = check_pin(m, _list(_pinned(cfg, params=("max_tokens",))), cfg)
    assert any("does not support ['temperature']" in p for p in no_temp.problems)
    lax = check_pin(m, _list(_pinned(cfg, params=("max_tokens",))), dry_cfg(routing={"require_parameters": False}))
    assert lax.ok and any("dropped silently" in w for w in lax.warnings)
    text_only = check_pin(m, _list(_pinned(cfg), modalities=("text",)), cfg)
    assert any("supports_images" in p for p in text_only.problems)


def test_check_pin_warns_about_precision_prices_and_caching(dry_cfg):
    cfg = dry_cfg()
    m = cfg.models.agents[0]
    low = check_pin(m, _list(_pinned(cfg, quant="fp4"), _ep("other/fp8", "Other")), cfg)
    assert low.ok and any("low-precision" in w and "other/fp8" in w for w in low.warnings)
    undeclared = check_pin(m, _list(_pinned(cfg, quant="unknown"), _ep("other/fp8", "Other")), cfg)
    assert any("declares no quantization" in w for w in undeclared.warnings)
    proprietary = check_pin(m, _list(_pinned(cfg, quant="unknown"), _ep(m.endpoint + "/flex", PROVIDER, quant="unknown")), cfg)
    assert not any("quantization" in w for w in proprietary.warnings)
    dearer = check_pin(m, _list(_pinned(cfg, prompt=0.30)), cfg)
    assert any("config prices" in w and "$0.3" in w for w in dearer.warnings)
    uncached = check_pin(m, _list(_pinned(cfg, caching=False)), cfg)
    assert any("implicit prompt caching" in w for w in uncached.warnings)
    judge = check_pin(m, _list(_pinned(cfg, caching=False)), cfg, role="judge")
    assert not any("caching" in w for w in judge.warnings)


# --------------------------------------------------------------------------------------------
# snapshots and the diff between runs
# --------------------------------------------------------------------------------------------

def test_snapshots_are_kept_per_run_and_diffed(tmp_path, dry_cfg):
    cfg = dry_cfg()
    m = cfg.models.agents[0]
    runs = cfg.runs_root
    p1 = save_snapshot(runs / "a", m, _list(_pinned(cfg)), mock=False, ts=1000)
    p2 = save_snapshot(runs / "b", m, _list(_pinned(cfg, quant="fp4"), _ep("x/fp8", "X")), mock=False, ts=2000)
    pm = save_snapshot(runs / "c", m, _list(_ep(m.endpoint, "Mock")), mock=True, ts=3000)
    assert p1.parent == runs / "a" / "endpoints" and p1.name.startswith(m.slug + "__")
    assert previous_snapshots(runs, m, mock=False, exclude=p2) == [p1]
    assert previous_snapshots(runs, m, mock=False) == [p1, p2]
    assert previous_snapshots(runs, m, mock=True) == [pm]                 # mock and real lists are never compared
    assert latest_snapshot(runs / "b", m) == p2
    d = diff_pinned(load_snapshot(p1), load_snapshot(p2))
    assert d["changed"]["quantization"] == {"before": "fp8", "after": "fp4"} and d["endpoints_added"] == ["x/fp8"]
    assert d["pinned_was_listed"] and d["pinned_is_listed"] and "'fp8' -> 'fp4'" in describe_diff(d)
    gone = diff_pinned(load_snapshot(p1), {"endpoint": m.endpoint, "data": _list(_ep("x/fp8", "X"))})
    assert not gone["pinned_is_listed"] and "DISAPPEARED" in describe_diff(gone)
    same = diff_pinned(load_snapshot(p1), load_snapshot(p1))
    assert same["changed"] == {} and "no change" in describe_diff(same)


# --------------------------------------------------------------------------------------------
# who served a call
# --------------------------------------------------------------------------------------------

def test_served_matches_pin():
    m = ModelSpec(id="x", endpoint="google-ai-studio")
    assert served_matches_pin("Google AI Studio", m) is True
    assert served_matches_pin("DeepInfra", m) is False
    assert served_matches_pin(None, m) is None
    assert served_matches_pin("Google AI Studio", ModelSpec(id="x")) is None
    rec = {"provider_name": "Google AI Studio"}
    assert served_matches_pin("google ai studio", m, rec) is True and served_matches_pin("Google Vertex", m, rec) is False


async def test_openrouter_client_sends_the_pin_and_records_who_served(tmp_path, dry_cfg):
    import httpx
    cfg = dry_cfg(run={"mode": "live"})
    agent = cfg.models.agents[0]
    seen: dict = {}
    state = {"provider_in_response": True}

    def handler(request):
        path = request.url.path
        if path.endswith("/chat/completions"):
            seen["body"] = json.loads(request.content)
            seen["auth"] = request.headers.get("authorization")
            body = {"id": "gen-1", "model": agent.id, "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "cost": 0.000002}}
            if state["provider_in_response"]:
                body["provider"] = PROVIDER
            return httpx.Response(200, json=body)
        if path.endswith("/generation"):
            seen["generation_id"] = request.url.params.get("id")
            return httpx.Response(200, json={"data": {"id": "gen-1", "provider_name": PROVIDER, "model": agent.id}})
        if path.endswith("/endpoints") and "nobody" not in path:
            seen["endpoints_auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"data": _list(_pinned(cfg))})
        return httpx.Response(404, json={"error": "nope"})

    led = Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices)
    client = OpenRouterClient("test-key", led, routing=cfg.routing, models={m.id: m for m in cfg.models.agents + [cfg.models.judge]},
                              transport=httpx.MockTransport(handler))
    try:
        resp = await client.chat(model=agent.id, messages=[{"role": "user", "content": "hi"}], max_tokens=5, temperature=1.0,
                                 purpose="agent_step", meta={"rollout_id": "r", "step": 1})
        assert seen["body"]["provider"] == {"only": [agent.endpoint], "allow_fallbacks": False, "require_parameters": True,
                                            "data_collection": cfg.routing.data_collection}
        assert seen["body"]["model"] == agent.id and seen["body"]["temperature"] == 1.0 and seen["auth"] == "Bearer test-key"
        assert resp.served_provider == PROVIDER and resp.request_id == "gen-1" and resp.usage.cost_source == "openrouter"
        assert "generation_id" not in seen                      # the response named its provider: no lookup needed
        # a response without the provider field: the generation record is asked
        state["provider_in_response"] = False
        resp2 = await client.chat(model=agent.id, messages=[{"role": "user", "content": "hi"}], max_tokens=5, temperature=1.0,
                                  purpose="agent_step", meta={})
        assert resp2.served_provider == PROVIDER and seen["generation_id"] == "gen-1"
        rows = [json.loads(l) for l in (tmp_path / "l.jsonl").read_text().splitlines()]
        assert [r["served_provider"] for r in rows] == [PROVIDER, PROVIDER]
        assert rows[0]["endpoint"] == agent.endpoint and rows[0]["served_model"] == agent.id and rows[0]["request_id"] == "gen-1"
        # the endpoint list is public: fetched without the key
        data = await client.fetch_endpoints(agent.id)
        assert find_endpoint(data, agent.endpoint)["provider_name"] == PROVIDER and seen["endpoints_auth"] is None
        with pytest.raises(ModelCallError, match="could not fetch"):
            await client.fetch_endpoints("nobody/nothing")
    finally:
        await client.close()


# --------------------------------------------------------------------------------------------
# the mock client, the pre-flight and the probe in a dry run
# --------------------------------------------------------------------------------------------

async def test_mock_endpoints_preflight_and_probe(dry_cfg, tmp_path):
    cfg = dry_cfg(routing={"data_collection": "deny"})      # the render mentions the probe only under "deny"
    agent, judge = cfg.models.agents[0], cfg.models.judge
    client = make_client(cfg, Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices))
    data = await client.fetch_endpoints(agent.id)
    assert [e["tag"] for e in data["endpoints"]] == [agent.endpoint, "mock-cheap/fp4"]
    assert check_pin(agent, data, cfg).ok
    report = await run_preflight(client, cfg, cfg.run_dir)
    assert report["ok"] and report["mock"] and (cfg.run_dir / PREFLIGHT_FILE).exists()
    assert set(report["models"]) == {agent.id, judge.id} and report["models"][judge.id]["record"]["tag"] == "google-ai-studio"
    text = render_preflight(report, all_endpoints=True)
    assert f"pinned to {agent.endpoint} (Mock" in text and f"* {agent.endpoint}" in text and "mock-cheap/fp4" in text
    assert "every pin can be used" in text
    assert "data_collection = \"deny\"" in text
    # a second pre-flight diffs against the first: nothing changed
    report2 = await run_preflight(client, cfg, cfg.run_dir)
    d = report2["models"][agent.id]["diff"]
    assert d is not None and d["changed"] == {} and report2["models"][agent.id]["previous_snapshot"] == report["models"][agent.id]["snapshot"]
    results = await probe_pins(client, cfg, {agent.id: report["models"][agent.id]["record"]})   # the judge's record is fetched
    assert results[agent.id]["matches_pin"] is True and results[agent.id]["served_provider"] == "Mock"
    assert results[judge.id]["served_provider"] == "Mock" and results[judge.id]["ok"] and results[judge.id]["matches_pin"] is True


async def test_probe_failures_are_routing_errors(dry_cfg):
    cfg = dry_cfg()

    class Refusing:
        is_mock = True

        async def fetch_endpoints(self, model_id):
            return _list(_pinned(cfg), _ep("google-ai-studio", "Google AI Studio"))

        async def chat(self, **kw):
            raise ModelCallError("OpenRouter call failed: HTTP 404: No endpoints found matching your data policy")

    with pytest.raises(RoutingError, match="data_collection"):
        await probe_pins(Refusing(), cfg)

    class Elsewhere(Refusing):
        async def chat(self, *, model, **kw):
            from aupower.client import ChatResponse, Usage
            return ChatResponse(text="OK", usage=Usage(), model=model, latency_s=0.0, served_provider="Novita")

    with pytest.raises(RoutingError, match="not by the pinned endpoint"):
        await probe_pins(Elsewhere(), cfg)


def test_routing_summary_counts_who_served(dry_cfg):
    cfg = dry_cfg()
    agent, judge = cfg.models.agents[0], cfg.models.judge
    run = cfg.run_dir
    run.mkdir(parents=True, exist_ok=True)
    save_snapshot(run, agent, _list(_pinned(cfg)), mock=False)
    rows = [{"model": agent.id, "status": "ok", "served_provider": PROVIDER, "served_model": agent.id}] * 3
    rows += [{"model": agent.id, "status": "ok", "served_provider": "Novita", "served_model": agent.id},
             {"model": agent.id, "status": "ok", "served_provider": None},
             {"model": agent.id, "status": "error", "served_provider": None},
             {"model": judge.id, "status": "ok", "served_provider": "Google AI Studio"},
             {"model": judge.id, "status": "ok", "served_provider": "Google AI Studio"}]
    (run / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = routing_summary(cfg, run)
    a = s["models"][agent.id]
    assert a["calls"] == 5 and a["served_by"] == {PROVIDER: 3, "Novita": 1} and a["unknown_provider"] == 1 and a["served_elsewhere"] == 1
    assert a["pinned_record"]["provider_name"] == PROVIDER and a["provider_preferences"]["only"] == [agent.endpoint]
    j = s["models"][judge.id]
    assert j["calls"] == 2 and j["served_by"] == {"Google AI Studio": 2} and j["served_elsewhere"] == 0
    notes = routing_notes(s)
    assert "SERVED BY ANOTHER PROVIDER" in notes[0] and "1 with no provider recorded" in notes[0]
    assert f"pinned to {agent.endpoint} ({PROVIDER}" in notes[0]
    assert "google-ai-studio" in notes[1] and "ANOTHER" not in notes[1]


async def test_dry_pipeline_runs_the_preflight_and_probe_and_reports_routing(dry_cfg):
    cfg = dry_cfg()
    pipe = Pipeline(cfg)
    try:
        await pipe.run_stages(["plan", "baseline"])
        assert (cfg.run_dir / PREFLIGHT_FILE).exists() and (cfg.run_dir / PROBE_FILE).exists()
        probe = json.loads((cfg.run_dir / PROBE_FILE).read_text())["results"]
        assert all(v["ok"] and v["served_provider"] == "Mock" for v in probe.values())
        pipe.stage_analyze()
        routing = json.loads((cfg.run_dir / "results" / "routing.json").read_text())
        a = routing["models"][cfg.models.agents[0].id]
        assert a["calls"] > 1 and a["served_by"] == {"Mock": a["calls"]} and a["served_elsewhere"] == 0 and routing["probe"]
        assert "routing:" in (cfg.run_dir / "results" / "summary.txt").read_text()
        rows = [json.loads(l) for l in (cfg.run_dir / "ledger.jsonl").read_text().splitlines()]
        assert rows[0]["purpose"] == "routing_probe" and all(r["served_provider"] == "Mock" and r["endpoint"] for r in rows)
    finally:
        await pipe.close()


async def test_pipeline_refuses_to_start_on_a_bad_pin(dry_cfg):
    agent = dry_cfg().models.agents[0].id
    cfg = dry_cfg(models={"agents": [{"id": agent, "endpoint": "deepinfra/fp8", "max_tokens": 10_000_000}]})   # above any output cap
    pipe = Pipeline(cfg)
    try:
        with pytest.raises(RoutingError, match="caps output"):
            await pipe.run_stages(["plan", "baseline"])
        assert not (cfg.run_dir / PROBE_FILE).exists()
    finally:
        await pipe.close()
