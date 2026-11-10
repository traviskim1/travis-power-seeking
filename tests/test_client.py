import json
import os

import json

import pytest

from aupower.client import (Ledger, MockClient, SpendingCapExceeded, make_client, make_ledger, prior_spend,
                            count_message_tokens_estimate)
from aupower.config import Pricing


def _prices():
    return {"m": Pricing(input_per_m=1.0, output_per_m=2.0, cached_input_per_m=0.1)}


def test_ledger_cost_estimate():
    led = Ledger.__new__(Ledger)
    led.prices = _prices()
    assert led.estimate_cost("m", 1_000_000, 500_000) == pytest.approx(1.0 + 1.0)
    assert led.estimate_cost("m", 1_000_000, 0, cached_tokens=500_000) == pytest.approx(0.5 + 0.05)
    assert led.estimate_cost("unknown", 10, 10) == 0.0


def test_token_estimate_counts_images():
    msgs = [{"role": "system", "content": "a" * 400}, {"role": "user", "content": [{"type": "text", "text": "b" * 40},
                                                                                     {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    tokens, n_images = count_message_tokens_estimate(msgs, image_tokens=1000)
    assert tokens == 100 + 10 + 1000 and n_images == 1


def test_ledger_records_and_enforces_cap(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path, cap_usd=0.01, prices=_prices(), image_tokens=10)
    msgs = [{"role": "user", "content": "x" * 4000}]      # ~1000 tokens -> $0.001 + output
    reserved = led.reserve("m", msgs, max_tokens=1000)     # 1000 * 2/M = 0.002 -> total 0.003
    assert reserved == pytest.approx(0.003)
    assert led.pending_usd == pytest.approx(0.003)
    led.record({"model": "m", "purpose": "agent_step", "prompt_tokens": 1000, "completion_tokens": 100, "cost_usd": 0.0012,
                "status": "ok"}, reserved)
    assert led.pending_usd == 0.0 and led.spent_usd == pytest.approx(0.0012) and led.n_calls == 1
    assert path.exists() and json.loads(path.read_text().splitlines()[0])["cost_usd"] == 0.0012
    # a call whose estimate would cross the cap is refused before it is made
    with pytest.raises(SpendingCapExceeded):
        led.reserve("m", [{"role": "user", "content": "x" * 40000}], max_tokens=1000)
    # actual spend crossing the cap stops the run
    with pytest.raises(SpendingCapExceeded):
        led.record({"model": "m", "purpose": "agent_step", "cost_usd": 0.02, "status": "ok"})
    # the ledger reloads its spend from disk
    led2 = Ledger(path, cap_usd=1.0, prices=_prices())
    assert led2.spent_usd == pytest.approx(0.0212) and led2.n_calls == 2
    assert led2.summary()["by_purpose"]["agent_step"]["calls"] == 2


@pytest.mark.asyncio
async def test_mock_client_is_deterministic_and_well_formed(dry_cfg, tmp_path):
    cfg = dry_cfg()
    led = Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices)
    client = make_client(cfg, led)
    assert isinstance(client, MockClient) and client.is_mock
    obs = "Current browser state:\nInteractive elements:\n[0] <input> \"Search\"\n[1] <button> \"Search\"\n[2] <a> \"Item 1\""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": obs}]}]
    a = await client.chat(model=cfg.models.agents[0].id, messages=msgs, max_tokens=100, temperature=1.0, purpose="agent_step",
                          meta={"rollout_id": "r1", "step": 1})
    b = await client.chat(model=cfg.models.agents[0].id, messages=msgs, max_tokens=100, temperature=1.0, purpose="agent_step",
                          meta={"rollout_id": "r1", "step": 1})
    assert a.text == b.text and "```json" in a.text
    assert a.usage.cost_usd > 0 and a.usage.cost_source == "mock"
    assert led.n_calls == 2


@pytest.mark.asyncio
async def test_mock_judge_is_monotone_in_the_cut_and_consistent_with_first_step(dry_cfg, tmp_path):
    cfg = dry_cfg()
    led = Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices)
    client = MockClient(led, cfg)
    judge = cfg.models.judge.id
    for rid in ("rollout-a", "rollout-b", "rollout-c"):
        for rubric in ("R1", "R2", "R3"):
            verdicts = []
            for cut in range(1, 11):
                text = f"Rubric ID: {rubric}\n... (trajectory had {cut} total step(s))."
                r = await client.chat(model=judge, messages=[{"role": "user", "content": text}], max_tokens=10, temperature=0,
                                      purpose="judge_rescore", meta={"rollout_id": rid, "rubric_id": rubric, "cut": cut, "n_steps_full": 10})
                verdicts.append('Status: "success"' in r.text)
            # once met, stays met
            assert verdicts == sorted(verdicts)
            fs = await client.chat(model=judge, messages=[{"role": "user", "content": f"Rubric ID: {rubric} (trajectory had 10 total step(s))"}],
                                   max_tokens=10, temperature=0, purpose="judge_first_step",
                                   meta={"rollout_id": rid, "rubric_id": rubric, "cut": 10, "n_steps_full": 10})
            if any(verdicts):
                first = verdicts.index(True) + 1
                assert f"First satisfied at step: {first}" in fs.text
            else:
                assert "First satisfied at step: none" in fs.text


def test_live_mode_without_key_fails_clearly(dry_cfg, monkeypatch, tmp_path):
    cfg = dry_cfg(run={"mode": "live"})
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import aupower.config as c
    monkeypatch.setattr(c, "ENV_PATH", tmp_path / "no.env")
    monkeypatch.setattr(c.load_dotenv, "__defaults__", (tmp_path / "no.env",))
    led = Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_client(cfg, led)


def test_dotenv_loader(tmp_path, monkeypatch):
    from aupower.config import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENROUTER_API_KEY='abc'\nexport OTHER=1\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loaded = load_dotenv(env)
    assert loaded == {"OPENROUTER_API_KEY": "abc", "OTHER": "1"}
    assert os.environ["OPENROUTER_API_KEY"] == "abc"


# --------------------------------------------------------------------------------------------
# The budget catcher
# --------------------------------------------------------------------------------------------

def _chat_kwargs(cfg):
    return dict(model=cfg.models.agents[0].id, messages=[{"role": "user", "content": "x" * 400}], max_tokens=50,
                temperature=0.0, purpose="agent_step", meta={"rollout_id": "r", "step": 1})


async def test_mock_calls_cost_nothing_unless_charged(dry_cfg, tmp_path):
    cfg = dry_cfg(mock={"charge_estimated_cost": False})
    led = Ledger(tmp_path / "l.jsonl", 10.0, cfg.cost.prices)
    resp = await make_client(cfg, led).chat(**_chat_kwargs(cfg))
    assert resp.usage.cost_usd == 0.0 and resp.usage.cost_source == "mock" and resp.usage.prompt_tokens > 0
    assert led.spent_usd == 0.0 and led.n_calls == 1
    cfg2 = dry_cfg(mock={"charge_estimated_cost": True})
    led2 = Ledger(tmp_path / "l2.jsonl", 10.0, cfg2.cost.prices)
    resp2 = await make_client(cfg2, led2).chat(**_chat_kwargs(cfg2))
    assert resp2.usage.cost_usd > 0 and led2.spent_usd == pytest.approx(resp2.usage.cost_usd)


def test_ledger_budget_counts_prior_spend_and_trips_once(tmp_path, dry_cfg):
    cfg = dry_cfg()
    prices, model = cfg.cost.prices, cfg.models.agents[0].id
    led = Ledger(tmp_path / "l.jsonl", 30.0, prices, image_tokens=10, prior_spent_usd=29.999, prior_by_run={"smoke": 29.999})
    assert led.total_spent_usd == pytest.approx(29.999) and led.remaining_usd == pytest.approx(0.001)
    msgs = [{"role": "user", "content": "x" * 40000}]     # ~10k tokens: about $0.0015 of input plus the output allowance
    with pytest.raises(SpendingCapExceeded) as ei:
        led.reserve(model, msgs, 1000)
    assert "budget $30.00" in str(ei.value) and "other runs" in str(ei.value) and led.tripped == str(ei.value)
    # once tripped, even a call that would fit is refused: the run is winding down
    with pytest.raises(SpendingCapExceeded):
        led.reserve(model, [{"role": "user", "content": "hi"}], 1)
    s = led.summary()
    assert s["prior_spent_usd"] == pytest.approx(29.999) and s["tripped"] and s["remaining_usd"] == pytest.approx(0.001)


def test_ledger_mock_entries_count_only_in_dry_mode(tmp_path, dry_cfg):
    prices = dry_cfg().cost.prices
    p = tmp_path / "l.jsonl"
    rows = [{"purpose": "agent_step", "cost_usd": 5.0, "mock": True, "status": "ok"},
            {"purpose": "agent_step", "cost_usd": 1.0, "mock": False, "status": "ok"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert Ledger(p, 30.0, prices, count_mock=True).spent_usd == pytest.approx(6.0)
    assert Ledger(p, 30.0, prices, count_mock=False).spent_usd == pytest.approx(1.0)


def test_prior_spend_scans_other_runs_real_entries_only(tmp_path, dry_cfg):
    runs = tmp_path / "runs"      # the dry_cfg fixture's runs_dir

    def write(name, rows):
        d = runs / name
        d.mkdir(parents=True)
        (d / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    write("a", [{"cost_usd": 2.5, "mock": False}, {"cost_usd": 100.0, "mock": True}])
    write("b", [{"cost_usd": 0.5, "mock": False}])
    (runs / "c").mkdir()
    assert prior_spend(runs, exclude=runs / "b") == {"a": 2.5}
    assert prior_spend(runs) == {"a": 2.5, "b": 0.5}
    assert prior_spend(tmp_path / "nowhere") == {}
    # live runs: other runs' real spend counts and mock entries never do; dry runs: only their own entries
    live = make_ledger(dry_cfg(run={"mode": "live", "name": "b"}))
    assert live.prior_spent_usd == pytest.approx(2.5) and live.prior_by_run == {"a": 2.5} and not live.count_mock
    assert live.spent_usd == pytest.approx(0.5) and live.total_spent_usd == pytest.approx(3.0)
    assert live.remaining_usd == pytest.approx(50.0 - 3.0)
    dry = make_ledger(dry_cfg(run={"name": "b"}))          # the fixture rehearses: mock entries count
    assert dry.prior_spent_usd == 0.0 and dry.count_mock and dry.spent_usd == pytest.approx(0.5)
    plain = make_ledger(dry_cfg(run={"name": "b"}, mock={"charge_estimated_cost": False}))
    assert plain.prior_spent_usd == 0.0 and not plain.count_mock
