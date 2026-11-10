import json

import pytest

from aupower.config import config_from_dict, load_config
from aupower.costs import estimate, measure_from_ledger, rollout_counts


SPEC_PRICES = {"input_per_m": 0.30, "output_per_m": 1.20, "cached_input_per_m": 0.006}   # the planning prices of section 8


def test_base_settings_reproduce_spec_section_8(tasks):
    # the base [experiment] settings are the full design of SPEC section 8; [profiles.initial] is a smaller first run.
    # The dollar figures of section 8 were computed at its planning prices, so those are put back for this check
    # (config.toml carries the pinned endpoint's current list price instead).
    agent_id = load_config().models.agents[0].id
    cfg = config_from_dict({"cost": {"prices": {agent_id: SPEC_PRICES}}})
    est = estimate(cfg, tasks)
    c = est.counts
    assert c["baseline_rollouts"] == 135 and c["trajectory_runs"] == 5 and c["fork_rollouts"] == 2700
    assert c["rollouts_total"] == 2840 and c["sampled_states"] == 15
    lines = {l.item: l for l in est.lines}
    # SPEC.md: about $0.47 per rollout without caching, about $0.17 with 80% cached
    assert lines["  (same, no caching, for reference)"].cost_usd / 2840 == pytest.approx(0.465, abs=0.002)
    assert lines[f"agent {cfg.models.agents[0].id}"[:38]].cost_usd / 2840 == pytest.approx(0.171, abs=0.002)   # labels are cut to 38 chars
    # SPEC.md: about $0.19 per rollout for re-scoring at each limit and $0.03 for first-step, both with 6 items;
    # easy tasks average 4.7 items, so scale by 6/m
    m = est.assumptions["mean_rubric_items"]
    assert m == pytest.approx(4.73, abs=0.01)
    assert lines["H scoring: re-score at each step limit"].cost_usd / 2835 * 6 / m == pytest.approx(0.189, abs=0.003)
    assert lines["T: first step per item (s0 rollouts)"].cost_usd / 135 * 6 / m == pytest.approx(0.0327, abs=0.002)
    assert est.total_usd < 1500
    # one budget for every profile: the prediction is compared with what is left of it
    assert est.cap_usd == cfg.cost.spending_cap_usd and est.within_cap == (est.total_usd <= est.cap_usd)


def test_initial_profile_is_a_smaller_first_run(tasks):
    # [profiles.initial] as configured: 3 trajectories x 1 state (halfway), 30 goals x 1 rollout, full context only
    cfg = load_config(profile="initial")
    est = estimate(cfg, tasks)
    c = est.counts
    assert c["baseline_rollouts"] == 45 and c["trajectory_runs"] == 3 and c["sampled_states"] == 3
    assert c["fork_rollouts"] == 90 and c["rollouts_total"] == 138
    assert c["rollouts_per_goal"] == 1 and c["fresh_context_forks"] is False
    assert est.within_cap and "fits within the remaining budget" in est.render()


def test_smoke_estimate(tasks):
    cfg = load_config(profile="smoke")
    est = estimate(cfg, tasks)
    assert est.counts["rollouts_total"] == 106 and est.counts["fork_rollouts"] == 60 and est.counts["baseline_rollouts"] == 45
    assert 20 < est.total_usd < 60
    assert est.cap_usd == cfg.cost.spending_cap_usd and est.within_cap == (est.total_usd <= est.cap_usd)
    text = est.render()
    assert ("fits within the remaining budget" in text) == est.within_cap
    assert ("EXCEEDS the remaining budget" in text) == (not est.within_cap)


def test_budget_is_one_number_for_every_profile():
    base = load_config().cost.spending_cap_usd
    assert base == 60.0
    assert load_config(profile="smoke").cost.spending_cap_usd == base
    assert load_config(profile="initial").cost.spending_cap_usd == base


def test_estimate_counts_what_is_already_spent(tasks):
    cfg = load_config(profile="smoke", overrides=["cost.spending_cap_usd=100.0"])
    fresh = estimate(cfg, tasks)
    assert fresh.within_cap and fresh.remaining_usd == 100.0
    spent = 100.0 - fresh.total_usd + 1.0          # leaves one dollar less than the prediction needs
    used = estimate(cfg, tasks, spent_usd=spent, spent_detail={"spent_this_run_usd": 0.0, "spent_other_runs_usd": spent})
    assert used.remaining_usd == pytest.approx(fresh.total_usd - 1.0) and not used.within_cap
    assert "other runs" in used.render() and "EXCEEDS the remaining budget" in used.render()
    assert used.to_dict()["remaining_usd"] == pytest.approx(fresh.total_usd - 1.0)


def test_estimate_respects_cap_and_modes(tasks):
    cfg = load_config(profile="smoke", overrides=["cost.spending_cap_usd=1.0"])
    assert not estimate(cfg, tasks).within_cap
    cheap = load_config(profile="initial", overrides=["judge.h_scoring_mode=first_step"])
    full = load_config(profile="initial")
    assert estimate(cheap, tasks).total_usd < estimate(full, tasks).total_usd
    bis = load_config(profile="initial", overrides=["judge.t_method=bisect"])
    assert estimate(bis, tasks).counts["judge_calls_total"] > estimate(full, tasks).counts["judge_calls_total"]


def test_counts_without_fresh_context_and_two_models(tasks):
    cfg = load_config(overrides=["experiment.fresh_context_forks=false"])
    assert rollout_counts(cfg, tasks)["fork_rollouts"] == 1350
    two = load_config(overrides=[
        'models.agents=[{"id":"deepseek/deepseek-v4.1-flash-1017"},{"id":"google/gemini-3.1-flash-lite-preview"}]'])
    assert rollout_counts(two, tasks)["rollouts_total"] == 2 * 2840


def test_measure_from_ledger(tmp_path):
    p = tmp_path / "ledger.jsonl"
    rows = [
        {"status": "ok", "purpose": "agent_step", "prompt_tokens": 10000, "completion_tokens": 500, "cached_tokens": 8000, "meta": {"rollout_id": "a"}},
        {"status": "ok", "purpose": "agent_step", "prompt_tokens": 20000, "completion_tokens": 1500, "cached_tokens": 16000, "meta": {"rollout_id": "a"}},
        {"status": "ok", "purpose": "judge_rescore", "prompt_tokens": 7000, "completion_tokens": 200},
        {"status": "error", "purpose": "agent_step", "prompt_tokens": 0, "completion_tokens": 0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = measure_from_ledger(p)
    assert m["steps_per_rollout"] == 2 and m["input_tokens_per_step"] == 15000 and m["output_tokens_per_step"] == 1000
    assert m["cache_fraction"] == pytest.approx(0.8)
    assert m["judge_input_tokens_full_trajectory"] == 7000 and m["judge_output_tokens_per_call"] == 200
