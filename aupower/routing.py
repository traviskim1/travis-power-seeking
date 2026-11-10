"""Provider pinning for OpenRouter, after SKILL.md ("use-openrouter-safely").

With OpenRouter's default routing a model slug is not a fixed artifact: the same slug is served by many
providers, at different quantizations, context windows and output caps, chosen per request by price, and
parameters a provider does not support are dropped silently. For research results that is an uncontrolled
variable. This module, with ``RoutingConfig`` and ``ModelSpec.endpoint`` in the config, makes every run:

* send one endpoint pin with every call (``provider.only = [tag]``, no fallbacks, parameters must be honoured,
  data collection denied): :meth:`aupower.config.RoutingConfig.provider_preferences`;
* check the pins before anything runs (:func:`check_pin`): the endpoint exists, serves the context and output
  length needed, honours the parameters sent, declares a precision, and prices as the config assumes;
* keep the whole endpoint list with the run (:func:`save_snapshot`, ``runs/<name>/endpoints/``) and diff the pinned
  record against the newest earlier snapshot under ``runs/`` (:func:`diff_pinned`): providers do not announce
  serving changes, so stored snapshots are the only change detector there is;
* start live runs with one minimal call per model through its pin (:func:`probe_pins`), so a pin that cannot be
  served fails before any rollout;
* record the provider that served every response in ``ledger.jsonl`` (``served_provider``, from the response or
  from OpenRouter's generation record) and summarise it next to the pins in ``results/routing.json``
  (:func:`routing_summary`), so a report can state what actually ran.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .config import Config, ModelSpec

log = logging.getLogger("aupower.routing")

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model}/endpoints"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
PROBE_PROMPT = "Reply with the single word OK."
SUMMARY_FIELDS = ("name", "model_id", "provider_name", "tag", "quantization", "context_length", "max_completion_tokens",
                  "max_prompt_tokens", "status", "service_tier", "supports_implicit_caching", "supported_parameters")
PRICE_FIELDS = ("prompt", "completion", "input_cache_read")
LOW_PRECISION = {"int2", "int3", "int4", "nf4", "fp4", "mxfp4", "fp6", "int8"}
PREFLIGHT_FILE = "routing_preflight.json"
PROBE_FILE = "routing_probe.json"


class RoutingError(RuntimeError):
    """A pin cannot be honoured (or is not set): the run must not start."""


# --------------------------------------------------------------------------------------------
# Endpoint records
# --------------------------------------------------------------------------------------------

def find_endpoint(data: dict, tag: str) -> dict | None:
    """The endpoint record with this tag in an endpoint list (``data`` of GET /models/{id}/endpoints)."""
    for e in data.get("endpoints") or []:
        if e.get("tag") == tag:
            return e
    return None


def _per_million(x: Any) -> float | None:
    try:
        return round(float(x) * 1_000_000, 6) if x is not None else None
    except (TypeError, ValueError):
        return None


def endpoint_summary(e: dict) -> dict[str, Any]:
    """The fields of an endpoint record worth keeping, diffing and reporting."""
    pricing = e.get("pricing") or {}
    out: dict[str, Any] = {k: e.get(k) for k in SUMMARY_FIELDS}
    out["pricing_per_m"] = {k: _per_million(pricing.get(k)) for k in PRICE_FIELDS}
    return out


def _n(x: Any) -> str:
    return f"{int(x):,}" if isinstance(x, (int, float)) else "?"


def _usd(x: Any) -> str:
    return f"${x:.4g}" if isinstance(x, (int, float)) else "?"


def describe_endpoint(s: dict) -> str:
    p = s.get("pricing_per_m") or {}
    caching = s.get("supports_implicit_caching")
    return (f"{s.get('tag')} ({s.get('provider_name')}; quantization {s.get('quantization') or 'unknown'}; "
            f"context {_n(s.get('context_length'))}; max output {_n(s.get('max_completion_tokens'))}; "
            f"{_usd(p.get('prompt'))}/{_usd(p.get('completion'))} per M tokens in/out, cache read "
            f"{_usd(p.get('input_cache_read'))}; implicit caching {'yes' if caching else 'no' if caching is False else '?'}; "
            f"serves {s.get('name')})")


# --------------------------------------------------------------------------------------------
# Checking a pin against the live list
# --------------------------------------------------------------------------------------------

@dataclass
class PinCheck:
    model: str
    endpoint: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record: dict | None = None              # summary of the pinned endpoint, when it is listed
    available: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parameters_sent(model: ModelSpec) -> list[str]:
    """The request parameters the clients send for this model (besides model and messages)."""
    return ["max_tokens"] + (["temperature"] if model.temperature is not None else [])


def check_pin(model: ModelSpec, data: dict, cfg: Config, *, role: str = "agent") -> PinCheck:
    """Verify a model's pin against OpenRouter's current endpoint list. ``role`` is "agent" or "judge" (it sets
    the context the endpoint must serve, from the planning assumptions)."""
    eps = data.get("endpoints") or []
    check = PinCheck(model.id, model.endpoint, available=[endpoint_summary(e) for e in eps])
    tags = [str(e.get("tag")) for e in eps]
    if not model.endpoint:
        msg = (f"{model.id} is not pinned: default routing would pick among {len(eps)} endpoints per request "
               f"({', '.join(tags) or 'none listed'})")
        (check.warnings if model.unpinned_ok else check.problems).append(
            msg + (" (allowed by unpinned_ok)" if model.unpinned_ok else "; set endpoint = \"<tag>\" on the model, or unpinned_ok = true"))
        return check
    e = find_endpoint(data, model.endpoint)
    if e is None:
        check.problems.append(f"endpoint {model.endpoint!r} is not in OpenRouter's current list for {model.id} "
                              f"({len(eps)} endpoints: {', '.join(tags) or 'none'})")
        return check
    rec = endpoint_summary(e)
    check.record = rec
    plan = cfg.cost.planning
    max_out = rec.get("max_completion_tokens")
    if isinstance(max_out, (int, float)) and max_out < model.max_tokens:
        check.problems.append(f"{model.endpoint} caps output at {_n(max_out)} tokens but max_tokens is {model.max_tokens:,}")
    need_ctx = plan.input_tokens_per_step if role == "agent" else plan.judge_input_tokens_full_trajectory
    ctx = rec.get("context_length")
    if isinstance(ctx, (int, float)) and ctx < need_ctx:
        check.problems.append(f"{model.endpoint} serves a {_n(ctx)}-token context but a {role} call is planned at "
                              f"{_n(need_ctx)} input tokens (cost.planning)")
    supported = set(rec.get("supported_parameters") or [])
    missing = [p for p in parameters_sent(model) if supported and p not in supported]
    if missing:
        if cfg.routing.require_parameters:
            check.problems.append(f"{model.endpoint} does not support {missing}; with routing.require_parameters the request "
                                  f"would not be routed there")
        else:
            check.warnings.append(f"{model.endpoint} does not support {missing}: those parameters would be dropped silently")
    modalities = (data.get("architecture") or {}).get("input_modalities")
    if model.supports_images and modalities and "image" not in modalities:
        check.problems.append(f"{model.id} accepts only {modalities} input but supports_images is true")
    status = rec.get("status")
    if status not in (None, 0):
        check.warnings.append(f"OpenRouter marks {model.endpoint} with status {status} (not the normal 0)")
    uptime = e.get("uptime_last_30m")
    if isinstance(uptime, (int, float)) and uptime < 95:
        check.warnings.append(f"{model.endpoint} reports {uptime}% uptime over the last 30 minutes; with fallbacks off, "
                              f"failed calls are errors")
    quant = rec.get("quantization")
    declared = sorted({str(x.get("quantization")) for x in eps if x.get("quantization") not in (None, "unknown")})
    if quant in LOW_PRECISION:
        better = [str(x.get("tag")) for x in eps if x.get("quantization") not in LOW_PRECISION and x.get("quantization") not in (None, "unknown")]
        check.warnings.append(f"{model.endpoint} is a {quant} (low-precision) endpoint" +
                              (f"; higher-precision endpoints exist: {', '.join(better)}" if better else ""))
    elif quant in (None, "unknown") and declared:
        check.warnings.append(f"{model.endpoint} declares no quantization while other endpoints of this model declare "
                              f"{', '.join(declared)} (labels are self-reported either way; the lab's own serving is usually the reference)")
    price = cfg.cost.prices.get(model.id)
    live = rec.get("pricing_per_m") or {}
    if price is not None:
        pairs = [("prompt", price.input_per_m), ("completion", price.output_per_m), ("input_cache_read", price.cached_input_per_m)]
        off = [(k, cfg_v, live.get(k)) for k, cfg_v in pairs
               if isinstance(live.get(k), (int, float)) and abs(live[k] - cfg_v) > max(1e-6, 0.01 * max(abs(cfg_v), abs(live[k])))]
        if off:
            check.warnings.append(f"config prices for {model.id} differ from {model.endpoint}'s: " +
                                  "; ".join(f"{k} ${cfg_v:g} vs ${lv:g} per M" for k, cfg_v, lv in off) +
                                  f" (the estimate uses the config; update [cost.prices.\"{model.id}\"])")
    if role == "agent" and rec.get("supports_implicit_caching") is False and plan.cache_fraction > 0:
        check.warnings.append(f"{model.endpoint} reports no implicit prompt caching, but cost.planning.cache_fraction is "
                              f"{plan.cache_fraction:g}: expect the agent calls to cost more than estimated")
    return check


# --------------------------------------------------------------------------------------------
# Snapshots of the endpoint list, and the diff between runs
# --------------------------------------------------------------------------------------------

def save_snapshot(run_dir: Path, model: ModelSpec, data: dict, *, mock: bool, ts: float | None = None) -> Path:
    """Keep the whole endpoint list with the run: runs/<name>/endpoints/<model>__<UTC time>.json."""
    ts = ts or time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts)) + f"{int((ts % 1) * 1000):03d}Z"
    folder = Path(run_dir) / "endpoints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model.slug}__{stamp}.json"
    n = 1
    while path.exists():                      # two fetches in the same millisecond: keep both
        n += 1
        path = folder / f"{model.slug}__{stamp}-{n}.json"
    payload = {"fetched_at": ts, "fetched_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)),
               "model": model.id, "endpoint": model.endpoint, "mock": bool(mock), "data": data}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_snapshot(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def previous_snapshots(runs_root: Path, model: ModelSpec, *, mock: bool, exclude: Path | None = None) -> list[Path]:
    """Earlier snapshots of this model's endpoint list under every run (oldest first), real and mock kept apart."""
    found: list[tuple[float, Path]] = []
    ex = Path(exclude).resolve() if exclude is not None else None
    for p in Path(runs_root).glob(f"*/endpoints/{model.slug}__*.json"):
        if ex is not None and p.resolve() == ex:
            continue
        try:
            payload = load_snapshot(p)
        except Exception:
            continue
        if bool(payload.get("mock")) != bool(mock) or payload.get("model") != model.id:
            continue
        found.append((float(payload.get("fetched_at") or 0.0), p))
    return [p for _, p in sorted(found)]


def latest_snapshot(run_dir: Path, model: ModelSpec) -> Path | None:
    """The newest snapshot of this model's list in one run (by the time recorded inside it)."""
    found: list[tuple[float, str, Path]] = []
    for p in Path(run_dir).glob(f"endpoints/{model.slug}__*.json"):
        try:
            found.append((float(load_snapshot(p).get("fetched_at") or 0.0), p.name, p))
        except Exception:
            continue
    return sorted(found)[-1][2] if found else None


def diff_pinned(previous: dict, current: dict) -> dict[str, Any]:
    """What changed for the pinned endpoint (and in the list) between two snapshots of the same model."""
    tag = current.get("endpoint") or previous.get("endpoint") or ""
    old = find_endpoint(previous.get("data") or {}, tag)
    new = find_endpoint(current.get("data") or {}, tag)
    changed: dict[str, Any] = {}
    if old is not None and new is not None:
        o, n = endpoint_summary(old), endpoint_summary(new)
        for k in o:
            if o[k] != n.get(k):
                changed[k] = {"before": o[k], "after": n.get(k)}
    tags_old = {str(e.get("tag")) for e in (previous.get("data") or {}).get("endpoints") or []}
    tags_new = {str(e.get("tag")) for e in (current.get("data") or {}).get("endpoints") or []}
    return {"previous_fetched_at_utc": previous.get("fetched_at_utc"), "endpoint": tag,
            "pinned_was_listed": old is not None, "pinned_is_listed": new is not None, "changed": changed,
            "endpoints_added": sorted(tags_new - tags_old), "endpoints_removed": sorted(tags_old - tags_new)}


def describe_diff(d: dict) -> str:
    since = d.get("previous_fetched_at_utc") or "the earlier snapshot"
    if not d.get("pinned_was_listed"):
        return f"since {since}: the pinned endpoint {d.get('endpoint')} was not listed then"
    if not d.get("pinned_is_listed"):
        return f"since {since}: the pinned endpoint {d.get('endpoint')} has DISAPPEARED from the list"
    parts = [f"{k}: {v['before']!r} -> {v['after']!r}" for k, v in (d.get("changed") or {}).items()]
    if d.get("endpoints_added"):
        parts.append(f"endpoints added {d['endpoints_added']}")
    if d.get("endpoints_removed"):
        parts.append(f"endpoints removed {d['endpoints_removed']}")
    return f"since {since}: " + ("; ".join(parts) if parts else "no change to the pinned endpoint or the list")


# --------------------------------------------------------------------------------------------
# Who served a call
# --------------------------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def served_matches_pin(served: str | None, model: ModelSpec, record: dict | None = None) -> bool | None:
    """Did the provider named in a response match the model's pin? None when it cannot be told (no provider
    recorded, or the model is not pinned). With the endpoint record the comparison is exact (provider_name);
    without it the provider part of the tag is compared loosely ("google-ai-studio" ~ "Google AI Studio")."""
    if not served or not model.endpoint:
        return None
    if record and record.get("provider_name"):
        return _norm(served) == _norm(record["provider_name"])
    a, b = _norm(served), _norm(model.provider_slug)
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


async def probe_pins(client, cfg: Config, records: dict[str, dict | None] | None = None) -> dict[str, Any]:
    """One minimal call per model through its pin, before any rollout: proves the pin can be served and that
    the provider named in the response is the pinned one. Raises RoutingError otherwise."""
    from .client import ModelCallError   # imported here: client imports this module
    records = dict(records or {})
    results: dict[str, Any] = {}
    for m in cfg.models.agents + [cfg.models.judge]:
        where = m.endpoint or "default routing (unpinned_ok)"
        if records.get(m.id) is None and m.endpoint:      # no pre-flight record: get the endpoint's own record to compare with
            try:
                ep = find_endpoint(await client.fetch_endpoints(m.id), m.endpoint)
                records[m.id] = endpoint_summary(ep) if ep else None
            except Exception as exc:
                log.warning("could not fetch the endpoint record of %s for the probe: %s", m.id, exc)
        try:
            resp = await client.chat(model=m.id, messages=[{"role": "user", "content": PROBE_PROMPT}], max_tokens=16,
                                     temperature=m.temperature, purpose="routing_probe", meta={"model": m.id, "endpoint": m.endpoint})
        except ModelCallError as exc:
            results[m.id] = {"ok": False, "endpoint": m.endpoint, "error": str(exc)}
            hint = ""
            if "404" in str(exc):
                hint = (" OpenRouter found no endpoint for the request. Check the tag with `python -m aupower endpoints`; "
                        "if the provider is excluded by routing.data_collection = \"deny\", either set it to \"allow\" "
                        "knowingly or pin a provider that does not retain prompts.")
            raise RoutingError(f"routing probe for {m.id} through {where} failed: {exc}.{hint}") from exc
        match = served_matches_pin(resp.served_provider, m, records.get(m.id))
        results[m.id] = {"ok": True, "endpoint": m.endpoint, "served_provider": resp.served_provider, "served_model": resp.model,
                         "matches_pin": match, "generation_id": resp.request_id, "cost_usd": resp.usage.cost_usd,
                         "reply": resp.text[:80]}
        log.info("routing probe %s via %s: served by %s (%s), model %r, $%.6f", m.id, where, resp.served_provider,
                 {True: "matches the pin", False: "DOES NOT MATCH THE PIN", None: "cannot be compared"}[match],
                 resp.model, resp.usage.cost_usd)
        if match is False:
            raise RoutingError(f"routing probe for {m.id} was served by {resp.served_provider!r}, not by the pinned "
                               f"endpoint {m.endpoint!r}: the pin is not being honoured; not starting")
    return results


# --------------------------------------------------------------------------------------------
# What a run's calls were served by: results/routing.json and the summary lines
# --------------------------------------------------------------------------------------------

def _empty(model: ModelSpec | None, model_id: str) -> dict[str, Any]:
    return {"endpoint": model.endpoint if model else "", "unpinned_ok": bool(model.unpinned_ok) if model else False,
            "provider_preferences": None, "calls": 0, "served_by": {}, "served_models": {}, "unknown_provider": 0,
            "served_elsewhere": 0, "pinned_record": None, "snapshot": None}


def routing_summary(cfg: Config, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    specs = {m.id: m for m in cfg.models.agents + [cfg.models.judge]}
    per: dict[str, dict[str, Any]] = {}
    for mid, m in specs.items():
        d = _empty(m, mid)
        d["provider_preferences"] = cfg.routing.provider_preferences(m)
        p = latest_snapshot(run_dir, m)
        if p is not None:
            try:
                payload = load_snapshot(p)
                rec = find_endpoint(payload.get("data") or {}, m.endpoint) if m.endpoint else None
                d["snapshot"] = str(p)
                d["snapshot_mock"] = bool(payload.get("mock"))
                d["pinned_record"] = endpoint_summary(rec) if rec else None
            except Exception:
                pass
        per[mid] = d
    ledger = run_dir / "ledger.jsonl"
    if ledger.exists():
        with open(ledger, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("status") != "ok":
                    continue
                mid = str(e.get("model"))
                d = per.get(mid)
                if d is None:
                    d = per[mid] = _empty(None, mid)
                d["calls"] += 1
                sp = e.get("served_provider")
                if sp:
                    d["served_by"][sp] = d["served_by"].get(sp, 0) + 1
                    spec = specs.get(mid)
                    if spec is not None and served_matches_pin(sp, spec, d.get("pinned_record")) is False:
                        d["served_elsewhere"] += 1
                else:
                    d["unknown_provider"] += 1
                sm = e.get("served_model")
                if sm:
                    d["served_models"][sm] = d["served_models"].get(sm, 0) + 1
    out: dict[str, Any] = {"mode": cfg.run.mode, "routing": asdict(cfg.routing), "models": per}
    for name, key in ((PREFLIGHT_FILE, "preflight"), (PROBE_FILE, "probe")):
        p = run_dir / name
        if p.exists():
            try:
                out[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                out[key] = None
    return out


def routing_notes(summary: dict[str, Any]) -> list[str]:
    """One line per model for summary.txt: the pin and who actually served the calls."""
    lines = []
    for mid, d in summary.get("models", {}).items():
        pin = d.get("endpoint") or "NOT PINNED (default routing)"
        rec = d.get("pinned_record") or {}
        detail = f" ({rec.get('provider_name')}, quantization {rec.get('quantization') or 'unknown'})" if rec else ""
        served = ", ".join(f"{k} {v}" for k, v in sorted(d["served_by"].items(), key=lambda kv: -kv[1])) or "none recorded"
        line = f"routing: {mid} pinned to {pin}{detail}: {d['calls']} calls, served by {served}"
        if d.get("unknown_provider"):
            line += f"; {d['unknown_provider']} with no provider recorded"
        if d.get("served_elsewhere"):
            line += f"; {d['served_elsewhere']} SERVED BY ANOTHER PROVIDER THAN THE PIN"
        if d.get("snapshot_mock"):
            line += " [mock endpoint list: dry run]"
        lines.append(line)
    return lines


def write_routing_summary(cfg: Config, run_dir: Path, results_dir: Path) -> dict[str, Any]:
    s = routing_summary(cfg, run_dir)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "routing.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    return s


# --------------------------------------------------------------------------------------------
# The mock endpoint list (dry runs)
# --------------------------------------------------------------------------------------------

def mock_endpoint_list(model: ModelSpec, cfg: Config) -> dict[str, Any]:
    """A well-formed fake endpoint list for dry runs: the pinned endpoint (or "mock") at the config's prices,
    served by "Mock", plus a cheaper low-precision one, so the pre-flight runs the same code path as live."""
    price = cfg.cost.prices.get(model.id)

    def rec(tag: str, provider: str, quant: str, prompt: float, completion: float, cache: float) -> dict[str, Any]:
        return {"name": f"{provider} | {model.id}-mock", "model_id": model.id, "model_name": f"Mock: {model.id}",
                "context_length": 1_048_576,
                "pricing": {"prompt": f"{prompt / 1e6:.12f}", "completion": f"{completion / 1e6:.12f}",
                            "input_cache_read": f"{cache / 1e6:.12f}", "discount": 0},
                "provider_name": provider, "tag": tag, "service_tier": None, "quantization": quant,
                "max_completion_tokens": 393_216, "max_prompt_tokens": None,
                "supported_parameters": ["max_tokens", "temperature", "top_p", "seed", "stop", "response_format"],
                "status": 0, "uptime_last_30m": 100, "supports_implicit_caching": True}

    pinned = rec(model.endpoint or "mock", "Mock", "fp8", price.input_per_m if price else 0.0,
                 price.output_per_m if price else 0.0, price.cached_input_per_m if price else 0.0)
    other = rec("mock-cheap/fp4", "Mock Cheap", "fp4", 0.006, 0.5, 0.005)
    return {"id": model.id, "name": f"Mock: {model.id}", "created": 0, "description": "mock endpoint list (dry run)",
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "endpoints": [pinned, other]}


# --------------------------------------------------------------------------------------------
# The pre-flight: every model's list fetched, pin checked, snapshot saved and diffed
# --------------------------------------------------------------------------------------------

def model_roles(cfg: Config) -> list[tuple[str, ModelSpec]]:
    return [("agent", m) for m in cfg.models.agents] + [("judge", cfg.models.judge)]


async def run_preflight(client, cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Fetch the endpoint list of every configured model through ``client`` (the mock in dry runs), check each
    pin, keep the list with the run and diff the pinned record against the newest earlier snapshot of the same
    model under runs_dir. Writes runs/<name>/routing_preflight.json and returns it; ``ok`` says whether every
    pin can be used."""
    if not cfg.routing.preflight:
        log.warning("routing.preflight is off: the pins are not checked against OpenRouter's endpoint lists")
        return {"skipped": True, "ok": True, "models": {}}
    run_dir = Path(run_dir)
    mock = bool(client.is_mock)
    report: dict[str, Any] = {"ts": time.time(), "fetched_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                              "mode": cfg.run.mode, "mock": mock, "ok": True, "routing": asdict(cfg.routing), "models": {}}
    for role, m in model_roles(cfg):
        data = await client.fetch_endpoints(m.id)
        path = save_snapshot(run_dir, m, data, mock=mock)
        check = check_pin(m, data, cfg, role=role)
        prev = previous_snapshots(cfg.runs_root, m, mock=mock, exclude=path)
        diff = None
        if prev:
            try:
                diff = diff_pinned(load_snapshot(prev[-1]), load_snapshot(path))
            except Exception as exc:
                log.warning("could not diff %s against %s: %s", path.name, prev[-1], exc)
        report["models"][m.id] = {"role": role, "endpoint": m.endpoint, "unpinned_ok": m.unpinned_ok, "ok": check.ok,
                                  "problems": check.problems, "warnings": check.warnings, "record": check.record,
                                  "available": check.available, "snapshot": str(path),
                                  "previous_snapshot": str(prev[-1]) if prev else None, "diff": diff}
        report["ok"] = bool(report["ok"] and check.ok)
        where = describe_endpoint(check.record) if check.record else (m.endpoint or "NOT PINNED")
        log.info("%s %s -> %s%s", role, m.id, where, " [mock list]" if mock else "")
        for w in check.warnings:
            log.warning("%s %s: %s", role, m.id, w)
        for p in check.problems:
            log.error("%s %s: %s", role, m.id, p)
        if diff is not None:
            serious = bool(diff.get("changed")) or not diff.get("pinned_is_listed")
            (log.warning if serious else log.info)("%s %s: %s", role, m.id, describe_diff(diff))
    (run_dir / PREFLIGHT_FILE).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def preflight_problems(report: dict[str, Any]) -> list[str]:
    return [f"{mid}: {p}" for mid, e in (report.get("models") or {}).items() for p in e.get("problems", [])]


def preflight_records(report: dict[str, Any]) -> dict[str, dict | None]:
    return {mid: e.get("record") for mid, e in (report.get("models") or {}).items()}


def render_preflight(report: dict[str, Any], *, all_endpoints: bool = False) -> str:
    """The pre-flight as text for the console: the pin of every model, problems, warnings and changes."""
    if report.get("skipped"):
        return "provider pins: the pre-flight check is off (routing.preflight = false)"
    source = "MOCK endpoint lists (dry run)" if report.get("mock") else f"OpenRouter endpoint lists fetched {report.get('fetched_at_utc')}"
    lines = [f"provider pins ({source}):"]
    for mid, e in (report.get("models") or {}).items():
        head = f"  {e.get('role')} {mid}: "
        if e.get("record"):
            lines.append(head + "pinned to " + describe_endpoint(e["record"]))
        else:
            lines.append(head + (f"pinned to {e['endpoint']!r}" if e.get("endpoint") else "NOT PINNED (default routing)"))
        for p in e.get("problems", []):
            lines.append(f"    PROBLEM: {p}")
        for w in e.get("warnings", []):
            lines.append(f"    warning: {w}")
        if e.get("diff"):
            lines.append(f"    {describe_diff(e['diff'])}")
        if all_endpoints:
            lines.append(f"    all {len(e.get('available') or [])} endpoints (* = pinned):")
            for s in e.get("available") or []:
                lines.append(f"      {'* ' if s.get('tag') == e.get('endpoint') else '  '}{describe_endpoint(s)}")
        lines.append(f"    snapshot: {e.get('snapshot')}")
    r = report.get("routing") or {}
    lines.append(f"  every request carries provider = {{only: [pin], allow_fallbacks: {str(r.get('allow_fallbacks')).lower()}, "
                 f"require_parameters: {str(r.get('require_parameters')).lower()}, data_collection: {r.get('data_collection')!r}}}")
    if r.get("data_collection") == "deny":
        lines.append("  data_collection = \"deny\": providers' data policies are not in the endpoint lists, so this is checked "
                     "by the probe (a 404 there means the pinned provider is excluded; set \"allow\" knowingly or pin another)")
    lines.append("  " + ("every pin can be used" if report.get("ok") else "A PIN CANNOT BE USED AS CONFIGURED: see the problems above"))
    return "\n".join(lines)
