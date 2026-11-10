"""Model clients and the call ledger.

* :class:`Ledger` logs every model call (tokens, cost, purpose) to ``ledger.jsonl`` and enforces the
  budget (``cost.spending_cap_usd``): a call that would push real spend over the budget is refused before
  it is made, the run stops if recorded spend crosses it, and once tripped every later call is refused.
  Real spend recorded by other runs under runs_dir counts too (:func:`make_ledger`).
* :class:`OpenRouterClient` talks to OpenRouter's OpenAI-compatible chat endpoint. It is the only
  code that performs network calls to a model, and it gets the key from ``aupower.config``.
* :class:`MockClient` returns fake but well-formed agent actions and judge verdicts for dry runs. It
  is deterministic given the seeds, and consistent across judge modes (an item met at step f is
  reported met at every cut >= f).
* :func:`make_client` is the one place that chooses between them.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .config import Config, ModelSpec, Pricing, RoutingConfig, get_openrouter_api_key, API_KEY_ENV_VAR
from .record import API_LOG_NAME, CallLog, redact_images
from .routing import ENDPOINTS_URL, GENERATION_URL, mock_endpoint_list
from .tasks import derive_seed

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class SpendingCapExceeded(RuntimeError):
    """Raised when a call would exceed, or has exceeded, the configured spending cap."""


class ModelCallError(RuntimeError):
    """A model call failed after retries."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: str = "estimated"   # "openrouter" (reported by the API), "estimated" (from prices) or "mock"


@dataclass
class ChatResponse:
    text: str
    usage: Usage
    model: str
    latency_s: float
    finish_reason: str | None = None
    request_id: str | None = None
    served_provider: str | None = None   # who served the call: the response's provider field, else the generation record


# --------------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------------

def count_message_tokens_estimate(messages: list[dict], image_tokens: float) -> tuple[int, int]:
    """Rough token count of a message list before sending: ~4 characters per token plus a fixed
    allowance per image. Returns (text_tokens, n_images)."""
    chars = 0
    images = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    chars += len(part.get("text", ""))
                elif part.get("type") == "image_url":
                    images += 1
    return chars // 4 + int(images * image_tokens), images


class Ledger:
    """Append-only JSONL log of every model call, with the running spend and the budget.

    The budget is real money: ``prior_spent_usd`` is what other runs have already spent, ``spent_usd``
    what this run has spent. :meth:`reserve` refuses a call whose estimated cost would take the total past
    the budget; :meth:`record` stops the run if the recorded spend has crossed it. Either way ``tripped``
    is set and every later call is refused at once, so the run winds down without another paid call.
    Mock entries (dry runs) count only when ``count_mock`` is true.
    """

    def __init__(self, path: Path, cap_usd: float, prices: dict[str, Pricing], image_tokens: float = 1000.0, *,
                 count_mock: bool = True, prior_spent_usd: float = 0.0, prior_by_run: dict[str, float] | None = None):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)
        self.prices = prices
        self.image_tokens = image_tokens
        self.count_mock = count_mock
        self.prior_spent_usd = float(prior_spent_usd)
        self.prior_by_run = dict(prior_by_run or {})
        self.spent_usd = 0.0
        self.pending_usd = 0.0
        self.n_calls = 0
        self.by_purpose: dict[str, dict[str, float]] = {}
        self.tripped: str | None = None    # the reason, from the first time the budget refused or stopped a call
        self._load_existing()

    # ---- persistence ----
    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._tally(e)

    def _tally(self, e: dict) -> None:
        if e.get("mock") and not self.count_mock:
            return
        self.n_calls += 1
        self.spent_usd += float(e.get("cost_usd") or 0.0)
        p = self.by_purpose.setdefault(e.get("purpose", "?"), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                              "cached_tokens": 0, "cost_usd": 0.0})
        p["calls"] += 1
        p["prompt_tokens"] += int(e.get("prompt_tokens") or 0)
        p["completion_tokens"] += int(e.get("completion_tokens") or 0)
        p["cached_tokens"] += int(e.get("cached_tokens") or 0)
        p["cost_usd"] += float(e.get("cost_usd") or 0.0)

    # ---- pricing ----
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        p = self.prices.get(model)
        if p is None:
            return 0.0
        uncached = max(0, prompt_tokens - cached_tokens)
        return (uncached * p.input_per_m + cached_tokens * p.cached_input_per_m
                + completion_tokens * p.output_per_m) / 1_000_000.0

    # ---- budget ----
    @property
    def total_spent_usd(self) -> float:
        """Spend that counts against the budget: this run plus other runs."""
        return self.prior_spent_usd + self.spent_usd

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.total_spent_usd - self.pending_usd

    def describe(self) -> str:
        s = f"spent ${self.spent_usd:.4f} in this run"
        if self.prior_spent_usd:
            s += f" and ${self.prior_spent_usd:.4f} in other runs ({', '.join(f'{k} ${v:.2f}' for k, v in self.prior_by_run.items())})"
        return s

    def exhausted_message(self) -> str:
        return f"budget ${self.cap_usd:.2f} (cost.spending_cap_usd) is already used up: {self.describe()}"

    def reserve(self, model: str, messages: list[dict], max_tokens: int) -> float:
        """Check the budget before a call. Returns the reserved estimate (released by record())."""
        if self.tripped:
            raise SpendingCapExceeded(self.tripped)
        est_prompt, _ = count_message_tokens_estimate(messages, self.image_tokens)
        est = self.estimate_cost(model, est_prompt, max_tokens)
        if self.total_spent_usd + self.pending_usd + est > self.cap_usd:
            self.tripped = (f"budget ${self.cap_usd:.2f} (cost.spending_cap_usd) reached: {self.describe()}, "
                            f"${self.pending_usd:.4f} in flight, and the next call could cost up to ${est:.4f}")
            raise SpendingCapExceeded(self.tripped)
        self.pending_usd += est
        return est

    def release(self, reserved: float) -> None:
        self.pending_usd = max(0.0, self.pending_usd - reserved)

    def record(self, entry: dict, reserved: float = 0.0) -> None:
        """Append a call to the ledger and update the running spend. Raises if the budget is now exceeded."""
        self.release(reserved)
        entry = {"ts": time.time(), **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._tally(entry)
        if self.total_spent_usd > self.cap_usd:
            self.tripped = (f"budget ${self.cap_usd:.2f} (cost.spending_cap_usd) exceeded: {self.describe()} "
                            f"after {self.n_calls} calls")
            raise SpendingCapExceeded(self.tripped)

    def summary(self) -> dict[str, Any]:
        return {"calls": self.n_calls, "spent_usd": round(self.spent_usd, 6), "cap_usd": self.cap_usd,
                "prior_spent_usd": round(self.prior_spent_usd, 6), "total_spent_usd": round(self.total_spent_usd, 6),
                "remaining_usd": round(self.remaining_usd, 6), "tripped": self.tripped,
                "by_purpose": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                               for k, v in self.by_purpose.items()}}


def prior_spend(runs_root: Path, exclude: Path | None = None) -> dict[str, float]:
    """Real (non-mock) spend recorded by every other run's ledger.jsonl under runs_root, by run name."""
    out: dict[str, float] = {}
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return out
    ex = Path(exclude).resolve() if exclude is not None else None
    for led in sorted(runs_root.glob("*/ledger.jsonl")):
        if ex is not None and led.parent.resolve() == ex:
            continue
        total = 0.0
        with open(led, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("mock"):
                    continue
                total += float(e.get("cost_usd") or 0.0)
        if total > 0:
            out[led.parent.name] = round(total, 6)
    return out


def make_ledger(cfg: Config, path: Path | None = None) -> Ledger:
    """The run's ledger with the budget set up. Live mode: real spend recorded by other runs under runs_dir
    counts too, mock entries never do. Dry mode: nothing is real, so other runs are ignored and this run's
    own mock entries count only when rehearsing the budget stop (mock.charge_estimated_cost)."""
    live = cfg.run.mode == "live"
    prior = prior_spend(cfg.runs_root, exclude=cfg.run_dir) if live else {}
    return Ledger(path or (cfg.run_dir / "ledger.jsonl"), cfg.cost.spending_cap_usd, cfg.cost.prices,
                  cfg.cost.planning.image_tokens, count_mock=(not live) and cfg.mock.charge_estimated_cost,
                  prior_spent_usd=sum(prior.values()), prior_by_run=prior)


def budget_status(cfg: Config) -> dict[str, Any]:
    """Where the budget stands before a run: what this run and other runs have spent, and what is left."""
    led = make_ledger(cfg)
    return {"cap_usd": led.cap_usd, "spent_this_run_usd": round(led.spent_usd, 6),
            "spent_other_runs_usd": round(led.prior_spent_usd, 6), "other_runs": led.prior_by_run,
            "total_spent_usd": round(led.total_spent_usd, 6), "remaining_usd": round(led.remaining_usd, 6)}


# --------------------------------------------------------------------------------------------
# Client interface
# --------------------------------------------------------------------------------------------

class ModelClient(ABC):
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    @property
    @abstractmethod
    def is_mock(self) -> bool: ...

    @abstractmethod
    async def chat(self, *, model: str, messages: list[dict], max_tokens: int, temperature: float | None,
                   purpose: str, meta: dict | None = None) -> ChatResponse: ...

    @abstractmethod
    async def fetch_endpoints(self, model_id: str) -> dict:
        """OpenRouter's endpoint list for a model (the ``data`` object of GET /models/{id}/endpoints)."""

    async def close(self) -> None:
        return None


def _ledger_entry(model: str, purpose: str, meta: dict | None, usage: Usage, latency_s: float,
                  status: str, error: str | None = None, mock: bool = False, request_id: str | None = None,
                  served_provider: str | None = None, served_model: str | None = None, endpoint: str | None = None) -> dict:
    return {
        "model": model, "purpose": purpose, "meta": meta or {}, "status": status, "error": error, "mock": mock,
        "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
        "cached_tokens": usage.cached_tokens, "reasoning_tokens": usage.reasoning_tokens,
        "cost_usd": usage.cost_usd, "cost_source": usage.cost_source, "latency_s": round(latency_s, 3),
        "request_id": request_id,
        # provenance (SKILL.md): the endpoint the call was pinned to, and who actually served it
        "endpoint": endpoint, "served_provider": served_provider, "served_model": served_model,
    }


# --------------------------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------------------------

class OpenRouterClient(ModelClient):
    RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, api_key: str, ledger: Ledger, *, timeout_s: float = 300.0, max_retries: int = 5,
                 base_url: str = OPENROUTER_URL, routing: RoutingConfig | None = None,
                 models: dict[str, ModelSpec] | None = None, transport=None, api_log: Path | None = None):
        super().__init__(ledger)
        import httpx  # imported here so dry runs never need network libraries
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0), transport=transport)
        self.routing = routing or RoutingConfig()
        self.models = dict(models or {})     # model id -> spec, for the endpoint pin of each call
        self.call_log = CallLog(api_log) if api_log else None    # every request and response, verbatim (images redacted)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/traviskim1/aupower",
            "X-Title": "aupower",
        }
        self.base_url = base_url
        self.max_retries = max_retries

    @property
    def is_mock(self) -> bool:
        return False

    def provider_preferences(self, model: str) -> dict:
        """The `provider` object sent with every request to `model`: its endpoint pin plus the routing rules."""
        return self.routing.provider_preferences(self.models.get(model) or ModelSpec(id=model))

    async def close(self) -> None:
        await self._http.aclose()

    def _log(self, **kw) -> None:
        if self.call_log is not None:
            try:
                self.call_log.write(**kw)
            except Exception as exc:   # the log must never break a run
                import logging
                logging.getLogger("aupower.client").warning("api call log: %s", exc)

    async def chat(self, *, model: str, messages: list[dict], max_tokens: int, temperature: float | None,
                   purpose: str, meta: dict | None = None) -> ChatResponse:
        reserved = self.ledger.reserve(model, messages, max_tokens)
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "provider": self.provider_preferences(model)}
        if temperature is not None:
            body["temperature"] = temperature
        spec = self.models.get(model)
        endpoint = spec.endpoint if spec else None
        logged_body = {**body, "messages": redact_images(messages)}
        t0 = time.monotonic()
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            t_att = time.monotonic()
            try:
                resp = await self._http.post(self.base_url, headers=self._headers, json=body)
            except Exception as exc:  # network errors
                last_error = f"{type(exc).__name__}: {exc}"
                self._log(kind="chat", purpose=purpose, meta=meta, model=model, endpoint=endpoint, attempt=attempt,
                          request={"url": self.base_url, "body": logged_body}, status=None,
                          latency_s=time.monotonic() - t_att, error=last_error)
                await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
                continue
            latency_att = time.monotonic() - t_att
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    last_error = f"bad JSON: {exc}"
                    self._log(kind="chat", purpose=purpose, meta=meta, model=model, endpoint=endpoint, attempt=attempt,
                              request={"url": self.base_url, "body": logged_body}, status=200, latency_s=latency_att,
                              response=resp.text[:2000], error=last_error)
                    await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
                    continue
                if "error" in data and not data.get("choices"):
                    last_error = f"api error: {data['error']}"
                    self._log(kind="chat", purpose=purpose, meta=meta, model=model, endpoint=endpoint, attempt=attempt,
                              request={"url": self.base_url, "body": logged_body}, status=200, latency_s=latency_att,
                              response=data, error=last_error)
                    await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
                    continue
                latency = time.monotonic() - t0
                choice = (data.get("choices") or [{}])[0]
                text = str((choice.get("message") or {}).get("content") or "")
                usage = self._parse_usage(model, data.get("usage") or {})
                gen_id = data.get("id")
                served = data.get("provider")
                self._log(kind="chat", purpose=purpose, meta=meta, model=model, endpoint=endpoint, attempt=attempt,
                          request={"url": self.base_url, "body": logged_body}, status=200, latency_s=latency_att,
                          response=data, request_id=str(gen_id) if gen_id else None)
                if served is None and self.routing.provenance_lookup and gen_id:
                    served = await self._served_provider(str(gen_id))
                entry = _ledger_entry(model, purpose, meta, usage, latency, "ok", request_id=gen_id,
                                      served_provider=served, served_model=data.get("model"), endpoint=endpoint)
                self.ledger.record(entry, reserved)
                return ChatResponse(text=text, usage=usage, model=str(data.get("model") or model), latency_s=latency,
                                    finish_reason=choice.get("finish_reason"), request_id=gen_id, served_provider=served)
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            try:
                err_body: Any = resp.json()
            except ValueError:
                err_body = resp.text[:2000]
            self._log(kind="chat", purpose=purpose, meta=meta, model=model, endpoint=endpoint, attempt=attempt,
                      request={"url": self.base_url, "body": logged_body}, status=resp.status_code, latency_s=latency_att,
                      response=err_body, error=last_error)
            if resp.status_code in self.RETRY_STATUSES:
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0 ** attempt + random.random())
                await asyncio.sleep(delay)
                continue
            break  # non-retryable (400, 401, 402, 403, 404 ...)
        latency = time.monotonic() - t0
        self.ledger.record(_ledger_entry(model, purpose, meta, Usage(cost_source="estimated"), latency, "error",
                                         error=last_error, endpoint=endpoint), reserved)
        raise ModelCallError(f"OpenRouter call failed for {model} ({purpose}): {last_error}")

    async def _served_provider(self, generation_id: str) -> str | None:
        """OpenRouter's documented route to the provider that served a call: GET /api/v1/generation?id=...
        (its ``provider_name``). The record can lag the response by a moment, hence a few short retries."""
        for attempt in range(1, 5):
            t0 = time.monotonic()
            try:
                resp = await self._http.get(GENERATION_URL, params={"id": generation_id}, headers=self._headers)
                data = resp.json() if resp.status_code == 200 else resp.text[:500]
                self._log(kind="generation", attempt=attempt, request={"url": GENERATION_URL, "params": {"id": generation_id}},
                          status=resp.status_code, latency_s=time.monotonic() - t0, response=data, request_id=generation_id)
                if resp.status_code == 200:
                    name = ((data or {}).get("data") or {}).get("provider_name")
                    if name:
                        return str(name)
            except Exception as exc:
                self._log(kind="generation", attempt=attempt, request={"url": GENERATION_URL, "params": {"id": generation_id}},
                          status=None, latency_s=time.monotonic() - t0, error=f"{type(exc).__name__}: {exc}", request_id=generation_id)
            await asyncio.sleep(0.5 * attempt)
        return None

    async def fetch_endpoints(self, model_id: str) -> dict:
        """OpenRouter's endpoint list for a model (public, no key needed): every provider serving it, with
        its quantization, context, output cap, prices and supported parameters."""
        url = ENDPOINTS_URL.format(model=model_id)
        headers = {k: v for k, v in self._headers.items() if k in ("HTTP-Referer", "X-Title")}
        last_error: str | None = None
        for attempt in range(3):
            try:
                resp = await self._http.get(url, headers=headers)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1.0 + attempt)
                continue
            if resp.status_code == 200:
                try:
                    data = (resp.json() or {}).get("data") or {}
                except ValueError as exc:
                    last_error = f"bad JSON: {exc}"
                    await asyncio.sleep(1.0 + attempt)
                    continue
                if not isinstance(data.get("endpoints"), list):
                    raise ModelCallError(f"OpenRouter's endpoint list for {model_id} has no 'endpoints' array")
                self._log(kind="endpoints", model=model_id, attempt=attempt + 1, request={"url": url}, status=200, latency_s=0.0,
                          response={"endpoints": len(data["endpoints"]), "note": "the full list is saved under endpoints/ by the pre-flight"})
                return data
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            self._log(kind="endpoints", model=model_id, attempt=attempt + 1, request={"url": url}, status=resp.status_code,
                      latency_s=0.0, error=last_error)
            if resp.status_code == 404:
                break
            await asyncio.sleep(1.0 + attempt)
        raise ModelCallError(f"could not fetch OpenRouter's endpoint list for {model_id}: {last_error}")

    def _parse_usage(self, model: str, u: dict) -> Usage:
        prompt = int(u.get("prompt_tokens") or 0)
        completion = int(u.get("completion_tokens") or 0)
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        reasoning = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        if u.get("cost") is not None:
            return Usage(prompt, completion, cached, reasoning, float(u["cost"]), "openrouter")
        return Usage(prompt, completion, cached, reasoning, self.ledger.estimate_cost(model, prompt, completion, cached),
                     "estimated")


# --------------------------------------------------------------------------------------------
# Mock client (dry runs)
# --------------------------------------------------------------------------------------------

_ELEMENT_RE = re.compile(r"^\[(\d+)\] <(\w+)", re.MULTILINE)
_TAB_RE = re.compile(r"^\s*(\d+)\. (?:\(active\) )?", re.MULTILINE)
_RUBRIC_RE = re.compile(r"Rubric ID: (\S+)")
_NSTEPS_RE = re.compile(r"trajectory had (\d+) total step")
_FAKE_HOSTS = ["https://www.shopmock.com", "https://www.wikimock.org", "https://travelmock.net",
               "https://www.newsmock.com", "https://recipesmock.io", "https://www.moviemock.com"]
_WORDS = ["search", "results", "item", "detail", "cart", "login", "article", "compare", "review", "booking"]


class MockClient(ModelClient):
    """Fake model. Agent calls return a plausible browser action; judge calls return verdicts that are
    monotone in the cut-off step and identical between the re-scoring and first-step modes."""

    def __init__(self, ledger: Ledger, cfg: Config):
        super().__init__(ledger)
        self.seed = cfg.seeds.base
        self.steps_mean = max(2.0, float(cfg.mock.steps_mean))
        self.item_met_prob = float(cfg.mock.item_met_prob)
        self.image_tokens = cfg.cost.planning.image_tokens
        self.charge = bool(cfg.mock.charge_estimated_cost)   # dry runs are free unless rehearsing the budget stop
        self.cfg = cfg
        self.models = {m.id: m for m in cfg.models.agents + [cfg.models.judge]}

    @property
    def is_mock(self) -> bool:
        return True

    async def chat(self, *, model: str, messages: list[dict], max_tokens: int, temperature: float | None,
                   purpose: str, meta: dict | None = None) -> ChatResponse:
        meta = meta or {}
        reserved = self.ledger.reserve(model, messages, max_tokens)
        t0 = time.monotonic()
        if purpose == "agent_step":
            text = self._agent_response(messages, meta)
        elif purpose in ("judge_rescore", "judge_first_step", "judge_bisect"):
            text = self._judge_response(messages, meta, purpose)
        else:
            text = "OK"
        prompt_tokens, _ = count_message_tokens_estimate(messages, self.image_tokens)
        completion_tokens = max(1, len(text) // 4)
        cost = self.ledger.estimate_cost(model, prompt_tokens, completion_tokens) if self.charge else 0.0
        usage = Usage(prompt_tokens, completion_tokens, 0, 0, cost, "mock")
        latency = time.monotonic() - t0
        spec = self.models.get(model)
        self.ledger.record(_ledger_entry(model, purpose, meta, usage, latency, "ok", mock=True, served_provider="Mock",
                                         served_model=model, endpoint=spec.endpoint if spec else None), reserved)
        return ChatResponse(text=text, usage=usage, model=model, latency_s=latency, finish_reason="stop", served_provider="Mock")

    async def fetch_endpoints(self, model_id: str) -> dict:
        """A fake but well-formed endpoint list: the model's pin served by "Mock" at the config's prices."""
        return mock_endpoint_list(self.models.get(model_id) or ModelSpec(id=model_id), self.cfg)

    # ---- agent ----
    @staticmethod
    def _last_user_text(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            if isinstance(c, str):
                return c
            return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
        return ""

    def _agent_response(self, messages: list[dict], meta: dict) -> str:
        rollout_id = str(meta.get("rollout_id", "r"))
        step = int(meta.get("step", 1))
        rng = random.Random(derive_seed(self.seed, "mock-agent", rollout_id, step))
        text = self._last_user_text(messages)
        elements = [(int(i), tag) for i, tag in _ELEMENT_RE.findall(text)]
        inputs = [i for i, tag in elements if tag in ("input", "textarea")]
        n_tabs = max(1, text.count("\n  tab "))
        if step >= 2 and rng.random() < 1.0 / self.steps_mean:
            action = {"action": "done", "args": {"answer": f"Mock final answer after {step} steps: the requested "
                                                              f"details were gathered from {rng.choice(_FAKE_HOSTS)}."}}
        else:
            roll = rng.random()
            if roll < 0.45 and elements:
                action = {"action": "click", "args": {"index": rng.choice(elements)[0]}}
            elif roll < 0.60 and inputs:
                action = {"action": "type", "args": {"index": rng.choice(inputs), "text": rng.choice(_WORDS) + " query",
                                                     "press_enter": True}}
            elif roll < 0.78:
                action = {"action": "navigate", "args": {"url": f"{rng.choice(_FAKE_HOSTS)}/{rng.choice(_WORDS)}-{rng.randint(1, 40)}"}}
            elif roll < 0.90:
                action = {"action": "scroll", "args": {"direction": rng.choice(["down", "down", "up"])}}
            elif roll < 0.95:
                action = {"action": "new_tab", "args": {"url": f"{rng.choice(_FAKE_HOSTS)}/{rng.choice(_WORDS)}-{rng.randint(1, 40)}"}}
            elif n_tabs > 1:
                action = {"action": "switch_tab", "args": {"index": rng.randrange(n_tabs)}}
            else:
                action = {"action": "wait", "args": {"seconds": 1}}
        thought = f"[mock step {step}] I will {action['action']} to make progress on the task."
        payload = {"thought": thought, **action}
        return f"{thought}\n\n```json\n{json.dumps(payload)}\n```"

    # ---- judge ----
    def _judge_response(self, messages: list[dict], meta: dict, purpose: str) -> str:
        text = self._last_user_text(messages)
        rubric_id = str(meta.get("rubric_id") or (_RUBRIC_RE.search(text).group(1) if _RUBRIC_RE.search(text) else "R?"))
        shown = int(meta.get("cut") or (_NSTEPS_RE.search(text).group(1) if _NSTEPS_RE.search(text) else 0))
        n_full = int(meta.get("n_steps_full") or shown or 1)
        rollout_id = str(meta.get("rollout_id", "r"))
        rng = random.Random(derive_seed(self.seed, "mock-judge", rollout_id, rubric_id))
        met_ever = rng.random() < self.item_met_prob
        first = rng.randint(1, max(1, n_full)) if met_ever else None
        success = first is not None and first <= shown
        thoughts = (f"[mock judge] Rubric {rubric_id}: the trajectory shows {shown} step(s); the item "
                    + (f"is first satisfied at step {first}." if success else "is not satisfied in these steps."))
        out = f"Thoughts: {thoughts}\nStatus: \"{'success' if success else 'failure'}\""
        if purpose == "judge_first_step":
            out += f"\nFirst satisfied at step: {first if success else 'none'}"
        return out


# --------------------------------------------------------------------------------------------
# Factory: the one place that decides mock vs. live
# --------------------------------------------------------------------------------------------

def make_client(cfg: Config, ledger: Ledger) -> ModelClient:
    if cfg.run.mode == "dry":
        return MockClient(ledger, cfg)
    key = get_openrouter_api_key()
    if not key:
        raise RuntimeError(
            f"run.mode is 'live' but no API key was found. Put {API_KEY_ENV_VAR}=<your key> in "
            f"{cfg.project_root / '.env'} (see .env.example), or use --mode dry.")
    return OpenRouterClient(key, ledger, routing=cfg.routing,
                            models={m.id: m for m in cfg.models.agents + [cfg.models.judge]},
                            api_log=cfg.run_dir / API_LOG_NAME)


def image_data_url(path: Path) -> str:
    data = Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
