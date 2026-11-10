"""Configuration: one TOML file (config.toml), optional profile, optional command-line overrides.

This module is also the only place that reads the OpenRouter API key. Every model call goes through
``aupower.client``, which gets the key from :func:`get_openrouter_api_key` here. The key is read from
the ``OPENROUTER_API_KEY`` environment variable, which :func:`load_dotenv` fills from the project's
``.env`` file. No key is ever stored in code.
"""
from __future__ import annotations

import copy
import json
import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
ENV_PATH = PROJECT_ROOT / ".env"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


# --------------------------------------------------------------------------------------------
# API key
# --------------------------------------------------------------------------------------------

def load_dotenv(path: Path = ENV_PATH) -> dict[str, str]:
    """Load KEY=VALUE lines from ``path`` into ``os.environ`` (without overriding existing values).

    Returns the variables that were read. Missing file -> nothing happens.
    """
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def get_openrouter_api_key() -> str | None:
    """The OpenRouter key from the environment (filled from .env). ``None`` when no key is present."""
    load_dotenv()
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    return key or None


# --------------------------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------------------------

@dataclass
class RunConfig:
    mode: str = "dry"
    name: str = "dev"
    browser: str = "auto"
    headless: bool = True
    viewport: list[int] = field(default_factory=lambda: [1280, 800])
    max_concurrent_rollouts: int = 4
    max_concurrent_judge_calls: int = 8
    history_screenshots: int = 3
    page_text_chars: int = 4000
    max_elements: int = 150
    tasks_path: str = "vendor/odysseys/data/odysseys.json"
    runs_dir: str = "runs"
    retry_failed_rollouts: bool = True
    action_timeout_s: float = 30.0
    screenshot_format: str = "jpeg"     # Playwright screenshots: "jpeg" (smaller judge payloads) or "png"
    screenshot_quality: int = 85

    @property
    def effective_browser(self) -> str:
        if self.browser == "auto":
            return "mock" if self.mode == "dry" else "playwright"
        return self.browser


@dataclass
class ModelSpec:
    id: str
    endpoint: str = ""                 # the OpenRouter endpoint every call to this model is pinned to: a tag from the
                                       # model's endpoint list ("provider_slug" or "provider_slug/variant", see
                                       # `python -m aupower endpoints`). "" = not pinned: refused in live mode unless
                                       # unpinned_ok is set, because default routing picks a different provider, and
                                       # possibly different weights, for every request (see README, "Provider pinning").
    unpinned_ok: bool = False          # the explicit, visible opt-out: let this model run live on default routing
    supports_images: bool = True
    temperature: float | None = None   # None: send no temperature, so the provider default is used
    max_tokens: int = 4096

    @property
    def provider_slug(self) -> str:
        """The provider part of the pinned endpoint tag ("deepinfra/fp8" -> "deepinfra")."""
        return self.endpoint.split("/", 1)[0] if self.endpoint else ""

    @property
    def slug(self) -> str:
        """Filesystem-safe name."""
        return self.id.replace("/", "__").replace(":", "_")


@dataclass
class ModelsConfig:
    agents: list[ModelSpec]
    judge: ModelSpec
    t_reference_model: str = ""

    @property
    def t_reference(self) -> ModelSpec:
        if self.t_reference_model:
            for m in self.agents:
                if m.id == self.t_reference_model:
                    return m
            raise ValueError(f"t_reference_model {self.t_reference_model!r} is not one of the agent models")
        return self.agents[0]


@dataclass
class ExperimentConfig:
    levels: list[str] = field(default_factory=lambda: ["easy"])
    n_trajectories: int = 5
    state_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    n_aux_goals: int = 30
    rollouts_per_goal: int = 3
    max_steps: int = 200
    step_limit_fractions: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 0.80, 0.90, 0.95, 1.00])
    fresh_context_forks: bool = True
    baseline_all_tasks: bool = True
    trajectory_task_ids: list[str] = field(default_factory=list)
    excluded_task_ids: list[str] = field(default_factory=list)
    start_url: str = ""


@dataclass
class SeedsConfig:
    base: int = 20261105


@dataclass
class JudgeConfig:
    h_scoring_mode: str = "rescore"
    t_method: str = "first_step"
    max_images: int = 0


@dataclass
class Pricing:
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float = 0.0


@dataclass
class PlanningAssumptions:
    steps_per_rollout: float = 50
    input_tokens_per_step: float = 25000
    output_tokens_per_step: float = 1500
    cache_fraction: float = 0.8
    judge_input_tokens_full_trajectory: float = 20000
    judge_output_tokens_per_call: float = 300
    image_tokens: float = 1000


@dataclass
class CostConfig:
    spending_cap_usd: float = 25.0
    prices: dict[str, Pricing] = field(default_factory=dict)
    planning: PlanningAssumptions = field(default_factory=PlanningAssumptions)

    def price_for(self, model_id: str) -> Pricing:
        if model_id in self.prices:
            return self.prices[model_id]
        raise KeyError(f"No price configured for model {model_id!r}; add it under [cost.prices] in config.toml")


@dataclass
class RoutingConfig:
    """Provider pinning for OpenRouter (SKILL.md, "use-openrouter-safely"). These fields and each model's
    `endpoint` make up the `provider` object sent with every request; the rest controls the checks around it."""
    allow_fallbacks: bool = False       # a pinned endpoint that is down is an error, never a silent switch of weights
    require_parameters: bool = True     # route only to an endpoint that honours every parameter sent (none dropped)
    data_collection: str = "deny"       # "deny": only providers that do not retain or train on prompts; "allow" otherwise
    preflight: bool = True              # before a run: fetch the endpoint lists, verify the pins, save the snapshot, diff
    probe: bool = True                  # live runs: one minimal paid call per model through its pin before any rollout
    provenance_lookup: bool = True      # when a response does not name its provider, ask GET /api/v1/generation for it

    def provider_preferences(self, model: "ModelSpec") -> dict[str, Any]:
        """The `provider` object for a request to `model`."""
        prefs: dict[str, Any] = {"require_parameters": self.require_parameters, "data_collection": self.data_collection}
        if model.endpoint:
            prefs["only"] = [model.endpoint]
            prefs["allow_fallbacks"] = self.allow_fallbacks
        return prefs


@dataclass
class MockConfig:
    steps_mean: float = 12
    item_met_prob: float = 0.6
    charge_estimated_cost: bool = False   # dry runs: charge mock calls their estimated price against the budget


@dataclass
class Config:
    run: RunConfig
    models: ModelsConfig
    experiment: ExperimentConfig
    seeds: SeedsConfig
    judge: JudgeConfig
    cost: CostConfig
    mock: MockConfig
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    profile: str | None = None
    source_path: str = str(DEFAULT_CONFIG_PATH)

    # ---- derived paths ----
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def tasks_path(self) -> Path:
        p = Path(self.run.tasks_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def runs_root(self) -> Path:
        p = Path(self.run.runs_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        e = self.experiment
        if self.run.mode not in ("dry", "live"):
            raise ValueError("run.mode must be 'dry' or 'live'")
        if self.run.browser not in ("auto", "mock", "playwright"):
            raise ValueError("run.browser must be 'auto', 'mock' or 'playwright'")
        if self.run.screenshot_format not in ("jpeg", "png"):
            raise ValueError("run.screenshot_format must be 'jpeg' or 'png'")
        if not self.models.agents:
            raise ValueError("models.agents must list at least one model")
        for lvl in e.levels:
            if lvl not in ("easy", "medium", "hard"):
                raise ValueError(f"unknown difficulty level {lvl!r}")
        if e.n_aux_goals < 1 or e.rollouts_per_goal < 1 or e.max_steps < 1 or e.n_trajectories < 1:
            raise ValueError("n_aux_goals, rollouts_per_goal, max_steps and n_trajectories must be >= 1")
        for f in e.state_fractions:
            if not 0 < f < 1:
                raise ValueError("state_fractions must lie strictly between 0 and 1")
        for p in e.step_limit_fractions:
            if not 0 < p <= 1:
                raise ValueError("step_limit_fractions must lie in (0, 1]")
        if self.judge.h_scoring_mode not in ("rescore", "first_step"):
            raise ValueError("judge.h_scoring_mode must be 'rescore' or 'first_step'")
        if self.judge.t_method not in ("first_step", "bisect"):
            raise ValueError("judge.t_method must be 'first_step' or 'bisect'")
        if self.cost.spending_cap_usd <= 0:
            raise ValueError("cost.spending_cap_usd must be positive")
        if self.routing.data_collection not in ("allow", "deny"):
            raise ValueError("routing.data_collection must be 'allow' or 'deny'")
        for m in self.models.agents + [self.models.judge]:
            self.cost.price_for(m.id)
            if self.run.mode == "live" and not m.endpoint and not m.unpinned_ok:
                raise ValueError(
                    f"model {m.id!r} has no `endpoint` pin. Live calls must name one OpenRouter endpoint "
                    f"(endpoint = \"provider_slug/variant\" from `python -m aupower endpoints`), or set "
                    f"unpinned_ok = true on the model to run it on default routing knowingly (README, 'Provider pinning')")
        _ = self.models.t_reference


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str, like: Any) -> Any:
    """Parse a command-line override string to the type of the existing value."""
    if isinstance(like, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, (list, dict)):
        return json.loads(value)
    return value


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` overrides (dotted paths; values coerced to the existing type)."""
    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        path, _, value = item.partition("=")
        keys = path.strip().split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        last = keys[-1]
        node[last] = _coerce(value.strip(), node.get(last)) if last in node else _coerce(value.strip(), None)
    return out


def _build(raw: dict, profile: str | None, source_path: Path) -> Config:
    models_raw = raw.get("models", {})
    agents = [ModelSpec(**m) for m in models_raw.get("agents", [])]
    judge = ModelSpec(**models_raw.get("judge", {"id": "google/gemini-3.1-flash-lite-preview"}))
    cost_raw = dict(raw.get("cost", {}))
    prices = {mid: Pricing(**p) for mid, p in cost_raw.pop("prices", {}).items()}
    planning = PlanningAssumptions(**cost_raw.pop("planning", {}))
    cfg = Config(
        run=RunConfig(**raw.get("run", {})),
        models=ModelsConfig(agents=agents, judge=judge, t_reference_model=models_raw.get("t_reference_model", "")),
        experiment=ExperimentConfig(**raw.get("experiment", {})),
        seeds=SeedsConfig(**raw.get("seeds", {})),
        judge=JudgeConfig(**raw.get("judge", {})),
        cost=CostConfig(prices=prices, planning=planning, **cost_raw),
        mock=MockConfig(**raw.get("mock", {})),
        routing=RoutingConfig(**raw.get("routing", {})),
        profile=profile,
        source_path=str(source_path),
    )
    cfg.validate()
    return cfg


def load_config(path: Path | str | None = None, profile: str | None = None,
                overrides: list[str] | None = None) -> Config:
    """Load config.toml, merge ``[profiles.<profile>]`` over it, then apply overrides."""
    source = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(source, "rb") as f:
        raw = tomllib.load(f)
    profiles = raw.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise KeyError(f"profile {profile!r} not found in {source} (have: {sorted(profiles)})")
        raw = _deep_merge(raw, profiles[profile])
    if overrides:
        raw = apply_overrides(raw, list(overrides))
    return _build(raw, profile, source)


def config_from_dict(raw: dict, profile: str | None = None) -> Config:
    """Build a Config from an already-merged dict (used by tests)."""
    base = tomllib.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    base.pop("profiles", None)
    return _build(_deep_merge(base, raw), profile, DEFAULT_CONFIG_PATH)
