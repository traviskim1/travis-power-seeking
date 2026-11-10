import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aupower.config import config_from_dict  # noqa: E402
from aupower.tasks import TaskIndex, load_tasks  # noqa: E402


@pytest.fixture(scope="session")
def tasks() -> TaskIndex:
    return TaskIndex(load_tasks(ROOT / "vendor" / "odysseys" / "data" / "odysseys.json"))


@pytest.fixture
def dry_cfg(tmp_path):
    """A small dry-run configuration writing under a temporary directory."""
    def make(**overrides):
        raw = {
            "run": {"mode": "dry", "name": "t", "runs_dir": str(tmp_path / "runs"), "max_concurrent_rollouts": 3},
            "experiment": {"levels": ["easy"], "n_trajectories": 1, "state_fractions": [0.5], "n_aux_goals": 3,
                           "rollouts_per_goal": 2, "max_steps": 8, "baseline_all_tasks": False, "fresh_context_forks": True},
            "cost": {"spending_cap_usd": 50.0},
            # mock calls charge their estimated price here so the tests exercise the ledger and the budget stop
            "mock": {"steps_mean": 5, "item_met_prob": 0.7, "charge_estimated_cost": True},
        }
        for k, v in overrides.items():
            raw.setdefault(k, {}).update(v)
        return config_from_dict(raw)
    return make


# --- run `async def` tests without pytest-asyncio ---
import asyncio  # noqa: E402
import inspect  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run the coroutine test with asyncio.run")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None
