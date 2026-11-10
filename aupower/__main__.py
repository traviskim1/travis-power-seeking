"""Command line: python -m aupower <command> [--profile smoke|initial] [--mode dry|live] [--set k=v ...]"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .client import OpenRouterClient, SpendingCapExceeded, budget_status, make_client, make_ledger
from .config import Config, load_config, get_openrouter_api_key, API_KEY_ENV_VAR
from .costs import CostEstimate, estimate, measure_from_ledger
from .pipeline import Pipeline, STAGES, SPENDING_STAGES, build_plan, planned_work, resolve_trajectories
from .record import API_LOG_NAME, log_invocation, read_calls
from .routing import RoutingError, render_preflight, run_preflight

EXIT_ABORTED = 1          # the confirmation was declined
EXIT_ERROR = 3
EXIT_BUDGET_STOP = 4      # the run ended because the budget (cost.spending_cap_usd) ran out; partial results saved
EXIT_ROUTING = 5          # a provider pin cannot be used as configured, or the probe through it failed; nothing was run


def _setup_logging(run_dir: Path | None) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="path to config.toml (default: the project's)")
    p.add_argument("--profile", default=None, help="profile from [profiles.*] in config.toml, e.g. smoke or initial")
    p.add_argument("--mode", choices=["dry", "live"], default=None, help="dry = mock model + mock browser; live = real calls")
    p.add_argument("--name", default=None, help="run name (runs/<name>)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value")
    p.add_argument("--trajectory", action="append", default=[], metavar="N|ID",
                   help="run (and judge) only these trajectories of the plan: 1-based positions (1, 2, ...) or task-id "
                        "prefixes; repeat or separate with commas. Finished work is reused; the analysis always covers "
                        "every trajectory that has data")


def _load(args: argparse.Namespace) -> Config:
    overrides = list(args.set)
    if args.mode:
        overrides.append(f"run.mode={args.mode}")
    if args.name:
        overrides.append(f"run.name={args.name}")
    return load_config(args.config, args.profile, overrides)


def _print_estimate(cfg: Config, from_ledger: str | None = None, trajectories: list[str] | None = None) -> CostEstimate:
    """The predicted cost of what this command still has to do: the selected trajectories (or all), minus the
    rollouts and judge results already under runs/<name>."""
    tasks = TaskIndex(load_tasks(cfg.tasks_path))
    measured = measure_from_ledger(Path(from_ledger)) if from_ledger else None
    status = budget_status(cfg)
    plan = build_plan(cfg, tasks)
    try:
        selected = resolve_trajectories(plan, trajectories)
    except ValueError as exc:
        sys.exit(str(exc))
    work = planned_work(cfg, tasks, plan, selected, cfg.run_dir)
    est = estimate(cfg, tasks, measured, spent_usd=status["total_spent_usd"], spent_detail=status,
                   counts_override=work["counts_override"])
    print(est.render())
    if measured:
        print(f"  (assumptions replaced by measurements from {from_ledger}: {json.dumps({k: round(v, 2) for k, v in measured.items()})})")
    return est


def _live_estimate(cfg: Config, trajectories: list[str] | None = None) -> CostEstimate | None:
    """Live mode only: refuse without a key or with the budget already used up; print the estimate; warn when
    the prediction exceeds what is left of the budget (the run will stop there, keeping what it has)."""
    if cfg.run.mode != "live":
        return None
    if not get_openrouter_api_key():
        sys.exit(f"No {API_KEY_ENV_VAR} found in {cfg.project_root / '.env'}; add it (see .env.example) or run with --mode dry.")
    est = _print_estimate(cfg, trajectories=trajectories)
    if est.remaining_usd <= 0:
        sys.exit(f"Refusing to start: the budget ${est.cap_usd:,.2f} is already used up (${est.spent_usd:,.2f} recorded "
                 f"under {cfg.runs_root}). Raise cost.spending_cap_usd in config.toml to continue.")
    if not est.within_cap:
        print(f"WARNING: the predicted cost ${est.total_usd:,.2f} exceeds the remaining budget ${est.remaining_usd:,.2f}. "
              "The run will stop when spend reaches the budget and keep everything collected until then.")
    return est


def _confirm(cfg: Config, est: CostEstimate | None, yes: bool) -> bool:
    """Live mode: the y/N prompt, asked after the estimate and the pins have been shown."""
    if cfg.run.mode != "live" or yes or est is None:
        return True
    if not sys.stdin.isatty():
        print("Live mode needs confirmation: re-run with --yes to accept the estimate and the pins above.", file=sys.stderr)
        return False
    reply = input(f"This will make paid API calls through the pinned endpoints above (predicted ${est.total_usd:,.2f}; "
                  f"budget ${est.cap_usd:,.2f}, remaining ${est.remaining_usd:,.2f}; the run stops at the budget). Proceed? [y/N] ")
    return reply.strip().lower() in ("y", "yes")


def _budget_stop_message(cfg: Config, exc: Exception) -> str:
    run = cfg.run_dir
    stage = None
    try:
        stage = json.loads((run / "budget_stop.json").read_text(encoding="utf-8")).get("stage")
    except Exception:
        pass
    return "\n".join([
        f"BUDGET STOP: {exc}",
        f"The run ended{' during the ' + str(stage) + ' stage' if stage else ''}. Everything collected so far is saved under {run}:",
        f"  partial results   {run / 'results'}  (summary.txt, delta_au.csv, per_goal.csv, T.json, rollouts.csv, routing.json)",
        f"  the stop record   {run / 'budget_stop.json'};  every call with its cost and provider: {run / 'ledger.jsonl'}",
        "To continue, raise cost.spending_cap_usd in config.toml and re-run the same command: finished rollouts and "
        "judge results are reused, rollouts that were cut short are re-run.",
    ])


def _calls(cfg: Config, args: argparse.Namespace) -> int:
    """`calls`: print entries of the run's API call log."""
    path = cfg.run_dir / API_LOG_NAME
    entries = read_calls(path, purpose=args.purpose, kind=args.kind, limit=args.last or None)
    if not entries:
        print(f"no logged calls in {path}" + (" matching the filter" if (args.purpose or args.kind) else ""))
        return 0
    for e in entries:
        if args.full:
            print(json.dumps(e, indent=1, ensure_ascii=False))
            continue
        resp = e.get("response") if isinstance(e.get("response"), dict) else {}
        choice = ((resp.get("choices") or [{}])[0].get("message") or {}) if resp else {}
        text = str(choice.get("content") or "")
        usage = resp.get("usage") or {}
        print(f"{e.get('ts_utc')}  {e.get('kind')}  {e.get('purpose') or ''}  {e.get('model') or ''} via {e.get('endpoint') or '-'}  "
              f"attempt {e.get('attempt')}  status {e.get('status')}  {e.get('latency_s')}s  "
              f"tokens {usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')}  cost {usage.get('cost', '?')}  "
              f"served by {resp.get('provider') if resp else '-'}  id {e.get('request_id') or '-'}")
        if e.get("error"):
            print(f"    error: {e['error']}")
        if text:
            print("    response: " + text[:300].replace("\n", " ") + ("..." if len(text) > 300 else ""))
    print(f"({len(entries)} shown; --full prints whole requests and responses; the file is {path})")
    return 0


def _endpoints(cfg: Config, args: argparse.Namespace) -> int:
    """`endpoints`: fetch OpenRouter's endpoint list for every configured model (public, no key, no cost), check
    the pins, save the snapshot under runs/<name>/endpoints/ and show what changed since the last one. With an
    explicit --mode dry the mock lists are used instead (no network)."""
    _setup_logging(None)
    logging.getLogger().setLevel(logging.WARNING)      # the rendered report below is the output
    ledger = make_ledger(cfg)
    if args.mode == "dry":
        client = make_client(cfg, ledger)
    else:
        client = OpenRouterClient(get_openrouter_api_key() or "", ledger, routing=cfg.routing,
                                  models={m.id: m for m in cfg.models.agents + [cfg.models.judge]})

    async def go() -> dict:
        try:
            return await run_preflight(client, cfg, cfg.run_dir)
        finally:
            await client.close()

    report = asyncio.run(go())
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(render_preflight(report, all_endpoints=args.all))
    return 0 if report.get("ok") else EXIT_ROUTING


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aupower", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run every stage (or --stages) of the method")
    _common(p_run)
    p_run.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    p_run.add_argument("--yes", action="store_true", help="accept the cost estimate and the pins without asking (live mode)")

    for name in STAGES:
        p = sub.add_parser(name, help=f"run only the {name} stage")
        _common(p)
        p.add_argument("--yes", action="store_true")

    p_est = sub.add_parser("estimate", help="predict token use and cost; check against what is left of the budget")
    _common(p_est)
    p_est.add_argument("--from-ledger", default=None, help="replace planning assumptions with measurements from a ledger.jsonl")

    p_led = sub.add_parser("ledger", help="show the spend recorded in a run's ledger and where the budget stands")
    _common(p_led)

    p_calls = sub.add_parser("calls", help="read a run's API call log back (api_calls.jsonl.gz): every request and response")
    _common(p_calls)
    p_calls.add_argument("--purpose", default=None, help="only calls with this purpose (agent_step, judge_rescore, judge_first_step, routing_probe)")
    p_calls.add_argument("--kind", default=None, help="chat, generation or endpoints")
    p_calls.add_argument("--last", type=int, default=5, help="how many of the newest calls to print (default 5; 0 = all)")
    p_calls.add_argument("--full", action="store_true", help="print the whole request and response, not a summary")

    p_ep = sub.add_parser("endpoints", help="fetch OpenRouter's endpoint lists, check the provider pins, save a snapshot "
                                            "(no key, no cost; --mode dry uses the mock lists)")
    _common(p_ep)
    p_ep.add_argument("--all", action="store_true", help="list every endpoint of each model, not only the pinned one")
    p_ep.add_argument("--json", action="store_true", help="print the full report as JSON")

    args = parser.parse_args(argv)
    cfg = _load(args)

    if args.command == "estimate":
        _setup_logging(None)
        est = _print_estimate(cfg, args.from_ledger, args.trajectory)
        return 0 if est.within_cap else 2

    if args.command == "ledger":
        print(json.dumps(make_ledger(cfg).summary(), indent=1))
        return 0

    if args.command == "endpoints":
        return _endpoints(cfg, args)

    if args.command == "calls":
        return _calls(cfg, args)

    stages = [s.strip() for s in args.stages.split(",")] if args.command == "run" else [args.command]
    _setup_logging(cfg.run_dir)
    log_invocation(cfg, list(argv) if argv is not None else sys.argv[1:], stages=stages, trajectories=args.trajectory)
    est = _live_estimate(cfg, args.trajectory)
    if cfg.run.mode == "dry":
        rehearsal = (" (mock calls charge their estimated price against the budget: a rehearsal of the budget stop)"
                     if cfg.mock.charge_estimated_cost else "")
        print(f"[dry run] mock model client and {cfg.run.effective_browser} browser; no network calls, no cost{rehearsal}.")
    needs_models = any(s in SPENDING_STAGES for s in stages)
    yes = bool(getattr(args, "yes", False))

    async def go() -> int:
        pipe = Pipeline(cfg, trajectories=args.trajectory)
        try:
            if needs_models:
                report = await pipe.preflight()        # free: endpoint lists, pin checks, snapshot and diff
                print(render_preflight(report))
                if not report.get("ok"):
                    print("Not starting: a provider pin cannot be used as configured (see above; "
                          "`python -m aupower endpoints --all` lists what OpenRouter serves now).", file=sys.stderr)
                    return EXIT_ROUTING
                if not _confirm(cfg, est, yes):
                    print("Aborted.", file=sys.stderr)
                    return EXIT_ABORTED
            await pipe.run_stages(stages)              # live: the routing probe first, then the stages
            return 0
        except RoutingError as exc:
            print(f"ROUTING: {exc}", file=sys.stderr)
            return EXIT_ROUTING
        except SpendingCapExceeded as exc:
            print(_budget_stop_message(cfg, exc), file=sys.stderr)
            return EXIT_BUDGET_STOP
        except Exception as exc:  # already logged
            print(f"run stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        finally:
            await pipe.close()
            print(f"ledger: {json.dumps(pipe.ledger.summary())}")

    return asyncio.run(go())


from .tasks import TaskIndex, load_tasks  # noqa: E402  (after the functions that use them, as before)

if __name__ == "__main__":
    sys.exit(main())
