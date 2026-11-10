#!/usr/bin/env bash
# SPEC.md section 8 smoke test: one trajectory, two sampled states, one rollout per goal, no fresh-context forks.
# Live mode: needs OPENROUTER_API_KEY in .env. Prints the cost estimate and asks for confirmation (pass --yes to skip).
# Stops by itself when real spend reaches cost.spending_cap_usd ($60), keeping everything collected so far.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
exec "$PY" -m aupower run --profile smoke --mode live "$@"
