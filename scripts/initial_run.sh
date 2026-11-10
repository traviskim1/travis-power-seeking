#!/usr/bin/env bash
# The initial run as configured in [profiles.initial] in config.toml: easy tasks, DeepSeek V4.1 Flash, 3 trajectories
# x 1 state (halfway), 30 goals x 1 rollout, full context only. SPEC.md section 8's full design (5 trajectories x 3
# states, 3 rollouts per goal, full and fresh context) is the base [experiment] setting: `python -m aupower run
# --mode live` with no --profile. Live mode needs OPENROUTER_API_KEY in .env; it prints the estimate and the provider pins
# (checked against OpenRouter's live endpoint lists), asks for confirmation, then probes each pin with one tiny call.
# `--trajectory 1` (then 2, then 3) runs one trajectory at a time, reusing everything already done (README, "Running
# it in stages"); the analysis after each stage covers every trajectory that has data.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
exec "$PY" -m aupower run --profile initial --mode live "$@"
