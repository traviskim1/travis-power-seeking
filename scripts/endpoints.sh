#!/usr/bin/env bash
# OpenRouter's endpoint lists for the configured models (public, no key, no cost): checks the provider pins,
# saves a snapshot under runs/<name>/endpoints/ and shows what changed since the last one. `--all` lists every
# endpoint of each model; `--profile initial` checks that profile's models; `--mode dry` uses the mock lists.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
exec "$PY" -m aupower endpoints "$@"
