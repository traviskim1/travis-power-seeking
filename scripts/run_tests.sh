#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
exec "$PY" -m pytest -q tests "$@"
