#!/usr/bin/env bash
# Full pipeline in dry-run mode (mock model, mock browser, no network, no cost). Default profile: smoke.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
PROFILE="${1:-smoke}"; shift || true
exec "$PY" -m aupower run --profile "$PROFILE" --mode dry --name "dry_$PROFILE" "$@"
