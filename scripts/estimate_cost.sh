#!/usr/bin/env bash
# Predicted token use and cost of the smoke test and the initial run, checked against what is left of the budget.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
"$PY" -m aupower estimate --profile smoke "$@" || true
echo
"$PY" -m aupower estimate --profile initial "$@" || true
