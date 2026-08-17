#!/usr/bin/env bash
# Smoke: the SAME code path at a size that finishes in ~2 minutes, printing exactly one
# FINAL line. This proves the path runs; it is NOT evidence about the paper, so its
# output is never reported as a result.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MIDSTEER_SMOKE=1

"$PY" "$REPO/experiments/run_e1_synth.py"
