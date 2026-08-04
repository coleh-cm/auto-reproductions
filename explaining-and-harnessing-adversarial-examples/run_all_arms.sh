#!/usr/bin/env bash
# run_all_arms.sh — run every arm at the paper's full configuration at every
# seed and write measured.json. Each arm prints one FINAL line.
#
# CPU sub-scale: this run uses a capped epoch budget (see run_all_arms.py);
# REPRODUCTION.md records the gap. Heavy arms that cannot finish are marked
# BLOCKED in measured.json (honest blocker, never a silent synthetic fallback).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" run_all_arms.py "$@"
