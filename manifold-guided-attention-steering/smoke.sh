#!/usr/bin/env bash
# smoke.sh — runs the MAGS code path at smoke size (CPU, couple of minutes).
# NOT evidence about the paper. Prints exactly one FINAL line.
# See smoke.py for what it exercises (fit -> steer -> grade on a tiny open model).
set -euo pipefail
cd "$(dirname "$0")"
export MAGS_SMOKE_MODEL="${MAGS_SMOKE_MODEL:-distilgpt2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
exec .venv/bin/python -m smoke
