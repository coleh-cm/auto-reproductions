#!/usr/bin/env bash
# smoke.sh — runs the MAGS code path at smoke size (CPU, couple of minutes).
# NOT evidence about the paper. Prints exactly one FINAL line.
# See smoke.py for what it exercises (fit -> steer -> grade on a tiny open model).
set -euo pipefail
cd "$(dirname "$0")"
export MAGS_SMOKE_MODEL="${MAGS_SMOKE_MODEL:-distilgpt2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
# Prefer a local .venv (sandbox dev env); fall back to `python` on PATH (Docker
# image bakes deps into the base interpreter).
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
exec "$PYTHON" -m smoke
