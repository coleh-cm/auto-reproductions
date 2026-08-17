#!/usr/bin/env bash
# Run every arm at the paper's full configuration at every seed, write measured.json.
# Seeds {0,1,2} are NOT optional. Each arm prints exactly one 'FINAL <arm>=<value>'
# (or =BLOCKED) line. Exit nonzero if measured.json is missing any arm/seed/metric.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$REPO/results"

echo "[run_all_arms] E1 synthetic closed-form invariants (CPU, REAL)..."
"$PY" "$REPO/experiments/run_e1_synth.py"

echo "[run_all_arms] E2 LLM concrete (model arm)..."
"$PY" "$REPO/experiments/run_e2_llm_concrete.py"

echo "[run_all_arms] E3 LLM safety (model arm)..."
"$PY" "$REPO/experiments/run_e3_llm_safety.py"

echo "[run_all_arms] E4 SDXL horse->motorcycle (model arm)..."
"$PY" "$REPO/experiments/run_e4_sdxl_h2m.py"

echo "[run_all_arms] E5 SDXL safety (model arm)..."
"$PY" "$REPO/experiments/run_e5_sdxl_safety.py"

echo "[run_all_arms] assembling measured.json..."
"$PY" "$REPO/assemble_measured.py"
