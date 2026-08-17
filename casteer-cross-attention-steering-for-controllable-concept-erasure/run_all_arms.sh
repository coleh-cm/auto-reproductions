#!/usr/bin/env bash
# Run every CASteer arm at the paper's full configuration at every seed and
# write measured.json. Each arm prints exactly one line:
#     FINAL <arm>=<value>
# or
#     FINAL <arm>=BLOCKED      (this environment cannot produce the value)
#
# On a CPU-only host the paper's full config (50 steps, >=1000 prompts, 3
# seeds) is infeasible without a GPU (paper used 8xV100, supplementary.tex:30),
# so every arm emits BLOCKED and measured.json records BLOCKED with a reason
# sidecar (measured_blocked_reasons.json). On a CUDA host the same script runs
# the arms for real. Seeds are not optional: all three {42,1234,2024} run.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi
exec "$PY" scripts/diffusion/run_all_arms.py
