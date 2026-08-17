#!/usr/bin/env bash
# Smoke test: run the SAME CASteer code path at a size that finishes in a
# couple of minutes on CPU. It exercises load -> estimate steering vectors
# (Algorithm 1) -> steer (Eq.7) -> generate, for the casteer_clip and sd14
# arms at 4 steps / 256x256 / seed 42. Prints exactly one FINAL line.
#
# This PROVES THE PATH RUNS. It is NOT evidence about the paper (the subset is
# tiny and the resolution/step count are reduced); never report its output as
# a result. A synthetic fixture is for smoke.sh only; this smoke uses a REAL
# SD-1.4 generation (no synthetic substitution).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi
exec "$PY" scripts/diffusion/smoke.py
