#!/usr/bin/env bash
# Smoke test: run the SAME CASteer code path at a size that finishes in a
# couple of minutes on CPU. It exercises load -> estimate steering vectors
# (Algorithm 1) -> steer (Eq.7) -> generate -> score (snoopy_cs CLIP metric),
# for the casteer_clip arm at 4 steps / 256x256 / seed 42. Prints exactly one
# stdout line: FINAL casteer_clip=<snoopy_cs>, using an arm name from
# claims.json and a real numeric measurement.
#
# This PROVES THE PATH RUNS. It is NOT evidence about the paper (the subset is
# tiny and the resolution/step count are reduced); never report its output as a
# result. A synthetic fixture is for smoke.sh only; this smoke uses a REAL
# SD-1.4 generation (no synthetic substitution).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then PY="python3"; fi
exec "$PY" scripts/diffusion/smoke.py
