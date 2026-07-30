#!/usr/bin/env bash
# Run every arm of "Distributionally Robust Linear Regression With Block Lewis
# Weights" at the paper's full configuration (ACS Income, m=51, d=10, 200/region)
# and print one ``FINAL <arm>=<value>`` line per arm -- the line the gate checks.
#
# The value is the arm's primary gate metric: iterations to 1% relative
# worst-group gap on ACS (base = initial ERM gap, the scale-invariant reading
# of the unstated 1%-reference; SPEC section 6 item 4).  The CVXPY reference arm
# prints OPT.  An arm that never reaches 1% prints ``FINAL <arm>=not_reached``
# (the paper marks the subgradient arm exactly this way,
# paper/experiments.tex:178).
#
# ACS data is downloaded by folktables on first use (census.gov) and cached
# under data/.  Tuning grids and budgets live in gdr/harness.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-.venv/bin/python}"
# Always use the venv interpreter if present; fall back to sys.executable-style
# discovery only if the venv is absent (never bare "python").
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || command -v python)"
fi

exec "$PY" -m gdr.harness all --instance acs --budget 100 --tune-budget 80
