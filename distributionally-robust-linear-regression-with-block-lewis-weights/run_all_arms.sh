#!/usr/bin/env bash
# run_all_arms.sh — runs every arm the paper compares, at the paper's full
# configuration (ACS Income: m=51, d=10, 200/region; the headline instance of
# paper/experiments.tex:145-189, Table tab:acs_runtime at :174-186).
#
# Each arm prints exactly one line  FINAL <arm name>=<value>
# where <value> is iterations-to-1%-relative-gap (base=init, the most natural
# reading of "relative suboptimality", SPEC section 6 item 4), or "not_reached",
# or the OPT value for the reference arm.  The gate greps these lines; arm names
# match arms.json exactly.
#
# The reference arm (reference_cvxpy) computes OPT via the CVXPY epigraph
# (paper/experiments.tex:44-50); every other arm warm-starts at the ERM
# (paper/experiments.tex:148) and is grid-tuned per paper/experiments.tex:81-92.
#
# Usage:  bash run_all_arms.sh [instance]   (default: acs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HERE}/.venv/bin/python"
INSTANCE="${1:-acs}"

cd "$HERE"
# `harness all` computes OPT once, then tunes+runs every optimizer arm, printing
# one FINAL line per arm (paper/experiments.tex:174-186).  Results JSON is written
# to results/<instance>_all.json (committed, not gitignored).
exec "$PY" -m gdr.harness all --instance "$INSTANCE" --out results
