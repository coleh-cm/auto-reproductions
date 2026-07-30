#!/usr/bin/env bash
# Smoke test: the SAME code path as run_all_arms.sh at a size that finishes in a
# couple of minutes, printing one FINAL line.  This proves the path runs end to
# end (data -> smoothed surrogate -> Lewis weights -> trust-region Newton -> gap
# metric).  Its number is NOT evidence about the paper -- never report it as a
# result.  (research-code skill: a toy config is the instrument, not the result.)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || command -v python)"
fi

exec "$PY" -m gdr.harness smoke
