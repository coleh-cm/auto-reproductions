#!/usr/bin/env bash
# smoke.sh — the same code path (smoothed surrogate -> Lewis weights ->
# trust-region Newton -> gap metric) at a size that finishes in a couple of
# minutes, printing one FINAL line.
#
# This proves the path runs.  Its number is NOT evidence about the paper and
# must never be reported as a result — it is a plumbing check only
# (research-code skill: "A toy config is the instrument you use for every
# later check").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${HERE}/.venv/bin/python"
cd "$HERE"
# Tiny self-contained problem (no ACS download, no data-module dependency):
# m=6 groups, d=3, 6 rows/group.  One ball-oracle arm, 3 outer iterations.
exec "$PY" -m gdr.harness smoke
