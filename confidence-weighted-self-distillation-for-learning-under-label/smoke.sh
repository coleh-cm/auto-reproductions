#!/usr/bin/env bash
# Smoke test: exercise the SAME code path as run_all_arms.sh at a size that
# finishes in a couple of seconds, printing exactly one FINAL line.
#
# This is NOT evidence about the paper. It proves the path runs end to end
# (data load -> corrupt -> init -> train -> evaluate -> print) without
# importing the paper's full 4000-step budget. Never report its output as a
# reproduction result; the real numbers come from run_all_arms.sh -> measured.json.
#
# The one stdout line is deliberately labelled 'FINAL smoke=' (not the
# 'FINAL accuracy=' contract line) so a reader cannot mistake it for a result.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"

# Reduced budget: 50 steps (vs 4000), CWSD arm (lambda=1, exercises the gate,
# target, and stop-grad path -- the code path the baseline arm does not touch).
out="$($PY run_experiment.py --lambda 1.0 --steps 50 --seed 0 2>/dev/null)"
rc=$?
if [ $rc -ne 0 ]; then
  echo "FINAL smoke=BLOCKED"
  exit 1
fi
val="$(printf '%s\n' "$out" | grep -E '^FINAL accuracy=' | head -1 | sed 's/^FINAL accuracy=//')"
if [ -z "$val" ]; then
  echo "FINAL smoke=BLOCKED"
  exit 1
fi
echo "FINAL smoke=${val}"
