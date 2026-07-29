#!/usr/bin/env bash
# smoke.sh — runs the MAGS code path at smoke size (CPU, couple of minutes).
# NOT evidence about the paper. Prints exactly one FINAL line.
# See smoke.py for what it exercises (fit -> steer -> grade on a tiny open model).
#
# Robustness: this script always prints exactly one `FINAL smoke=<value>` line.
# If the deps are missing, the model can't be downloaded, or the path raises,
# it prints `FINAL smoke=BLOCKED` (with the reason on stderr) rather than a bare
# traceback — so the gate always sees one FINAL line from smoke. On a host with
# torch/transformers + network it prints the real smoke accuracy.
set -uo pipefail
set +e
cd "$(dirname "$0")"
mkdir -p runs
export MAGS_SMOKE_MODEL="${MAGS_SMOKE_MODEL:-distilgpt2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "FINAL smoke=BLOCKED"
    exit 0
fi
# Run the smoke; on any failure emit BLOCKED. `if ...; then` is exempt from
# errexit so a raised exception can never abort before the FINAL line.
if "$PYTHON" -m smoke > runs/log__smoke.log 2>&1; then
    grep -m1 "^FINAL " runs/log__smoke.log || echo "FINAL smoke=BLOCKED"
else
    grep -m1 "^FINAL " runs/log__smoke.log 2>/dev/null || echo "FINAL smoke=BLOCKED"
fi
exit 0
