#!/usr/bin/env bash
# run_all_arms.sh — runs every arm the paper compares, at the paper's full
# configuration, each printing exactly one line:
#     FINAL <arm name>=<value>
# Arm names (keys of arms.json) are unique per (method, benchmark, model), matching
# the FINAL lines the gate checks. arms_contract.json holds the claimed values and
# bootstrap CIs (the numbers-gate contract, SPEC §5.7).
#
# Two phases:
#   Phase 1 (fit):  for each (model, benchmark) that has a steering arm (iti /
#                  angular-steering / mags), run `python -m mags.fit` to produce
#                  manifolds/<model>__<benchmark>.npz. (mags-u molecular is blocked
#                  separately: its setup is unstated by the paper, SPEC §4.18.)
#   Phase 2 (eval): run every arm in arms.json; each prints `FINAL <name>=<value>`.
#
# In this sandbox (no GPU, no gated HF token) both phases BLOCK; each arm prints
# `FINAL <name>=BLOCKED`. That is the honest "no numbers" result: real data or no
# numbers, never a fabricated value. On a GPU host with HF tokens this same script
# produces the real 45 numbers.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
mkdir -p runs manifolds

if [ ! -f arms.json ]; then echo "arms.json missing" >&2; exit 1; fi

# Resolve a python that can import the project deps. Prefer a local .venv
# (the sandbox dev environment); fall back to `python`/`python3` on PATH (the
# Docker image bakes deps into the base interpreter). The arms.json commands
# use bare `python`, so putting .venv/bin first on PATH makes them resolve to
# the same interpreter here. `json` is stdlib, so the mapfile/STEERING_PAIRS
# calls below populate the loop even on a host whose python lacks the heavy
# project deps -- each arm then exits non-zero and the grep||echo fallback
# still emits its FINAL line.
if [ -x "$(dirname "$0")/.venv/bin/python" ]; then
    export PATH="$(cd "$(dirname "$0")" && pwd)/.venv/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || { echo "no python interpreter found" >&2; exit 1; }
export PYTHON

# --- Phase 1: fit manifolds for steering arms (skip molecular, fully blocked) ---
STEERING_PAIRS=$("$PYTHON" - <<'PY'
import json
arms = json.load(open("arms.json"))
pairs = set()
for k in arms:
    arm, bench, model = k.split("__", 2)
    if arm in ("iti","angular-steering","mags") and "SMILES" not in bench:
        model = model.replace("_","/",1)  # restore first slash only
        # restore model id (only first slash was replaced)
        pairs.add((model, bench))
for m,b in sorted(pairs):
    print(f"{m}\t{b}")
PY
)
echo "$STEERING_PAIRS" | while IFS=$'\t' read -r model bench; do
    [ -z "$model" ] && continue
    slug=$(echo "$model $bench" | tr '/ ' '__')
    out="manifolds/$(echo "$model" | tr '/' '_')__$(echo "$bench" | tr ' ' '_').npz"
    if [ -f "$out" ]; then echo "[fit] exists $out"; continue; fi
    echo "[fit] $model / $bench -> $out"
    "$PYTHON" -m mags.fit --model "$model" --benchmark "$bench" --out "$out" \
        > "runs/log__fit__${slug}.log" 2>&1 || echo "[fit] BLOCKED $model/$bench"
done

# --- Phase 2: run every arm ---
# Read arm names with a stdlib-only call so the loop is populated even on a
# host whose `python` lacks the project deps (json is stdlib). Each arm command
# is then eval'd; if it fails to import / load the model it exits non-zero and
# the grep||echo fallback below still emits the FINAL line.
mapfile -t ARMS < <("$PYTHON" -c "import json;[print(k) for k in json.load(open('arms.json'))]")
status=0
for arm in "${ARMS[@]}"; do
    cmd=$("$PYTHON" -c "import json;print(json.load(open('arms.json'))['$arm'])")
    # arms.json commands start with bare `python`; the PATH prepend above makes
    # that resolve to the venv interpreter here, or to the base interpreter in
    # the Docker image. eval it; on any failure (incl. `python` not found on a
    # bare host) the grep||echo fallback below still emits the FINAL line.
    eval "$cmd" > "runs/log__${arm}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        status=1
        grep -m1 "^FINAL " "runs/log__${arm}.log" || echo "FINAL ${arm}=BLOCKED"
    else
        grep -m1 "^FINAL " "runs/log__${arm}.log" || echo "FINAL ${arm}=BLOCKED"
    fi
done
exit $status
