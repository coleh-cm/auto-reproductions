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

# --- Phase 1: fit manifolds for steering arms (skip molecular, fully blocked) ---
STEERING_PAIRS=$(.venv/bin/python - <<'PY'
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
    .venv/bin/python -m mags.fit --model "$model" --benchmark "$bench" --out "$out" \
        > "runs/log__fit__${slug}.log" 2>&1 || echo "[fit] BLOCKED $model/$bench"
done

# --- Phase 2: run every arm ---
mapfile -t ARMS < <(.venv/bin/python -c "import json;[print(k) for k in json.load(open('arms.json'))]")
status=0
for arm in "${ARMS[@]}"; do
    cmd=$(.venv/bin/python -c "import json;print(json.load(open('arms.json'))['$arm'])")
    eval "$cmd" > "runs/log__${arm}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        status=1
        grep -m1 "^FINAL " "runs/log__${arm}.log" || echo "FINAL ${arm}=BLOCKED"
    else
        grep -m1 "^FINAL " "runs/log__${arm}.log"
    fi
done
exit $status
