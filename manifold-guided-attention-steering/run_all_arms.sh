#!/usr/bin/env bash
# run_all_arms.sh — runs every arm the paper compares, at the paper's full
# configuration, each printing exactly one line:
#     FINAL <arm name>=<value>
# Arm names (keys of arms.json) are unique per (method, benchmark, model),
# matching the FINAL lines the gate checks. arms_contract.json holds the
# claimed values and bootstrap CIs (the numbers-gate contract, SPEC §5.7).
#
# Two phases:
#   Phase 1 (fit):  for each (model, benchmark) that has a steering arm (iti /
#                  angular-steering / mags), run `python -m mags.fit` to produce
#                  manifolds/<model>__<benchmark>.npz. (mags-u molecular is
#                  blocked separately: its setup is unstated by the paper, SPEC
#                  §4.18.)
#   Phase 2 (eval): run every arm in arms.json; each prints `FINAL <name>=<value>`.
#
# Robustness contract (the important part):
#   This script MUST print exactly one `FINAL <arm>=<value>` line per arm no
#   matter how it is invoked — `bash run_all_arms.sh`, `bash -e run_all_arms.sh`,
#   or under a harness that has already done `set -e`. A failing arm (no GPU, no
#   gated HF token, no torch in the base interpreter) must still emit
#   `FINAL <arm>=BLOCKED`. We therefore (a) explicitly disable errexit with
#   `set +e` (overrides a forced `bash -e`), and (b) run every fallible command
#   inside an `if`/`||` form, which is exempt from errexit. We never fabricate a
#   number: BLOCKED is the honest "no numbers" result. On a GPU host with HF
#   tokens this same script produces the real 45 numbers.

# Disable errexit regardless of how we were launched (`bash -e`, a harness
# `set -e`, etc.). We handle every failure explicitly below so a single blocked
# arm can never abort the script before the remaining arms print their FINAL
# line. (-u / pipefail stay on; they do not cause early exit on handled cmds.)
set -uo pipefail
set +e

cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
mkdir -p runs manifolds

if [ ! -f arms.json ]; then echo "arms.json missing" >&2; exit 1; fi

# Resolve a python that can at least run stdlib (json). The arm commands
# themselves need torch/transformers; in a no-GPU/no-deps sandbox they fail to
# import and each arm falls back to `FINAL <arm>=BLOCKED`.
if [ -x "$PWD/.venv/bin/python" ]; then
    export PATH="$PWD/.venv/bin:$PATH"
fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
HAVE_PYTHON=0
if command -v "$PYTHON" >/dev/null 2>&1; then HAVE_PYTHON=1; fi

# Build the arm map (arm<TAB>cmd) using stdlib json so the Phase-2 loop is
# populated even when the heavy project deps are missing. If python is entirely
# absent, fall back to a grep-based key extract so we can still emit a FINAL
# line for every arm (commands will be empty -> straight to BLOCKED).
ARMS_TSV="runs/_arms_map.tsv"
if [ "$HAVE_PYTHON" = 1 ]; then
    "$PYTHON" - <<'PY' > "$ARMS_TSV" 2>/dev/null || true
import json
arms = json.load(open("arms.json"))
for k, v in arms.items():
    print(k + "\t" + v)
PY
fi
if [ ! -s "$ARMS_TSV" ]; then
    : > "$ARMS_TSV"
    grep -o '"[A-Za-z0-9_.-]*__[^"]*"[[:space:]]*:' arms.json \
        | sed -e 's/^"//' -e 's/"[[:space:]]*:$//' >> "$ARMS_TSV" || true
fi

# --- Phase 1: fit manifolds for steering arms (skip molecular, fully blocked) ---
if [ "$HAVE_PYTHON" = 1 ]; then
    "$PYTHON" - <<'PY' > runs/_fit_pairs.tsv 2>/dev/null || true
import json
arms = json.load(open("arms.json"))
pairs = set()
for k in arms:
    arm, bench, model = k.split("__", 2)
    if arm in ("iti", "angular-steering", "mags") and "SMILES" not in bench:
        model = model.replace("_", "/", 1)  # restore first slash only
        pairs.add((model, bench))
for m, b in sorted(pairs):
    print(m + "\t" + b)
PY
    while IFS='	' read -r model bench; do
        [ -z "$model" ] && continue
        slug=$(echo "$model $bench" | tr '/ ' '__')
        out="manifolds/$(echo "$model" | tr '/' '_')__$(echo "$bench" | tr ' ' '_').npz"
        if [ -f "$out" ]; then echo "[fit] exists $out"; continue; fi
        echo "[fit] $model / $bench -> $out"
        if "$PYTHON" -m mags.fit --model "$model" --benchmark "$bench" --out "$out" \
                > "runs/log__fit__${slug}.log" 2>&1; then
            :
        else
            echo "[fit] BLOCKED $model/$bench"
        fi
    done < runs/_fit_pairs.tsv
fi

# --- Phase 2: run every arm; emit exactly one FINAL line per arm ---
while IFS='	' read -r arm cmd; do
    [ -z "$arm" ] && continue
    # Run the arm command (if we have one and a python to run it). Wrapped in
    # `if` so a non-zero exit can never trigger errexit — the FINAL-line
    # fallback below always runs.
    if [ "$HAVE_PYTHON" = 1 ] && [ -n "$cmd" ]; then
        if eval "$cmd" > "runs/log__${arm}.log" 2>&1; then
            :
        fi
    fi
    # Exactly one FINAL line for this arm: prefer the one the arm printed
    # (mags.run emits `FINAL <id>=<value>` on success or `=BLOCKED` on a
    # caught failure), else synthesize BLOCKED.
    line=$(grep -m1 "^FINAL " "runs/log__${arm}.log" 2>/dev/null) || true
    if [ -n "$line" ]; then
        printf '%s\n' "$line"
    else
        echo "FINAL ${arm}=BLOCKED"
    fi
done < "$ARMS_TSV"

# Exit 0: the gate keys off the FINAL lines, not the exit code, and a non-zero
# exit under a `set -e`/`&&`-chaining harness would discard the parsed output.
# The per-arm FINAL=BLOCKED lines carry the honest "no numbers" signal.
exit 0
