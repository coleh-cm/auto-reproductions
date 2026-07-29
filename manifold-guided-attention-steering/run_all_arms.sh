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
export PYTHONUNBUFFERED=1
mkdir -p runs manifolds

# --- offline fast-fail (round-4 gate fix) -------------------------------------
# The gate runs in a Docker image that HAS torch/transformers but has NO model
# cache and (typically) NO network. transformers' default ONLINE mode then hangs
# on every `from_pretrained` until the gate's overall timeout -> the script is
# killed before any `FINAL <arm>=...` line prints -> the gate reports all 45
# arms "missing a FINAL line" with `values: []` (this was the failure for the
# first THREE gate rounds; the prior fixes addressed missing-deps and errexit,
# which were real but not the actual gate failure mode). Default to OFFLINE
# when no HuggingFace token is discoverable (env var OR the `huggingface-cli
# login` token file) so an uncached model raises in <1s -> FINAL <arm>=BLOCKED.
# (mags/run.py and mags/fit.py set the same default at import, so this also
# protects arms invoked directly; setting it here covers the Phase-1 fit calls
# and makes the behaviour visible.) A real GPU host that ran `huggingface-cli
# login` is detected and left online so it can download; pre-cached models
# load under offline=1 too.
_hf_token_set() {
    [ -n "${HF_TOKEN:-}${HF_HUB_TOKEN:-}" ] && return 0
    local f="${HF_HOME:-$HOME/.cache/huggingface}/token"
    [ -s "$f" ] && return 0
    [ -s "$HOME/.huggingface/token" ] && return 0
    return 1
}
if ! _hf_token_set; then
    : "${HF_HUB_OFFLINE:=1}"; : "${TRANSFORMERS_OFFLINE:=1}"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
# Short HF timeouts as a backstop so even explicit online mode cannot hang the
# gate on a blackholed network (fail in ~10s instead of the ~75s TCP default).
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-10}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-10}"
# Per-command wall-clock backstops (only a safety net; in the gate every command
# fast-fails in <1s under offline mode, so these never trigger). On a real GPU
# host a single fit/eval can legitimately exceed these — override
# MAGS_FIT_TIMEOUT / MAGS_ARM_TIMEOUT to raise them.
FIT_TIMEOUT="${MAGS_FIT_TIMEOUT:-1800}"
ARM_TIMEOUT="${MAGS_ARM_TIMEOUT:-3600}"
if ! command -v timeout >/dev/null 2>&1; then
    # no coreutils `timeout`: define a no-op shim so the `timeout N cmd` calls below
    # just run the command unbounded (offline fast-fail still protects the gate).
    timeout() { shift; "$@"; }
fi

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

# --- capability probe: can this host run the paper's 8B/20B GPU arms? -------
# The paper's experiments REQUIRE GPU (RTX 4090 / H200, SPEC §C.1 / Appendix C).
# A CPU-only host — the automated gate, a CI runner — cannot load or run an
# 8B/20B model in any reasonable time, so the ONLY honest result there is
# BLOCKED for every arm. We detect "no CUDA GPU" with a single bounded torch
# import and, if true, emit all `FINAL <arm>=BLOCKED` lines immediately
# (sub-second) and exit 0. This is the fix for the recurring gate failure
# "arms missing a FINAL line / values: []":
#   - With NO HuggingFace token, the round-4 offline default already made an
#     uncached model fast-fail; BUT
#   - the gate carries an HF token (and possibly a partial model cache), which
#     defeats the offline default -> `from_pretrained` enters ONLINE mode and
#     hangs on a blackholed network (verified: a token-set run was still going
#     at 90s with only 29/45 FINAL lines). The gate's wall-clock budget then
#     kills the script mid-Phase-1, BEFORE any per-arm FINAL line prints, so the
#     gate reports every arm "missing a FINAL line" with `values: []`.
#   - The GPU probe is independent of token / cache / network state: a CPU host
#     fast-paths to all-BLOCKED in well under a second regardless of whether a
#     token is set. A real GPU host passes the probe and runs the full
#     Phase-1/Phase-2 pipeline for real. smoke.sh is unaffected (it runs a
#     CPU-tiny model on purpose via `python -m smoke`, not this script).
_probe_gpu() {
    [ "$HAVE_PYTHON" = 1 ] || { echo NOCUDA; return; }
    timeout 60 "$PYTHON" - <<'PY' 2>/dev/null || echo NOCUDA
try:
    import torch
except Exception:
    print("NOCUDA")
else:
    print("CUDA" if torch.cuda.is_available() else "NOCUDA")
PY
}
if [ "$(_probe_gpu)" != "CUDA" ]; then
    mkdir -p runs
    while IFS='	' read -r arm cmd; do
        [ -z "$arm" ] && continue
        reason="no CUDA GPU available; the paper's 8B/20B models require GPU (RTX 4090 / H200, SPEC §C.1). On a GPU host with cached models this fast-path is skipped."
        printf 'FINAL %s=BLOCKED\n' "$arm"
        { printf 'FINAL %s=BLOCKED\n' "$arm"; \
          printf 'BLOCKED[%s]: %s\n' "$arm" "$reason"; } > "runs/log__${arm}.log"
        printf '{"arm": "%s", "blocked_reason": "%s"}\n' "$arm" "$reason" \
            > "runs/BLOCKED__${arm}.json" 2>/dev/null || true
    done < "$ARMS_TSV"
    exit 0
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
        if timeout "$FIT_TIMEOUT" "$PYTHON" -m mags.fit --model "$model" --benchmark "$bench" --out "$out" \
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
        if eval "timeout \"$ARM_TIMEOUT\" $cmd" > "runs/log__${arm}.log" 2>&1; then
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
