#!/usr/bin/env bash
# run_all_arms.sh — runs every arm the paper compares, at the paper's full
# configuration, each printing exactly one line:
#     FINAL <arm name>=<value>
# Arm names (keys of arms.json) are unique per (method, benchmark, model),
# matching the FINAL lines the gate checks. arms_contract.json holds the
# claimed values and bootstrap CIs (the numbers-gate contract, SPEC §5.7).
#
# Two phases (only reached when the capability gate below passes for a model):
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
#   cached model, no torch in the base interpreter) must still emit
#   `FINAL <arm>=BLOCKED`. We therefore (a) explicitly disable errexit with
#   `set +e` (overrides a forced `bash -e`), and (b) run every fallible command
#   inside an `if`/`||` form, which is exempt from errexit. We never fabricate a
#   number: BLOCKED is the honest "no numbers" result.
#
#   The capability gate is a two-tier FAST-PATH designed to emit every FINAL
#   line in well under a second on a host that cannot reproduce the paper, so
#   the gate's wall-clock budget can never kill this script before a FINAL line
#   prints (the recurring "arms missing a FINAL line / values: []" failure):
#     Tier 1 — MODEL-CACHE CHECK (no torch, no network, instant filesystem stat):
#       For each distinct model in arms.json, test whether its weights are
#       already present in the HuggingFace hub cache. The paper's models are
#       Llama-3.1-8B-Instruct (gated), Gemma-4-E4B-it, and GPT-OSS-20B; on the
#       automated gate / a CI runner NONE of them are cached, so every arm is
#       `FINAL <arm>=BLOCKED` in <1s with zero torch imports and zero network
#       calls. This is independent of CUDA, HuggingFace tokens, and network
#       state, so it cannot hang. (A real reproducer pre-caches the models;
#       the README documents this. We never attempt an in-run download, which
#       is what hung earlier gate rounds.)
#     Tier 2 — CUDA CHECK (one bounded torch import, only if >=1 model cached):
#       Cached 8B/20B models cannot run on CPU, so a CPU-only host with a cached
#       model still BLOCKs. Reached only when Tier 1 already proved a model is
#       present, so the torch import cost is paid at most once and only on a
#       host that can actually use it.

# Disable errexit regardless of how we were launched (`bash -e`, a harness
# `set -e`, etc.). We handle every failure explicitly below so a single blocked
# arm can never abort the script before the remaining arms print their FINAL
# line. (-u / pipefail stay on; they do not cause early exit on handled cmds.)
set -uo pipefail
set +e

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p runs manifolds

# Force OFFLINE for any real model load: Tier 1 already proved the model is
# cached, so offline load uses the cache; a partial cache raises in <1s instead
# of hanging on a blackholed network. (We never download inside this script.)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-10}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-10}"
# Per-command wall-clock backstops (only a safety net; on a no-cache host Tier 1
# fast-fails in <1s, so these never trigger). On a real GPU host a single
# fit/eval can legitimately exceed these — override MAGS_FIT_TIMEOUT /
# MAGS_ARM_TIMEOUT to raise them.
FIT_TIMEOUT="${MAGS_FIT_TIMEOUT:-1800}"
ARM_TIMEOUT="${MAGS_ARM_TIMEOUT:-3600}"
if ! command -v timeout >/dev/null 2>&1; then
    # no coreutils `timeout`: define a no-op shim so the `timeout N cmd` calls below
    # just run the command unbounded (Tier-1 cache check still protects the gate).
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

# Build the full arm map: `model<TAB>arm<TAB>cmd` using stdlib json. The model
# id is parsed from the arm's `--model <id>` argument (authoritative), NOT from
# the arm key, so the cache check tests the real HuggingFace repo id. If python
# is entirely absent, fall back to a grep key extract with empty model/cmd
# (every arm goes straight to Tier-1 BLOCKED).
ARMS_FULL="runs/_arms_full.tsv"
ARM_KEYS="runs/_arms_keys.tsv"
if [ "$HAVE_PYTHON" = 1 ]; then
    "$PYTHON" - <<'PY' > "$ARMS_FULL" 2>/dev/null || true
import json, re
arms = json.load(open("arms.json"))
for k, v in arms.items():
    m = re.search(r"--model\s+(\S+)", v)
    mid = m.group(1) if m else ""
    print(f"{mid}\t{k}\t{v}")
PY
fi
if [ ! -s "$ARMS_FULL" ]; then
    : > "$ARMS_FULL"
    grep -o '"[A-Za-z0-9_.-]*__[^"]*"[[:space:]]*:' arms.json \
        | sed -e 's/^"//' -e 's/"[[:space:]]*:$//' >> "$ARMS_FULL" || true
    # no python -> model/cmd unknowable; emit keys with empty model/cmd
    : > "$ARMS_FULL"
    while IFS= read -r key; do
        [ -n "$key" ] && printf '\t%s\t\n' "$key" >> "$ARMS_FULL"
    done < <(grep -o '"[A-Za-z0-9_.-]*__[^"]*"[[:space:]]*:' arms.json \
              | sed -e 's/^"//' -e 's/"[[:space:]]*:$//')
fi
# Keys-only list (used for the Tier-1 BLOCKED fan-out when no python).
cut -f2 "$ARMS_FULL" > "$ARM_KEYS" 2>/dev/null || true

# --- Tier 1: MODEL-CACHE CHECK (filesystem, no torch, no network) ---------------
# Returns 0 iff the model's weights are present in the HuggingFace hub cache.
# Layout: $HF_HUB_CACHE/models--<org>--<name>/snapshots/<hash>/<weights>.
# This is a pure stat/find: instant, and independent of CUDA/tokens/network.
_hf_hub_dir() {
    printf '%s' "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
}
_model_cached() {
    local mid="$1"
    [ -n "$mid" ] || return 1
    local hub base snap
    hub="$(_hf_hub_dir)"
    base="$hub/models--${mid//\//--}"
    [ -d "$base/snapshots" ] || return 1
    # any snapshot revision dir?
    snap=$(find "$base/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)
    [ -n "$snap" ] || return 1
    # weight files present? (safetensors / bin / gguf / consolidated pth)
    find "$snap" -maxdepth 1 -type f \( \
            -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' \
            -o -name 'consolidated*.pth' -o -name 'pytorch_model*.bin' \) \
        -print -quit 2>/dev/null | grep -q .
}

# Distinct models referenced by arms.json (in declaration order, de-duplicated).
mapfile -t ALL_MODELS < <(cut -f1 "$ARMS_FULL" | awk 'NF' | awk '!seen[$0]++')
# A model id is empty when python was unavailable (we could not parse --model);
# treat empty as "not cached" so those arms BLOCK at Tier 1 (honest: we cannot
# even identify the model, so we cannot run it).

USABLE_MODELS=()
_model_usable() {  # echo 1 if model usable (cached), else 0
    local m="$1"
    if [ -z "$m" ]; then echo 0; return; fi
    if _model_cached "$m"; then echo 1; else echo 0; fi
}

# Build a model->usable lookup so we only stat each model once.
declare -A USABLE
for m in "${ALL_MODELS[@]}"; do
    if [ "$(_model_usable "$m")" = 1 ]; then
        USABLE["$m"]=1
        USABLE_MODELS+=("$m")
    else
        USABLE["$m"]=0
    fi
done

# --- Tier 2: CUDA CHECK (only if at least one model is cached) ------------------
# Cached 8B/20B models cannot run on CPU. One bounded torch import decides
# whether any cached model can actually execute. Skipped entirely when no
# model is cached (the gate case) -> zero torch imports, zero hang risk.
HAVE_CUDA=0
if [ "${#USABLE_MODELS[@]}" -gt 0 ] && [ "$HAVE_PYTHON" = 1 ]; then
    if timeout 60 "$PYTHON" - <<'PY' 2>/dev/null
try:
    import torch
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch.cuda.is_available() else 2)
PY
    then
        HAVE_CUDA=1
    fi
fi

# --- Emit FINAL=BLOCKED for every arm we cannot run, partition the rest -------
# An arm is runnable only if its model is cached (Tier 1) AND CUDA is present
# (Tier 2). Anything else gets an immediate, honest `FINAL <arm>=BLOCKED` and is
# removed from the Phase-1/Phase-2 work list. This guarantees the gate sees a
# FINAL line for EVERY arm within seconds, regardless of CUDA/token/network.
RUNNABLE="runs/_arms_runnable.tsv"
: > "$RUNNABLE"
_emit_blocked() {
    local arm="$1" reason="$2"
    printf 'FINAL %s=BLOCKED\n' "$arm"
    { printf 'FINAL %s=BLOCKED\n' "$arm"; printf 'BLOCKED[%s]: %s\n' "$arm" "$reason"; } \
        > "runs/log__${arm}.log" 2>/dev/null || true
    printf '{"arm": "%s", "blocked_reason": "%s"}\n' "$arm" "${reason//\"/\\\"}" \
        > "runs/BLOCKED__${arm}.json" 2>/dev/null || true
}

while IFS=$'\t' read -r model arm cmd; do
    [ -n "$arm" ] || continue
    if [ -z "$model" ]; then
        _emit_blocked "$arm" "could not resolve model id from arm command (no python to parse arms.json); cannot run."
        continue
    fi
    if [ "${USABLE[$model]:-0}" != 1 ]; then
        case "$arm" in
            *__SMILES-molecular-generation__*)
                _emit_blocked "$arm" "Molecular task (Table 3) is a stretch target: the target protein, prompt template, SMILES contrastive corpus, affinity cutoff, and AutoDock-GPU params are all UNSTATED by the paper (SPEC §4.18); GPT-OSS-20B needs >=40GB VRAM and is not cached here. Not implemented for real data." ;;
            *)
                _emit_blocked "$arm" "model $model is not present in the HuggingFace cache ($(_hf_hub_dir)); the paper's 8B/20B models require a GPU host with the model pre-downloaded (SPEC §C.1). No in-run download is attempted (would hang an offline gate)." ;;
        esac
        continue
    fi
    if [ "$HAVE_CUDA" != 1 ]; then
        _emit_blocked "$arm" "model $model is cached but no CUDA GPU is available; the paper's 8B/20B models require GPU (RTX 4090 / H200, SPEC §C.1) and cannot run on CPU."
        continue
    fi
    printf '%s\t%s\t%s\n' "$model" "$arm" "$cmd" >> "$RUNNABLE"
done < "$ARMS_FULL"

# If nothing is runnable (the gate case), we are already done: every arm has a
# FINAL line. Exit 0 so a `set -e` harness does not discard the output.
if [ ! -s "$RUNNABLE" ]; then
    exit 0
fi

# --- Phase 1: fit manifolds for the runnable steering arms --------------------
# Only for (model, benchmark) pairs that survived the gate. Each fit is bounded
# by FIT_TIMEOUT; a failure is logged and the corresponding steering arms will
# BLOCK at Phase 2 (missing manifold file) — never aborts the script.
if [ "$HAVE_PYTHON" = 1 ]; then
    : > runs/_fit_pairs.tsv
    while IFS=$'\t' read -r model arm cmd; do
        case "$arm" in
            iti__*|angular-steering__*|mags__*) ;;
            *) continue ;;
        esac
        case "$arm" in *__SMILES-molecular-generation__*) continue ;; esac
        bench=$(printf '%s' "$arm" | sed -E 's/^[^_]+__//; s/__[A-Za-z0-9._-]+$//')
        printf '%s\t%s\n' "$model" "$bench" >> runs/_fit_pairs.tsv
    done < "$RUNNABLE"
    sort -u runs/_fit_pairs.tsv -o runs/_fit_pairs.tsv
    while IFS=$'\t' read -r model bench; do
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

# --- Phase 2: run every runnable arm; emit exactly one FINAL line per arm -----
while IFS=$'\t' read -r model arm cmd; do
    [ -z "$arm" ] && continue
    if [ "$HAVE_PYTHON" = 1 ] && [ -n "$cmd" ]; then
        if eval "timeout \"$ARM_TIMEOUT\" $cmd" > "runs/log__${arm}.log" 2>&1; then
            :
        fi
    fi
    line=$(grep -m1 "^FINAL " "runs/log__${arm}.log" 2>/dev/null) || true
    if [ -n "$line" ]; then
        printf '%s\n' "$line"
    else
        echo "FINAL ${arm}=BLOCKED"
    fi
done < "$RUNNABLE"

# Exit 0: the gate keys off the FINAL lines, not the exit code, and a non-zero
# exit under a `set -e`/`&&`-chaining harness would discard the parsed output.
# The per-arm FINAL=BLOCKED lines carry the honest "no numbers" signal.
exit 0
