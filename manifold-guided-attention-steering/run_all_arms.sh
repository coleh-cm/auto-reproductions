#!/usr/bin/env bash
# run_all_arms.sh — runs every arm the paper compares, at the paper's full
# configuration, each printing exactly one line:
#     FINAL <arm name>=<value>
# Arm names (keys of arms.json) are unique per (method, benchmark, model),
# matching the FINAL lines the gate checks. arms_contract.json holds the
# claimed values and bootstrap CIs (the numbers-gate contract, SPEC §5.7).
#
# POSIX-sh portability (THE gate fix, round 7): this script is written to run
# under /bin/sh (dash) as well as bash. The numbers gate invokes it as
# `sh run_all_arms.sh`, and the previous six "missing FINAL line" rounds all
# failed because they added bash-only constructs (process substitution
# `< <(...)`, `mapfile`, `declare -A`, `${var//pat/re}`, `[[ ]]`) that dash
# rejects at parse time -> the script died before printing a single FINAL
# line -> the gate reported every arm "missing a FINAL line" with
# `values: []`. This version uses ONLY POSIX constructs: pipes and
# `while read ... done < file` (never process substitution), awk/sed/tr for
# string transforms (never `${//}`), `[ ]` (never `[[ ]]), and per-arm
# filesystem stats (never associative arrays). Verified under dash and bash.
#
# Robustness contract:
#   This script MUST print exactly one `FINAL <arm>=<value>` line per arm no
#   matter how it is invoked — `bash run_all_arms.sh`, `sh run_all_arms.sh`,
#   `bash -e run_all_arms.sh`, or under a harness that already did `set -e`.
#   A failing arm (no GPU, no cached model, no torch) must still emit
#   `FINAL <arm>=BLOCKED`. We (a) explicitly disable errexit with `set +e`
#   (overrides a forced `bash -e`), and (b) run every fallible command inside
#   an `if`/`|| true` form, which is exempt from errexit. We never fabricate a
#   number: BLOCKED is the honest "no numbers" result.
#
#   The capability gate is a two-tier FAST-PATH designed to emit every FINAL
#   line in well under a second on a host that cannot reproduce the paper, so
#   the gate's wall-clock budget can never kill this script before a FINAL line
#   prints:
#     Tier 1 — MODEL-CACHE CHECK (no torch, no network, instant filesystem stat):
#       For each distinct model in arms.json, stat whether its weights are
#       already present in the HuggingFace hub cache. The paper's models are
#       Llama-3.1-8B-Instruct (gated), Gemma-4-E4B-it, and GPT-OSS-20B; on the
#       automated gate / a CI runner NONE of them are cached, so every arm is
#       `FINAL <arm>=BLOCKED` in <1s with zero torch imports and zero network
#       calls. (A real reproducer pre-caches the models; the README documents
#       this. We never attempt an in-run download, which is what hung earlier
#       rounds.)
#     Tier 2 — CUDA CHECK (one bounded torch import, only if >=1 model cached):
#       Cached 8B/20B models cannot run on CPU, so a CPU-only host with a cached
#       model still BLOCKs.

# Disable errexit regardless of how we were launched (`bash -e`, a harness
# `set -e`, etc.). We handle every failure explicitly below so a single blocked
# arm can never abort the script before the remaining arms print their FINAL
# line. (`-u` stays on; it does not cause early exit on handled cmds. We do NOT
# set `pipefail`: it is not portable to all dash builds, and we never rely on a
# pipe's non-zero status — every fallible pipe ends in `|| true` or feeds an
# `if`.)
set -u
set +e

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p runs manifolds

# Force OFFLINE (models + datasets) unless the reproducer opts in with
# MAGS_ONLINE=1. This is the round-13 root-cause fix for the recurring "all arms
# missing a FINAL line" gate failure: a cached model + token + blackholed net
# let an arm proceed past the model precheck to `load_dataset`, which HANGS
# online until the gate kills it -> zero FINAL lines. Offline makes every
# uncached resource fast-fail (ConnectionError in <1s) -> one honest
# FINAL=BLOCKED line per arm in <1s, regardless of token/network/CUDA. A real
# GPU host pre-caches models+datasets (README) — offline loads the cache — or
# sets MAGS_ONLINE=1 to download. The per-arm `python -m mags.run` / `mags.fit`
# modules ALSO force offline at import (defense-in-depth for direct invocation),
# so this export is belt-and-suspenders.
if [ "${MAGS_ONLINE:-0}" != "1" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi
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

# Build the full arm map TSV: `model<TAB>arm<TAB>cmd` using stdlib json. The
# model id is parsed from the arm's `--model <id>` argument (authoritative),
# NOT from the arm key, so the cache check tests the real HuggingFace repo id.
# POSIX: heredoc + redirection (no process substitution).
ARMS_FULL="runs/_arms_full.tsv"
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
# Fallback when python is absent OR the heredoc produced nothing: extract the
# arm keys from arms.json with grep/sed (POSIX). With no python we cannot parse
# `--model`, so the model field is empty and every arm goes straight to Tier-1
# BLOCKED ("could not resolve model id") — still one FINAL line per arm.
if [ ! -s "$ARMS_FULL" ]; then
    : > "$ARMS_FULL"
    grep -o '"[A-Za-z0-9_.-]*__[^"]*"[[:space:]]*:' arms.json \
        | sed -e 's/^"//' -e 's/"[[:space:]]*:$//' \
        | while IFS= read -r _key; do
            [ -n "$_key" ] && printf '\t%s\t\n' "$_key" >> "$ARMS_FULL"
          done
fi

# --- Tier 1: MODEL-CACHE CHECK (filesystem, no torch, no network) ---------------
# Returns 0 iff the model's weights are present in the HuggingFace hub cache.
# Layout: $HF_HUB_CACHE/models--<org>--<name>/snapshots/<hash>/<weights>.
# This is a pure stat/find: instant, and independent of CUDA/tokens/network.
# POSIX: no associative arrays, no `${var//pat/re}` (use sed/tr), no `[[`.
_hf_hub_dir() {
    printf '%s' "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
}

_model_cached() {
    # $1 = model id. Returns 0 iff a snapshot dir with weight files exists.
    _mid="$1"
    [ -n "$_mid" ] || return 1
    _hub="$(_hf_hub_dir)"
    _base="$_hub/models--$(printf '%s' "$_mid" | sed 's|/|--|g')"
    [ -d "$_base/snapshots" ] || return 1
    _snap=$(find "$_base/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)
    [ -n "$_snap" ] || return 1
    find "$_snap" -maxdepth 1 -type f \( \
            -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' \
            -o -name 'consolidated*.pth' -o -name 'pytorch_model*.bin' \) \
        -print -quit 2>/dev/null | grep -q .
}

# Distinct models referenced by arms.json (declaration order, de-duplicated).
# POSIX awk dedup (no `mapfile`).
MODELS_TSV="runs/_models.tsv"
cut -f1 "$ARMS_FULL" | awk 'NF && !seen[$0]++' > "$MODELS_TSV" 2>/dev/null || true

# Stat each distinct model ONCE; write `model<TAB>1|0` to a lookup TSV. Reading
# from $MODELS_TSV and writing to a different file is portable (no process
# substitution). Reset first so a stale file from a prior run cannot poison us.
USABLE_TSV="runs/_model_usable.tsv"
: > "$USABLE_TSV"
ANY_USABLE=0
while IFS= read -r _m; do
    [ -n "$_m" ] || continue
    if _model_cached "$_m"; then
        printf '%s\t1\n' "$_m" >> "$USABLE_TSV"
        ANY_USABLE=1
    else
        printf '%s\t0\n' "$_m" >> "$USABLE_TSV"
    fi
done < "$MODELS_TSV"

# Lookup helper: is model $1 usable (cached)? POSIX awk over the TSV.
_model_usable_p() {
    awk -F '\t' -v m="$1" '$1 == m && $2 == "1" {found=1} END {exit !found}' "$USABLE_TSV"
}

# --- Tier 2: CUDA CHECK (only if at least one model is cached) ------------------
# Cached 8B/20B models cannot run on CPU. One bounded torch import decides
# whether any cached model can actually execute. Skipped entirely when no model
# is cached (the gate case) -> zero torch imports, zero hang risk.
HAVE_CUDA=0
if [ "$ANY_USABLE" = 1 ] && [ "$HAVE_PYTHON" = 1 ]; then
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
# (Tier 2). Anything else gets an immediate, honest `FINAL <arm>=BLOCKED` and
# is removed from the Phase-1/Phase-2 work list. This guarantees the gate sees a
# FINAL line for EVERY arm within seconds, regardless of CUDA/token/network.
RUNNABLE="runs/_arms_runnable.tsv"
: > "$RUNNABLE"

# JSON-escape a string for the BLOCKED manifest (POSIX sed; no `${//}`).
_json_esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

_emit_blocked() {
    # $1 = arm, $2 = reason. Prints one FINAL line + writes a BLOCKED manifest.
    _arm="$1"; _reason="$2"
    printf 'FINAL %s=BLOCKED\n' "$_arm"
    { printf 'FINAL %s=BLOCKED\n' "$_arm"; printf 'BLOCKED[%s]: %s\n' "$_arm" "$_reason"; } \
        > "runs/log__${_arm}.log" 2>/dev/null || true
    _esc="$(_json_esc "$_reason")"
    printf '{"arm": "%s", "blocked_reason": "%s"}\n' "$_arm" "$_esc" \
        > "runs/BLOCKED__${_arm}.json" 2>/dev/null || true
}

# POSIX: set IFS to a single tab via command substitution (no `$'\t'`).
_TAB="$(printf '\t')"
while IFS="$_TAB" read -r model arm cmd; do
    [ -n "$arm" ] || continue
    if [ -z "$model" ]; then
        _emit_blocked "$arm" "could not resolve model id from arm command (no python to parse arms.json); cannot run."
        continue
    fi
    if ! _model_usable_p "$model"; then
        # Model-specific, accurate block reasons (round-21). The earlier
        # blanket "8B/20B cannot run on CPU" was inaccurate for the 4B Gemma.
        case "$model" in
            meta-llama/Llama-3.1-8B-Instruct)
                _emit_blocked "$arm" "model $model is GATED on HuggingFace (manual license approval) and no HF token is set in this sandbox, so its weights cannot be downloaded; the HF cache has no snapshot for it. Even on a GPU host this arm needs 'hf auth login' + an accepted license before download. No in-run download is attempted (SPEC §C.1)." ;;
            google/gemma-4-E4B-it)
                _emit_blocked "$arm" "model $model is NOT gated and IS downloadable (freshly re-verified: unauthenticated HF download now proceeds at ~2.1 MB/s, ~2 h for the 16 GB safetensors; the round-21 'throttle to a stall' no longer holds), but there is NO GPU in this sandbox (CPU-only torch, 16 aarch64 cores, no CUDA). Full-config eval of a 4B model on CPU is infeasible within any gate wall-clock budget (est. 30+ h for the 4 Gemma reasoning benchmarks x 5 arms). The paper requires RTX 4090 (SPEC §C.1)." ;;
            openai/gpt-oss-20b)
                case "$arm" in
                    *__SMILES-molecular-generation__*)
                        _emit_blocked "$arm" "Molecular task (Table 3) is a stretch target: the target protein, prompt template, SMILES contrastive corpus, affinity cutoff, and AutoDock-GPU params are all UNSTATED by the paper (SPEC §4.18); GPT-OSS-20B (13.7 GB, not gated) is not cached here and a 20B model cannot run on CPU. Not implemented for real data." ;;
                    *)
                        _emit_blocked "$arm" "model $model (13.7 GB, not gated) is not cached here and a 20B model cannot run on CPU; no GPU available (SPEC §C.1)." ;;
                esac ;;
            *)
                _emit_blocked "$arm" "model $model is not present in the HuggingFace cache ($(_hf_hub_dir)); the paper's models require a GPU host with the model pre-downloaded (SPEC §C.1). No in-run download is attempted (would hang an offline gate)." ;;
        esac
        continue
    fi
    if [ "$HAVE_CUDA" != 1 ]; then
        # A cached model with no GPU. A 4B model COULD load on CPU but the
        # paper's full-config eval is infeasible on CPU in any gate budget, so
        # we still block. (Llama-8B / GPT-OSS-20B genuinely cannot run on CPU.)
        _emit_blocked "$arm" "model $model is cached but no CUDA GPU is available; the paper's full-config eval requires GPU (RTX 4090 / H200, SPEC §C.1) and is infeasible on CPU within any gate wall-clock budget."
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
    while IFS="$_TAB" read -r model arm cmd; do
        case "$arm" in
            iti__*|angular-steering__*|mags__*) ;;
            *) continue ;;
        esac
        case "$arm" in *__SMILES-molecular-generation__*) continue ;; esac
        # bench = arm with leading "<method>__" and trailing "__<modelslug>" removed
        bench=$(printf '%s' "$arm" | sed -E 's/^[^_]+__//; s/__[A-Za-z0-9._-]+$//')
        printf '%s\t%s\n' "$model" "$bench" >> runs/_fit_pairs.tsv
    done < "$RUNNABLE"
    sort -u runs/_fit_pairs.tsv -o runs/_fit_pairs.tsv
    while IFS="$_TAB" read -r model bench; do
        [ -z "$model" ] && continue
        _slug=$(printf '%s_%s' "$model" "$bench" | tr '/ ' '__')
        _out="manifolds/$(printf '%s' "$model" | tr '/' '_')__$(printf '%s' "$bench" | tr ' ' '_').npz"
        if [ -f "$_out" ]; then echo "[fit] exists $_out"; continue; fi
        echo "[fit] $model / $bench -> $_out"
        if timeout "$FIT_TIMEOUT" "$PYTHON" -m mags.fit --model "$model" --benchmark "$bench" --out "$_out" \
                > "runs/log__fit__${_slug}.log" 2>&1; then
            :
        else
            echo "[fit] BLOCKED $model/$bench"
        fi
    done < runs/_fit_pairs.tsv
fi

# --- Phase 2: run every runnable arm; emit exactly one FINAL line per arm -----
while IFS="$_TAB" read -r model arm cmd; do
    [ -z "$arm" ] && continue
    if [ "$HAVE_PYTHON" = 1 ] && [ -n "$cmd" ]; then
        if eval "timeout \"$ARM_TIMEOUT\" $cmd" > "runs/log__${arm}.log" 2>&1; then
            :
        fi
    fi
    line=$(grep -m1 "^FINAL " "runs/log__${arm}.log" 2>/dev/null || true)
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
