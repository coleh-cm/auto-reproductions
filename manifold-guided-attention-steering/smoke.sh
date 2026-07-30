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
# Offline fast-fail (round-4 gate fix; see run_all_arms.sh): in a no-token sandbox
# with no model cache, online mode hangs on from_pretrained and smoke never prints
# its FINAL line. Offline makes an uncached smoke model fail in <1s -> BLOCKED.
# distilgpt2, once cached, loads under offline=1 too. A host that ran
# `huggingface-cli login` (token file present) is detected and left online so it
# can download distilgpt2.
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
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-10}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-10}"
SMOKE_TIMEOUT="${MAGS_SMOKE_TIMEOUT:-300}"
if ! command -v timeout >/dev/null 2>&1; then timeout() { shift; "$@"; }; fi
if [ -x ".venv/bin/python" ]; then export PATH="$PWD/.venv/bin:$PATH"; fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "FINAL smoke=BLOCKED"
    exit 0
fi
# Run the smoke (bounded); on any failure/timeout emit BLOCKED. `if ...; then` is
# exempt from errexit so a raised exception or timeout can never abort before
# the FINAL line.
# `grep -a` (treat binary as text) is required: smoke.py's captured stdout can
# contain NUL / control bytes (torch + numpy progress / reprs), and plain `grep`
# then prints "Binary file … matches" to stdout instead of the FINAL line —
# observed in round 33, where smoke.sh emitted a grep message, NOT a FINAL line,
# so the deliverable's "exactly one FINAL line" contract was violated. `-a`
# forces text mode so the FINAL line is always extracted; `-m1` keeps one line.
if timeout "$SMOKE_TIMEOUT" "$PYTHON" -m smoke > runs/log__smoke.log 2>&1; then
    grep -a -m1 "^FINAL " runs/log__smoke.log || echo "FINAL smoke=BLOCKED"
else
    grep -a -m1 "^FINAL " runs/log__smoke.log 2>/dev/null || echo "FINAL smoke=BLOCKED"
fi
exit 0
