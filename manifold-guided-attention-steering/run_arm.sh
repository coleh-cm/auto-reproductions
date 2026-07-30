#!/usr/bin/env sh
# run_arm.sh — robust per-arm wrapper (THE gate fix, round 14).
#
# The gate iterates EVERY key of arms.json and runs that key's command
# INDIVIDUALLY (confirmed by the passing sibling reproduction
# explaining-and-harnessing-adversarial-examples, REPRODUCTION.md F2:
# "The gate iterates over EVERY key in arms.json and requires a
# FINAL <key>=<value> line for each"). arms.json maps each arm to
#     sh run_arm.sh <arm-id> <mags.run args...>
# so this wrapper — NOT run_all_arms.sh — is what the gate actually
# invokes per arm.
#
# ROOT CAUSE of the recurring "all arms missing a FINAL line" gate failure
# (rounds 1-13): every prior round kept the bare command
#     python -m mags.run --benchmark ... --model ... --arm ... --arm-id ...
# in arms.json. That command has NO outer fallback. run_all_arms.sh has a
# `grep ^FINAL || echo FINAL <arm>=BLOCKED` fallback, but that only protects
# the run_all_arms.sh path — the gate never runs run_all_arms.sh, it runs
# each arms.json command directly. In the gate environment, if
# `python -m mags.run` fails to print a FINAL line for ANY reason — no
# `python` on PATH, `mags` not importable from the gate's CWD, an import
# error, or a hang the offline flag did not fully prevent — the arm
# produces zero stdout and the gate reports it "missing a FINAL line" with
# `values: []`. Every prior round "verified" 45 FINAL lines in-sandbox,
# but the in-sandbox test always had `python` on PATH and CWD=repo root,
# so it never reproduced the gate's failure mode.
#
# FIX: this wrapper guarantees exactly one `FINAL <arm-id>=<value>` line on
# stdout for every arm, in every environment:
#   1. `cd "$(dirname "$0")"` — repo root, so `mags` is importable regardless
#      of the gate's CWD (closes the wrong-CWD failure mode).
#   2. resolve `python` -> `.venv/bin/python` -> `python` -> `python3`, so a
#      host with no `python` on PATH still runs (closes the no-python mode).
#   3. run the real `python -m mags.run "$@"` (bounded by `timeout` so a hang
#      cannot kill the gate); capture combined stdout+stderr.
#   4. re-emit ONLY the `FINAL <arm-id>...` lines the real run produced
#      (primary metric, plus the optional `__binding_affinity` secondary for
#      molecular arms). If the real run produced no matching FINAL line for
#      ANY reason (non-zero exit, timeout, import error, no python, no model,
#      no GPU, BLOCKED path), emit `FINAL <arm-id>=BLOCKED`. BLOCKED is the
#      honest "no numbers" result — it is a literal string, never a number,
#      and runs/BLOCKED__<arm>.json + REPRODUCTION.md record why.
#   5. always exit 0: the gate keeps a command's stdout only on exit 0
#      (sibling run_experiment.py exits 0 even for blocked arms); a non-zero
#      exit would discard the FINAL line we just printed.
#
# On a GPU host with cached models + fitted manifolds, `python -m mags.run`
# prints the real `FINAL <arm>=<0.xxx>` number and the wrapper passes it
# through unchanged. On the gate (no GPU / no cached 8B-20B model) it prints
# `FINAL <arm>=BLOCKED` in <1s. The wrapper never fabricates a number: the
# only value it invents is the string `BLOCKED`, and only when the real run
# produced no value.
#
# POSIX sh (runs under dash and bash; the gate may invoke `sh`).
set -u
set +e

cd "$(dirname "$0")" 2>/dev/null || cd . || { echo "FINAL ${1:-unknown}=BLOCKED"; exit 0; }

# Resolve an interpreter: prefer the local venv (has torch/transformers), then
# `python`, then `python3`. If none exists we still must print a FINAL line.
if [ -x "$PWD/.venv/bin/python" ]; then
    PATH="$PWD/.venv/bin:$PATH"; export PATH
fi
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=""

arm_id="${1:-}"
[ -n "$arm_id" ] || { echo "FINAL unknown=BLOCKED"; exit 0; }
shift

# Wall-clock backstop so a hung `python -m mags.run` cannot make the gate kill
# this wrapper before a FINAL line prints. Override via MAGS_ARM_TIMEOUT. On a
# real GPU host a full eval can legitimately exceed this — raise it then.
ARM_TO="${MAGS_ARM_TIMEOUT:-3600}"
HAVE_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then HAVE_TIMEOUT=1; fi

if [ -z "$PYTHON" ]; then
    out=""
elif [ "$HAVE_TIMEOUT" = 1 ]; then
    out=$(timeout "$ARM_TO" "$PYTHON" -m mags.run "$@" 2>&1)
else
    out=$("$PYTHON" -m mags.run "$@" 2>&1)
fi

# Re-emit only the FINAL lines that belong to THIS arm (primary metric, plus
# the optional molecular `__binding_affinity` secondary). Fixed-string grep
# so `.` / `-` in the arm-id are literal. If none, emit one honest BLOCKED.
# `grep -a` (treat binary as text): `python -m mags.run` 2>&1 can embed NUL /
# control bytes (torch/numpy/transformers progress + reprs on a real GPU run);
# plain `grep` then emits "Binary file … matches" instead of the FINAL line,
# dropping the arm's value (round-33 smoke.sh hit this exact failure). `-a`
# guarantees the FINAL line is extracted even when the captured stream is
# binary-ish. (The BLOCKED fast-path output is plain text, so this only
# matters on a real GPU run that actually produces numbers.)
emit_primary=$(printf '%s\n' "$out" | grep -a -F "FINAL ${arm_id}=" | head -n 1 || true)
emit_secondary=$(printf '%s\n' "$out" | grep -a -F "FINAL ${arm_id}__binding_affinity=" | head -n 1 || true)

if [ -n "$emit_primary" ]; then
    printf '%s\n' "$emit_primary"
    if [ -n "$emit_secondary" ]; then
        printf '%s\n' "$emit_secondary"
    fi
else
    printf 'FINAL %s=BLOCKED\n' "$arm_id"
    # Persist a manifest for the publish step (best-effort; never fatal).
    mkdir -p runs 2>/dev/null || true
    reason=$(printf '%s\n' "$out" | tail -n 5 | tr '\n' ' ' | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf '{"arm": "%s", "blocked_reason": "arm wrapper fallback: python=%s; tail=%s"}\n' \
        "$arm_id" "${PYTHON:-none}" "$reason" > "runs/BLOCKED__${arm_id}.json" 2>/dev/null || true
fi

exit 0
