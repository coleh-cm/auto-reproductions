#!/usr/bin/env bash
# Run every arm of the paper's comparison (Table 1) at the paper's full
# configuration, each printing exactly one line `FINAL <arm name>=<value>`.
#
# Arms (definition of record: arms.json / SPEC.md §7):
#   baseline_ce  -- cross-entropy baseline, lambda = 0
#   cwsd         -- CWSD (the paper's method), lambda = 1
# Both arms are the SAME program under a different --lambda (paper §5),
# seed 0, every other flag at its default (the paper's stated values; the
# one unstated CWSD hyperparameter --s defaults to 0.15, see SPEC §4 item 1).
#
# run_experiment.py prints `FINAL accuracy=<float>` (the paper's output
# contract). This wrapper rewrites that to `FINAL <arm name>=<value>` so the
# arm name the gate checks matches arms.json exactly. If an arm fails to
# produce the contract line, the wrapper exits non-zero (no silent OK on an
# empty result).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a Python interpreter that has numpy + scikit-learn available.
resolve_py() {
    if [ -x "$HERE/.venv/bin/python" ] && "$HERE/.venv/bin/python" -c 'import numpy, sklearn' >/dev/null 2>&1; then
        printf '%s\n' "$HERE/.venv/bin/python"
        return
    fi
    if python3 -c 'import numpy, sklearn' >/dev/null 2>&1; then
        printf 'python3\n'
        return
    fi
    # Fall back: build a venv from the pinned requirements.
    python3 -m venv "$HERE/.venv" >/dev/null
    "$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt" >/dev/null
    printf '%s\n' "$HERE/.venv/bin/python"
}

PY="$(resolve_py)"

# run_arm <arm_name> <args...>
run_arm() {
    local name="$1"; shift
    local line val
    line="$("$PY" "$HERE/run_experiment.py" "$@")"
    case "$line" in
        "FINAL accuracy="*)
            val="${line#FINAL accuracy=}"
            ;;
        *)
            echo "arm '$name' did not print the 'FINAL accuracy=<float>' contract line; got: '$line'" >&2
            exit 1
            ;;
    esac
    printf 'FINAL %s=%s\n' "$name" "$val"
}

run_arm baseline_ce --lambda 0.0
run_arm cwsd      --lambda 1.0
