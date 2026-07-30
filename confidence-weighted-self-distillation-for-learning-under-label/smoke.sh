#!/usr/bin/env bash
# Smoke test: exercise the SAME code path (data -> forward -> target -> loss ->
# grads -> SGD -> evaluate) at a size that finishes in a couple of minutes,
# printing exactly one FINAL line. This proves the path runs; it is NOT
# evidence about the paper and its output must never be reported as a result.
#
# Uses the CWSD arm (lambda = 1) so the target/gate path is exercised, with a
# short step count. The full-config run is run_all_arms.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_py() {
    if [ -x "$HERE/.venv/bin/python" ] && "$HERE/.venv/bin/python" -c 'import numpy, sklearn' >/dev/null 2>&1; then
        printf '%s\n' "$HERE/.venv/bin/python"; return
    fi
    if python3 -c 'import numpy, sklearn' >/dev/null 2>&1; then
        printf 'python3\n'; return
    fi
    python3 -m venv "$HERE/.venv" >/dev/null
    "$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt" >/dev/null
    printf '%s\n' "$HERE/.venv/bin/python"
}

PY="$(resolve_py)"
"$PY" "$HERE/run_experiment.py" --lambda 1.0 --steps 200
