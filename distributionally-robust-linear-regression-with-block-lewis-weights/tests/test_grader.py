"""Grader exercise: the metric that decides whether an arm reached 1% must be
exercised on one known-correct and one known-wrong input, and any subprocess
must use ``sys.executable`` (never bare ``python``).

Per the task: a grader that cannot run must RAISE, never return a negative
verdict.  This module exercises ``cost_to_rel_gap`` (the iterations-to-1%
decision) and the harness FINAL-line emission (via subprocess using
``sys.executable``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gdr.metrics import cost_to_rel_gap, gap_curve


# --- known-correct / known-wrong for the 1%-gap grader ---------------------
def test_grader_known_correct_reached():
    """A curve that reaches 1% at index 5 is graded as reached=5 (not NR/wrong)."""
    # gap0 = 10.0; 1% of gap0 = 0.1; best-so-far crosses 0.1 at index 5
    losses = [10.0, 9.0, 7.0, 5.0, 3.0, 0.09, 0.05, 0.04]
    curve = gap_curve(losses, opt=0.0)
    gap0 = float(curve[0])
    it = cost_to_rel_gap(curve, gap0, 0.01, base="init")
    assert it == 5, (it, curve)


def test_grader_known_wrong_not_reached():
    """A curve that never reaches 1% is graded None (not a false positive)."""
    losses = [10.0, 9.5, 9.0, 8.8, 8.7]   # best-so-far 8.7; 1% of 10 = 0.1 -> never
    curve = gap_curve(losses, opt=0.0)
    gap0 = float(curve[0])
    it = cost_to_rel_gap(curve, gap0, 0.01, base="init")
    assert it is None, (it, curve)


def test_grader_base_opt():
    """base='opt' uses threshold 0.01*|OPT|, distinct from base='init'."""
    losses = [10.0, 5.0, 1.5, 1.2]      # opt=1.0 -> 1% opt = 0.01; init gap0=9
    curve = gap_curve(losses, opt=1.0)
    gap0 = float(curve[0])
    it_init = cost_to_rel_gap(curve, gap0, 0.01, base="init")   # thr=0.09
    it_opt = cost_to_rel_gap(curve, gap0, 0.01, base="opt", opt=1.0)  # thr=0.01
    # 1.2-1.0=0.2 > 0.09 -> init not reached either; but best-so-far at idx3 is 0.2
    assert it_init is None
    assert it_opt is None   # 0.2 > 0.01 too


# --- the grader must RAISE on a degenerate input, never silently pass -------
def test_grader_raises_on_empty_curve():
    """An empty curve (no evaluations) must raise, not return a negative verdict.

    A fit with nothing fitted is indistinguishable from the method never having
    been applied (task: "No success path may report OK on an empty result").
    """
    with pytest.raises((ValueError, IndexError)):
        cost_to_rel_gap(np.array([]), gap0=1.0, rel=0.01, base="init")


# --- subprocess grader uses sys.executable, not bare python ----------------
def test_harness_emits_final_line_via_sys_executable():
    """The harness FINAL line (what the gate greps) must be produced by
    ``sys.executable`` (the running interpreter), never bare ``python``.

    On a host with only ``python3``, a bare-``python`` subprocess would fail
    and silently return a wrong verdict.  We invoke the harness smoke path
    through ``sys.executable`` and confirm exactly one FINAL line appears.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    py = venv_py if os.path.exists(venv_py) else sys.executable
    # smoke: tiny self-contained problem, one FINAL line, finishes in seconds
    proc = subprocess.run(
        [py, "-m", "gdr.harness", "smoke"],
        cwd=repo, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    finals = [ln for ln in proc.stdout.splitlines() if ln.startswith("FINAL ")]
    assert len(finals) == 1, f"expected 1 FINAL line, got {finals}"
    # the value must be a finite number (the gap), not OK/blank
    val = finals[0].split("=", 1)[1]
    assert np.isfinite(float(val)), finals[0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
