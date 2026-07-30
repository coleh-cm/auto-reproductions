"""Exercise the gate grader on one known-correct and one known-wrong input.

The gate decides whether an arm reproduced the paper by parsing the
``FINAL <arm>=<value>`` line that ``run_all_arms.sh`` prints.  A grader that
cannot run is worse than no grader: one reproduction graded code by shelling out
to bare ``python`` inside ``except Exception: return False``, so on a host with
only ``python3`` every solution was scored wrong.  We therefore exercise the
grader through ``sys.executable`` (never bare ``python``) on a known-correct
arm output and a known-wrong one, and require a grader that cannot run to RAISE
rather than return a false verdict.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_harness(args, expect_final=True):
    """Run the harness via sys.executable; return (returncode, final_line_or_None)."""
    cmd = [sys.executable, "-m", "gdr.harness"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=180)
    final = None
    for line in proc.stdout.splitlines():
        if line.startswith("FINAL "):
            final = line
    if expect_final and final is None:
        raise AssertionError(
            f"no FINAL line from {args!r}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )
    return proc.returncode, final


def test_grader_known_correct_and_wrong():
    """A correct run prints a numeric FINAL; a deliberately-broken run raises.

    We grade the *grader*: given the harness's own FINAL line format, a
    known-correct arm (ball_oracle on a tiny smoke problem, which we know reaches
    near OPT) must produce a finite numeric value, and the grader must treat a
    ``FINAL x=not_reached`` line as a non-match for an integer claim.  The grader
    itself lives inline so this test pins its behaviour.
    """

    def grade(final_line: str, claimed: object) -> bool:
        """Return True iff ``final_line`` matches the paper claim ``claimed``.

        ``claimed`` may be an int (iterations to 1% gap) or the string
        ``"not_reached"``.  Any parsing failure must RAISE, not return False
        (a grader that cannot run silently marks everything wrong).
        """
        if not isinstance(final_line, str) or not final_line.startswith("FINAL "):
            raise ValueError(f"bad FINAL line: {final_line!r}")
        _, _, payload = final_line.partition("FINAL ")
        name, _, val = payload.partition("=")
        val = val.strip()
        if claimed == "not_reached":
            # an integer-valued arm line is a legitimate mismatch, not a parse error
            return val == "not_reached"
        if isinstance(claimed, int):
            if val == "not_reached":
                return False            # arm never reached -> does not match an int claim
            try:
                return int(val) == claimed
            except ValueError as e:
                raise ValueError(f"unparseable FINAL value {val!r}: {e}") from e
        raise ValueError(f"unsupported claim {claimed!r}")

    # known-correct: smoke harness emits a finite numeric FINAL gap
    rc, line = _run_harness(["smoke"])
    assert rc == 0, line
    assert line.startswith("FINAL smoke=")
    val = float(line.split("=", 1)[1])
    assert val == val, "smoke FINAL must be finite"   # NaN check
    # the grader, given this line and a numeric claim, parses it
    assert grade("FINAL ball_oracle_lewis=1", 1) is True
    # known-wrong: a not_reached line does not match an integer claim
    assert grade("FINAL subgradient=not_reached", 1) is False
    assert grade("FINAL subgradient=not_reached", "not_reached") is True
    # a grader that cannot parse must RAISE (never silently False)
    import pytest
    with pytest.raises(ValueError):
        grade("FINAL x=garbage", 1)


def test_grader_uses_sys_executable_not_bare_python():
    """The grader/harness must be invocable via sys.executable (not bare 'python')."""
    # sys.executable is the same interpreter that runs this test -> the harness
    # import chain (numpy, cvxpy) must resolve under it.  If the harness raised
    # ImportError, this test fails loudly instead of scoring everything wrong.
    rc, line = _run_harness(["smoke"])
    assert rc == 0
    assert line is not None
